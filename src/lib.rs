//! # vmem
//!
//! Read and write the memory of another process, resolve multi-level pointer
//! chains, AOB/signature-scan, patch code, and inject Cheat-Engine-style hooks
//! — on Linux, from safe Rust.
//!
//! Cross-process reads and writes use `process_vm_readv` / `process_vm_writev`
//! (no `ptrace`-stop required); code patching falls back to `/proc/<pid>/mem`
//! so even read-only `.text` pages can be modified; and remote allocation and
//! hooking briefly `ptrace`-attach to inject an `mmap`.
//!
//! ```no_run
//! use vmem::Process;
//!
//! let proc = Process::by_name("game")?;
//! let module = proc.module("game")?;
//!
//! // Cheat Engine pointer:  "game"+0x10F2A30 -> 0x10 -> 0x8 -> 0x0
//! let hp = proc
//!     .pointer(module.base + 0x10F2A30)
//!     .offsets(&[0x10, 0x8, 0x0])
//!     .read::<i32>()?;
//! println!("HP = {hp}");
//! # Ok::<(), vmem::Error>(())
//! ```
//!
//! ## Feature tour
//! * [`Process`] — a cheap, `Copy` handle to a target; module bases are
//!   resolved on demand from `/proc/<pid>/maps`.
//! * Typed [`Process::read`] / [`Process::write`] are sound: the `T: Pod`
//!   bound (from `bytemuck`) guarantees every bit pattern is valid, so there
//!   is no UB on padding or invalid discriminants.
//! * [`Pointer`] — a fluent pointer-chain resolver with an explicit,
//!   documented dereference convention (and a toggle for the other one).
//! * [`Scatter`] — reads many disjoint addresses in a **single syscall** by
//!   handing the kernel multiple `iovec`s.
//! * [`Pattern`] / [`Process::scan`] — AOB / signature scanning with `??`
//!   wildcards, plus a RIP-relative resolver ([`Process::resolve_rip`]).
//! * [`Patch`] — reversible byte/code patches that revert on drop.
//! * [`Asm`] and the [`asm64!`](crate::asm64) macro — a focused x86-64 encoder
//!   for cave code.
//! * [`RemoteMem`] / [`Process::hook`] — remote allocation and detour hooks.
//!
//! ## Permissions
//! Every cross-process operation needs the right to `ptrace` the target: the
//! same UID with `/proc/sys/kernel/yama/ptrace_scope = 0`, the `cap_sys_ptrace`
//! capability, or root. Insufficient rights surface as [`Error::Permission`].
//!
//! ## A note on danger
//! Writing another process's memory — and especially [patching
//! code](Process::patch) or [installing hooks](Process::hook) — can corrupt or
//! crash the target. These operations are memory-safe for *this* process (no
//! Rust UB), which is why the API is safe, but they are inherently unsafe in
//! *effect*. Test against a process you own.
//!
//! ## Portability
//! The crate is Linux-only. On Windows the shape is identical: swap the backend
//! behind [`Process::read_bytes`] / [`Process::write_bytes`] for `OpenProcess`
//! with `ReadProcessMemory` / `WriteProcessMemory`, and everything above
//! ([`Pointer`], [`Scatter`], [`Pattern`], [`Asm`]) is unchanged.

#![cfg(target_os = "linux")]
#![recursion_limit = "512"]
#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

mod asm;
mod inject;
pub use asm::{Asm, AsmError, Reg};
pub use inject::{Hook, RemoteMem, prot};

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};

use bytemuck::Pod;

/// Errors produced by this crate.
///
/// This enum is `#[non_exhaustive]`: matching on it must include a `_` arm so
/// that future additions are not breaking changes.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// No process matched the requested name or pid.
    #[error("process '{0}' not found")]
    ProcessNotFound(String),

    /// The named module is not mapped in the target process.
    #[error("module '{module}' not found in pid {pid}")]
    ModuleNotFound {
        /// The module basename that was searched for.
        module: String,
        /// The target pid that was searched.
        pid: i32,
    },

    /// `EPERM`/`EACCES`. Usually Yama: see
    /// `/proc/sys/kernel/yama/ptrace_scope`, grant `cap_sys_ptrace`, or elevate.
    #[error(
        "permission denied accessing pid {pid} \
             (check /proc/sys/kernel/yama/ptrace_scope or grant cap_sys_ptrace)"
    )]
    Permission {
        /// The target pid that could not be accessed.
        pid: i32,
    },

    /// The address (or part of the range) is not mapped in the target.
    #[error("address {addr:#x} (+{len}) is not fully mapped in the target")]
    Unmapped {
        /// Start of the offending range.
        addr: usize,
        /// Length of the offending range, in bytes.
        len: usize,
    },

    /// The kernel transferred fewer bytes than requested (a short read/write,
    /// typically because the range straddles a mapped/unmapped boundary).
    #[error("partial transfer at {addr:#x}: wanted {wanted} bytes, moved {moved}")]
    Partial {
        /// Start of the transfer.
        addr: usize,
        /// Bytes requested.
        wanted: usize,
        /// Bytes actually moved.
        moved: usize,
    },

    /// A signature scan found no match.
    #[error("pattern '{0}' not found")]
    PatternNotFound(String),

    /// A `rel32` relative branch cannot reach its target (> ±2 GiB); use an
    /// absolute jump instead.
    #[error("rel32 from {from:#x} to {to:#x} is out of range; use an absolute jmp")]
    BranchTooFar {
        /// Source address of the branch.
        from: usize,
        /// Destination address of the branch.
        to: usize,
    },

    /// The region to patch is smaller than the encoding requires.
    #[error("patch region of {got} byte(s) is too small; need at least {need}")]
    PatchTooSmall {
        /// Minimum number of bytes required.
        need: usize,
        /// Number of bytes available.
        got: usize,
    },

    /// An underlying OS/I/O error not covered by a more specific variant.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Crate result alias.
///
/// The error type defaults to [`Error`] but is a parameter, so a call site can
/// pin a more precise error when needed.
pub type Result<T, E = Error> = std::result::Result<T, E>;

// ---------------------------------------------------------------------------
// Process discovery
// ---------------------------------------------------------------------------

/// A handle to a target process. Cloning is cheap (just the pid).
#[derive(Clone, Copy, Debug)]
pub struct Process {
    pid: i32,
}

impl Process {
    /// Wrap an existing pid, verifying it exists.
    ///
    /// Existence is checked once, via `/proc/<pid>`; the pid may still die
    /// afterwards (a later operation then fails with [`Error::Unmapped`] or
    /// [`Error::Io`]).
    ///
    /// # Errors
    /// [`Error::ProcessNotFound`] if `/proc/<pid>` does not exist.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// assert_eq!(proc.pid(), 1234);
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn by_pid(pid: i32) -> Result<Self> {
        if fs::metadata(format!("/proc/{pid}")).is_ok() {
            Ok(Self { pid })
        } else {
            Err(Error::ProcessNotFound(pid.to_string()))
        }
    }

    /// First process whose `comm` (or `/proc/<pid>/cmdline` basename) matches.
    ///
    /// `comm` is truncated to 15 bytes by the kernel, so for long executable
    /// names we also check the `cmdline` basename. When several processes
    /// match, the lowest pid is returned (see [`all_by_name`](Self::all_by_name)
    /// for every match).
    ///
    /// # Errors
    /// [`Error::ProcessNotFound`] if nothing matches.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_name("game")?;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn by_name(name: &str) -> Result<Self> {
        Self::all_by_name(name)
            .into_iter()
            .next()
            .map(|pid| Self { pid })
            .ok_or_else(|| Error::ProcessNotFound(name.to_string()))
    }

    /// Every pid matching `name`, ascending.
    ///
    /// Matching follows the same `comm`/`cmdline`-basename rule as
    /// [`by_name`](Self::by_name). Returns an empty vector if `/proc` cannot be
    /// read or nothing matches (this is an inventory, not a fallible lookup).
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// for pid in Process::all_by_name("chrome") {
    ///     println!("chrome pid {pid}");
    /// }
    /// ```
    pub fn all_by_name(name: &str) -> Vec<i32> {
        let mut pids = Vec::new();
        let Ok(dir) = fs::read_dir("/proc") else {
            return pids;
        };
        for entry in dir.flatten() {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|s| s.parse::<i32>().ok())
            else {
                continue;
            };
            let comm = fs::read_to_string(format!("/proc/{pid}/comm"));
            let comm_match = comm.as_deref().map(str::trim).ok() == Some(name);
            let cmd_match = || {
                fs::read(format!("/proc/{pid}/cmdline"))
                    .ok()
                    .and_then(|b| {
                        let first = b.split(|&c| c == 0).next()?.to_vec();
                        let s = String::from_utf8(first).ok()?;
                        Some(s.rsplit('/').next().unwrap_or(&s).to_owned())
                    })
                    .as_deref()
                    == Some(name)
            };
            if comm_match || cmd_match() {
                pids.push(pid);
            }
        }
        pids.sort_unstable();
        pids
    }

    /// The target's pid.
    #[inline]
    pub fn pid(&self) -> i32 {
        self.pid
    }
}

// ---------------------------------------------------------------------------
// Memory maps / modules
// ---------------------------------------------------------------------------

/// One line of `/proc/<pid>/maps`: a single contiguous virtual-memory region.
#[derive(Clone, Debug)]
pub struct MapRegion {
    /// First address of the region (inclusive).
    pub start: usize,
    /// One past the last address of the region (exclusive).
    pub end: usize,
    /// The four permission characters, e.g. `"r-xp"` (read, write, execute,
    /// private/shared).
    pub perms: String,
    /// Backing file path, if the region is file-backed; `None` for anonymous
    /// mappings and special regions like `[heap]` or `[stack]`.
    pub path: Option<String>,
}

impl MapRegion {
    /// Whether the region is readable (`r` permission).
    #[inline]
    pub fn readable(&self) -> bool {
        self.perms.as_bytes().first() == Some(&b'r')
    }
    /// Whether the region is writable (`w` permission).
    #[inline]
    pub fn writable(&self) -> bool {
        self.perms.as_bytes().get(1) == Some(&b'w')
    }
    /// Whether the region is executable (`x` permission).
    #[inline]
    pub fn executable(&self) -> bool {
        self.perms.as_bytes().get(2) == Some(&b'x')
    }
    /// Length of the region in bytes (`end - start`).
    #[inline]
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    /// Whether the region is empty (`start == end`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// A loaded module (executable or shared object): its base, total span, path.
#[derive(Clone, Debug)]
pub struct Module {
    /// The module's file basename, e.g. `"libc.so.6"`.
    pub name: String,
    /// Lowest mapped address of the module.
    pub base: usize,
    /// `base .. base+size` covers every contiguous mapping of the file.
    pub size: usize,
    /// The module's full path on disk.
    pub path: String,
}

impl Module {
    /// Whether `addr` falls within `base .. base+size`.
    #[inline]
    pub fn contains(&self, addr: usize) -> bool {
        (self.base..self.base.saturating_add(self.size)).contains(&addr)
    }
}

impl Process {
    /// Parse `/proc/<pid>/maps` into one [`MapRegion`] per line.
    ///
    /// # Errors
    /// [`Error::Io`] if the maps file cannot be read (e.g. the process exited).
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// for region in proc.maps()? {
    ///     if region.executable() {
    ///         println!("{:#x}..{:#x} {}", region.start, region.end, region.perms);
    ///     }
    /// }
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn maps(&self) -> Result<Vec<MapRegion>> {
        let text = fs::read_to_string(format!("/proc/{}/maps", self.pid))?;
        let mut out = Vec::new();
        for line in text.lines() {
            // start-end perms offset dev inode pathname
            let mut p = line
                .splitn(6, char::is_whitespace)
                .filter(|s| !s.is_empty());
            let Some(range) = p.next() else { continue };
            let perms = p.next().unwrap_or("").to_string();
            // skip offset, dev, inode
            let (_off, _dev, _inode) = (p.next(), p.next(), p.next());
            let path = p
                .next()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let Some((s, e)) = range.split_once('-') else {
                continue;
            };
            let (Ok(start), Ok(end)) = (usize::from_str_radix(s, 16), usize::from_str_radix(e, 16))
            else {
                continue;
            };
            out.push(MapRegion {
                start,
                end,
                perms,
                path,
            });
        }
        Ok(out)
    }

    /// Resolve a module by file basename (e.g. `"libc.so.6"`, `"game"`).
    ///
    /// The returned [`Module`] spans from the lowest to the highest address of
    /// every mapping backed by that file, so `base .. base+size` covers all of
    /// its segments.
    ///
    /// # Errors
    /// [`Error::ModuleNotFound`] if no mapping is backed by a file with that
    /// basename, or [`Error::Io`] if the maps file cannot be read.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_name("game")?;
    /// let m = proc.module("game")?;
    /// println!("{} base={:#x} size={:#x}", m.name, m.base, m.size);
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn module(&self, name: &str) -> Result<Module> {
        let maps = self.maps()?;
        let mut base: Option<usize> = None;
        let mut end = 0usize;
        let mut path = String::new();
        for m in &maps {
            let Some(p) = &m.path else { continue };
            let matches = p.rsplit('/').next() == Some(name);
            if matches {
                base = Some(base.map_or(m.start, |b| b.min(m.start)));
                end = end.max(m.end);
                path = p.clone();
            }
        }
        match base {
            Some(base) => Ok(Module {
                name: name.to_string(),
                base,
                size: end - base,
                path,
            }),
            None => Err(Error::ModuleNotFound {
                module: name.to_string(),
                pid: self.pid,
            }),
        }
    }

    /// Every distinct file-backed module (by path) currently mapped, ascending
    /// by path.
    ///
    /// Anonymous regions and special maps (`[heap]`, `[stack]`, …) are skipped.
    ///
    /// # Errors
    /// [`Error::Io`] if the maps file cannot be read.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// for m in proc.modules()? {
    ///     println!("{:#x} {}", m.base, m.path);
    /// }
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn modules(&self) -> Result<Vec<Module>> {
        let maps = self.maps()?;
        let mut acc: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        for m in &maps {
            let Some(p) = &m.path else { continue };
            if !p.starts_with('/') {
                continue; // skip [heap], [stack], anon
            }
            let e = acc.entry(p.clone()).or_insert((m.start, m.end));
            e.0 = e.0.min(m.start);
            e.1 = e.1.max(m.end);
        }
        Ok(acc
            .into_iter()
            .map(|(path, (base, end))| Module {
                name: path.rsplit('/').next().unwrap_or(&path).to_string(),
                base,
                size: end - base,
                path,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Raw I/O backend
// ---------------------------------------------------------------------------

impl Process {
    fn classify(&self, addr: usize, len: usize, errno: i32) -> Error {
        match errno {
            libc::EPERM | libc::EACCES => Error::Permission { pid: self.pid },
            libc::EFAULT | libc::ENOMEM | libc::EIO => Error::Unmapped { addr, len },
            other => Error::Io(std::io::Error::from_raw_os_error(other)),
        }
    }

    /// Read `buf.len()` bytes from `addr` into `buf`.
    ///
    /// Uses `process_vm_readv` — one syscall, no `ptrace`-stop. Reading zero
    /// bytes always succeeds.
    ///
    /// # Errors
    /// [`Error::Permission`] if `ptrace` access is denied, [`Error::Unmapped`]
    /// if the range is not mapped, [`Error::Partial`] on a short transfer, or
    /// [`Error::Io`] for any other OS error.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let mut buf = [0u8; 16];
    /// proc.read_bytes(0x55_5555_0000, &mut buf)?;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn read_bytes(&self, addr: usize, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let local = libc::iovec {
            iov_base: buf.as_mut_ptr().cast(),
            iov_len: buf.len(),
        };
        let remote = libc::iovec {
            iov_base: addr as *mut _,
            iov_len: buf.len(),
        };
        // SAFETY: both iovecs describe valid, correctly-sized buffers; the
        // local one is uniquely borrowed for the duration of the call.
        let n = unsafe { libc::process_vm_readv(self.pid, &local, 1, &remote, 1, 0) };
        if n < 0 {
            return Err(self.classify(addr, buf.len(), errno()));
        }
        if n as usize != buf.len() {
            return Err(Error::Partial {
                addr,
                wanted: buf.len(),
                moved: n as usize,
            });
        }
        Ok(())
    }

    /// Write `buf` to `addr`.
    ///
    /// Uses `process_vm_writev`. This path **cannot** write to read-only pages
    /// (e.g. `.text`); use [`write_force`](Self::write_force) or
    /// [`write_bytes_mem`](Self::write_bytes_mem) for those. Writing zero bytes
    /// always succeeds.
    ///
    /// # Errors
    /// [`Error::Permission`] if `ptrace` access is denied, [`Error::Unmapped`]
    /// if the range is not mapped (or not writable), [`Error::Partial`] on a
    /// short transfer, or [`Error::Io`] for any other OS error.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// proc.write_bytes(0x55_5555_0000, &[0xDE, 0xAD])?;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn write_bytes(&self, addr: usize, buf: &[u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let local = libc::iovec {
            iov_base: buf.as_ptr() as *mut _,
            iov_len: buf.len(),
        };
        let remote = libc::iovec {
            iov_base: addr as *mut _,
            iov_len: buf.len(),
        };
        // SAFETY: see read_bytes; the local buffer is only read.
        let n = unsafe { libc::process_vm_writev(self.pid, &local, 1, &remote, 1, 0) };
        if n < 0 {
            return Err(self.classify(addr, buf.len(), errno()));
        }
        if n as usize != buf.len() {
            return Err(Error::Partial {
                addr,
                wanted: buf.len(),
                moved: n as usize,
            });
        }
        Ok(())
    }

    /// Fallback read path via `/proc/<pid>/mem`.
    ///
    /// Same permission rules as the `vm_*` syscalls but works when those are
    /// unavailable (old kernels, seccomp filters). Slower: one syscall per
    /// call, plus the open.
    ///
    /// # Errors
    /// [`Error::Permission`] if the mem file cannot be opened, [`Error::Unmapped`]
    /// if the range is not mapped, or [`Error::Io`] for any other OS error.
    pub fn read_bytes_mem(&self, addr: usize, buf: &mut [u8]) -> Result<()> {
        let mut f = fs::File::open(format!("/proc/{}/mem", self.pid)).map_err(|e| {
            match e.raw_os_error() {
                Some(libc::EACCES) | Some(libc::EPERM) => Error::Permission { pid: self.pid },
                _ => Error::Io(e),
            }
        })?;
        f.seek(SeekFrom::Start(addr as u64))?;
        f.read_exact(buf).map_err(|e| match e.raw_os_error() {
            Some(libc::EIO) => Error::Unmapped {
                addr,
                len: buf.len(),
            },
            _ => Error::Io(e),
        })
    }

    /// Write via `/proc/<pid>/mem`.
    ///
    /// Unlike [`write_bytes`](Self::write_bytes) (which goes through
    /// `process_vm_writev` and **cannot** touch read-only pages), this path
    /// uses the kernel's `FOLL_FORCE` semantics and can modify read-only
    /// mappings such as `.text` — exactly how a debugger writes a breakpoint.
    /// This is what makes code patching work.
    ///
    /// # Errors
    /// [`Error::Permission`] if the mem file cannot be opened for writing,
    /// [`Error::Unmapped`] if the range is not mapped, or [`Error::Io`] for any
    /// other OS error.
    pub fn write_bytes_mem(&self, addr: usize, buf: &[u8]) -> Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .open(format!("/proc/{}/mem", self.pid))
            .map_err(|e| match e.raw_os_error() {
                Some(libc::EACCES) | Some(libc::EPERM) => Error::Permission { pid: self.pid },
                _ => Error::Io(e),
            })?;
        f.seek(SeekFrom::Start(addr as u64))?;
        f.write_all(buf).map_err(|e| match e.raw_os_error() {
            Some(libc::EIO) | Some(libc::EFAULT) => Error::Unmapped {
                addr,
                len: buf.len(),
            },
            _ => Error::Io(e),
        })
    }

    /// Write that succeeds even on read-only code pages.
    ///
    /// Tries the fast `process_vm_writev` path first, then falls back to
    /// `/proc/<pid>/mem` when the target page isn't writable (the `EFAULT` ->
    /// [`Error::Unmapped`] case). All patching goes through here.
    ///
    /// # Errors
    /// Whatever [`write_bytes`](Self::write_bytes) or
    /// [`write_bytes_mem`](Self::write_bytes_mem) return.
    pub fn write_force(&self, addr: usize, buf: &[u8]) -> Result<()> {
        match self.write_bytes(addr, buf) {
            Err(Error::Unmapped { .. }) => self.write_bytes_mem(addr, buf),
            other => other,
        }
    }

    /// Typed read. `T: Pod` guarantees the bytes are a valid `T`.
    ///
    /// Works for any plain-old-data type — integers, floats, fixed arrays, and
    /// `#[derive(Pod)]` structs.
    ///
    /// # Errors
    /// Whatever [`read_bytes`](Self::read_bytes) returns.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let hp: i32 = proc.read(0x55_5555_0000)?;
    /// let pos: [f32; 3] = proc.read(0x55_5555_0010)?;
    /// # let _ = (hp, pos);
    /// # Ok::<(), vmem::Error>(())
    /// ```
    #[inline]
    pub fn read<T: Pod>(&self, addr: usize) -> Result<T> {
        let mut val = T::zeroed();
        self.read_bytes(addr, bytemuck::bytes_of_mut(&mut val))?;
        Ok(val)
    }

    /// Typed write.
    ///
    /// # Errors
    /// Whatever [`write_bytes`](Self::write_bytes) returns.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// proc.write::<i32>(0x55_5555_0000, 9999)?;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    #[inline]
    pub fn write<T: Pod>(&self, addr: usize, val: T) -> Result<()> {
        self.write_bytes(addr, bytemuck::bytes_of(&val))
    }

    /// Read `len` bytes into a fresh `Vec`.
    ///
    /// # Errors
    /// Whatever [`read_bytes`](Self::read_bytes) returns.
    pub fn read_vec(&self, addr: usize, len: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; len];
        self.read_bytes(addr, &mut v)?;
        Ok(v)
    }

    /// Read a NUL-terminated string (up to `max` bytes), lossily decoded as
    /// UTF-8.
    ///
    /// The string is read incrementally in small chunks and the scan stops at
    /// the first NUL, at `max` bytes, or at the first unreadable chunk —
    /// whichever comes first. Reading in chunks means a short string near the
    /// end of a mapping is returned successfully even though a single
    /// `max`-byte read would have run off into unmapped space.
    ///
    /// # Errors
    /// [`Error::Permission`], [`Error::Unmapped`], or [`Error::Io`] — but only
    /// if the *very first* chunk is unreadable. Once any bytes have been read,
    /// a later unreadable chunk simply ends the string instead of erroring.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let name = proc.read_cstring(0x55_5555_0000, 64)?;
    /// println!("{name}");
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn read_cstring(&self, addr: usize, max: usize) -> Result<String> {
        const CHUNK: usize = 64;
        let mut out = Vec::new();
        let mut cur = addr;
        let mut remaining = max;
        while remaining > 0 {
            let want = CHUNK.min(remaining);
            let mut buf = vec![0u8; want];
            match self.read_bytes(cur, &mut buf) {
                Ok(()) => {}
                // Already captured a prefix: stop here instead of failing.
                Err(Error::Unmapped { .. }) | Err(Error::Partial { .. }) if !out.is_empty() => {
                    break;
                }
                Err(e) => return Err(e),
            }
            if let Some(pos) = buf.iter().position(|&b| b == 0) {
                out.extend_from_slice(&buf[..pos]);
                return Ok(String::from_utf8_lossy(&out).into_owned());
            }
            out.extend_from_slice(&buf);
            cur = cur.wrapping_add(want);
            remaining -= want;
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// Start a fluent pointer chain rooted at `base` (an *address*, e.g.
    /// `module.base + static_offset`).
    ///
    /// See [`Pointer`] for the dereference convention.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_name("game")?;
    /// let base = proc.module("game")?.base;
    /// let hp = proc.pointer(base + 0x10F2A30).offsets(&[0x10, 0x8, 0x0]).read::<i32>()?;
    /// # let _ = hp;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    #[inline]
    pub fn pointer(&self, base: usize) -> Pointer<'_> {
        Pointer {
            proc: self,
            base,
            offsets: Vec::new(),
            offset_first: false,
        }
    }

    /// Begin a batched scatter read (one syscall, many addresses).
    ///
    /// See [`Scatter`].
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let mut s = proc.scatter();
    /// let a = s.add_typed::<i32>(0x1000);
    /// let b = s.add_typed::<i32>(0x2000);
    /// let out = s.run()?;
    /// let (va, vb): (i32, i32) = (vmem::pod_at(&out, a), vmem::pod_at(&out, b));
    /// # let _ = (va, vb);
    /// # Ok::<(), vmem::Error>(())
    /// ```
    #[inline]
    pub fn scatter(&self) -> Scatter<'_> {
        Scatter {
            proc: self,
            items: Vec::new(),
        }
    }
}

#[inline]
fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Pointer chains
// ---------------------------------------------------------------------------

/// A multi-level pointer chain.
///
/// Default (Cheat-Engine) convention, for `base` + `[o0, o1, .., oN]`:
///
/// ```text
/// addr = base
/// for off in offsets:        // deref, THEN add offset
///     addr = read_usize(addr)
///     addr += off
/// // `addr` is the address of the final value; `.read()` dereferences once more
/// ```
///
/// This matches Cheat Engine when you fold the static offset into `base` and
/// list the CE offsets top-to-bottom. If your notes use the *other* convention
/// (add the offset *before* each deref), call [`Pointer::offset_first`].
///
/// A `Pointer` is a lazy builder: nothing is read until you call
/// [`resolve`](Self::resolve), [`read`](Self::read), or [`write`](Self::write).
#[derive(Clone, Debug)]
#[must_use = "a Pointer is inert until you call .resolve(), .read(), or .write()"]
pub struct Pointer<'p> {
    proc: &'p Process,
    base: usize,
    offsets: Vec<usize>,
    offset_first: bool,
}

impl<'p> Pointer<'p> {
    /// Append one offset (chainable).
    #[inline]
    pub fn offset(mut self, off: usize) -> Self {
        self.offsets.push(off);
        self
    }

    /// Append many offsets (chainable).
    #[inline]
    pub fn offsets(mut self, offs: &[usize]) -> Self {
        self.offsets.extend_from_slice(offs);
        self
    }

    /// Switch to the "add offset before each dereference" convention.
    #[inline]
    pub fn offset_first(mut self) -> Self {
        self.offset_first = true;
        self
    }

    /// Resolve to the final **address** (without reading the value there).
    ///
    /// * default: `addr = deref(addr); addr += off` for each offset.
    /// * [`offset_first`](Self::offset_first): `addr += off; deref` for each
    ///   offset, skipping the trailing deref so the result is an address.
    ///
    /// Pointer arithmetic wraps (never panics); a dangling link surfaces when
    /// the intermediate deref reads unmapped memory.
    ///
    /// # Errors
    /// Whatever [`Process::read`] returns for an intermediate dereference
    /// (commonly [`Error::Unmapped`] when a link is stale).
    pub fn resolve(&self) -> Result<usize> {
        let mut addr = self.base;
        let last = self.offsets.len().wrapping_sub(1);
        for (i, &off) in self.offsets.iter().enumerate() {
            if self.offset_first {
                addr = addr.wrapping_add(off);
                if i != last {
                    addr = self.proc.read::<usize>(addr)?;
                }
            } else {
                addr = self.proc.read::<usize>(addr)?;
                addr = addr.wrapping_add(off);
            }
        }
        Ok(addr)
    }

    /// Resolve and read the value at the end of the chain.
    ///
    /// # Errors
    /// Whatever [`resolve`](Self::resolve) or [`Process::read`] return.
    #[inline]
    pub fn read<T: Pod>(&self) -> Result<T> {
        self.proc.read(self.resolve()?)
    }

    /// Resolve and write a value at the end of the chain.
    ///
    /// # Errors
    /// Whatever [`resolve`](Self::resolve) or [`Process::write`] return.
    #[inline]
    pub fn write<T: Pod>(&self, val: T) -> Result<()> {
        self.proc.write(self.resolve()?, val)
    }
}

// ---------------------------------------------------------------------------
// Batched (scatter/gather) reads
// ---------------------------------------------------------------------------

/// Collects several independent reads and issues them in one (or, past
/// `IOV_MAX`, a few) `process_vm_readv` syscall(s).
///
/// Queue reads with [`add`](Self::add) / [`add_typed`](Self::add_typed) — each
/// returns an index — then call [`run`](Self::run) and pull values back out
/// with [`pod_at`].
#[derive(Debug)]
#[must_use = "a Scatter is inert until you call .run()"]
pub struct Scatter<'p> {
    proc: &'p Process,
    items: Vec<(usize, usize)>, // (addr, len)
}

impl<'p> Scatter<'p> {
    /// Queue a read of `len` bytes at `addr`; returns its result index.
    pub fn add(&mut self, addr: usize, len: usize) -> usize {
        self.items.push((addr, len));
        self.items.len() - 1
    }

    /// Queue a typed read of `size_of::<T>()` bytes; returns its result index.
    ///
    /// Pair with [`pod_at::<T>`](pod_at) on the result to recover the value.
    pub fn add_typed<T: Pod>(&mut self, addr: usize) -> usize {
        self.add(addr, std::mem::size_of::<T>())
    }

    /// Execute the batched read. Returns one byte buffer per queued read, in
    /// the order they were added.
    ///
    /// Queues larger than `IOV_MAX` (1024) are split across several syscalls
    /// automatically.
    ///
    /// # Errors
    /// [`Error::Permission`], [`Error::Unmapped`], [`Error::Partial`], or
    /// [`Error::Io`] — reported against the first address of the failing chunk.
    /// Because the kernel reports only a total byte count, a partial transfer
    /// cannot be attributed to an exact slot.
    pub fn run(self) -> Result<Vec<Vec<u8>>> {
        const IOV_MAX: usize = 1024;
        let mut bufs: Vec<Vec<u8>> = self.items.iter().map(|&(_, len)| vec![0u8; len]).collect();

        for chunk_start in (0..self.items.len()).step_by(IOV_MAX) {
            let chunk_end = (chunk_start + IOV_MAX).min(self.items.len());
            let range = chunk_start..chunk_end;

            let locals: Vec<libc::iovec> = bufs[range.clone()]
                .iter_mut()
                .map(|b| libc::iovec {
                    iov_base: b.as_mut_ptr().cast(),
                    iov_len: b.len(),
                })
                .collect();
            let remotes: Vec<libc::iovec> = self.items[range.clone()]
                .iter()
                .map(|&(addr, len)| libc::iovec {
                    iov_base: addr as *mut _,
                    iov_len: len,
                })
                .collect();
            let want: usize = self.items[range.clone()].iter().map(|&(_, l)| l).sum();

            // SAFETY: locals/remotes are valid for the call; each local buffer
            // is uniquely borrowed via bufs and lives until after the syscall.
            let n = unsafe {
                libc::process_vm_readv(
                    self.proc.pid,
                    locals.as_ptr(),
                    locals.len() as libc::c_ulong,
                    remotes.as_ptr(),
                    remotes.len() as libc::c_ulong,
                    0,
                )
            };
            if n < 0 {
                let (addr, len) = self.items[chunk_start];
                return Err(self.proc.classify(addr, len, errno()));
            }
            if n as usize != want {
                let (addr, _) = self.items[chunk_start];
                return Err(Error::Partial {
                    addr,
                    wanted: want,
                    moved: n as usize,
                });
            }
        }
        Ok(bufs)
    }
}

/// Interpret a scatter result slot as a `T`.
///
/// The read is unaligned, so the slot need not be aligned for `T`.
///
/// # Panics
/// Panics if `index` is out of bounds, or if the slot's length is not exactly
/// `size_of::<T>()` (i.e. it was not queued with the matching size — prefer
/// [`Scatter::add_typed`]).
///
/// # Examples
/// ```
/// let bufs = vec![1u32.to_le_bytes().to_vec()];
/// let v: u32 = vmem::pod_at(&bufs, 0);
/// assert_eq!(v, 1);
/// ```
pub fn pod_at<T: Pod>(bufs: &[Vec<u8>], index: usize) -> T {
    bytemuck::pod_read_unaligned(&bufs[index])
}

// ---------------------------------------------------------------------------
// AoB / signature scanning
// ---------------------------------------------------------------------------

/// A byte signature with wildcards, e.g. parsed from `"48 8B ?? 89 ?? ??"`.
///
/// Each position is either a concrete byte or a wildcard that matches any byte.
/// Build one with [`parse`](Self::parse) (IDA-style hex string) or
/// [`from_mask`](Self::from_mask) (bytes + `"xx?x"`-style mask).
#[derive(Clone, Debug)]
pub struct Pattern(Vec<Option<u8>>);

impl Pattern {
    /// Parse an IDA-style pattern: hex bytes separated by whitespace, with
    /// `??`, `?`, `**`, or `*` for a wildcard byte.
    ///
    /// # Errors
    /// [`Error::Io`] (`InvalidInput`) if a token is neither a wildcard nor a
    /// valid two-digit hex byte.
    ///
    /// # Examples
    /// ```
    /// use vmem::Pattern;
    /// let p = Pattern::parse("48 8B ?? 89")?;
    /// assert_eq!(p.len(), 4);
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn parse(sig: &str) -> Result<Self> {
        let mut bytes = Vec::new();
        for tok in sig.split_whitespace() {
            if matches!(tok, "??" | "?" | "**" | "*") {
                bytes.push(None);
            } else {
                let b = u8::from_str_radix(tok, 16).map_err(|_| {
                    Error::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("bad pattern token '{tok}'"),
                    ))
                })?;
                bytes.push(Some(b));
            }
        }
        Ok(Pattern(bytes))
    }

    /// Build from a code-style pattern + mask.
    ///
    /// `x` (or `X`) in the mask means "this byte must match"; any other
    /// character (conventionally `?`) is a wildcard, and the corresponding
    /// placeholder byte in `bytes` is ignored.
    ///
    /// # Errors
    /// [`Error::Io`] (`InvalidInput`) if `bytes` and `mask` differ in length.
    ///
    /// # Examples
    /// ```
    /// use vmem::Pattern;
    /// let p = Pattern::from_mask(b"\x48\x8B\x00\x89", "xx?x")?;
    /// assert_eq!(p.len(), 4);
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn from_mask(bytes: &[u8], mask: &str) -> Result<Self> {
        if bytes.len() != mask.len() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "pattern and mask differ in length",
            )));
        }
        Ok(Pattern(
            bytes
                .iter()
                .zip(mask.chars())
                .map(|(&b, m)| if m == 'x' || m == 'X' { Some(b) } else { None })
                .collect(),
        ))
    }

    /// Number of bytes (concrete + wildcard) in the pattern.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Whether the pattern has no bytes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    fn matches_at(&self, hay: &[u8], i: usize) -> bool {
        self.0
            .iter()
            .enumerate()
            .all(|(j, pat)| pat.is_none_or(|b| hay[i + j] == b))
    }

    fn first_match(&self, hay: &[u8]) -> Option<usize> {
        let n = self.0.len();
        if n == 0 || hay.len() < n {
            return None;
        }
        (0..=hay.len() - n).find(|&i| self.matches_at(hay, i))
    }

    fn all_matches(&self, hay: &[u8]) -> Vec<usize> {
        let n = self.0.len();
        if n == 0 || hay.len() < n {
            return Vec::new();
        }
        (0..=hay.len() - n)
            .filter(|&i| self.matches_at(hay, i))
            .collect()
    }
}

impl Process {
    /// Scan a single readable region for `pattern`; returns the absolute
    /// address of the first match.
    ///
    /// Reads in 1 MiB chunks with overlap, so matches straddling chunk
    /// boundaries are still found. Sub-page holes inside an `r` region are
    /// skipped. A non-readable or empty input yields `Ok(None)`.
    ///
    /// # Errors
    /// [`Error::Permission`] or [`Error::Io`] from the underlying reads (an
    /// [`Error::Unmapped`]/[`Error::Partial`] on a sub-range is treated as a
    /// hole and skipped, not an error).
    pub fn scan_region(&self, region: &MapRegion, pattern: &Pattern) -> Result<Option<usize>> {
        if !region.readable() || pattern.is_empty() {
            return Ok(None);
        }
        self.scan_chunks(region, pattern.len(), |base, window| {
            pattern.first_match(window).map(|off| base + off)
        })
    }

    /// Walk a readable region in overlapping 1 MiB chunks, invoking
    /// `on_window(base_addr, &window)` for each. The window carries the trailing
    /// `plen - 1` bytes of the previous chunk so a match straddling a chunk
    /// boundary is seen whole. Unreadable sub-ranges are skipped (treated as
    /// holes). The walk stops early and returns the value if a callback yields
    /// `Some`.
    fn scan_chunks(
        &self,
        region: &MapRegion,
        plen: usize,
        mut on_window: impl FnMut(usize, &[u8]) -> Option<usize>,
    ) -> Result<Option<usize>> {
        const CHUNK: usize = 1 << 20; // 1 MiB
        let mut pos = region.start;
        let mut carry: Vec<u8> = Vec::new();
        let mut carry_addr = region.start;

        while pos < region.end {
            let want = CHUNK.min(region.end - pos);
            let mut buf = match self.read_vec(pos, want) {
                Ok(b) => b,
                // a sub-page can be unreadable even in an "r" region; skip it
                Err(Error::Unmapped { .. }) | Err(Error::Partial { .. }) => {
                    carry.clear();
                    pos += want;
                    carry_addr = pos;
                    continue;
                }
                Err(e) => return Err(e),
            };

            let mut window = std::mem::take(&mut carry);
            let base_addr = carry_addr;
            window.append(&mut buf);

            if let Some(hit) = on_window(base_addr, &window) {
                return Ok(Some(hit));
            }

            // keep the last plen-1 bytes to catch a boundary-straddling match
            let keep = plen.saturating_sub(1).min(window.len());
            carry_addr = base_addr + (window.len() - keep);
            carry = window[window.len() - keep..].to_vec();
            pos += want;
        }
        Ok(None)
    }

    /// Scan every readable, file-backed region of `module` for `sig`.
    ///
    /// # Errors
    /// [`Error::ModuleNotFound`] if the module is not mapped, plus anything
    /// [`Pattern::parse`] or [`scan_region`](Self::scan_region) return.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_name("game")?;
    /// if let Some(addr) = proc.scan_module("game", "DE AD ?? EF")? {
    ///     println!("found @ {addr:#x}");
    /// }
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn scan_module(&self, module: &str, sig: &str) -> Result<Option<usize>> {
        let pat = Pattern::parse(sig)?;
        let m = self.module(module)?;
        for region in self.maps()? {
            if region.start < m.base || region.end > m.base + m.size {
                continue;
            }
            if let Some(a) = self.scan_region(&region, &pat)? {
                return Ok(Some(a));
            }
        }
        Ok(None)
    }

    /// Scan all readable regions of the process for `sig`.
    ///
    /// # Errors
    /// Anything [`Pattern::parse`], [`maps`](Self::maps), or
    /// [`scan_region`](Self::scan_region) return.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let hit = proc.scan("48 8B ?? 89 ** ?")?;
    /// # let _ = hit;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn scan(&self, sig: &str) -> Result<Option<usize>> {
        let pat = Pattern::parse(sig)?;
        for region in self.maps()? {
            if let Some(a) = self.scan_region(&region, &pat)? {
                return Ok(Some(a));
            }
        }
        Ok(None)
    }

    /// Collect **every** match of `pattern` within a region (deduplicated and
    /// ascending). Boundary-straddling matches are found via chunk overlap.
    ///
    /// # Errors
    /// Same as [`scan_region`](Self::scan_region).
    pub fn scan_region_all(&self, region: &MapRegion, pattern: &Pattern) -> Result<Vec<usize>> {
        if !region.readable() || pattern.is_empty() {
            return Ok(Vec::new());
        }
        let mut hits = Vec::new();
        self.scan_chunks(region, pattern.len(), |base, window| {
            for off in pattern.all_matches(window) {
                hits.push(base + off);
            }
            None
        })?;
        hits.sort_unstable();
        hits.dedup();
        Ok(hits)
    }

    /// Every match of `sig` across all readable regions (deduplicated,
    /// ascending).
    ///
    /// # Errors
    /// Anything [`Pattern::parse`], [`maps`](Self::maps), or
    /// [`scan_region_all`](Self::scan_region_all) return.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// for addr in proc.scan_all("DE C0 AD 0B")? {
    ///     println!("{addr:#x}");
    /// }
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn scan_all(&self, sig: &str) -> Result<Vec<usize>> {
        let pat = Pattern::parse(sig)?;
        let mut hits = Vec::new();
        for region in self.maps()? {
            hits.extend(self.scan_region_all(&region, &pat)?);
        }
        Ok(hits)
    }

    /// First match of `sig` within only the **executable** regions of `module`.
    ///
    /// This is what you want for code signatures, since data regions can
    /// contain coincidental byte matches.
    ///
    /// # Errors
    /// [`Error::ModuleNotFound`] if the module is not mapped, plus anything
    /// [`Pattern::parse`] or [`scan_region`](Self::scan_region) return.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_name("game")?;
    /// let site = proc.scan_code("game", "29 48 10 ?? ?? ?? ??")?;
    /// # let _ = site;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn scan_code(&self, module: &str, sig: &str) -> Result<Option<usize>> {
        let pat = Pattern::parse(sig)?;
        let m = self.module(module)?;
        for region in self.maps()? {
            let in_module = region.start >= m.base && region.end <= m.base + m.size;
            if !(in_module && region.executable()) {
                continue;
            }
            if let Some(a) = self.scan_region(&region, &pat)? {
                return Ok(Some(a));
            }
        }
        Ok(None)
    }

    /// Resolve an x86-64 RIP-relative reference.
    ///
    /// `instr_addr` is where the instruction starts, `disp_offset` is the byte
    /// offset of the `disp32` field within it, and `instr_len` is the full
    /// instruction length. Returns `instr_addr + instr_len + disp32`, i.e. the
    /// address the instruction points at. Typical use: find a
    /// `mov reg,[rip+x]` via [`scan`](Self::scan), then turn it into the
    /// absolute data address.
    ///
    /// # Errors
    /// Whatever [`read`](Self::read) returns when fetching the `disp32` field.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// // `48 8B 05 disp32` = mov rax,[rip+disp32]: disp at +3, length 7.
    /// let data = proc.resolve_rip(0x1000, 3, 7)?;
    /// # let _ = data;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn resolve_rip(
        &self,
        instr_addr: usize,
        disp_offset: usize,
        instr_len: usize,
    ) -> Result<usize> {
        let disp: i32 = self.read(instr_addr.wrapping_add(disp_offset))?;
        Ok(instr_addr
            .wrapping_add(instr_len)
            .wrapping_add(disp as isize as usize))
    }
}

// ---------------------------------------------------------------------------
// Patching
// ---------------------------------------------------------------------------

/// A reversible in-memory patch.
///
/// Created enabled (the new bytes are already written). It remembers the
/// original bytes, so you can [`disable`](Self::disable)/[`enable`](Self::enable)/
/// [`toggle`](Self::toggle) it at will. **By default it restores the original
/// bytes when dropped** — call [`persist`](Self::persist) to make it permanent.
///
/// All writes go through [`Process::write_force`], so patching read-only code
/// pages works.
#[derive(Debug)]
#[must_use = "a Patch reverts the original bytes when dropped; bind it to a variable \
              (or call .persist()) to keep the patch applied"]
pub struct Patch<'p> {
    proc: &'p Process,
    addr: usize,
    original: Vec<u8>,
    patched: Vec<u8>,
    enabled: bool,
    restore_on_drop: bool,
}

impl<'p> Patch<'p> {
    /// Target address.
    #[inline]
    pub fn addr(&self) -> usize {
        self.addr
    }
    /// Patch length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.patched.len()
    }
    /// Whether the patch is zero-length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.patched.is_empty()
    }
    /// Whether the patched bytes are currently written.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    /// The saved original bytes.
    #[inline]
    pub fn original(&self) -> &[u8] {
        &self.original
    }
    /// The replacement bytes.
    #[inline]
    pub fn patched(&self) -> &[u8] {
        &self.patched
    }

    /// Write the patched bytes (idempotent).
    ///
    /// # Errors
    /// Whatever [`Process::write_force`] returns.
    pub fn enable(&mut self) -> Result<()> {
        self.proc.write_force(self.addr, &self.patched)?;
        self.enabled = true;
        Ok(())
    }
    /// Restore the original bytes (idempotent).
    ///
    /// # Errors
    /// Whatever [`Process::write_force`] returns.
    pub fn disable(&mut self) -> Result<()> {
        self.proc.write_force(self.addr, &self.original)?;
        self.enabled = false;
        Ok(())
    }
    /// Flip between patched and original.
    ///
    /// # Errors
    /// Whatever [`enable`](Self::enable) / [`disable`](Self::disable) return.
    pub fn toggle(&mut self) -> Result<()> {
        if self.enabled {
            self.disable()
        } else {
            self.enable()
        }
    }

    /// Keep the patch applied after this handle is dropped.
    pub fn persist(&mut self) {
        self.restore_on_drop = false;
    }
    /// Control whether dropping reverts the patch (default `true`).
    pub fn set_restore_on_drop(&mut self, yes: bool) {
        self.restore_on_drop = yes;
    }
}

impl<'p> Drop for Patch<'p> {
    fn drop(&mut self) {
        if self.restore_on_drop && self.enabled {
            let _ = self.proc.write_force(self.addr, &self.original);
        }
    }
}

impl Process {
    /// Overwrite `addr` with `new`, returning a reversible [`Patch`] that has
    /// already saved the original bytes and written the new ones.
    ///
    /// # Errors
    /// [`Error::Unmapped`] if `addr` cannot be read, or whatever
    /// [`write_force`](Self::write_force) returns.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let mut p = proc.patch(0x1000, &[0x90, 0x90, 0x90])?;
    /// p.disable()?; // restore
    /// p.persist();  // ...or keep it applied past drop
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn patch(&self, addr: usize, new: &[u8]) -> Result<Patch<'_>> {
        let original = self.read_vec(addr, new.len())?;
        self.write_force(addr, new)?;
        Ok(Patch {
            proc: self,
            addr,
            original,
            patched: new.to_vec(),
            enabled: true,
            restore_on_drop: true,
        })
    }

    /// Replace `len` bytes at `addr` with `0x90` NOPs (reversible).
    ///
    /// # Errors
    /// Same as [`patch`](Self::patch).
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let _godmode = proc.nop(0x1000, 3)?; // reverts on drop
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn nop(&self, addr: usize, len: usize) -> Result<Patch<'_>> {
        self.patch(addr, &vec![0x90u8; len])
    }

    /// Find `sig` in `module` and patch `new` at `match + match_offset`.
    ///
    /// # Errors
    /// [`Error::PatternNotFound`] if the signature does not match, plus anything
    /// [`scan_module`](Self::scan_module) or [`patch`](Self::patch) return.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_name("game")?;
    /// let _p = proc.patch_pattern("game", "29 48 10 ?? ?? ?? ??", 0, &[0x90; 7])?;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn patch_pattern(
        &self,
        module: &str,
        sig: &str,
        match_offset: usize,
        new: &[u8],
    ) -> Result<Patch<'_>> {
        let at = self
            .scan_module(module, sig)?
            .ok_or_else(|| Error::PatternNotFound(sig.to_string()))?;
        self.patch(at + match_offset, new)
    }

    /// Write a 5-byte near `jmp rel32` from `from` to `to`, padding any slack
    /// up to `patch_len` with NOPs.
    ///
    /// `patch_len` must be >= 5 and should equal the number of original
    /// instruction bytes you're clobbering so you don't leave a torn
    /// instruction behind.
    ///
    /// # Errors
    /// [`Error::PatchTooSmall`] if `patch_len < 5`, [`Error::BranchTooFar`] if
    /// the target is out of `rel32` range, or whatever [`patch`](Self::patch)
    /// returns.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let _j = proc.write_jmp(0x1000, 0x1100, 5)?;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn write_jmp(&self, from: usize, to: usize, patch_len: usize) -> Result<Patch<'_>> {
        self.write_branch(0xE9, from, to, patch_len)
    }

    /// Like [`write_jmp`](Self::write_jmp) but emits `call rel32` (`0xE8`).
    ///
    /// # Errors
    /// Same as [`write_jmp`](Self::write_jmp).
    pub fn write_call(&self, from: usize, to: usize, patch_len: usize) -> Result<Patch<'_>> {
        self.write_branch(0xE8, from, to, patch_len)
    }

    fn write_branch(&self, op: u8, from: usize, to: usize, patch_len: usize) -> Result<Patch<'_>> {
        if patch_len < 5 {
            return Err(Error::PatchTooSmall {
                need: 5,
                got: patch_len,
            });
        }
        let rel = (to as i64) - (from as i64 + 5);
        if rel < i32::MIN as i64 || rel > i32::MAX as i64 {
            return Err(Error::BranchTooFar { from, to });
        }
        let mut bytes = Vec::with_capacity(patch_len);
        bytes.push(op);
        bytes.extend_from_slice(&(rel as i32).to_le_bytes());
        bytes.resize(patch_len, 0x90);
        self.patch(from, &bytes)
    }

    /// Write a 14-byte absolute indirect jump (`FF 25 00000000` + qword target)
    /// — reaches anywhere in the 64-bit space, unlike `rel32`. Pads to
    /// `patch_len` (>= 14) with NOPs.
    ///
    /// # Errors
    /// [`Error::PatchTooSmall`] if `patch_len < 14`, or whatever
    /// [`patch`](Self::patch) returns.
    pub fn write_jmp_abs(&self, from: usize, to: usize, patch_len: usize) -> Result<Patch<'_>> {
        if patch_len < 14 {
            return Err(Error::PatchTooSmall {
                need: 14,
                got: patch_len,
            });
        }
        let mut bytes = vec![0xFF, 0x25, 0x00, 0x00, 0x00, 0x00];
        bytes.extend_from_slice(&(to as u64).to_le_bytes());
        bytes.resize(patch_len, 0x90);
        self.patch(from, &bytes)
    }

    /// Install a detour at `from` to `to`, clobbering `patch_len` bytes.
    ///
    /// Uses a 5-byte `rel32` jump when the target is in range, otherwise a
    /// 14-byte absolute jump. Returns a reversible [`Patch`].
    ///
    /// # Errors
    /// [`Error::PatchTooSmall`] if `patch_len` is too small for the chosen
    /// encoding, or whatever [`patch`](Self::patch) returns.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let _d = proc.detour(0x1000, 0x7F00_0000_0000, 14)?; // far -> absolute
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn detour(&self, from: usize, to: usize, patch_len: usize) -> Result<Patch<'_>> {
        let rel = (to as i64) - (from as i64 + 5);
        let near_ok = patch_len >= 5 && (i32::MIN as i64..=i32::MAX as i64).contains(&rel);
        if near_ok {
            self.write_jmp(from, to, patch_len)
        } else {
            self.write_jmp_abs(from, to, patch_len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_parse_and_match() {
        let p = Pattern::parse("48 8B ?? 89").unwrap();
        assert_eq!(p.len(), 4);
        let hay = [0x00, 0x48, 0x8B, 0xFF, 0x89, 0x00];
        assert_eq!(p.first_match(&hay), Some(1));
    }

    #[test]
    fn pattern_no_match() {
        let p = Pattern::parse("DE AD BE EF").unwrap();
        assert_eq!(p.first_match(&[0, 1, 2, 3]), None);
    }

    #[test]
    fn pattern_star_wildcards() {
        // ?, ??, *, ** all mean "any byte"
        let p = Pattern::parse("48 * 89 ** ?").unwrap();
        assert_eq!(p.len(), 5);
        let hay = [0x48, 0x11, 0x89, 0x22, 0x33];
        assert!(p.first_match(&hay).is_some());
        let bad = [0x48, 0x11, 0x8A, 0x22, 0x33]; // 0x89 byte differs
        assert!(p.first_match(&bad).is_none());
    }

    #[test]
    fn pattern_parse_rejects_bad_token() {
        assert!(Pattern::parse("48 ZZ").is_err());
    }

    #[test]
    fn map_region_perm_helpers() {
        let r = MapRegion {
            start: 0x1000,
            end: 0x2000,
            perms: "r-xp".into(),
            path: None,
        };
        assert!(r.readable());
        assert!(!r.writable());
        assert!(r.executable());
        assert_eq!(r.len(), 0x1000);
        assert!(!r.is_empty());
    }

    #[test]
    fn self_read_roundtrip() {
        // Read our own memory: vm_readv on self always allowed.
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        let value: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let addr = &value as *const u64 as usize;
        let got: u64 = me.read(addr).unwrap();
        assert_eq!(got, value);
    }

    #[test]
    fn self_read_cstring() {
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        let s = b"hello\0world";
        let addr = s.as_ptr() as usize;
        assert_eq!(me.read_cstring(addr, 64).unwrap(), "hello");
    }

    #[test]
    fn self_scatter() {
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        let a: u32 = 0x1111_2222;
        let b: u32 = 0x3333_4444;
        let mut s = me.scatter();
        let ia = s.add_typed::<u32>(&a as *const _ as usize);
        let ib = s.add_typed::<u32>(&b as *const _ as usize);
        let bufs = s.run().unwrap();
        assert_eq!(pod_at::<u32>(&bufs, ia), a);
        assert_eq!(pod_at::<u32>(&bufs, ib), b);
    }

    #[test]
    fn self_pointer_chain() {
        // value <- &value <- &&value ; chain from &&value with offsets [0,0]
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        let value: i64 = -42;
        let p1: *const i64 = &value;
        let p2: *const *const i64 = &p1;
        let root = &p2 as *const _ as usize;
        // resolve: deref root -> p2 ; +0 ; deref -> p1 ; +0 -> &value (address)
        let got: i64 = me.pointer(root).offsets(&[0, 0]).read().unwrap();
        assert_eq!(got, value);
    }

    #[test]
    fn pattern_from_mask_and_all() {
        let p = Pattern::from_mask(b"\x48\x8B\x00\x89", "xx?x").unwrap();
        assert_eq!(p.len(), 4);
        let hay = [0x48, 0x8B, 0x11, 0x89, 0x48, 0x8B, 0xFF, 0x89];
        assert_eq!(p.all_matches(&hay), vec![0, 4]);
    }

    #[test]
    fn patch_roundtrip_and_drop_restores() {
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        let buf: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let addr = buf.as_ptr() as usize;
        {
            let mut p = me.patch(addr, &[0xAA, 0xBB]).unwrap();
            assert_eq!(&buf[..2], &[0xAA, 0xBB]);
            p.disable().unwrap();
            assert_eq!(&buf[..2], &[0, 1]);
            p.enable().unwrap();
            assert_eq!(&buf[..2], &[0xAA, 0xBB]);
        } // drop restores original
        assert_eq!(&buf[..2], &[0, 1]);
    }

    #[test]
    fn patch_persist_keeps_bytes() {
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        let buf: Vec<u8> = vec![0xFF; 4];
        let addr = buf.as_ptr() as usize;
        {
            let mut p = me.nop(addr, 4).unwrap();
            p.persist();
        }
        assert_eq!(&buf[..], &[0x90, 0x90, 0x90, 0x90]);
    }

    #[test]
    fn jmp_rel32_encoding() {
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        let code: Vec<u8> = vec![0xCC; 8];
        let from = code.as_ptr() as usize;
        let to = from + 0x100;
        let mut p = me.write_jmp(from, to, 8).unwrap();
        assert_eq!(code[0], 0xE9);
        let rel = i32::from_le_bytes(code[1..5].try_into().unwrap());
        assert_eq!(rel as i64, 0x100 - 5);
        assert_eq!(&code[5..8], &[0x90, 0x90, 0x90]); // NOP slack
        p.disable().unwrap();
        assert_eq!(&code[..], &[0xCC; 8]);
    }

    #[test]
    fn jmp_abs_encoding() {
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        let code: Vec<u8> = vec![0u8; 14];
        let from = code.as_ptr() as usize;
        let to = 0xDEAD_BEEF_1234_5678u64 as usize;
        let _p = me.write_jmp_abs(from, to, 14).unwrap();
        assert_eq!(&code[..6], &[0xFF, 0x25, 0, 0, 0, 0]);
        assert_eq!(
            u64::from_le_bytes(code[6..14].try_into().unwrap()),
            to as u64
        );
    }

    #[test]
    fn scan_region_finds_known_needle() {
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        let mut data = vec![0xABu8; 64 * 1024];
        let needle = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let at = 40_000;
        data[at..at + needle.len()].copy_from_slice(&needle);
        let addr = data.as_ptr() as usize;
        let region = me
            .maps()
            .unwrap()
            .into_iter()
            .find(|r| r.start <= addr && addr + data.len() <= r.end)
            .expect("a mapped region containing the buffer");
        let pat = Pattern::parse("11 22 33 44 55 66 77 88").unwrap();
        let all = me.scan_region_all(&region, &pat).unwrap();
        assert!(all.contains(&(addr + at)));
        // first match is the lowest hit address
        assert_eq!(me.scan_region(&region, &pat).unwrap(), all.first().copied());
    }

    #[test]
    fn scan_region_all_spans_chunk_boundaries() {
        // Exercises the >1 MiB multi-chunk path and the carry that catches a
        // match straddling a chunk boundary.
        let me = Process::by_pid(std::process::id() as i32).unwrap();
        const MB: usize = 1 << 20;
        let size = 2 * MB + 8192;
        let mut data = vec![0xABu8; size];
        let needle = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let offsets = [64usize, MB - 3, MB + 64, 2 * MB - 5];
        for &o in &offsets {
            data[o..o + needle.len()].copy_from_slice(&needle);
        }
        let addr = data.as_ptr() as usize;
        let region = me
            .maps()
            .unwrap()
            .into_iter()
            .find(|r| r.start <= addr && addr + size <= r.end)
            .expect("a mapped region containing the buffer");
        let pat = Pattern::parse("11 22 33 44 55 66 77 88").unwrap();
        let all = me.scan_region_all(&region, &pat).unwrap();
        for &o in &offsets {
            assert!(all.contains(&(addr + o)), "missing needle at +{o:#x}");
        }
        assert_eq!(me.scan_region(&region, &pat).unwrap(), all.first().copied());
    }
}
