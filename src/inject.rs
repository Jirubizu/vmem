//! Remote allocation and code injection via `ptrace`.
//!
//! `process_vm_writev` can write memory but cannot *allocate* it, so to get a
//! scratch/code region in the target (Cheat Engine's `alloc(...)`) we briefly
//! `ptrace`-attach, plant a `syscall` instruction at the stopped thread's RIP,
//! and drive an `mmap` (then restore everything). [`Hook`] builds on this to
//! install a detour into a freshly-allocated cave, mirroring a CE auto-assembler
//! script.
//!
//! Requirements: permission to `ptrace` the target (same rules as the read/write
//! paths — Yama `ptrace_scope`, or `cap_sys_ptrace`). Caveat: the inject step
//! stops a single thread; for a multithreaded target there is a small window
//! where another thread could execute the 2 clobbered bytes at the stopped RIP.

use std::ffi::c_void;

use crate::{Error, Patch, Process, Result, errno};

const SYS_MMAP: u64 = 9;
const SYS_MUNMAP: u64 = 11;

/// Memory-protection flags for [`Process::alloc`].
///
/// These mirror the `PROT_*` bits and can be OR-ed together; the
/// [`RWX`](self::prot::RWX) and [`RW`](self::prot::RW) combinations are provided
/// for convenience.
pub mod prot {
    /// Pages may be read (`PROT_READ`).
    pub const READ: i32 = 1;
    /// Pages may be written (`PROT_WRITE`).
    pub const WRITE: i32 = 2;
    /// Pages may be executed (`PROT_EXEC`).
    pub const EXEC: i32 = 4;
    /// Read + write + execute — a classic code cave.
    pub const RWX: i32 = READ | WRITE | EXEC;
    /// Read + write — a data scratch buffer.
    pub const RW: i32 = READ | WRITE;
}

#[inline]
unsafe fn raw_ptrace(req: u32, pid: i32, addr: *mut c_void, data: *mut c_void) -> i64 {
    // SAFETY: forwarded to the caller, who guarantees the request/arg
    // combination is valid for the (stopped) target thread.
    unsafe { libc::ptrace(req as _, pid, addr, data) as i64 }
}

/// Block until `pid` reports a ptrace stop.
///
/// Retries across `EINTR` and turns an unexpected exit into an error instead of
/// silently proceeding with stale register state.
fn wait_stopped(pid: i32) -> Result<()> {
    loop {
        let mut status = 0i32;
        let r = unsafe { libc::waitpid(pid, &mut status, 0) };
        if r < 0 {
            let e = errno();
            if e == libc::EINTR {
                continue;
            }
            return Err(Error::Io(std::io::Error::from_raw_os_error(e)));
        }
        if libc::WIFEXITED(status) || libc::WIFSIGNALED(status) {
            return Err(Error::Io(std::io::Error::other(
                "target process exited during ptrace operation",
            )));
        }
        return Ok(());
    }
}

/// An attach/detach session over a target thread.
#[derive(Debug)]
struct Tracer {
    pid: i32,
}

impl Tracer {
    fn attach(pid: i32) -> Result<Self> {
        let null = std::ptr::null_mut::<c_void>();
        let r = unsafe { raw_ptrace(libc::PTRACE_ATTACH, pid, null, null) };
        if r < 0 {
            return Err(match errno() {
                libc::EPERM | libc::EACCES => Error::Permission { pid },
                e => Error::Io(std::io::Error::from_raw_os_error(e)),
            });
        }
        wait_stopped(pid)?;
        Ok(Self { pid })
    }

    fn getregs(&self) -> Result<libc::user_regs_struct> {
        let mut regs: libc::user_regs_struct = unsafe { std::mem::zeroed() };
        let r = unsafe {
            raw_ptrace(
                libc::PTRACE_GETREGS,
                self.pid,
                std::ptr::null_mut(),
                &mut regs as *mut _ as *mut c_void,
            )
        };
        if r < 0 {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }
        Ok(regs)
    }

    fn setregs(&self, regs: &libc::user_regs_struct) -> Result<()> {
        let r = unsafe {
            raw_ptrace(
                libc::PTRACE_SETREGS,
                self.pid,
                std::ptr::null_mut(),
                regs as *const _ as *mut c_void,
            )
        };
        if r < 0 {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }
        Ok(())
    }

    fn singlestep(&self) -> Result<()> {
        let null = std::ptr::null_mut::<c_void>();
        let r = unsafe { raw_ptrace(libc::PTRACE_SINGLESTEP, self.pid, null, null) };
        if r < 0 {
            return Err(Error::Io(std::io::Error::last_os_error()));
        }
        wait_stopped(self.pid)?;
        Ok(())
    }

    /// Execute one syscall in the target with up to 6 args, returning RAX.
    fn inject_syscall(&self, proc: &Process, nr: u64, args: [u64; 6]) -> Result<i64> {
        let saved = self.getregs()?;
        let rip = saved.rip;
        // back up and plant a `syscall` (0F 05) at RIP
        let backup = proc.read_vec(rip as usize, 2)?;
        proc.write_bytes_mem(rip as usize, &[0x0F, 0x05])?;

        let mut regs = saved;
        regs.rax = nr;
        regs.orig_rax = (-1i64) as u64; // defeat syscall-restart logic
        regs.rdi = args[0];
        regs.rsi = args[1];
        regs.rdx = args[2];
        regs.r10 = args[3];
        regs.r8 = args[4];
        regs.r9 = args[5];
        regs.rip = rip;
        self.setregs(&regs)?;
        self.singlestep()?;
        let after = self.getregs()?;

        // restore original code and registers
        proc.write_bytes_mem(rip as usize, &backup)?;
        self.setregs(&saved)?;
        Ok(after.rax as i64)
    }
}

impl Drop for Tracer {
    fn drop(&mut self) {
        let null = std::ptr::null_mut::<c_void>();
        unsafe { raw_ptrace(libc::PTRACE_DETACH, self.pid, null, null) };
    }
}

/// A region allocated inside the target process.
///
/// Freed (via an injected `munmap`) on drop unless [`leak`](Self::leak)ed.
/// Because dropping frees the region — and any code or hook that points into it
/// — you must bind it to a variable for as long as you need it.
#[derive(Debug)]
#[must_use = "the allocation is freed (munmap) as soon as this handle is dropped; \
              bind it to a variable or call .leak()"]
pub struct RemoteMem<'p> {
    proc: &'p Process,
    /// Base address of the region in the target.
    pub addr: usize,
    /// Length of the region in bytes.
    pub len: usize,
    freed: bool,
}

impl<'p> RemoteMem<'p> {
    /// Write `bytes` at `offset` within the region.
    ///
    /// # Errors
    /// Whatever [`Process::write_force`] returns.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        self.proc.write_force(self.addr + offset, bytes)
    }
    /// Read `len` bytes from `offset` within the region.
    ///
    /// # Errors
    /// Whatever [`Process::read_vec`] returns.
    pub fn read(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        self.proc.read_vec(self.addr + offset, len)
    }
    /// Free the region now (otherwise it happens on drop). Idempotent.
    ///
    /// # Errors
    /// [`Error::Permission`] if the target can no longer be `ptrace`d, or any
    /// error from the injected `munmap`.
    pub fn free(&mut self) -> Result<()> {
        if self.freed {
            return Ok(());
        }
        let tracer = Tracer::attach(self.proc.pid())?;
        tracer.inject_syscall(
            self.proc,
            SYS_MUNMAP,
            [self.addr as u64, self.len as u64, 0, 0, 0, 0],
        )?;
        self.freed = true;
        Ok(())
    }
    /// Keep the allocation alive past this handle (don't free on drop), and
    /// return its base address.
    pub fn leak(mut self) -> usize {
        self.freed = true;
        self.addr
    }
}

impl<'p> Drop for RemoteMem<'p> {
    fn drop(&mut self) {
        if !self.freed {
            let _ = self.free();
        }
    }
}

impl Process {
    /// Allocate `len` bytes in the target with the given protection (see
    /// [`prot`]).
    ///
    /// Backed by an injected `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`. This briefly
    /// `ptrace`-attaches the target; on a heavily multithreaded target there is
    /// a small race window on the clobbered bytes at the stopped thread's RIP.
    ///
    /// # Errors
    /// [`Error::Permission`] if the target cannot be `ptrace`d, or [`Error::Io`]
    /// carrying the `mmap` errno on failure.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::{Process, prot};
    /// let proc = Process::by_pid(1234)?;
    /// let scratch = proc.alloc(0x100, prot::RW)?;
    /// scratch.write(0, &[1, 2, 3, 4])?;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn alloc(&self, len: usize, protection: i32) -> Result<RemoteMem<'_>> {
        let tracer = Tracer::attach(self.pid())?;
        let flags = (libc::MAP_PRIVATE | libc::MAP_ANONYMOUS) as u64;
        let ret = tracer.inject_syscall(
            self,
            SYS_MMAP,
            [0, len as u64, protection as u64, flags, (-1i64) as u64, 0],
        )?;
        drop(tracer);
        if ret < 0 && ret > -4096 {
            return Err(Error::Io(std::io::Error::from_raw_os_error(-ret as i32)));
        }
        Ok(RemoteMem {
            proc: self,
            addr: ret as usize,
            len,
            freed: false,
        })
    }

    /// Allocate read/write/execute memory (a code cave).
    ///
    /// Shorthand for [`alloc(len, prot::RWX)`](Self::alloc).
    ///
    /// # Errors
    /// Same as [`alloc`](Self::alloc).
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_pid(1234)?;
    /// let cave = proc.alloc_rwx(0x1000)?;
    /// # let _ = cave;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn alloc_rwx(&self, len: usize) -> Result<RemoteMem<'_>> {
        self.alloc(len, prot::RWX)
    }

    /// Find `len` consecutive `filler` bytes inside an executable region of
    /// `module` — a "free" code cave needing no allocation (just write rights).
    ///
    /// Common fillers: `0x00`, `0xCC` (int3 padding), `0x90` (nop). Returns the
    /// absolute address of the start of the run, or `None` if no run is long
    /// enough.
    ///
    /// # Errors
    /// [`Error::ModuleNotFound`] if the module is not mapped, or [`Error::Io`]
    /// if the maps file cannot be read.
    ///
    /// # Examples
    /// ```no_run
    /// use vmem::Process;
    /// let proc = Process::by_name("game")?;
    /// if let Some(cave) = proc.find_code_cave("game", 64, 0x00)? {
    ///     println!("cave @ {cave:#x}");
    /// }
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn find_code_cave(&self, module: &str, len: usize, filler: u8) -> Result<Option<usize>> {
        let m = self.module(module)?;
        for region in self.maps()? {
            let in_mod = region.start >= m.base && region.end <= m.base + m.size;
            if !(in_mod && region.executable()) {
                continue;
            }
            let bytes = match self.read_vec(region.start, region.len()) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut run = 0usize;
            let mut start = 0usize;
            for (i, &b) in bytes.iter().enumerate() {
                if b == filler {
                    if run == 0 {
                        start = i;
                    }
                    run += 1;
                    if run >= len {
                        return Ok(Some(region.start + start));
                    }
                } else {
                    run = 0;
                }
            }
        }
        Ok(None)
    }
}

/// An installed detour into an allocated code cave — the CE auto-assembler
/// pattern.
///
/// The cave runs your code, then the stolen original instructions, then jumps
/// back. Reverts the detour **and** frees the cave on drop unless
/// [`persist`](Self::persist)ed (field order guarantees the detour is removed
/// before the cave is freed).
#[derive(Debug)]
#[must_use = "a Hook reverts the detour and frees the cave when dropped; bind it \
              to a variable (or call .persist()) to keep the hook installed"]
pub struct Hook<'p> {
    // field order matters: detour reverts BEFORE the cave is freed
    detour: Patch<'p>,
    cave: Option<RemoteMem<'p>>,
    target: usize,
    cave_addr: usize,
}

impl<'p> Hook<'p> {
    /// Address of the detoured site in the target.
    pub fn target(&self) -> usize {
        self.target
    }
    /// Address of the injected cave the detour jumps to.
    pub fn cave_addr(&self) -> usize {
        self.cave_addr
    }
    /// Whether the detour jump is currently installed.
    pub fn is_enabled(&self) -> bool {
        self.detour.is_enabled()
    }
    /// Re-install the jump.
    ///
    /// # Errors
    /// Whatever [`Patch::enable`] returns.
    pub fn enable(&mut self) -> Result<()> {
        self.detour.enable()
    }
    /// Remove the jump (restore original bytes); the cave stays allocated.
    ///
    /// # Errors
    /// Whatever [`Patch::disable`] returns.
    pub fn disable(&mut self) -> Result<()> {
        self.detour.disable()
    }
    /// Flip the detour between installed and removed.
    ///
    /// # Errors
    /// Whatever [`Patch::toggle`] returns.
    pub fn toggle(&mut self) -> Result<()> {
        self.detour.toggle()
    }
    /// Keep the hook and cave in place after this handle drops.
    pub fn persist(&mut self) {
        self.detour.persist();
        if let Some(cave) = self.cave.take() {
            cave.leak();
        }
    }
}

impl Process {
    /// Install a code-cave hook at `target`, stealing `steal_len` bytes.
    ///
    /// `build` writes your custom assembly; the stolen original instructions and
    /// a jump back are appended automatically. `steal_len` must be at least 5
    /// (for a `rel32` jump) and must land on an instruction boundary — and the
    /// stolen bytes must be position-independent (no RIP-relative operands), as
    /// they execute from the cave. If `target` and the cave are more than ±2 GiB
    /// apart, use `steal_len >= 14` so an absolute jump fits.
    ///
    /// # Errors
    /// [`Error::PatchTooSmall`] if `steal_len < 5`, plus anything
    /// [`read_vec`](Self::read_vec), [`alloc_rwx`](Self::alloc_rwx),
    /// [`Asm::assemble`](crate::Asm::assemble), or [`detour`](Self::detour)
    /// return.
    ///
    /// ```no_run
    /// use vmem::{Process, Reg};
    /// let proc = Process::by_name("game")?;
    /// let site = proc.scan_code("game", "29 48 10 ?? ?? ?? ??")?.unwrap();
    /// // Neutralize a damage write by zeroing the damage register before it runs.
    /// let hook = proc.hook(site, 7, |a| {
    ///     a.xor_rr(Reg::Rcx, Reg::Rcx);
    /// })?;
    /// # Ok::<(), vmem::Error>(())
    /// ```
    pub fn hook(
        &self,
        target: usize,
        steal_len: usize,
        build: impl FnOnce(&mut crate::Asm),
    ) -> Result<Hook<'_>> {
        if steal_len < 5 {
            return Err(Error::PatchTooSmall {
                need: 5,
                got: steal_len,
            });
        }
        let stolen = self.read_vec(target, steal_len)?;
        let cave = self.alloc_rwx(0x1000)?;

        let mut a = crate::Asm::new();
        build(&mut a);
        a.raw(&stolen); // run the original instructions we displaced
        a.jmp_abs((target + steal_len) as u64); // ... then return

        let bytes = a
            .assemble(cave.addr as u64)
            .map_err(|e| Error::Io(std::io::Error::other(e.to_string())))?;
        cave.write(0, &bytes)?;

        let detour = self.detour(target, cave.addr, steal_len)?;
        let cave_addr = cave.addr;
        Ok(Hook {
            detour,
            cave: Some(cave),
            target,
            cave_addr,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Child, Command};

    // Spawn a victim that spins in pure userspace so ptrace can attach cleanly.
    fn victim() -> Child {
        Command::new("sh")
            .arg("-c")
            .arg("while :; do :; done")
            .spawn()
            .unwrap()
    }

    #[test]
    fn alloc_write_free() {
        let mut child = victim();
        std::thread::sleep(std::time::Duration::from_millis(80));
        let proc = Process::by_pid(child.id() as i32).unwrap();

        let mem = proc.alloc_rwx(0x1000).unwrap();
        assert!(mem.addr > 0x1000);
        let code = [0x48u8, 0x31, 0xC0, 0xC3]; // xor rax,rax; ret
        mem.write(0, &code).unwrap();
        assert_eq!(mem.read(0, 4).unwrap(), code);
        drop(mem); // frees via injected munmap

        let _ = child.kill();
        let _ = child.wait();
    }
}
