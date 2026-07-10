//! The raw memory-access backend: cross-process reads/writes via
//! `process_vm_readv`/`writev` and the `/proc/<pid>/mem` fallback, plus the
//! shared errno-classification helpers used by every backend.

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};

use bytemuck::Pod;

use crate::{Error, Process, Result};

impl Process {
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
        #[cfg(feature = "kernel")]
        if let crate::backend::Backend::Kernel(d) = crate::backend::backend() {
            return d.read(self.pid, addr, buf);
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
            return Err(classify(self.pid, addr, buf.len(), errno()));
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
    /// **Kernel backend:** under the `kernel` feature with `/dev/vmem` active,
    /// this routes through the driver, which *can* write read-only pages. Do
    /// not rely on a read-only rejection here when that backend may be selected.
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
        #[cfg(feature = "kernel")]
        if let crate::backend::Backend::Kernel(d) = crate::backend::backend() {
            return d.write(self.pid, addr, buf);
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
            return Err(classify(self.pid, addr, buf.len(), errno()));
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
}

/// Map an OS `errno` from a memory-access syscall or the kernel driver's ioctl
/// onto the crate's error taxonomy. Shared by both backends so the same failure
/// yields the same [`Error`] regardless of which path served the request.
#[inline]
pub(crate) fn classify(pid: i32, addr: usize, len: usize, errno: i32) -> Error {
    match errno {
        libc::EPERM | libc::EACCES => Error::Permission { pid },
        libc::EFAULT | libc::ENOMEM | libc::EIO => Error::Unmapped { addr, len },
        other => Error::Io(std::io::Error::from_raw_os_error(other)),
    }
}

#[inline]
pub(crate) fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn classify_maps_errno_to_error() {
        assert!(matches!(
            classify(7, 0, 0, libc::EPERM),
            Error::Permission { pid: 7 }
        ));
        assert!(matches!(
            classify(7, 0, 0, libc::EACCES),
            Error::Permission { pid: 7 }
        ));
        assert!(matches!(
            classify(0, 0x1000, 16, libc::EFAULT),
            Error::Unmapped {
                addr: 0x1000,
                len: 16
            }
        ));
        assert!(matches!(
            classify(0, 0, 0, libc::EIO),
            Error::Unmapped { .. }
        ));
        assert!(matches!(
            classify(0, 0, 0, libc::ENOMEM),
            Error::Unmapped { .. }
        ));
        // ESRCH is deliberately NOT Unmapped — it falls through to Io, matching
        // the syscall path so both backends classify an errno identically.
        assert!(matches!(classify(0, 0, 0, libc::ESRCH), Error::Io(_)));
    }
}
