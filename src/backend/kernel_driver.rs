//! Thin ioctl client for the `/dev/vmem` kernel module.
//!
//! The [`VmemIo`] layout and the `VMEM_RW` ioctl number are the ABI contract
//! with the module (`kmod/vmem_main.rs`); they MUST match byte for byte.

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;

use crate::{Error, Result, classify, errno};

/// ioctl request payload. `#[repr(C)]`; field order and size are the ABI.
#[repr(C)]
struct VmemIo {
    pid: i32,
    write: u32, // 0 = read, 1 = write
    addr: u64,
    len: u64,
    ubuf: u64, // userspace pointer to the data buffer
}

/// `_IOWR('V', nr, struct vmem_io)` — Linux asm-generic ioctl encoding.
const fn iowr(nr: u32, size: usize) -> libc::c_ulong {
    ((3u32 << 30) | ((b'V' as u32) << 8) | nr | ((size as u32) << 16)) as libc::c_ulong
}

const VMEM_RW: libc::c_ulong = iowr(0, std::mem::size_of::<VmemIo>());

/// Maximum bytes per ioctl; larger transfers are chunked. Matches the module's
/// cap, so a single request never allocates an unbounded kernel bounce buffer.
const MAX_LEN: usize = 1 << 20;

/// Direction of a `VMEM_RW` transfer — replaces a bare `write: bool` at the
/// call sites so `read`/`write` read as intent, not a boolean flag.
#[derive(Clone, Copy)]
enum Direction {
    Read,
    Write,
}

impl Direction {
    /// The ABI `write` flag: `1` for a write op, `0` for a read.
    fn is_write(self) -> bool {
        matches!(self, Direction::Write)
    }
}

/// An open handle to the `/dev/vmem` char device.
pub(crate) struct KernelDriver {
    file: File,
}

impl KernelDriver {
    /// Open the device, or fail (so the caller can fall back to syscalls).
    pub(crate) fn open(path: &str) -> std::io::Result<Self> {
        Ok(Self {
            file: OpenOptions::new().read(true).write(true).open(path)?,
        })
    }

    /// One `VMEM_RW` ioctl over a `<= MAX_LEN` slice. Returns bytes moved.
    fn ioctl_once(&self, pid: i32, addr: usize, ptr: *mut u8, len: usize, dir: Direction) -> isize {
        let mut req = VmemIo {
            pid,
            write: u32::from(dir.is_write()),
            addr: addr as u64,
            len: len as u64,
            ubuf: ptr as u64,
        };
        // SAFETY: `self.file` is a live O_RDWR handle to /dev/vmem; `req`
        // outlives the call; `ptr` is valid for `len` bytes (mut on read,
        // shared on write) as guaranteed by `rw`'s callers.
        unsafe { libc::ioctl(self.file.as_raw_fd(), VMEM_RW, &mut req) as isize }
    }

    /// Transfer `len` bytes, chunking at `MAX_LEN`. Mirrors the syscall path's
    /// error contract: [`Error::Partial`] once any bytes have moved, otherwise
    /// a classified error.
    fn rw(&self, pid: i32, addr: usize, ptr: *mut u8, len: usize, dir: Direction) -> Result<()> {
        let mut done = 0usize;
        while done < len {
            let chunk = (len - done).min(MAX_LEN);
            // SAFETY: `done < len` and the buffer is valid for `len` bytes, so
            // `ptr + done` stays in bounds.
            let n = self.ioctl_once(pid, addr + done, unsafe { ptr.add(done) }, chunk, dir);
            if n < 0 {
                if done > 0 {
                    return Err(Error::Partial {
                        addr,
                        wanted: len,
                        moved: done,
                    });
                }
                return Err(classify(pid, addr, len, errno()));
            }
            let moved = n as usize;
            done += moved;
            if moved != chunk {
                return Err(Error::Partial {
                    addr,
                    wanted: len,
                    moved: done,
                });
            }
        }
        Ok(())
    }

    /// Read `buf.len()` bytes from the target's `addr` into `buf`.
    pub(crate) fn read(&self, pid: i32, addr: usize, buf: &mut [u8]) -> Result<()> {
        self.rw(pid, addr, buf.as_mut_ptr(), buf.len(), Direction::Read)
    }

    /// Write `buf` to the target's `addr` (read-only pages included).
    pub(crate) fn write(&self, pid: i32, addr: usize, buf: &[u8]) -> Result<()> {
        // The module only reads from this buffer on a write op.
        self.rw(
            pid,
            addr,
            buf.as_ptr().cast_mut(),
            buf.len(),
            Direction::Write,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vmem_rw_abi_constant_is_stable() {
        // Pins the ioctl number and payload size against silent ABI drift; the
        // kernel module MUST use the same _IOWR('V', 0, struct vmem_io).
        assert_eq!(std::mem::size_of::<VmemIo>(), 32);
        assert_eq!(VMEM_RW, 0xC020_5600);
    }
}
