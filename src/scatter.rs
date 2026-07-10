//! Batched scatter/gather reads ([`Scatter`]).

use bytemuck::Pod;

use crate::{Error, Process, Result, classify, errno};

impl Process {
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
    /// Under the `kernel` backend there is no batched ioctl, so `run` issues one
    /// read per slot — correctness is identical, but the single-syscall win does
    /// not apply.
    ///
    /// # Errors
    /// [`Error::Permission`], [`Error::Unmapped`], [`Error::Partial`], or
    /// [`Error::Io`] — reported against the first address of the failing chunk.
    /// Because the kernel reports only a total byte count, a partial transfer
    /// cannot be attributed to an exact slot.
    pub fn run(self) -> Result<Vec<Vec<u8>>> {
        #[cfg(feature = "kernel")]
        if let crate::backend::Backend::Kernel(_) = crate::backend::backend() {
            return self
                .items
                .iter()
                .map(|&(addr, len)| self.proc.read_vec(addr, len))
                .collect();
        }
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
                return Err(classify(self.proc.pid, addr, len, errno()));
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
