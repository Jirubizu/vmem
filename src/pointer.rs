//! Multi-level pointer chains ([`Pointer`]).

use bytemuck::Pod;

use crate::{Process, Result};

impl Process {
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
}

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
    /// (commonly [`Error::Unmapped`](crate::Error::Unmapped) when a link is stale).
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
