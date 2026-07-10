//! Reversible byte/code patches ([`Patch`]) and the branch/detour writers.

use crate::{Error, Process, Result};

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
}
