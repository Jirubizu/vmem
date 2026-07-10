//! AOB / signature scanning ([`Pattern`]) and RIP-relative resolution.

use crate::{Error, MapRegion, Process, Result};

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
    /// [`Error::InvalidPattern`] if a token is neither a wildcard nor a valid
    /// two-digit hex byte.
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
                let b = u8::from_str_radix(tok, 16)
                    .map_err(|_| Error::InvalidPattern(format!("bad token '{tok}'")))?;
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
    /// [`Error::InvalidPattern`] if `bytes` and `mask` differ in length.
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
            return Err(Error::InvalidPattern(
                "pattern and mask differ in length".into(),
            ));
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
        assert!(matches!(
            Pattern::parse("48 ZZ"),
            Err(Error::InvalidPattern(_))
        ));
        assert!(matches!(
            Pattern::from_mask(b"\x48", "xx"),
            Err(Error::InvalidPattern(_))
        ));
    }

    #[test]
    fn pattern_from_mask_and_all() {
        let p = Pattern::from_mask(b"\x48\x8B\x00\x89", "xx?x").unwrap();
        assert_eq!(p.len(), 4);
        let hay = [0x48, 0x8B, 0x11, 0x89, 0x48, 0x8B, 0xFF, 0x89];
        assert_eq!(p.all_matches(&hay), vec![0, 4]);
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
