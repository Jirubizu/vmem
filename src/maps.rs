//! Parsing `/proc/<pid>/maps` into [`MapRegion`]s and resolving [`Module`]s.

use std::collections::BTreeMap;
use std::fs;

use crate::{Error, Process, Result};

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
        let mut acc: BTreeMap<String, (usize, usize)> = Default::default();
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
