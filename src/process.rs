//! Process discovery: turning a name or pid into a [`Process`] handle.

use std::fs;

use crate::{Error, Result};

/// A handle to a target process. Cloning is cheap (just the pid).
#[derive(Clone, Copy, Debug)]
pub struct Process {
    pub(crate) pid: i32,
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
