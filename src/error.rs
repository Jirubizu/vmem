//! The crate's [`Error`] taxonomy and its [`Result`] alias.

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

    /// A signature or mask string was malformed: a bad hex token in
    /// [`Pattern::parse`](crate::Pattern::parse), or a pattern/mask length
    /// mismatch in [`Pattern::from_mask`](crate::Pattern::from_mask).
    #[error("invalid pattern: {0}")]
    InvalidPattern(String),

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
