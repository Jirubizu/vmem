//! Backend selection: userspace syscalls (default) or the `/dev/vmem` kernel
//! driver.
//!
//! This module exists only under the `kernel` feature. The device handle is
//! *process-global* (one `/dev/vmem` for the whole program, not per target), so
//! [`crate::Process`] stays `Copy` and every public signature is unchanged — the
//! waist functions in `lib.rs` just consult [`backend`] before falling through
//! to their existing syscall path.

use std::sync::LazyLock;

mod kernel_driver;
pub(crate) use kernel_driver::KernelDriver;

/// The memory-access backend chosen for this process.
pub(crate) enum Backend {
    /// `process_vm_readv`/`writev` + `/proc/<pid>/mem` (the crate's default).
    Syscall,
    /// The loaded `/dev/vmem` kernel module.
    Kernel(KernelDriver),
}

/// The backend for this process, selected once on first use.
///
/// Selection order:
/// 1. `VMEM_BACKEND=syscall` forces the userspace path.
/// 2. `VMEM_BACKEND=kernel` forces the driver (falling back to syscalls if
///    `/dev/vmem` cannot be opened).
/// 3. Otherwise: use the driver if `/dev/vmem` is present, else syscalls.
pub(crate) fn backend() -> &'static Backend {
    static BACKEND: LazyLock<Backend> = LazyLock::new(select);
    &BACKEND
}

fn select() -> Backend {
    const DEV: &str = "/dev/vmem";
    match std::env::var("VMEM_BACKEND").ok().as_deref() {
        Some("syscall") => return Backend::Syscall,
        Some("kernel") => {
            return KernelDriver::open(DEV)
                .map(Backend::Kernel)
                .unwrap_or(Backend::Syscall);
        }
        _ => {}
    }
    // Auto-detection default. The crate's own unit tests validate
    // backend-independent logic against self-memory, so under `cfg(test)` the
    // auto path stays on syscalls: a kernel module that merely happens to be
    // loaded on the build host cannot reroute (and destabilize) them. An
    // explicit `VMEM_BACKEND=kernel` still opts in above; the kernel path is
    // exercised by `examples/kernel_ab`.
    #[cfg(test)]
    {
        Backend::Syscall
    }
    #[cfg(not(test))]
    {
        KernelDriver::open(DEV)
            .map(Backend::Kernel)
            .unwrap_or(Backend::Syscall)
    }
}
