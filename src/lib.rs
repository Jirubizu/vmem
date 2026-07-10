//! # vmem
//!
//! Read and write the memory of another process, resolve multi-level pointer
//! chains, AOB/signature-scan, patch code, and inject Cheat-Engine-style hooks
//! — on Linux, from safe Rust.
//!
//! Cross-process reads and writes use `process_vm_readv` / `process_vm_writev`
//! (no `ptrace`-stop required); code patching falls back to `/proc/<pid>/mem`
//! so even read-only `.text` pages can be modified; and remote allocation and
//! hooking briefly `ptrace`-attach to inject an `mmap`.
//!
//! ```no_run
//! use vmem::Process;
//!
//! let proc = Process::by_name("game")?;
//! let module = proc.module("game")?;
//!
//! // Cheat Engine pointer:  "game"+0x10F2A30 -> 0x10 -> 0x8 -> 0x0
//! let hp = proc
//!     .pointer(module.base + 0x10F2A30)
//!     .offsets(&[0x10, 0x8, 0x0])
//!     .read::<i32>()?;
//! println!("HP = {hp}");
//! # Ok::<(), vmem::Error>(())
//! ```
//!
//! ## Feature tour
//! * [`Process`] — a cheap, `Copy` handle to a target; module bases are
//!   resolved on demand from `/proc/<pid>/maps`.
//! * Typed [`Process::read`] / [`Process::write`] are sound: the `T: Pod`
//!   bound (from `bytemuck`) guarantees every bit pattern is valid, so there
//!   is no UB on padding or invalid discriminants.
//! * [`Pointer`] — a fluent pointer-chain resolver with an explicit,
//!   documented dereference convention (and a toggle for the other one).
//! * [`Scatter`] — reads many disjoint addresses in a **single syscall** by
//!   handing the kernel multiple `iovec`s.
//! * [`Pattern`] / [`Process::scan`] — AOB / signature scanning with `??`
//!   wildcards, plus a RIP-relative resolver ([`Process::resolve_rip`]).
//! * [`Patch`] — reversible byte/code patches that revert on drop.
//! * [`Asm`] and the [`asm64!`](crate::asm64) macro — a focused x86-64 encoder
//!   for cave code.
//! * [`RemoteMem`] / [`Process::hook`] — remote allocation and detour hooks.
//!
//! ## Permissions
//! Every cross-process operation needs the right to `ptrace` the target: the
//! same UID with `/proc/sys/kernel/yama/ptrace_scope = 0`, the `cap_sys_ptrace`
//! capability, or root. Insufficient rights surface as [`Error::Permission`].
//!
//! ## A note on danger
//! Writing another process's memory — and especially [patching
//! code](Process::patch) or [installing hooks](Process::hook) — can corrupt or
//! crash the target. These operations are memory-safe for *this* process (no
//! Rust UB), which is why the API is safe, but they are inherently unsafe in
//! *effect*. Test against a process you own.
//!
//! ## Portability
//! The crate is Linux-only. On Windows the shape is identical: swap the backend
//! behind [`Process::read_bytes`] / [`Process::write_bytes`] for `OpenProcess`
//! with `ReadProcessMemory` / `WriteProcessMemory`, and everything above
//! ([`Pointer`], [`Scatter`], [`Pattern`], [`Asm`]) is unchanged.

#![cfg(target_os = "linux")]
#![recursion_limit = "512"]
#![deny(missing_docs)]
#![warn(missing_debug_implementations)]

mod asm;
#[cfg(feature = "kernel")]
mod backend;
mod error;
mod inject;
mod io;
mod maps;
mod patch;
mod pointer;
mod process;
mod scan;
mod scatter;

pub use asm::{Asm, AsmError, Reg};
pub use error::{Error, Result};
pub use inject::{Hook, RemoteMem, prot};
pub use maps::{MapRegion, Module};
pub use patch::Patch;
pub use pointer::Pointer;
pub use process::Process;
pub use scan::Pattern;
pub use scatter::{Scatter, pod_at};

pub(crate) use io::{classify, errno};
