//! A/B correctness harness for the kernel backend.
//!
//! It reads and writes *this* process's own memory through the `vmem` backend
//! and checks against a known sentinel — ground-truth correct regardless of
//! which backend is active. Run it under each backend and compare:
//!
//! ```text
//! cargo build --features kernel --example kernel_ab
//! VMEM_BACKEND=syscall ./target/debug/examples/kernel_ab
//! sudo VMEM_BACKEND=kernel ./target/debug/examples/kernel_ab   # module loaded
//! ```
//!
//! Exits non-zero on any mismatch, so it doubles as a smoke test.

use std::hint::black_box;

use vmem::Process;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let requested = std::env::var("VMEM_BACKEND").unwrap_or_default();
    let dev_present = std::path::Path::new("/dev/vmem").exists();
    println!("VMEM_BACKEND={requested:?}   /dev/vmem present: {dev_present}");

    // Fail loudly rather than silently falling back, so a "kernel" run that
    // actually exercised the syscall path can't masquerade as a pass.
    if requested == "kernel" && !dev_present {
        return Err("VMEM_BACKEND=kernel but /dev/vmem is absent — load the module first".into());
    }

    let proc = Process::by_pid(std::process::id() as i32)?;

    // --- READ: a heap sentinel, forced to memory ---
    let sentinel: Box<u64> = Box::new(0xDEAD_BEEF_CAFE_BABE);
    let addr = (&*sentinel as *const u64) as usize;
    let got: u64 = proc.read(addr)?;
    black_box(&sentinel);
    assert_eq!(
        got, *sentinel,
        "read mismatch: backend returned wrong bytes"
    );
    println!("read   {got:#018x} == {:#018x}   OK", *sentinel);

    // --- WRITE: flip a heap cell, observe it through the target ---
    let cell: Box<u32> = Box::new(0);
    let caddr = (&*cell as *const u32) as usize;
    proc.write::<u32>(caddr, 0x1234_5678)?;
    black_box(&cell);
    assert_eq!(*cell, 0x1234_5678, "write not observed in target memory");
    println!("write  {:#010x} observed           OK", *cell);

    // --- UNMAPPED: a bogus address must classify as Unmapped on any backend ---
    match proc.read::<u8>(0x1) {
        Err(vmem::Error::Unmapped { .. }) => println!("unmap  0x1 -> Error::Unmapped      OK"),
        other => return Err(format!("expected Unmapped, got {other:?}").into()),
    }

    println!("kernel_ab: PASS");
    Ok(())
}
