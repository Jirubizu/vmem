//! Cheat-Engine auto-assembler style injection:
//!   aobscanmodule -> alloc -> write assembly -> detour -> revert.
//! Usage: cargo run --example inject -- <process> <module>
use vmem::{Process, Reg, asm64};

fn main() -> vmem::Result<()> {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "game".into());
    let module = args.next().unwrap_or_else(|| name.clone());
    let proc = Process::by_name(&name)?;

    // 1) aobscanmodule(INJECT, module, "29 48 10 ?? ?? ?? ??")
    //    scan_code only looks in executable regions of the module.
    let site = proc
        .scan_code(&module, "29 48 10 ?? ?? ?? ??")?
        .expect("damage signature not found");
    println!("INJECT @ {site:#x}");

    // ---- Option A: define memory + write raw assembly yourself ----
    // alloc(cave, 0x1000) — real mmap injected into the target via ptrace.
    let cave = proc.alloc_rwx(0x1000)?;
    let code = asm64! {
        pushad;
        movabs rax, 0x270Fu64;     // 9999
        store [rbx + 0x10], rax;   // mov [rbx+0x10], rax
        popad;
        ret;
    }
    .assemble(cave.addr as u64)
    .unwrap();
    cave.write(0, &code)?;
    println!("cave @ {:#x}: wrote {} bytes of asm", cave.addr, code.len());
    cave.read(0, 4)
        .map(|b| println!("  first bytes: {b:02X?}"))?;

    // ---- Option B: full detour hook (the CE [ENABLE] block) ----
    // Steal 7 bytes (the damage-write instruction); our code runs first, then
    // the original stolen instruction, then a jump back — all wired for us.
    let mut hook = proc.hook(site, 7, |a| {
        // zero the damage register (rcx) before the original write executes
        a.xor_rr(Reg::Rcx, Reg::Rcx);
    })?;
    println!("hook: {:#x} -> cave {:#x}", hook.target(), hook.cave_addr());

    std::thread::sleep(std::time::Duration::from_secs(5));

    // [DISABLE]: restore original bytes. Dropping the hook also frees the cave.
    hook.disable()?;
    // hook.persist();  // ...or keep it installed and leak the cave on drop
    println!("hook disabled, original code restored");
    Ok(())
}
