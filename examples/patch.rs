//! AOB scan + patch example (a "godmode" style toggle).
//! Usage: cargo run --example patch -- <process> <module>
use vmem::Process;

fn main() -> vmem::Result<()> {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "game".into());
    let module = args.next().unwrap_or_else(|| name.clone());

    let proc = Process::by_name(&name)?;

    // 1. Find a "sub [rax+10], ecx" style damage instruction by signature.
    //    `??` wildcards the bytes that vary across builds.
    let sig = "29 48 10 ?? ?? ?? ??";
    let hit = proc
        .scan_code(&module, sig)?
        .expect("damage signature not found");
    println!("damage write @ {hit:#x}");

    // 2. NOP it out so damage is never applied. The returned Patch reverts
    //    automatically when dropped — call .persist() to keep it.
    let mut godmode = proc.nop(hit, 3)?;
    println!("godmode ON  (bytes -> {:02X?})", godmode.patched());

    std::thread::sleep(std::time::Duration::from_secs(5));

    godmode.disable()?;
    println!("godmode OFF (restored {:02X?})", godmode.original());

    // 3. Alternative: redirect a function to your own code cave with a detour.
    //    let cave = /* address of your injected stub */ 0;
    //    let _hook = proc.detour(hit, cave, 5)?; // 5-byte rel32, or abs if far

    // 4. Count every occurrence of a global magic value, in one place:
    for (i, a) in proc.scan_all("DE C0 AD 0B")?.into_iter().enumerate() {
        println!("  match {i} @ {a:#x}");
    }
    Ok(())
}
