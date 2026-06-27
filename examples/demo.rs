//! Usage: cargo run --example demo -- <process-name> <module> [hex_static_off] [off,off,..]
use vmem::Process;

fn main() -> vmem::Result<()> {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "game".into());
    let module = args.next().unwrap_or_else(|| name.clone());
    let static_off = args
        .next()
        .and_then(|s| usize::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x10F2A30);
    let offsets: Vec<usize> = args
        .next()
        .map(|s| {
            s.split(',')
                .filter_map(|p| usize::from_str_radix(p.trim().trim_start_matches("0x"), 16).ok())
                .collect()
        })
        .unwrap_or_else(|| vec![0x10, 0x8, 0x0]);

    let proc = Process::by_name(&name)?;
    println!("pid = {}", proc.pid());

    let m = proc.module(&module)?;
    println!("module {} base={:#x} size={:#x}", m.name, m.base, m.size);

    let chain = proc.pointer(m.base + static_off).offsets(&offsets);
    let addr = chain.resolve()?;
    let value: i32 = proc.read(addr)?;
    println!("value @ {addr:#x} = {value}");

    // Example: scatter-read three i32s near it in one syscall.
    let mut s = proc.scatter();
    let i0 = s.add_typed::<i32>(addr);
    let i1 = s.add_typed::<i32>(addr + 4);
    let i2 = s.add_typed::<i32>(addr + 8);
    let bufs = s.run()?;
    println!(
        "batch: {} {} {}",
        vmem::pod_at::<i32>(&bufs, i0),
        vmem::pod_at::<i32>(&bufs, i1),
        vmem::pod_at::<i32>(&bufs, i2),
    );
    Ok(())
}
