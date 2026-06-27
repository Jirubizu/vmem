# vmem

Read and write another process's memory, resolve multi-level pointer chains, AOB-scan,
patch code, and inject Cheat-Engine-style hooks — on Linux, from Rust.

```toml
[dependencies]
vmem = "0.1"   # Linux only; pulls in libc + bytemuck + thiserror
```

Everything below is exercised by the test suite (`cargo test`) and the runnable
examples in `examples/` (`demo`, `patch`, `inject`).

> **Permissions.** All cross-process access needs the right to `ptrace` the target:
> same UID with `/proc/sys/kernel/yama/ptrace_scope = 0`, or `cap_sys_ptrace`, or root.
> Errors surface as `Error::Permission`.

---

## 1. Find a process

```rust
use vmem::Process;

let proc = Process::by_name("game")?;        // first match by comm / cmdline basename
let proc = Process::by_pid(1234)?;           // wrap a known pid
let pids = Process::all_by_name("chrome");   // every match, ascending
```

## 2. Modules and maps

```rust
let m = proc.module("game")?;                // base + total span + path
println!("{} base={:#x} size={:#x}", m.name, m.base, m.size);

for region in proc.maps()? {                 // parsed /proc/<pid>/maps
    if region.readable() { /* region.start .. region.end, region.perms, region.path */ }
}
for module in proc.modules()? { /* every distinct file-backed module */ }
```

## 3. Read and write

`read`/`write` are typed and sound — the `T: Pod` bound (from `bytemuck`) guarantees
every bit pattern is valid, so there is no UB.

```rust
let hp: i32   = proc.read(addr)?;
let pos: [f32; 3] = proc.read(addr)?;        // any Pod type, including arrays/structs
proc.write::<i32>(addr, 9999)?;

let raw  = proc.read_vec(addr, 64)?;         // Vec<u8>
let name = proc.read_cstring(addr, 32)?;     // NUL-terminated, lossy UTF-8
proc.write_bytes(addr, &[0xDE, 0xAD])?;

// /proc/<pid>/mem fallback (old kernels / seccomp); write_bytes_mem can hit RO pages.
proc.read_bytes_mem(addr, &mut buf)?;
```

## 4. Multi-level pointer chains

Default convention matches Cheat Engine when you fold the static offset into the base
and list CE's offsets top-to-bottom: `addr = deref(addr); addr += off` per offset, and
the final result is the address of the value.

```rust
// CE: "game"+0x10F2A30 -> 0x10 -> 0x8 -> 0x0
let hp = proc
    .pointer(m.base + 0x10F2A30)
    .offsets(&[0x10, 0x8, 0x0])
    .read::<i32>()?;

proc.pointer(base).offset(0x20).offset(0x4).write::<f32>(1.0)?;

let addr = proc.pointer(base).offsets(&offs).resolve()?;  // just the final address

// If your notes use the other convention (add offset *before* each deref):
let v = proc.pointer(base).offsets(&offs).offset_first().read::<i32>()?;
```

## 5. Batched reads (one syscall, many addresses)

`process_vm_readv` takes scatter/gather iovecs, so independent reads collapse into a
single kernel round-trip.

```rust
let mut s = proc.scatter();
let hp  = s.add_typed::<i32>(addr_hp);
let mp  = s.add_typed::<i32>(addr_mp);
let pos = s.add(addr_pos, 12);
let out = s.run()?;                          // one syscall
let hp_val: i32 = vmem::pod_at(&out, hp);
```

## 6. AOB / signature scanning

Patterns accept `??`, `?`, `**`, or `*` as a wildcard byte; or build from a
code+mask pair.

```rust
proc.scan("48 8B ?? 89 ** ?")?;                       // first match, whole process
proc.scan_module("game", "DE AD ?? EF")?;             // within a module
proc.scan_code("game", "29 48 10 ?? ?? ?? ??")?;      // executable regions only
let all = proc.scan_all("DE C0 AD 0B")?;              // every match (deduped)

use vmem::Pattern;
let p = Pattern::from_mask(b"\x48\x8B\x00\x89", "xx?x")?;
let hits = proc.scan_region_all(&region, &p)?;

// RIP-relative: turn a `mov reg,[rip+disp32]` match into the absolute data address.
let data = proc.resolve_rip(instr_addr, /*disp_offset*/ 3, /*instr_len*/ 7)?;
```

## 7. Patching (reversible)

Every patch saves the original bytes and **reverts on drop** (call `.persist()` to keep
it). Writes route through `write_force`, which falls back to `/proc/<pid>/mem` so even
read-only `.text` pages can be patched — the way a debugger plants a breakpoint.

```rust
let mut p = proc.patch(addr, &[0x90, 0x90, 0x90])?; // raw bytes
p.disable()?;  p.enable()?;  p.toggle()?;           // flip live
p.persist();                                        // leave it applied

let _god = proc.nop(damage_addr, 3)?;               // NOP an instruction
let _p   = proc.patch_pattern("game", "29 48 10 ?? ?? ?? ??", 0, &[0x90;7])?;

// jumps / detours (rel32 with NOP-padded slack, or 14-byte absolute):
let _j = proc.write_jmp(from, to, 5)?;
let _a = proc.write_jmp_abs(from, to, 14)?;
let _d = proc.detour(from, to, 5)?;                 // auto-picks near vs absolute
```

## 8. The assembler and `asm64!`

A focused x86-64 encoder for cave code. Build with the `asm64!` macro (or the `Asm`
builder), then `assemble(base_addr)` to resolve relative jumps/labels for the address
the code will live at.

```rust
use vmem::{asm64, Asm, Reg};

let code = asm64! {
    pushad;                       // save all GP regs
    movabs rax, 0x270Fu64;        // movabs rax, 9999
    store [rbx + 0x10], rax;      // mov [rbx+0x10], rax
    load  rcx, [rbx + 0x08];      // mov rcx, [rbx+0x8]
    xor   rdx, rdx;
    add   rsi, 0x4;
    popad;
    ret;
}.assemble(0x1000)?;
```

Supported statements: `push/pop r`, `pushad/popad`, `movabs r, imm`, `mov d, s`,
`xor d, s`, `add/sub r, imm`, `store [b + disp], s`, `load d, [b + disp]`, `ret`,
`nop`, `int3`, `raw a, b, ..`, `label name`, `jmp/call expr` (rel32 to an absolute),
`jmp_label/call_label name`, `jmp_abs expr` (14-byte absolute). The builder also has
`jmp`, `call`, `jmp_abs`, labels, and `raw` for anything not covered.

## 9. Define memory in the target (`alloc`)

`process_vm_writev` can't allocate, so this injects an `mmap` via `ptrace` (attach,
plant a `syscall`, single-step, restore). The region is freed on drop unless leaked.

```rust
use vmem::prot;

let cave = proc.alloc_rwx(0x1000)?;          // RWX page (== alloc(.., prot::RWX))
cave.write(0, &code)?;
let back = cave.read(0, code.len())?;
let addr = cave.leak();                       // keep it; skip munmap-on-drop

let scratch = proc.alloc(0x100, prot::RW)?;  // RW data buffer
```

No-allocation alternative — reuse padding already in the target:

```rust
let cave = proc.find_code_cave("game", 64, 0x00)?; // 64 zero bytes in an exec region
```

## 10. Hooks — the Cheat Engine auto-assembler equivalent

`hook(target, steal_len, build)` does the whole `[ENABLE]` dance: alloc a cave, write
your code followed by the stolen original instructions and a jump back, then detour the
target into the cave. The returned `Hook` reverts the detour **and** frees the cave on
drop (or `.persist()`).

```rust
let site = proc.scan_code("game", "29 48 10 ?? ?? ?? ??")?.unwrap();
let mut hook = proc.hook(site, 7, |a| {
    a.xor_rr(vmem::Reg::Rcx, vmem::Reg::Rcx);  // zero damage before original write
})?;
// ... active ...
hook.disable()?;   // restore
// hook.persist(); // or keep installed
```

### Side by side

```
; ---- Cheat Engine ----                 // ---- vmem ----
[ENABLE]                                  let site = proc
aobscanmodule(INJECT,game,29 48 10 ..)        .scan_code("game","29 48 10 ?? ?? ?? ??")?
alloc(newmem,256)                             .unwrap();
newmem:                                   let mut hook = proc.hook(site, 7, |a| {
  xor ecx,ecx                                 a.xor_rr(Reg::Rcx, Reg::Rcx);
  mov [rax+10],ecx   ; original          })?; // stolen bytes + jmp-back appended
  jmp return
INJECT:
  jmp newmem
  nop 2
return:
registersymbol(INJECT)

[DISABLE]                                 hook.disable()?;   // (drop also frees cave)
INJECT: db 29 48 10 ..
dealloc(newmem)
```

> **Caveats for `hook`/`alloc`.** Stolen bytes must be position-independent (no
> RIP-relative operands) since they run from the cave; `steal_len` must be ≥ 5 and end
> on an instruction boundary (use ≥ 14 if the cave is > ±2 GiB away so an absolute jump
> fits). The inject step stops one thread — for heavily multithreaded targets there is a
> small race window on the 2 clobbered bytes at the stopped RIP.

## Windows

The shape is identical: swap the backend (`OpenProcess` + `ReadProcessMemory`/
`WriteProcessMemory`, `VirtualAllocEx` for `alloc`, `VirtualProtectEx` around code
patches). `Pointer`, `Scatter`, `Pattern`/scanning, `Asm`/`asm64!`, and `Hook` are
platform-independent.
