//! A tiny x86-64 machine-code builder for code caves and detours, plus the
//! [`asm64!`](crate::asm64) macro for terse authoring.
//!
//! This is deliberately *not* a full assembler — it covers the instructions you
//! actually need when writing Cheat-Engine-style hooks: save/restore registers,
//! move immediates and registers, simple `[base+disp]` loads/stores, arithmetic,
//! and jumps/calls (relative with labels, or absolute). Anything exotic you can
//! drop in with [`Asm::raw`].
//!
//! Relative jumps and labels are resolved by [`Asm::assemble`], which takes the
//! address the code will live at (e.g. the cave returned by
//! [`crate::RemoteMem`]).
//!
//! ```
//! use vmem::{Asm, Reg, asm64};
//!
//! // movabs rax, 0xDEAD; mov [rbx+0x10], rax; ret
//! let code = asm64! {
//!     movabs rax, 0xDEAD;
//!     store [rbx + 0x10], rax;
//!     ret;
//! };
//! let bytes = code.assemble(0x1000).unwrap(); // base address it will be written to
//! assert_eq!(bytes[0..2], [0x48, 0xB8]); // REX.W + movabs
//! ```

use std::collections::HashMap;

/// x86-64 general-purpose registers (64-bit), encoded `0..=15`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Reg {
    /// `rax` (accumulator).
    Rax = 0,
    /// `rcx` (counter).
    Rcx = 1,
    /// `rdx` (data).
    Rdx = 2,
    /// `rbx` (base).
    Rbx = 3,
    /// `rsp` (stack pointer).
    Rsp = 4,
    /// `rbp` (base pointer).
    Rbp = 5,
    /// `rsi` (source index).
    Rsi = 6,
    /// `rdi` (destination index).
    Rdi = 7,
    /// `r8`.
    R8 = 8,
    /// `r9`.
    R9 = 9,
    /// `r10`.
    R10 = 10,
    /// `r11`.
    R11 = 11,
    /// `r12`.
    R12 = 12,
    /// `r13`.
    R13 = 13,
    /// `r14`.
    R14 = 14,
    /// `r15`.
    R15 = 15,
}

impl Reg {
    #[inline]
    fn hi(self) -> bool {
        (self as u8) >= 8
    }
    #[inline]
    fn low3(self) -> u8 {
        (self as u8) & 7
    }
}

#[derive(Debug)]
enum Reloc {
    /// A 4-byte rel32 field at `at`, pointing to a label.
    Label { at: usize, name: String, end: usize },
    /// A 4-byte rel32 field at `at`, pointing to an absolute address.
    Abs { at: usize, target: u64, end: usize },
}

/// Accumulates machine code with deferred relocation of relative jumps.
///
/// Build with the chainable methods (or the [`asm64!`](crate::asm64) macro),
/// then call [`assemble`](Self::assemble) with the address the code will live
/// at to get the finished bytes with all relative jumps resolved.
///
/// ```
/// use vmem::{Asm, Reg};
/// let mut a = Asm::new();
/// a.push(Reg::Rax).ret();
/// assert_eq!(a.assemble(0x1000).unwrap(), [0x50, 0xC3]);
/// ```
#[derive(Default, Debug)]
#[must_use = "an Asm only produces machine code once you call .assemble()"]
pub struct Asm {
    buf: Vec<u8>,
    labels: HashMap<String, usize>,
    relocs: Vec<Reloc>,
}

impl Asm {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current encoded length in bytes (before relocation).
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    /// Whether no bytes have been emitted yet.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Append raw, already-encoded bytes (e.g. stolen original instructions).
    pub fn raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.buf.extend_from_slice(bytes);
        self
    }

    /// Emit a single `nop` (`0x90`).
    pub fn nop(&mut self) -> &mut Self {
        self.buf.push(0x90);
        self
    }
    /// Emit `n` `nop` bytes.
    pub fn nops(&mut self, n: usize) -> &mut Self {
        self.buf.extend(std::iter::repeat_n(0x90, n));
        self
    }
    /// Emit `int3` (`0xCC`), a debugger breakpoint.
    pub fn int3(&mut self) -> &mut Self {
        self.buf.push(0xCC);
        self
    }
    /// Emit `ret` (`0xC3`).
    pub fn ret(&mut self) -> &mut Self {
        self.buf.push(0xC3);
        self
    }

    /// Emit `push r` (64-bit).
    pub fn push(&mut self, r: Reg) -> &mut Self {
        if r.hi() {
            self.buf.push(0x41);
        }
        self.buf.push(0x50 + r.low3());
        self
    }
    /// Emit `pop r` (64-bit).
    pub fn pop(&mut self, r: Reg) -> &mut Self {
        if r.hi() {
            self.buf.push(0x41);
        }
        self.buf.push(0x58 + r.low3());
        self
    }

    /// Save the full GP set (minus `rsp`) with a run of `push`es. Handy at cave
    /// entry; pair with [`popad`](Self::popad) at exit.
    pub fn pushad(&mut self) -> &mut Self {
        for r in PUSH_ORDER {
            self.push(r);
        }
        self
    }
    /// Restore what [`pushad`](Self::pushad) saved (pops in reverse order).
    pub fn popad(&mut self) -> &mut Self {
        for r in PUSH_ORDER.iter().rev() {
            self.pop(*r);
        }
        self
    }

    /// Emit `movabs r, imm64` (load a full 64-bit immediate).
    pub fn mov_imm(&mut self, r: Reg, imm: u64) -> &mut Self {
        self.buf.push(0x48 | (r.hi() as u8)); // REX.W (+B)
        self.buf.push(0xB8 + r.low3());
        self.buf.extend_from_slice(&imm.to_le_bytes());
        self
    }
    /// Emit `mov dst, src` (64-bit register-to-register).
    pub fn mov_rr(&mut self, dst: Reg, src: Reg) -> &mut Self {
        self.rex_w(src, dst);
        self.buf.push(0x89);
        self.modrm_reg(src, dst);
        self
    }
    /// Emit `xor dst, src` (64-bit). `xor r, r` is the idiomatic register zero.
    pub fn xor_rr(&mut self, dst: Reg, src: Reg) -> &mut Self {
        self.rex_w(src, dst);
        self.buf.push(0x31);
        self.modrm_reg(src, dst);
        self
    }
    /// Emit `add r, imm32` (64-bit).
    pub fn add_imm(&mut self, r: Reg, imm: i32) -> &mut Self {
        self.alu_imm(0, r, imm)
    }
    /// Emit `sub r, imm32` (64-bit).
    pub fn sub_imm(&mut self, r: Reg, imm: i32) -> &mut Self {
        self.alu_imm(5, r, imm)
    }

    /// Emit `mov [base + disp], src` (64-bit store).
    pub fn store(&mut self, base: Reg, disp: i32, src: Reg) -> &mut Self {
        // REX.W | R(src) | B(base)
        self.buf
            .push(0x48 | ((src.hi() as u8) << 2) | (base.hi() as u8));
        self.buf.push(0x89);
        self.mem_modrm(src, base, disp);
        self
    }
    /// Emit `mov dst, [base + disp]` (64-bit load).
    pub fn load(&mut self, dst: Reg, base: Reg, disp: i32) -> &mut Self {
        self.buf
            .push(0x48 | ((dst.hi() as u8) << 2) | (base.hi() as u8));
        self.buf.push(0x8B);
        self.mem_modrm(dst, base, disp);
        self
    }

    /// Define a label at the current position, referable by
    /// [`jmp_label`](Self::jmp_label) / [`call_label`](Self::call_label).
    pub fn label(&mut self, name: &str) -> &mut Self {
        self.labels.insert(name.to_string(), self.buf.len());
        self
    }
    /// Emit `jmp <label>` (rel32, resolved at assemble time).
    pub fn jmp_label(&mut self, name: &str) -> &mut Self {
        self.rel_branch(0xE9, None);
        let at = self.buf.len() - 4;
        self.relocs.push(Reloc::Label {
            at,
            name: name.into(),
            end: self.buf.len(),
        });
        self
    }
    /// Emit `call <label>` (rel32, resolved at assemble time).
    pub fn call_label(&mut self, name: &str) -> &mut Self {
        self.rel_branch(0xE8, None);
        let at = self.buf.len() - 4;
        self.relocs.push(Reloc::Label {
            at,
            name: name.into(),
            end: self.buf.len(),
        });
        self
    }
    /// Emit `jmp <absolute>` as rel32 (the target must be within ±2 GiB of the
    /// code's final site, or [`assemble`](Self::assemble) errors).
    pub fn jmp(&mut self, target: u64) -> &mut Self {
        self.rel_branch(0xE9, None);
        let at = self.buf.len() - 4;
        self.relocs.push(Reloc::Abs {
            at,
            target,
            end: self.buf.len(),
        });
        self
    }
    /// Emit `call <absolute>` as rel32 (same ±2 GiB caveat as
    /// [`jmp`](Self::jmp)).
    pub fn call(&mut self, target: u64) -> &mut Self {
        self.rel_branch(0xE8, None);
        let at = self.buf.len() - 4;
        self.relocs.push(Reloc::Abs {
            at,
            target,
            end: self.buf.len(),
        });
        self
    }
    /// Emit `jmp <absolute>` via a 14-byte `FF 25` RIP-indirect jump — reaches
    /// anywhere in the 64-bit address space (no range limit).
    pub fn jmp_abs(&mut self, target: u64) -> &mut Self {
        self.buf.extend_from_slice(&[0xFF, 0x25, 0, 0, 0, 0]);
        self.buf.extend_from_slice(&target.to_le_bytes());
        self
    }

    /// Resolve labels/relative targets for a final load address and return the
    /// finished bytes.
    ///
    /// `base` is where the first byte will live in the target (e.g.
    /// [`RemoteMem::addr`](crate::RemoteMem::addr)).
    ///
    /// # Errors
    /// [`AsmError::UndefinedLabel`] if a `jmp_label`/`call_label` references a
    /// label that was never defined, or [`AsmError::Rel32OutOfRange`] if a
    /// relative branch cannot reach its target from `base`.
    ///
    /// # Examples
    /// ```
    /// use vmem::{Asm, Reg};
    /// let mut a = Asm::new();
    /// a.xor_rr(Reg::Rax, Reg::Rax).ret();
    /// assert_eq!(a.assemble(0x1000)?, [0x48, 0x31, 0xC0, 0xC3]);
    /// # Ok::<(), vmem::AsmError>(())
    /// ```
    pub fn assemble(&self, base: u64) -> Result<Vec<u8>, AsmError> {
        let mut out = self.buf.clone();
        for r in &self.relocs {
            let (at, end, target) = match r {
                Reloc::Label { at, name, end } => {
                    let off = *self
                        .labels
                        .get(name)
                        .ok_or_else(|| AsmError::UndefinedLabel(name.clone()))?;
                    (*at, *end, base + off as u64)
                }
                Reloc::Abs { at, target, end } => (*at, *end, *target),
            };
            let rip = base + end as u64; // rel is relative to end of instruction
            let rel = target as i64 - rip as i64;
            if rel < i32::MIN as i64 || rel > i32::MAX as i64 {
                return Err(AsmError::Rel32OutOfRange { rel });
            }
            out[at..at + 4].copy_from_slice(&(rel as i32).to_le_bytes());
        }
        Ok(out)
    }

    // --- encoding helpers ---

    fn rex_w(&mut self, reg: Reg, rm: Reg) {
        self.buf
            .push(0x48 | ((reg.hi() as u8) << 2) | (rm.hi() as u8));
    }
    fn modrm_reg(&mut self, reg: Reg, rm: Reg) {
        self.buf.push(0xC0 | (reg.low3() << 3) | rm.low3());
    }
    fn alu_imm(&mut self, op_ext: u8, r: Reg, imm: i32) -> &mut Self {
        self.buf.push(0x48 | (r.hi() as u8));
        self.buf.push(0x81);
        self.buf.push(0xC0 | (op_ext << 3) | r.low3());
        self.buf.extend_from_slice(&imm.to_le_bytes());
        self
    }
    /// ModRM+SIB+disp32 for `[base + disp]` with register field `reg`.
    /// Always uses mod=10 (disp32), which sidesteps the rbp/r13 mod=00 quirk;
    /// rsp/r12 still require a SIB byte.
    fn mem_modrm(&mut self, reg: Reg, base: Reg, disp: i32) {
        let rm = base.low3();
        self.buf.push(0x80 | (reg.low3() << 3) | rm); // mod=10
        if rm == 4 {
            self.buf.push(0x24); // SIB: scale=0 index=none base=rsp/r12
        }
        self.buf.extend_from_slice(&disp.to_le_bytes());
    }
    fn rel_branch(&mut self, op: u8, _placeholder: Option<()>) {
        self.buf.push(op);
        self.buf.extend_from_slice(&[0, 0, 0, 0]);
    }
}

const PUSH_ORDER: [Reg; 15] = [
    Reg::Rax,
    Reg::Rcx,
    Reg::Rdx,
    Reg::Rbx,
    Reg::Rbp,
    Reg::Rsi,
    Reg::Rdi,
    Reg::R8,
    Reg::R9,
    Reg::R10,
    Reg::R11,
    Reg::R12,
    Reg::R13,
    Reg::R14,
    Reg::R15,
];

/// Errors from [`Asm::assemble`].
///
/// This enum is `#[non_exhaustive]`: matching on it must include a `_` arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AsmError {
    /// A `jmp_label`/`call_label` referenced a label that was never defined.
    #[error("undefined label '{0}'")]
    UndefinedLabel(String),
    /// A relative branch's target is more than ±2 GiB from the code's final
    /// site; use [`Asm::jmp_abs`] instead.
    #[error("rel32 target out of range (delta {rel})")]
    Rel32OutOfRange {
        /// The computed relative delta that overflowed `i32`.
        rel: i64,
    },
}

/// Map a register identifier to a [`Reg`]. Used by [`asm64!`](crate::asm64).
#[macro_export]
macro_rules! reg {
    (rax) => {
        $crate::Reg::Rax
    };
    (rcx) => {
        $crate::Reg::Rcx
    };
    (rdx) => {
        $crate::Reg::Rdx
    };
    (rbx) => {
        $crate::Reg::Rbx
    };
    (rsp) => {
        $crate::Reg::Rsp
    };
    (rbp) => {
        $crate::Reg::Rbp
    };
    (rsi) => {
        $crate::Reg::Rsi
    };
    (rdi) => {
        $crate::Reg::Rdi
    };
    (r8) => {
        $crate::Reg::R8
    };
    (r9) => {
        $crate::Reg::R9
    };
    (r10) => {
        $crate::Reg::R10
    };
    (r11) => {
        $crate::Reg::R11
    };
    (r12) => {
        $crate::Reg::R12
    };
    (r13) => {
        $crate::Reg::R13
    };
    (r14) => {
        $crate::Reg::R14
    };
    (r15) => {
        $crate::Reg::R15
    };
}

/// Terse assembler. Returns an [`Asm`]; call `.assemble(base_addr)` for bytes.
///
/// Supported statements (each ends with `;`):
/// `push r` · `pop r` · `pushad` · `popad` · `movabs r, expr` · `mov d, s` ·
/// `xor d, s` · `add r, expr` · `sub r, expr` · `store [b + disp], s` ·
/// `load d, [b + disp]` · `ret` · `nop` · `int3` · `raw a, b, ..` ·
/// `label name` · `jmp expr` · `call expr` · `jmp_abs expr` ·
/// `jmp_label name` · `call_label name`.
///
/// ```
/// use vmem::{asm64, Reg};
/// let code = asm64! {
///     pushad;
///     xor rcx, rcx;
///     popad;
///     ret;
/// };
/// let bytes = code.assemble(0x2000).unwrap();
/// assert_eq!(bytes[0], 0x50);             // push rax
/// assert_eq!(*bytes.last().unwrap(), 0xC3); // ret
/// ```
#[macro_export]
macro_rules! asm64 {
    (@s $a:ident;) => {};
    (@s $a:ident; pushad ; $($r:tt)*) => { $a.pushad(); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; popad ; $($r:tt)*) => { $a.popad(); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; push $x:ident ; $($r:tt)*) => { $a.push($crate::reg!($x)); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; pop $x:ident ; $($r:tt)*) => { $a.pop($crate::reg!($x)); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; movabs $d:ident , $imm:expr ; $($r:tt)*) => { $a.mov_imm($crate::reg!($d), ($imm) as u64); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; mov $d:ident , $s:ident ; $($r:tt)*) => { $a.mov_rr($crate::reg!($d), $crate::reg!($s)); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; xor $d:ident , $s:ident ; $($r:tt)*) => { $a.xor_rr($crate::reg!($d), $crate::reg!($s)); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; add $d:ident , $imm:expr ; $($r:tt)*) => { $a.add_imm($crate::reg!($d), ($imm) as i32); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; sub $d:ident , $imm:expr ; $($r:tt)*) => { $a.sub_imm($crate::reg!($d), ($imm) as i32); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; store [ $b:ident + $disp:expr ] , $s:ident ; $($r:tt)*) => { $a.store($crate::reg!($b), ($disp) as i32, $crate::reg!($s)); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; load $d:ident , [ $b:ident + $disp:expr ] ; $($r:tt)*) => { $a.load($crate::reg!($d), $crate::reg!($b), ($disp) as i32); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; ret ; $($r:tt)*) => { $a.ret(); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; nop ; $($r:tt)*) => { $a.nop(); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; int3 ; $($r:tt)*) => { $a.int3(); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; raw $($byte:expr),+ ; $($r:tt)*) => { $a.raw(&[$(($byte) as u8),+]); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; label $name:ident ; $($r:tt)*) => { $a.label(stringify!($name)); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; jmp_label $name:ident ; $($r:tt)*) => { $a.jmp_label(stringify!($name)); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; call_label $name:ident ; $($r:tt)*) => { $a.call_label(stringify!($name)); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; jmp_abs $t:expr ; $($r:tt)*) => { $a.jmp_abs(($t) as u64); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; jmp $t:expr ; $($r:tt)*) => { $a.jmp(($t) as u64); $crate::asm64!(@s $a; $($r)*); };
    (@s $a:ident; call $t:expr ; $($r:tt)*) => { $a.call(($t) as u64); $crate::asm64!(@s $a; $($r)*); };
    // public entry — MUST be last so the @s arms above win for internal calls
    ($($t:tt)*) => {{
        let mut __a = $crate::Asm::new();
        $crate::asm64!(@s __a; $($t)*);
        __a
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asm(f: impl FnOnce(&mut Asm)) -> Vec<u8> {
        let mut a = Asm::new();
        f(&mut a);
        a.assemble(0).unwrap()
    }

    #[test]
    fn basic_encodings() {
        assert_eq!(
            asm(|a| {
                a.push(Reg::Rax);
            }),
            [0x50]
        );
        assert_eq!(
            asm(|a| {
                a.push(Reg::R15);
            }),
            [0x41, 0x57]
        );
        assert_eq!(
            asm(|a| {
                a.pop(Reg::Rax);
            }),
            [0x58]
        );
        assert_eq!(
            asm(|a| {
                a.ret();
            }),
            [0xC3]
        );
        assert_eq!(
            asm(|a| {
                a.mov_imm(Reg::Rax, 0x1122334455667788);
            }),
            [0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
        );
        assert_eq!(
            asm(|a| {
                a.mov_imm(Reg::R8, 1);
            }),
            [0x49, 0xB8, 1, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| {
                a.mov_rr(Reg::Rbx, Reg::Rax);
            }),
            [0x48, 0x89, 0xC3]
        );
        assert_eq!(
            asm(|a| {
                a.xor_rr(Reg::Rax, Reg::Rax);
            }),
            [0x48, 0x31, 0xC0]
        );
        assert_eq!(
            asm(|a| {
                a.add_imm(Reg::Rax, 0x10);
            }),
            [0x48, 0x81, 0xC0, 0x10, 0, 0, 0]
        );
        assert_eq!(
            asm(|a| {
                a.sub_imm(Reg::Rax, 0x10);
            }),
            [0x48, 0x81, 0xE8, 0x10, 0, 0, 0]
        );
    }

    #[test]
    fn mem_operands() {
        // mov [rbx+0x10], rax
        assert_eq!(
            asm(|a| {
                a.store(Reg::Rbx, 0x10, Reg::Rax);
            }),
            [0x48, 0x89, 0x83, 0x10, 0, 0, 0]
        );
        // mov [rsp+0x10], rax  -> needs SIB 0x24
        assert_eq!(
            asm(|a| {
                a.store(Reg::Rsp, 0x10, Reg::Rax);
            }),
            [0x48, 0x89, 0x84, 0x24, 0x10, 0, 0, 0]
        );
        // mov rax, [rbx+0x10]
        assert_eq!(
            asm(|a| {
                a.load(Reg::Rax, Reg::Rbx, 0x10);
            }),
            [0x48, 0x8B, 0x83, 0x10, 0, 0, 0]
        );
        // mov [r13+0x4], r9
        assert_eq!(
            asm(|a| {
                a.store(Reg::R13, 4, Reg::R9);
            }),
            [0x4D, 0x89, 0x8D, 4, 0, 0, 0]
        );
    }

    #[test]
    fn jmp_abs_indirect() {
        let b = asm(|a| {
            a.jmp_abs(0xDEAD_BEEF_1234_5678);
        });
        assert_eq!(&b[..6], &[0xFF, 0x25, 0, 0, 0, 0]);
        assert_eq!(
            u64::from_le_bytes(b[6..14].try_into().unwrap()),
            0xDEAD_BEEF_1234_5678
        );
    }

    #[test]
    fn rel32_label_resolution() {
        // jmp forward over a nop to a label
        let mut a = Asm::new();
        a.jmp_label("end");
        a.nop();
        a.label("end");
        a.ret();
        let bytes = a.assemble(0x4000).unwrap();
        // E9 rel32 | 90 | C3 ; rel = (5+1) - 5 = 1
        assert_eq!(bytes[0], 0xE9);
        assert_eq!(i32::from_le_bytes(bytes[1..5].try_into().unwrap()), 1);
        assert_eq!(&bytes[5..], &[0x90, 0xC3]);
    }

    #[test]
    fn undefined_label_errors() {
        let mut a = Asm::new();
        a.jmp_label("nowhere");
        assert!(matches!(
            a.assemble(0x1000),
            Err(AsmError::UndefinedLabel(_))
        ));
    }

    #[test]
    fn macro_smoke() {
        let code = asm64! {
            pushad;
            movabs rax, 0x1000u64;
            store [rbx + 0x10], rax;
            xor rax, rax;
            popad;
            ret;
        };
        let bytes = code.assemble(0x2000).unwrap();
        assert_eq!(bytes[0], 0x50); // push rax
        assert_eq!(*bytes.last().unwrap(), 0xC3); // ret
    }
}
