//! SSE and SSE2 (the x86-64 baseline), opcode map 0F. XMM registers are handled as `u128`
//! (little-endian); the upper half of YMM is left as it is, as the legacy encodings do.

use crate::cpu::*;
use crate::exec::{Ctx, Rm, R};

/// Element-wise integer operation on lanes of `w` bytes.
fn lanes(a: u128, b: u128, w: u32, f: impl Fn(u64, u64) -> u64) -> u128 {
    let bits = w * 8;
    let m = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let mut r = 0u128;
    for i in 0..16 / w {
        let x = (a >> (i * bits)) as u64 & m;
        let y = (b >> (i * bits)) as u64 & m;
        r |= ((f(x, y) & m) as u128) << (i * bits);
    }
    r
}

fn sxl(v: u64, w: u32) -> i64 {
    match w {
        1 => v as i8 as i64,
        2 => v as i16 as i64,
        4 => v as i32 as i64,
        _ => v as i64,
    }
}

fn ps(a: u128) -> [f32; 4] {
    std::array::from_fn(|i| f32::from_bits((a >> (i * 32)) as u32))
}
fn from_ps(v: [f32; 4]) -> u128 {
    v.iter().enumerate().fold(0, |r, (i, x)| r | (x.to_bits() as u128) << (i * 32))
}
fn pd(a: u128) -> [f64; 2] {
    [f64::from_bits(a as u64), f64::from_bits((a >> 64) as u64)]
}
fn from_pd(v: [f64; 2]) -> u128 {
    v[0].to_bits() as u128 | (v[1].to_bits() as u128) << 64
}

/// ADD MUL SUB MIN DIV MAX (opcodes 58 59 5C 5D 5E 5F), with x86's operand order for MIN / MAX
/// (the second operand when either is a NaN or both are zero).
macro_rules! arith {
    ($name:ident, $t:ty) => {
        fn $name(op: u8, x: $t, y: $t) -> $t {
            match op {
                0x58 => x + y,
                0x59 => x * y,
                0x5C => x - y,
                0x5D => {
                    if x < y {
                        x
                    } else {
                        y
                    }
                }
                0x5E => x / y,
                _ => {
                    if x > y {
                        x
                    } else {
                        y
                    }
                }
            }
        }
    };
}
arith!(arith32, f32);
arith!(arith64, f64);

/// CMPPS-family predicate `p` (0..7) → all ones / zero.
fn compare<T: PartialOrd>(p: u8, x: T, y: T) -> bool {
    let unordered = x.partial_cmp(&y).is_none();
    match p & 7 {
        0 => x == y,
        1 => x < y,
        2 => x <= y,
        3 => unordered,
        4 => x != y,
        5 => !(x < y),
        6 => !(x <= y),
        _ => !unordered,
    }
}

fn f32_to_i32(x: f32, trunc: bool) -> u32 {
    let r = if trunc { x.trunc() } else { x.round_ties_even() };
    if r.is_nan() || r >= 2147483648.0 || r < -2147483648.0 {
        0x8000_0000
    } else {
        r as i32 as u32
    }
}
fn f64_to_i32(x: f64, trunc: bool) -> u32 {
    let r = if trunc { x.trunc() } else { x.round_ties_even() };
    if r.is_nan() || r >= 2147483648.0 || r < -2147483648.0 {
        0x8000_0000
    } else {
        r as i32 as u32
    }
}
fn f64_to_i64(x: f64, trunc: bool) -> u64 {
    let r = if trunc { x.trunc() } else { x.round_ties_even() };
    if r.is_nan() || r >= 9223372036854775808.0 || r < -9223372036854775808.0 {
        0x8000_0000_0000_0000
    } else {
        r as i64 as u64
    }
}

fn saturate_s(v: i64, w: u32) -> u64 {
    let (lo, hi) = match w {
        1 => (i8::MIN as i64, i8::MAX as i64),
        2 => (i16::MIN as i64, i16::MAX as i64),
        _ => (i32::MIN as i64, i32::MAX as i64),
    };
    v.clamp(lo, hi) as u64
}
fn saturate_u(v: i64, w: u32) -> u64 {
    v.clamp(0, if w == 1 { 0xFF } else { 0xFFFF }) as u64
}

/// PACKSSWB / PACKUSWB / PACKSSDW: the lanes of `a` then `b` (`w` bytes each), saturated to half.
fn pack(a: u128, b: u128, w: u32, unsigned: bool) -> u128 {
    let n = 16 / w;
    let half = w / 2;
    let mut r = 0u128;
    for i in 0..2 * n {
        let src = if i < n { a >> (i * w * 8) } else { b >> ((i - n) * w * 8) };
        let v = sxl(src as u64 & ((1u64 << (w * 8)) - 1), w);
        let s = if unsigned { saturate_u(v, half) } else { saturate_s(v, half) };
        r |= ((s & ((1u64 << (half * 8)) - 1)) as u128) << (i * half * 8);
    }
    r
}

/// PUNPCKL* / PUNPCKH*: interleave the low or high lanes of `a` and `b`.
fn unpack(a: u128, b: u128, w: u32, high: bool) -> u128 {
    let bits = w * 8;
    let n = 16 / w;
    let m = if bits == 64 { u128::from(u64::MAX) } else { (1u128 << bits) - 1 };
    let mut r = 0u128;
    for i in 0..n / 2 {
        let s = if high { i + n / 2 } else { i };
        r |= ((a >> (s * bits)) & m) << (2 * i * bits);
        r |= ((b >> (s * bits)) & m) << ((2 * i + 1) * bits);
    }
    r
}

/// Shift every lane (`w` bytes) by `count` bits: 0 left, 1 logical right, 2 arithmetic right.
fn shift_lanes(a: u128, w: u32, count: u64, kind: u8) -> u128 {
    let bits = (w * 8) as u64;
    lanes(a, 0, w, |x, _| match kind {
        0 => {
            if count >= bits {
                0
            } else {
                x << count
            }
        }
        1 => {
            if count >= bits {
                0
            } else {
                x >> count
            }
        }
        _ => (sxl(x, w) >> count.min(bits - 1)) as u64,
    })
}

impl Ctx<'_> {
    /// The 128 bits of vector register `r` the instruction works on (see `Ctx::half`).
    #[inline]
    pub(crate) fn x(&self, r: u8) -> u128 {
        let v = &self.cpu.v[r as usize];
        let h = self.half as usize * 2;
        v[h] as u128 | (v[h + 1] as u128) << 64
    }

    /// Write them. A VEX instruction on XMM registers clears the upper half of the YMM register.
    #[inline]
    pub(crate) fn set_x(&mut self, r: u8, v: u128) {
        let h = self.half as usize * 2;
        let reg = &mut self.cpu.v[r as usize];
        reg[h] = v as u64;
        reg[h + 1] = (v >> 64) as u64;
        if self.vex && !self.l {
            reg[2] = 0;
            reg[3] = 0;
        }
    }

    /// The first source of a two-source instruction: the destination itself, or the register a
    /// VEX prefix names.
    #[inline]
    pub(crate) fn src1(&self, reg: u8) -> u128 {
        self.x(if self.vex { self.vvvv } else { reg })
    }

    /// An XMM or memory operand, `bytes` wide in memory (zero-extended).
    pub(crate) fn rd_x(&self, rm: Rm, bytes: u8) -> R<u128> {
        Ok(match rm {
            Rm::Reg(r) => self.x(r),
            _ => {
                let a = self.ea(rm);
                match bytes {
                    4 => self.mem.read_u32(a)? as u128,
                    8 => self.mem.read_u64(a)? as u128,
                    _ => u128::from_le_bytes(self.mem.read_n(a + 16 * self.half as u64)?),
                }
            }
        })
    }

    /// The low 128 bits of an operand whichever half is worked on (a shift count).
    fn rd_x_low(&mut self, rm: Rm) -> R<u128> {
        let half = std::mem::take(&mut self.half);
        let v = self.rd_x(rm, 16);
        self.half = half;
        v
    }

    pub(crate) fn wr_x(&mut self, rm: Rm, bytes: u8, v: u128) -> R<()> {
        match rm {
            Rm::Reg(r) => self.set_x(r, v),
            _ => {
                let a = self.ea(rm);
                match bytes {
                    4 => self.mem.write_u32(a, v as u32)?,
                    8 => self.mem.write_u64(a, v as u64)?,
                    _ => self.mem.write(a + 16 * self.half as u64, &v.to_le_bytes())?,
                }
            }
        }
        Ok(())
    }

    /// Flags of UCOMISS / COMISD and friends.
    fn fcompare<T: PartialOrd>(&mut self, a: T, b: T) {
        self.cpu.rflags &= !(CF | PF | ZF | OF | SF | AF);
        self.cpu.rflags |= match a.partial_cmp(&b) {
            None => ZF | PF | CF,
            Some(std::cmp::Ordering::Less) => CF,
            Some(std::cmp::Ordering::Equal) => ZF,
            Some(std::cmp::Ordering::Greater) => 0,
        };
    }
}

const LOW32: u128 = 0xFFFF_FFFF;
const LOW64: u128 = 0xFFFF_FFFF_FFFF_FFFF;

/// Execute the 0F-map instruction `op`. `false`: not one this module implements.
pub(crate) fn exec(c: &mut Ctx, op: u8) -> R<bool> {
    // the mandatory prefix: none (packed single), 66 (packed double / integer), F3, F2
    let p66 = c.opsz16 && c.rep == 0;
    let f3 = c.rep == 0xF3;
    let f2 = c.rep == 0xF2;
    let none = !c.opsz16 && c.rep == 0;
    // scalar operand width in memory
    let scalar = if f3 { 4 } else { 8 };
    match op {
        // movups / movupd / movss / movsd, movaps / movapd, movdqa / movdqu: load
        0x10 | 0x28 | 0x6F => {
            if op == 0x6F && !(p66 || f3) || op == 0x28 && (f3 || f2) {
                return Ok(false);
            }
            let (reg, rm) = c.modrm()?;
            if op == 0x10 && (f3 || f2) {
                let v = c.rd_x(rm, scalar)?;
                let keep = if f3 { LOW32 } else { LOW64 };
                let r = match rm {
                    Rm::Reg(_) => (c.src1(reg) & !keep) | (v & keep),
                    _ => v,
                };
                c.set_x(reg, r);
            } else {
                let v = c.rd_x(rm, 16)?;
                c.set_x(reg, v);
            }
        }
        // … store
        0x11 | 0x29 | 0x7F | 0x2B | 0xE7 => {
            if op == 0x7F && !(p66 || f3) || matches!(op, 0x29 | 0x2B) && (f3 || f2) || op == 0xE7 && !p66 {
                return Ok(false);
            }
            let (reg, rm) = c.modrm()?;
            let v = c.x(reg);
            if op == 0x11 && (f3 || f2) {
                let keep = if f3 { LOW32 } else { LOW64 };
                match rm {
                    Rm::Reg(r) => {
                        let merged = (if c.vex { c.x(c.vvvv) } else { c.x(r) } & !keep) | (v & keep);
                        c.set_x(r, merged);
                    }
                    _ => c.wr_x(rm, scalar, v)?,
                }
            } else {
                c.wr_x(rm, 16, v)?;
            }
        }
        // movlps / movlpd load (register form: movhlps)
        0x12 if none || p66 => {
            let (reg, rm) = c.modrm()?;
            let lo = match rm {
                Rm::Reg(r) => c.x(r) >> 64,
                _ => c.rd_x(rm, 8)?,
            };
            let v = (c.src1(reg) & !LOW64) | (lo & LOW64);
            c.set_x(reg, v);
        }
        0x13 if none || p66 => {
            let (reg, rm) = c.modrm()?;
            let v = c.x(reg);
            c.wr_x(rm, 8, v & LOW64)?;
        }
        // unpcklps / unpckhps / unpcklpd / unpckhpd
        0x14 | 0x15 if none || p66 => {
            let (reg, rm) = c.modrm()?;
            let b = c.rd_x(rm, 16)?;
            let v = unpack(c.src1(reg), b, if p66 { 8 } else { 4 }, op == 0x15);
            c.set_x(reg, v);
        }
        // movhps / movhpd load (register form: movlhps)
        0x16 if none || p66 => {
            let (reg, rm) = c.modrm()?;
            let hi = c.rd_x(rm, 8)? & LOW64;
            let v = (c.src1(reg) & LOW64) | hi << 64;
            c.set_x(reg, v);
        }
        0x17 if none || p66 => {
            let (reg, rm) = c.modrm()?;
            let v = c.x(reg) >> 64;
            c.wr_x(rm, 8, v)?;
        }
        // cvtsi2ss / cvtsi2sd
        0x2A if f3 || f2 => {
            let sz = if c.rex & 8 != 0 { 8 } else { 4 };
            let (reg, rm) = c.modrm()?;
            let i = crate::exec::sx(c.rd_rm(rm, sz)?, sz) as i64;
            let old = c.src1(reg);
            let v = if f3 { (old & !LOW32) | (i as f32).to_bits() as u128 } else { (old & !LOW64) | (i as f64).to_bits() as u128 };
            c.set_x(reg, v);
        }
        // cvttss2si / cvttsd2si / cvtss2si / cvtsd2si
        0x2C | 0x2D if f3 || f2 => {
            let sz = if c.rex & 8 != 0 { 8 } else { 4 };
            let (reg, rm) = c.modrm()?;
            let v = c.rd_x(rm, scalar)?;
            let x = if f3 { f32::from_bits(v as u32) as f64 } else { f64::from_bits(v as u64) };
            let trunc = op == 0x2C;
            let r = if sz == 8 { f64_to_i64(x, trunc) } else { f64_to_i32(x, trunc) as u64 };
            c.sreg(reg, sz, r);
        }
        // ucomiss / comiss / ucomisd / comisd
        0x2E | 0x2F if none || p66 => {
            let (reg, rm) = c.modrm()?;
            let a = c.x(reg);
            if p66 {
                let b = c.rd_x(rm, 8)?;
                c.fcompare(f64::from_bits(a as u64), f64::from_bits(b as u64));
            } else {
                let b = c.rd_x(rm, 4)?;
                c.fcompare(f32::from_bits(a as u32), f32::from_bits(b as u32));
            }
        }
        // movmskps / movmskpd
        0x50 if none || p66 => {
            let (reg, rm) = c.modrm()?;
            let Rm::Reg(r) = rm else { return Ok(false) };
            let v = c.x(r);
            let m = if p66 { (v >> 63) as u64 & 1 | ((v >> 127) as u64 & 1) << 1 } else { (0..4).fold(0, |m, i| m | ((v >> (i * 32 + 31)) as u64 & 1) << i) };
            // the upper half of a YMM source: its bits above the lower half's
            let m = if c.half == 1 { c.greg(reg, 8) | m << if p66 { 2 } else { 4 } } else { m };
            c.sreg(reg, 8, m);
        }
        // sqrt, rsqrt (ps / ss), rcp (ps / ss)
        0x51..=0x53 => {
            if op != 0x51 && (p66 || f2) {
                return Ok(false);
            }
            let (reg, rm) = c.modrm()?;
            let single = |x: f32| match op {
                0x51 => x.sqrt(),
                0x52 => 1.0 / x.sqrt(),
                _ => 1.0 / x,
            };
            let old = c.src1(reg);
            let v = if none {
                from_ps(ps(c.rd_x(rm, 16)?).map(single))
            } else if p66 {
                from_pd(pd(c.rd_x(rm, 16)?).map(f64::sqrt))
            } else if f3 {
                (old & !LOW32) | single(f32::from_bits(c.rd_x(rm, 4)? as u32)).to_bits() as u128
            } else {
                (old & !LOW64) | f64::from_bits(c.rd_x(rm, 8)? as u64).sqrt().to_bits() as u128
            };
            c.set_x(reg, v);
        }
        // andps andnps orps xorps (and the pd forms), pand pandn por pxor
        0x54..=0x57 | 0xDB | 0xDF | 0xEB | 0xEF => {
            if op >= 0xDB && !p66 || op < 0xDB && (f3 || f2) {
                return Ok(false);
            }
            let (reg, rm) = c.modrm()?;
            let b = c.rd_x(rm, 16)?;
            let a = c.src1(reg);
            let v = match op {
                0x54 | 0xDB => a & b,
                0x55 | 0xDF => !a & b,
                0x56 | 0xEB => a | b,
                _ => a ^ b,
            };
            c.set_x(reg, v);
        }
        // add mul sub min div max
        0x58 | 0x59 | 0x5C..=0x5F => {
            let (reg, rm) = c.modrm()?;
            let a = c.src1(reg);
            let v = if none {
                let (x, y) = (ps(a), ps(c.rd_x(rm, 16)?));
                from_ps(std::array::from_fn(|i| arith32(op, x[i], y[i])))
            } else if p66 {
                let (x, y) = (pd(a), pd(c.rd_x(rm, 16)?));
                from_pd(std::array::from_fn(|i| arith64(op, x[i], y[i])))
            } else if f3 {
                let y = f32::from_bits(c.rd_x(rm, 4)? as u32);
                (a & !LOW32) | arith32(op, f32::from_bits(a as u32), y).to_bits() as u128
            } else {
                let y = f64::from_bits(c.rd_x(rm, 8)? as u64);
                (a & !LOW64) | arith64(op, f64::from_bits(a as u64), y).to_bits() as u128
            };
            c.set_x(reg, v);
        }
        // cvtps2pd cvtpd2ps cvtss2sd cvtsd2ss
        0x5A => {
            let (reg, rm) = c.modrm()?;
            let a = c.src1(reg);
            let v = if none {
                let x = ps(c.rd_x(rm, 8)?);
                from_pd([x[0] as f64, x[1] as f64])
            } else if p66 {
                let x = pd(c.rd_x(rm, 16)?);
                from_ps([x[0] as f32, x[1] as f32, 0.0, 0.0])
            } else if f3 {
                (a & !LOW64) | (f32::from_bits(c.rd_x(rm, 4)? as u32) as f64).to_bits() as u128
            } else {
                (a & !LOW32) | (f64::from_bits(c.rd_x(rm, 8)? as u64) as f32).to_bits() as u128
            };
            c.set_x(reg, v);
        }
        // cvtdq2ps cvtps2dq cvttps2dq
        0x5B if !f2 => {
            let (reg, rm) = c.modrm()?;
            let b = c.rd_x(rm, 16)?;
            let v = if none {
                from_ps(std::array::from_fn(|i| (b >> (i * 32)) as u32 as i32 as f32))
            } else {
                ps(b).iter().enumerate().fold(0u128, |r, (i, &x)| r | (f32_to_i32(x, f3) as u128) << (i * 32))
            };
            c.set_x(reg, v);
        }
        // punpckl*, packsswb, pcmpgt*, packuswb, punpckh*, packssdw, punpck*qdq
        0x60..=0x6D if p66 => {
            let (reg, rm) = c.modrm()?;
            let b = c.rd_x(rm, 16)?;
            let a = c.src1(reg);
            let v = match op {
                0x60..=0x62 => unpack(a, b, 1 << (op - 0x60), false),
                0x63 => pack(a, b, 2, false),
                0x64..=0x66 => {
                    let w = 1 << (op - 0x64);
                    lanes(a, b, w, |x, y| if sxl(x, w) > sxl(y, w) { !0 } else { 0 })
                }
                0x67 => pack(a, b, 2, true),
                0x68..=0x6A => unpack(a, b, 1 << (op - 0x68), true),
                0x6B => pack(a, b, 4, false),
                0x6C => unpack(a, b, 8, false),
                _ => unpack(a, b, 8, true),
            };
            c.set_x(reg, v);
        }
        // movd / movq xmm, r/m
        0x6E if p66 => {
            let sz = if c.rex & 8 != 0 { 8 } else { 4 };
            let (reg, rm) = c.modrm()?;
            let v = c.rd_rm(rm, sz)?;
            c.set_x(reg, v as u128);
        }
        // pshufd / pshufhw / pshuflw
        0x70 if p66 || f3 || f2 => {
            let (reg, rm) = c.modrm()?;
            let order = c.fetch8()? as u32;
            let b = c.rd_x(rm, 16)?;
            let v = if p66 {
                (0..4).fold(0u128, |r, i| r | ((b >> (((order >> (i * 2)) & 3) * 32)) & LOW32) << (i * 32))
            } else {
                let (half, other) = if f3 { ((b >> 64) as u64, b & LOW64) } else { (b as u64, b & !LOW64) };
                let s = (0..4).fold(0u64, |r, i| r | ((half >> (((order >> (i * 2)) & 3) * 16)) & 0xFFFF) << (i * 16));
                if f3 {
                    other | (s as u128) << 64
                } else {
                    other | s as u128
                }
            };
            c.set_x(reg, v);
        }
        // shifts by imm8: psrlw psraw psllw / psrld psrad pslld / psrlq psrldq psllq pslldq
        0x71..=0x73 if p66 => {
            let (reg, rm) = c.modrm()?;
            let Rm::Reg(r) = rm else { return Ok(false) };
            let count = c.fetch8()? as u64;
            let w = 2 << (op - 0x71);
            // the VEX form writes the register the prefix names
            let dst = if c.vex { c.vvvv } else { r };
            let a = c.x(r);
            let v = match reg & 7 {
                2 => shift_lanes(a, w, count, 1),
                4 if op != 0x73 => shift_lanes(a, w, count, 2),
                6 => shift_lanes(a, w, count, 0),
                3 if op == 0x73 => {
                    if count > 15 {
                        0
                    } else {
                        a >> (count * 8)
                    }
                }
                7 if op == 0x73 => {
                    if count > 15 {
                        0
                    } else {
                        a << (count * 8)
                    }
                }
                _ => return Ok(false),
            };
            c.set_x(dst, v);
        }
        // pcmpeqb / w / d
        0x74..=0x76 if p66 => {
            let (reg, rm) = c.modrm()?;
            let b = c.rd_x(rm, 16)?;
            let v = lanes(c.src1(reg), b, 1 << (op - 0x74), |x, y| if x == y { !0 } else { 0 });
            c.set_x(reg, v);
        }
        // emms
        0x77 if none => {}
        // movd / movq r/m, xmm; movq xmm, xmm/m64
        0x7E if p66 => {
            let sz = if c.rex & 8 != 0 { 8 } else { 4 };
            let (reg, rm) = c.modrm()?;
            let v = c.x(reg) as u64;
            c.wr_rm(rm, sz, v)?;
        }
        0x7E if f3 => {
            let (reg, rm) = c.modrm()?;
            let v = c.rd_x(rm, 8)? & LOW64;
            c.set_x(reg, v);
        }
        // cmpps / cmppd / cmpss / cmpsd
        0xC2 => {
            let (reg, rm) = c.modrm()?;
            let p = c.fetch8()?;
            let a = c.src1(reg);
            let v = if none {
                let (x, y) = (ps(a), ps(c.rd_x(rm, 16)?));
                (0..4).fold(0u128, |r, i| r | if compare(p, x[i], y[i]) { LOW32 << (i * 32) } else { 0 })
            } else if p66 {
                let (x, y) = (pd(a), pd(c.rd_x(rm, 16)?));
                (0..2).fold(0u128, |r, i| r | if compare(p, x[i], y[i]) { LOW64 << (i * 64) } else { 0 })
            } else if f3 {
                let y = f32::from_bits(c.rd_x(rm, 4)? as u32);
                (a & !LOW32) | if compare(p, f32::from_bits(a as u32), y) { LOW32 } else { 0 }
            } else {
                let y = f64::from_bits(c.rd_x(rm, 8)? as u64);
                (a & !LOW64) | if compare(p, f64::from_bits(a as u64), y) { LOW64 } else { 0 }
            };
            c.set_x(reg, v);
        }
        // pinsrw / pextrw
        0xC4 if p66 => {
            let (reg, rm) = c.modrm()?;
            let i = (c.fetch8()? & 7) as u32 * 16;
            let w = match rm {
                Rm::Reg(r) => c.greg(r, 2),
                _ => c.rd_rm(rm, 2)?,
            } as u128;
            let v = (c.src1(reg) & !(0xFFFF << i)) | w << i;
            c.set_x(reg, v);
        }
        0xC5 if p66 => {
            let (reg, rm) = c.modrm()?;
            let Rm::Reg(r) = rm else { return Ok(false) };
            let i = (c.fetch8()? & 7) as u32 * 16;
            let v = (c.x(r) >> i) as u64 & 0xFFFF;
            c.sreg(reg, 8, v);
        }
        // shufps / shufpd
        0xC6 if none || p66 => {
            let (reg, rm) = c.modrm()?;
            let order = c.fetch8()? as u32;
            let b = c.rd_x(rm, 16)?;
            let a = c.src1(reg);
            let v = if p66 {
                ((a >> ((order & 1) * 64)) & LOW64) | ((b >> (((order >> 1) & 1) * 64)) & LOW64) << 64
            } else {
                let pick = |src: u128, i: u32| (src >> (((order >> (i * 2)) & 3) * 32)) & LOW32;
                pick(a, 0) | pick(a, 1) << 32 | pick(b, 2) << 64 | pick(b, 3) << 96
            };
            c.set_x(reg, v);
        }
        // movq xmm/m64, xmm
        0xD6 if p66 => {
            let (reg, rm) = c.modrm()?;
            let v = c.x(reg) & LOW64;
            match rm {
                Rm::Reg(r) => c.set_x(r, v),
                _ => c.wr_x(rm, 8, v)?,
            }
        }
        // pmovmskb
        0xD7 if p66 => {
            let (reg, rm) = c.modrm()?;
            let Rm::Reg(r) = rm else { return Ok(false) };
            let v = c.x(r);
            let m = (0..16).fold(0u64, |m, i| m | ((v >> (i * 8 + 7)) as u64 & 1) << i);
            let m = if c.half == 1 { c.greg(reg, 8) | m << 16 } else { m };
            c.sreg(reg, 8, m);
        }
        // cvttpd2dq / cvtdq2pd / cvtpd2dq
        0xE6 if p66 || f3 || f2 => {
            let (reg, rm) = c.modrm()?;
            let v = if f3 {
                let b = c.rd_x(rm, 8)?;
                from_pd([b as u32 as i32 as f64, (b >> 32) as u32 as i32 as f64])
            } else {
                let x = pd(c.rd_x(rm, 16)?);
                f64_to_i32(x[0], p66) as u128 | (f64_to_i32(x[1], p66) as u128) << 32
            };
            c.set_x(reg, v);
        }
        // the packed-integer block
        0xD1..=0xFE if p66 => {
            let (reg, rm) = c.modrm()?;
            let b = c.rd_x(rm, 16)?;
            let a = c.src1(reg);
            // a shift's count is the low quadword of its operand, for both halves
            let count = c.rd_x_low(rm)? as u64;
            let v = match op {
                0xD1..=0xD3 => shift_lanes(a, 2 << (op - 0xD1), count, 1),
                0xE1 | 0xE2 => shift_lanes(a, 2 << (op - 0xE1), count, 2),
                0xF1..=0xF3 => shift_lanes(a, 2 << (op - 0xF1), count, 0),
                0xD4 => lanes(a, b, 8, u64::wrapping_add),
                0xD5 => lanes(a, b, 2, u64::wrapping_mul),
                0xD8 | 0xD9 => lanes(a, b, 1 << (op - 0xD8), u64::saturating_sub),
                0xDA => lanes(a, b, 1, u64::min),
                0xDC | 0xDD => {
                    let w = 1 << (op - 0xDC);
                    lanes(a, b, w, |x, y| saturate_u((x + y) as i64, w))
                }
                0xDE => lanes(a, b, 1, u64::max),
                0xE0 | 0xE3 => lanes(a, b, if op == 0xE0 { 1 } else { 2 }, |x, y| (x + y + 1) >> 1),
                0xE4 => lanes(a, b, 2, |x, y| (x * y) >> 16),
                0xE5 => lanes(a, b, 2, |x, y| ((sxl(x, 2) * sxl(y, 2)) >> 16) as u64),
                0xE8 | 0xE9 => {
                    let w = 1 << (op - 0xE8);
                    lanes(a, b, w, |x, y| saturate_s(sxl(x, w) - sxl(y, w), w))
                }
                0xEA => lanes(a, b, 2, |x, y| sxl(x, 2).min(sxl(y, 2)) as u64),
                0xEC | 0xED => {
                    let w = 1 << (op - 0xEC);
                    lanes(a, b, w, |x, y| saturate_s(sxl(x, w) + sxl(y, w), w))
                }
                0xEE => lanes(a, b, 2, |x, y| sxl(x, 2).max(sxl(y, 2)) as u64),
                0xF4 => lanes(a, b, 8, |x, y| (x & 0xFFFF_FFFF) * (y & 0xFFFF_FFFF)),
                0xF5 => {
                    (0..4).fold(0u128, |r, i| {
                        let t = |s: u128, k: u32| sxl((s >> ((2 * i + k) * 16)) as u64 & 0xFFFF, 2);
                        let sum = (t(a, 0) * t(b, 0) + t(a, 1) * t(b, 1)) as u32;
                        r | (sum as u128) << (i * 32)
                    })
                }
                0xF6 => {
                    let sad = |shift: u32| (0..8).map(|i| ((a >> (shift + i * 8)) as u8).abs_diff((b >> (shift + i * 8)) as u8) as u128).sum::<u128>();
                    sad(0) | sad(64) << 64
                }
                0xF8..=0xFB => lanes(a, b, 1 << (op - 0xF8), u64::wrapping_sub),
                0xFC..=0xFE => lanes(a, b, 1 << (op - 0xFC), u64::wrapping_add),
                _ => return Ok(false),
            };
            c.set_x(reg, v);
        }
        _ => return Ok(false),
    }
    Ok(true)
}
