//! VEX-encoded instructions and the three-byte opcode maps: AVX / AVX2 forms of the SSE
//! instructions (run by `sse`, once per 128-bit half), and BMI1, BMI2, MOVBE, CRC32, ADX.

use crate::cpu::*;
use crate::exec::{mask, sign_bit, Ctx, Rm, R};
use crate::sse::{from_pd, from_ps, lanes, pd, ps, saturate_s, saturate_u, shift_lanes, sxl, LOW32, LOW64};

/// Run `kernel` (one instruction on 128-bit operands) on the lower halves and, for a 256-bit
/// instruction, again on the upper ones: it decodes its operands itself, so the instruction's
/// bytes are read a second time.
fn per_half(c: &mut Ctx, kernel: impl Fn(&mut Ctx) -> R<bool>) -> R<bool> {
    let start = c.pc;
    if !kernel(c)? {
        return Ok(false);
    }
    if c.l {
        c.pc = start;
        c.half = 1;
        kernel(c)?;
        c.half = 0;
    }
    Ok(true)
}

impl Ctx<'_> {
    /// All 256 bits of a vector register, as its two halves.
    fn y(&self, r: u8) -> [u128; 2] {
        let v = &self.cpu.v[r as usize];
        [v[0] as u128 | (v[1] as u128) << 64, v[2] as u128 | (v[3] as u128) << 64]
    }

    /// Write a vector register whole; a 128-bit VEX result has a zero upper half.
    fn set_y(&mut self, r: u8, v: [u128; 2]) {
        self.cpu.v[r as usize] = [v[0] as u64, (v[0] >> 64) as u64, v[1] as u64, (v[1] >> 64) as u64];
    }

    /// A whole register or 32 bytes of memory.
    fn rd_y(&self, rm: Rm) -> R<[u128; 2]> {
        Ok(match rm {
            Rm::Reg(r) => self.y(r),
            _ => {
                let a = self.ea(rm);
                [u128::from_le_bytes(self.mem.read_n(a)?), u128::from_le_bytes(self.mem.read_n(a + 16)?)]
            }
        })
    }

    /// The low `bytes` of a register or that many bytes of memory, whatever half is worked on.
    fn rd_low(&self, rm: Rm, bytes: u8) -> R<u128> {
        Ok(match rm {
            Rm::Reg(r) => self.y(r)[0],
            _ => {
                let a = self.ea(rm);
                match bytes {
                    1 => self.mem.read_u8(a)? as u128,
                    2 => self.mem.read_u16(a)? as u128,
                    4 => self.mem.read_u32(a)? as u128,
                    8 => self.mem.read_u64(a)? as u128,
                    _ => u128::from_le_bytes(self.mem.read_n(a)?),
                }
            }
        })
    }
}

/// Element `i` (of `w` bytes) of a vector held as bytes.
fn element(v: &[u8], i: usize, w: usize) -> u64 {
    let mut b = [0u8; 8];
    b[..w].copy_from_slice(&v[i * w..(i + 1) * w]);
    u64::from_le_bytes(b)
}

fn bytes_of(v: [u128; 2]) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[..16].copy_from_slice(&v[0].to_le_bytes());
    b[16..].copy_from_slice(&v[1].to_le_bytes());
    b
}

fn halves_of(b: &[u8; 32]) -> [u128; 2] {
    [u128::from_le_bytes(b[..16].try_into().unwrap()), u128::from_le_bytes(b[16..].try_into().unwrap())]
}

/// Half-precision → single.
fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = ((h >> 15) as u32, ((h >> 10) & 0x1F) as u32, (h & 0x3FF) as u32);
    let bits = match exp {
        0 if frac == 0 => 0,
        0 => {
            // subnormal: normalise
            let shift = frac.leading_zeros() - 21;
            ((113 - shift) << 23) | ((frac << shift) & 0x3FF) << 13
        }
        31 => 0x7F80_0000 | frac << 13 | if frac != 0 { 0x40_0000 } else { 0 },
        _ => (exp + 112) << 23 | frac << 13,
    };
    f32::from_bits(sign << 31 | bits)
}

/// Single → half-precision, round to nearest even.
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7F_FFFF;
    if exp == 255 {
        return sign | 0x7C00 | if frac != 0 { 0x200 | (frac >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00;
    }
    // the significand with its leading one, shifted to 10 fraction bits (more for subnormals)
    let (m, shift) = if e <= 0 { (frac | 0x80_0000, 14 - e) } else { (frac, 13) };
    if shift > 24 {
        return sign;
    }
    let half = 1u32 << (shift - 1);
    let rounded = (m >> shift) + ((m & half != 0 && (m & (half - 1) != 0 || (m >> shift) & 1 != 0)) as u32);
    sign | ((e.max(0) as u16) << 10).wrapping_add(rounded as u16)
}

/// Round as ROUNDPS / ROUNDSD's immediate says (bit 2: the current mode, which is nearest).
fn round_f64(x: f64, mode: u8) -> f64 {
    match if mode & 4 != 0 { 0 } else { mode & 3 } {
        0 => x.round_ties_even(),
        1 => x.floor(),
        2 => x.ceil(),
        _ => x.trunc(),
    }
}

fn round_f32(x: f32, mode: u8) -> f32 {
    match if mode & 4 != 0 { 0 } else { mode & 3 } {
        0 => x.round_ties_even(),
        1 => x.floor(),
        2 => x.ceil(),
        _ => x.trunc(),
    }
}

/// Scalar instructions of map 0F: they work on the low element whatever the vector length says.
fn is_scalar(c: &Ctx, op: u8) -> bool {
    match op {
        0x2E | 0x2F => true,
        0x10 | 0x11 | 0x2A | 0x2C | 0x2D | 0x51..=0x53 | 0x58..=0x5F | 0xC2 => c.rep != 0,
        _ => false,
    }
}

/// Execute the instruction `op` of opcode map `map` (1: 0F, 2: 0F 38, 3: 0F 3A). `false`: not
/// implemented.
pub(crate) fn exec(c: &mut Ctx, map: u8, op: u8) -> R<bool> {
    match map {
        1 => map_0f(c, op),
        2 => map_0f38(c, op),
        3 => map_0f3a(c, op),
        _ => Ok(false),
    }
}

/// A VEX instruction of map 0F: the SSE instruction, on one half or on both.
fn map_0f(c: &mut Ctx, op: u8) -> R<bool> {
    // vzeroupper / vzeroall
    if op == 0x77 {
        for v in c.cpu.v.iter_mut() {
            if c.l {
                *v = [0; 4];
            } else {
                v[2] = 0;
                v[3] = 0;
            }
        }
        return Ok(true);
    }
    if is_scalar(c, op) {
        c.l = false;
    }
    // conversions between four doubles and four singles / dwords: one side is 128 bits wide
    if c.l && (op == 0x5A && c.rep == 0 || op == 0xE6) {
        let (reg, rm) = c.modrm()?;
        let out = match (op, c.opsz16, c.rep) {
            // vcvtps2pd ymm, xmm/m128
            (0x5A, false, _) => {
                let x = ps(c.rd_low(rm, 16)?);
                [from_pd([x[0] as f64, x[1] as f64]), from_pd([x[2] as f64, x[3] as f64])]
            }
            // vcvtdq2pd ymm, xmm/m128
            (0xE6, _, 0xF3) => {
                let b = c.rd_low(rm, 16)?;
                let d = |i: u32| (b >> (32 * i)) as u32 as i32 as f64;
                [from_pd([d(0), d(1)]), from_pd([d(2), d(3)])]
            }
            // vcvtpd2ps / vcvtpd2dq / vcvttpd2dq xmm, ymm/m256
            _ => {
                let y = c.rd_y(rm)?;
                let x = [pd(y[0])[0], pd(y[0])[1], pd(y[1])[0], pd(y[1])[1]];
                let lo = if op == 0x5A {
                    from_ps(x.map(|v| v as f32))
                } else {
                    x.iter().enumerate().fold(0u128, |r, (i, &v)| r | (crate::sse::f64_to_i32(v, c.opsz16) as u128) << (32 * i))
                };
                [lo, 0]
            }
        };
        c.set_y(reg, out);
        return Ok(true);
    }
    let start = c.pc;
    if !crate::sse::exec(c, op)? {
        return Ok(false);
    }
    if c.l {
        // the same instruction again, on the upper halves
        c.pc = start;
        c.half = 1;
        crate::sse::exec(c, op)?;
        c.half = 0;
    }
    Ok(true)
}

/// Flags of the BMI instructions: ZF and SF of the result, CF as given, OF clear.
fn bmi_flags(c: &mut Ctx, r: u64, sz: u8, carry: bool) {
    c.cpu.rflags &= !(CF | OF | ZF | SF);
    c.cpu.set_flag(CF, carry);
    c.cpu.set_flag(ZF, r & mask(sz) == 0);
    c.cpu.set_flag(SF, r & sign_bit(sz) != 0);
}

/// The instructions of map 0F 38 that work lane by lane: one 128-bit half.
fn lanewise_38(c: &mut Ctx, op: u8) -> R<bool> {
    let w = c.rex & 8 != 0;
    let (reg, rm) = c.modrm()?;
    let b = c.rd_x(rm, 16)?;
    let a = c.src1(reg);
    let byte = |v: u128, i: u32| (v >> (8 * i)) as u8;
    let word = |v: u128, i: u32| (v >> (16 * i)) as u16 as i16 as i64;
    let dword = |v: u128, i: u32| (v >> (32 * i)) as u32 as i32 as i64;
    let v: u128 = match op {
        // pshufb
        0x00 => (0..16).fold(0, |r, i| r | if byte(b, i) & 0x80 != 0 { 0 } else { (byte(a, (byte(b, i) & 15) as u32) as u128) << (8 * i) }),
        // phaddw phaddd phaddsw / phsubw phsubd phsubsw: pairs of a, then pairs of b
        0x01..=0x03 | 0x05..=0x07 => {
            let sub = op >= 5;
            let (bytes, n) = if op & 3 == 2 { (4, 2) } else { (2, 4) };
            (0..2 * n).fold(0, |r, i| {
                let src = if i < n { a } else { b };
                let k = 2 * (i % n);
                let (x, y) = if bytes == 2 { (word(src, k), word(src, k + 1)) } else { (dword(src, k), dword(src, k + 1)) };
                let t = if sub { x - y } else { x + y };
                let t = if op & 3 == 3 { saturate_s(t, 2) } else { t as u64 };
                let m = if bytes == 2 { 0xFFFF } else { 0xFFFF_FFFF };
                r | ((t & m) as u128) << (8 * bytes * i)
            })
        }
        // pmaddubsw: unsigned bytes of a × signed bytes of b, neighbours added with saturation
        0x04 => (0..8).fold(0, |r, i| {
            let p = |k: u32| byte(a, k) as i64 * byte(b, k) as i8 as i64;
            r | ((saturate_s(p(2 * i) + p(2 * i + 1), 2) & 0xFFFF) as u128) << (16 * i)
        }),
        // psignb / w / d
        0x08..=0x0A => {
            let w = 1 << (op - 8);
            lanes(a, b, w, |x, y| match sxl(y, w) {
                0 => 0,
                s if s < 0 => x.wrapping_neg(),
                _ => x,
            })
        }
        // pmulhrsw
        0x0B => lanes(a, b, 2, |x, y| ((((sxl(x, 2) * sxl(y, 2)) >> 14) + 1) >> 1) as u64),
        // vpermilps / vpermilpd by a control vector
        0x0C if c.vex => (0..4).fold(0, |r, i| r | ((a >> (32 * ((b >> (32 * i)) as u32 & 3))) & LOW32) << (32 * i)),
        0x0D if c.vex => (0..2).fold(0, |r, i| r | ((a >> (64 * ((b >> (64 * i + 1)) as u32 & 1))) & LOW64) << (64 * i)),
        // pabsb / w / d
        0x1C..=0x1E => {
            let w = 1 << (op - 0x1C);
            lanes(b, 0, w, |x, _| sxl(x, w).unsigned_abs())
        }
        // pmuldq
        0x28 => lanes(a, b, 8, |x, y| (x as u32 as i32 as i64).wrapping_mul(y as u32 as i32 as i64) as u64),
        0x29 => lanes(a, b, 8, |x, y| if x == y { !0 } else { 0 }),
        // movntdqa
        0x2A => b,
        // packusdw
        0x2B => (0..8).fold(0, |r, i| {
            let v = dword(if i < 4 { a } else { b }, i % 4);
            r | ((saturate_u(v, 2) & 0xFFFF) as u128) << (16 * i)
        }),
        0x37 => lanes(a, b, 8, |x, y| if (x as i64) > (y as i64) { !0 } else { 0 }),
        // pminsb pminsd pminuw pminud pmaxsb pmaxsd pmaxuw pmaxud
        0x38..=0x3F => {
            let (w, signed) = match op & 3 {
                0 => (1, true),
                1 => (4, true),
                2 => (2, false),
                _ => (4, false),
            };
            let max = op >= 0x3C;
            lanes(a, b, w, |x, y| {
                let first = if signed { sxl(x, w) < sxl(y, w) } else { x < y };
                if first != max {
                    x
                } else {
                    y
                }
            })
        }
        0x40 => lanes(a, b, 4, u64::wrapping_mul),
        // vpsrlvd/q, vpsravd, vpsllvd/q: every element by its own count
        0x45..=0x47 if c.vex => {
            let bytes = if w { 8 } else { 4 };
            let kind = [1, 2, 0][(op - 0x45) as usize];
            lanes(a, b, bytes, |x, n| shift_lanes(x as u128, bytes, n, kind) as u64)
        }
        // FMA: 132 / 213 / 231 forms of fmaddsub fmsubadd fmadd fmsub fnmadd fnmsub
        0x96..=0x9F | 0xA6..=0xAF | 0xB6..=0xBF if c.vex => {
            let dst = c.x(reg);
            // (multiplicand, multiplier, addend)
            let (x, y, z) = match op >> 4 {
                9 => (dst, b, a),
                0xA => (a, dst, b),
                _ => (a, b, dst),
            };
            let kind = op & 0xF;
            let scalar = kind >= 9 && kind & 1 == 1;
            // per element: negate the product? subtract the addend?
            let signs = |i: usize| match kind {
                6 => (false, i % 2 == 0),
                7 => (false, i % 2 == 1),
                8 | 9 => (false, false),
                0xA | 0xB => (false, true),
                0xC | 0xD => (true, false),
                _ => (true, true),
            };
            let r = if w {
                let (x, y, z) = (pd(x), pd(y), pd(z));
                from_pd(std::array::from_fn(|i| {
                    let (np, nz) = signs(i);
                    (if np { -x[i] } else { x[i] }).mul_add(y[i], if nz { -z[i] } else { z[i] })
                }))
            } else {
                let (x, y, z) = (ps(x), ps(y), ps(z));
                from_ps(std::array::from_fn(|i| {
                    let (np, nz) = signs(i);
                    (if np { -x[i] } else { x[i] }).mul_add(y[i], if nz { -z[i] } else { z[i] })
                }))
            };
            if scalar {
                let keep = if w { LOW64 } else { LOW32 };
                (dst & !keep) | (r & keep)
            } else {
                r
            }
        }
        _ => return Ok(false),
    };
    c.set_x(reg, v);
    Ok(true)
}

/// pmovsx / pmovzx: the low elements of the source, widened to fill the destination.
fn widen(c: &mut Ctx, op: u8) -> R<bool> {
    let (from, to): (usize, usize) = match op & 7 {
        0 => (1, 2),
        1 => (1, 4),
        2 => (1, 8),
        3 => (2, 4),
        4 => (2, 8),
        5 => (4, 8),
        _ => return Ok(false),
    };
    let signed = op < 0x30;
    let (reg, rm) = c.modrm()?;
    let total = if c.l { 32 } else { 16 };
    let n = total / to;
    let src = c.rd_low(rm, (n * from) as u8)?.to_le_bytes();
    let mut out = [0u8; 32];
    for i in 0..n {
        let v = element(&src, i, from);
        let v = if signed { sxl(v, from as u32) as u64 } else { v };
        out[i * to..(i + 1) * to].copy_from_slice(&v.to_le_bytes()[..to]);
    }
    let out = halves_of(&out);
    if c.vex {
        c.set_y(reg, out);
    } else {
        c.set_x(reg, out[0]);
    }
    Ok(true)
}

/// A gather: the elements whose mask element (in vvvv) has its sign bit set are loaded from
/// base + index × scale, the indexes being a vector register (VSIB). The mask ends up zero.
fn gather(c: &mut Ctx, op: u8) -> R<bool> {
    let m = c.fetch8()?;
    if m >> 6 == 3 || m & 7 != 4 {
        return Ok(false);
    }
    let reg = ((m >> 3) & 7) | ((c.rex & 4) << 1);
    let sib = c.fetch8()?;
    let index_reg = ((sib >> 3) & 7) | ((c.rex & 2) << 2);
    let scale = sib >> 6;
    let mut base = if sib & 7 == 5 && m >> 6 == 0 { c.fetch32()? as i32 as i64 as u64 } else { c.cpu.r[((sib & 7) | ((c.rex & 1) << 3)) as usize] };
    match m >> 6 {
        1 => base = base.wrapping_add(c.fetch8()? as i8 as i64 as u64),
        2 => base = base.wrapping_add(c.fetch32()? as i32 as i64 as u64),
        _ => {}
    }
    let base = base.wrapping_add(c.seg);
    let data: usize = if c.rex & 8 != 0 { 8 } else { 4 };
    let index: usize = if op & 1 == 0 { 4 } else { 8 };
    // as many elements as fit both the index vector and the destination
    let n = if c.l { 32 } else { 16 } / data.max(index);
    let indexes = bytes_of(c.y(index_reg));
    let mut mask = bytes_of(c.y(c.vvvv));
    let mut out = bytes_of(c.y(reg));
    for i in 0..n {
        if mask[i * data + data - 1] & 0x80 != 0 {
            let k = sxl(element(&indexes, i, index), index as u32) as u64;
            let at = base.wrapping_add(k << scale);
            c.mem.read(at, &mut out[i * data..(i + 1) * data])?;
        }
    }
    // what lies above the elements written is cleared, in the destination and the mask
    out[n * data..].fill(0);
    mask.fill(0);
    let vvvv = c.vvvv;
    c.set_y(reg, halves_of(&out));
    c.set_y(vvvv, halves_of(&mask));
    Ok(true)
}

/// pcmpestri / pcmpestrm / pcmpistri / pcmpistrm: the SSE4.2 string compares.
fn string_compare(c: &mut Ctx, op: u8) -> R<bool> {
    let (reg, rm) = c.modrm()?;
    let imm = c.fetch8()?;
    let (a, b) = (c.x(reg).to_le_bytes(), c.rd_x(rm, 16)?.to_le_bytes());
    let w: usize = if imm & 1 != 0 { 2 } else { 1 };
    let n = 16 / w;
    let signed = imm & 2 != 0;
    let get = |v: &[u8; 16], i: usize| {
        let x = element(v, i, w);
        if signed {
            sxl(x, w as u32)
        } else {
            x as i64
        }
    };
    let explicit = op & 2 == 0;
    let length = |v: &[u8; 16], reg: usize| {
        if explicit {
            let r = c.cpu.r[reg];
            let len = if c.rex & 8 != 0 { (r as i64).unsigned_abs() } else { (r as u32 as i32).unsigned_abs() as u64 };
            len.min(n as u64) as usize
        } else {
            (0..n).find(|&i| element(v, i, w) == 0).unwrap_or(n)
        }
    };
    let (la, lb) = (length(&a, RAX), length(&b, RDX));
    let mut res = 0u32;
    for i in 0..n {
        let bit = match (imm >> 2) & 3 {
            // equal any: the element of b is one of a's
            0 => i < lb && (0..la).any(|j| get(&a, j) == get(&b, i)),
            // ranges: it lies in one of a's pairs
            1 => i < lb && (0..la / 2).any(|j| get(&a, 2 * j) <= get(&b, i) && get(&b, i) <= get(&a, 2 * j + 1)),
            // equal each: the strings agree here (beyond both ends they do)
            2 => (i < la && i < lb && get(&a, i) == get(&b, i)) || (i >= la && i >= lb),
            // equal ordered: a starts here in b
            _ => (0..n - i).all(|j| j >= la || (i + j < lb && get(&a, j) == get(&b, i + j))),
        };
        res |= (bit as u32) << i;
    }
    let all = (1u32 << n) - 1;
    let res = match (imm >> 4) & 3 {
        1 => !res & all,
        3 => res ^ ((1u32 << lb) - 1),
        _ => res,
    };
    if op & 1 != 0 {
        // the index of the first (or last) match, or the number of elements
        let index = if res == 0 {
            n as u32
        } else if imm & 0x40 != 0 {
            31 - res.leading_zeros()
        } else {
            res.trailing_zeros()
        };
        c.sreg(RCX as u8, 8, index as u64);
    } else {
        let mask = if imm & 0x40 != 0 {
            (0..n).fold(0u128, |m, i| if res >> i & 1 != 0 { m | ((1u128 << (8 * w)) - 1) << (8 * w * i) } else { m })
        } else {
            res as u128
        };
        c.set_x(0, mask);
    }
    c.cpu.rflags &= !(CF | ZF | SF | OF | AF | PF);
    c.cpu.set_flag(CF, res != 0);
    c.cpu.set_flag(ZF, lb < n);
    c.cpu.set_flag(SF, la < n);
    c.cpu.set_flag(OF, res & 1 != 0);
    Ok(true)
}

fn map_0f38(c: &mut Ctx, op: u8) -> R<bool> {
    let vector = c.opsz16 && c.rep == 0;
    match op {
        0x90..=0x93 if vector && c.vex => return gather(c, op),
        0x00..=0x0D | 0x1C..=0x1E | 0x28..=0x2B | 0x37..=0x40 | 0x45..=0x47 | 0x96..=0x9F | 0xA6..=0xAF | 0xB6..=0xBF if vector => {
            // a scalar FMA form works on the low element only
            if op >= 0x96 && op & 0xF >= 9 && op & 1 == 1 {
                c.l = false;
            }
            return per_half(c, |c| lanewise_38(c, op));
        }
        0x20..=0x25 | 0x30..=0x35 if vector => return widen(c, op),
        // pblendvb / blendvps / blendvpd: by the sign bits of XMM0's elements
        0x10 | 0x14 | 0x15 if vector && !c.vex => {
            let (reg, rm) = c.modrm()?;
            let b = c.rd_x(rm, 16)?;
            let (a, m) = (c.x(reg), c.x(0));
            let w = match op {
                0x10 => 1,
                0x14 => 4,
                _ => 8,
            };
            let v = lanes(m, 0, w, |x, _| if sxl(x, w) < 0 { !0 } else { 0 });
            c.set_x(reg, (a & !v) | (b & v));
            return Ok(true);
        }
        // ptest, vtestps, vtestpd: ZF = no common bits, CF = nothing of b outside a
        0x17 | 0x0E | 0x0F if vector => {
            let (reg, rm) = c.modrm()?;
            let (a, b) = if c.l { (c.y(reg), c.rd_y(rm)?) } else { ([c.x(reg), 0], [c.rd_x(rm, 16)?, 0]) };
            let signs = match op {
                0x0E => lanes(!0, 0, 4, |_, _| 0x8000_0000),
                0x0F => lanes(!0, 0, 8, |_, _| 1 << 63),
                _ => !0,
            };
            let zf = (0..2).all(|i| a[i] & b[i] & signs == 0);
            let cf = (0..2).all(|i| !a[i] & b[i] & signs == 0);
            c.cpu.rflags &= !(CF | ZF | SF | OF | PF | AF);
            c.cpu.set_flag(ZF, zf);
            c.cpu.set_flag(CF, cf);
            return Ok(true);
        }
        // vcvtph2ps
        0x13 if vector && c.vex => {
            let (reg, rm) = c.modrm()?;
            let src = c.rd_low(rm, if c.l { 16 } else { 8 })?;
            let f = |i: u32| f16_to_f32((src >> (16 * i)) as u16);
            let out = [from_ps([f(0), f(1), f(2), f(3)]), if c.l { from_ps([f(4), f(5), f(6), f(7)]) } else { 0 }];
            c.set_y(reg, out);
            return Ok(true);
        }
        // vpermps / vpermd ymm, ymm (indexes), ymm/m256
        0x16 | 0x36 if vector && c.vex && c.l => {
            let (reg, rm) = c.modrm()?;
            let src = bytes_of(c.rd_y(rm)?);
            let index = bytes_of(c.y(c.vvvv));
            let mut out = [0u8; 32];
            for i in 0..8 {
                let k = element(&index, i, 4) as usize & 7;
                out[4 * i..4 * i + 4].copy_from_slice(&src[4 * k..4 * k + 4]);
            }
            c.set_y(reg, halves_of(&out));
            return Ok(true);
        }
        // broadcasts: vbroadcastss / sd / f128 / i128, vpbroadcastb / w / d / q
        0x18 | 0x19 | 0x1A | 0x5A | 0x58 | 0x59 | 0x78 | 0x79 if vector && c.vex => {
            let w: usize = match op {
                0x18 | 0x58 => 4,
                0x19 | 0x59 => 8,
                0x1A | 0x5A => 16,
                0x78 => 1,
                _ => 2,
            };
            let (reg, rm) = c.modrm()?;
            let src = c.rd_low(rm, w as u8)?.to_le_bytes();
            let mut out = [0u8; 32];
            for chunk in out[..if c.l { 32 } else { 16 }].chunks_mut(w) {
                chunk.copy_from_slice(&src[..w]);
            }
            c.set_y(reg, halves_of(&out));
            return Ok(true);
        }
        // vmaskmovps / pd, vpmaskmovd / q: load (2C 2D 8C) or store (2E 2F 8E) the elements
        // whose mask element (in vvvv) has its sign bit set
        0x2C..=0x2F | 0x8C | 0x8E if vector && c.vex => {
            let w: usize = match op {
                0x2C | 0x2E => 4,
                0x2D | 0x2F => 8,
                _ => {
                    if c.rex & 8 != 0 {
                        8
                    } else {
                        4
                    }
                }
            };
            let store = matches!(op, 0x2E | 0x2F | 0x8E);
            let (reg, rm) = c.modrm()?;
            if matches!(rm, Rm::Reg(_)) {
                return Ok(false);
            }
            let at = c.ea(rm);
            let mask = bytes_of(c.y(c.vvvv));
            let mut reg_bytes = if store { bytes_of(c.y(reg)) } else { [0u8; 32] };
            for i in 0..if c.l { 32 } else { 16 } / w {
                if mask[i * w + w - 1] & 0x80 == 0 {
                    continue;
                }
                let a = at + (i * w) as u64;
                if store {
                    c.mem.write(a, &reg_bytes[i * w..(i + 1) * w])?;
                } else {
                    c.mem.read(a, &mut reg_bytes[i * w..(i + 1) * w])?;
                }
            }
            if !store {
                c.set_y(reg, halves_of(&reg_bytes));
            }
            return Ok(true);
        }
        // phminposuw
        0x41 if vector && !c.l => {
            let (reg, rm) = c.modrm()?;
            let b = c.rd_x(rm, 16)?;
            let (value, index) = (0..8).map(|i| ((b >> (16 * i)) as u16, i)).min().unwrap();
            c.set_x(reg, value as u128 | (index as u128) << 16);
            return Ok(true);
        }
        _ => {}
    }
    let sz = if c.rex & 8 != 0 { 8 } else { 4 };
    let m = mask(sz);
    let bits = sz as u64 * 8;
    match (op, c.vex) {
        // movbe r, m / movbe m, r
        (0xF0 | 0xF1, false) if c.rep == 0 => {
            let sz = c.opsize();
            let (reg, rm) = c.modrm()?;
            let swap = |v: u64| match sz {
                2 => (v as u16).swap_bytes() as u64,
                4 => (v as u32).swap_bytes() as u64,
                _ => v.swap_bytes(),
            };
            if op == 0xF0 {
                let v = c.rd_rm(rm, sz)?;
                c.sreg(reg, sz, swap(v));
            } else {
                let v = c.greg(reg, sz);
                c.wr_rm(rm, sz, swap(v))?;
            }
        }
        // crc32 r, r/m (CRC-32C)
        (0xF0 | 0xF1, false) if c.rep == 0xF2 => {
            let src = if op == 0xF0 { 1 } else { c.opsize() };
            let (reg, rm) = c.modrm()?;
            let v = c.rd_rm(rm, src)?;
            let mut crc = c.greg(reg, 4) as u32;
            for i in 0..src {
                crc ^= (v >> (8 * i)) as u8 as u32;
                for _ in 0..8 {
                    crc = (crc >> 1) ^ (0x82F6_3B78 & (crc & 1).wrapping_neg());
                }
            }
            c.sreg(reg, 8, crc as u64);
        }
        // adcx (66) / adox (F3): add with CF, or with OF, as the carry; no other flag changes
        (0xF6, false) if c.opsz16 || c.rep == 0xF3 => {
            let flag = if c.rep == 0xF3 { OF } else { CF };
            let (reg, rm) = c.modrm()?;
            let wide = c.greg(reg, sz) as u128 + c.rd_rm(rm, sz)? as u128 + c.cpu.flag(flag) as u128;
            c.cpu.set_flag(flag, wide >> bits != 0);
            c.sreg(reg, sz, wide as u64 & m);
        }
        // andn r, vvvv, r/m
        (0xF2, true) if c.rep == 0 && !c.opsz16 => {
            let (reg, rm) = c.modrm()?;
            let r = !c.greg(c.vvvv, sz) & c.rd_rm(rm, sz)? & m;
            bmi_flags(c, r, sz, false);
            c.sreg(reg, sz, r);
        }
        // blsr / blsmsk / blsi vvvv, r/m
        (0xF3, true) if c.rep == 0 && !c.opsz16 => {
            let (reg, rm) = c.modrm()?;
            let v = c.rd_rm(rm, sz)?;
            let (r, carry) = match reg & 7 {
                1 => (v.wrapping_sub(1) & v, v == 0),
                2 => (v.wrapping_sub(1) ^ v, v == 0),
                3 => (v.wrapping_neg() & v, v != 0),
                _ => return Ok(false),
            };
            bmi_flags(c, r, sz, carry);
            c.sreg(c.vvvv, sz, r & m);
        }
        // bzhi r, r/m, vvvv; pext / pdep r, vvvv, r/m
        (0xF5, true) if !c.opsz16 => {
            let (reg, rm) = c.modrm()?;
            let v = c.rd_rm(rm, sz)?;
            let other = c.greg(c.vvvv, sz);
            match c.rep {
                0 => {
                    let n = other & 0xFF;
                    let r = if n < bits { v & ((1u64 << n) - 1) } else { v };
                    bmi_flags(c, r, sz, n >= bits);
                    c.sreg(reg, sz, r);
                }
                // pext: the bits of `other` the mask `v` selects, packed
                0xF3 => {
                    let (mut r, mut k) = (0u64, 0);
                    for i in (0..bits).filter(|i| v >> i & 1 != 0) {
                        r |= (other >> i & 1) << k;
                        k += 1;
                    }
                    c.sreg(reg, sz, r);
                }
                // pdep: the low bits of `other`, spread to where the mask `v` has bits
                _ => {
                    let (mut r, mut k) = (0u64, 0);
                    for i in (0..bits).filter(|i| v >> i & 1 != 0) {
                        r |= (other >> k & 1) << i;
                        k += 1;
                    }
                    c.sreg(reg, sz, r);
                }
            }
        }
        // mulx r, vvvv, r/m: RDX × r/m, high half to r, low half to vvvv, no flags
        (0xF6, true) if c.rep == 0xF2 => {
            let (reg, rm) = c.modrm()?;
            let p = c.greg(RDX as u8, sz) as u128 * c.rd_rm(rm, sz)? as u128;
            let vvvv = c.vvvv;
            c.sreg(vvvv, sz, p as u64 & m);
            c.sreg(reg, sz, (p >> bits) as u64 & m);
        }
        // bextr r, r/m, vvvv; shlx (66) / sarx (F3) / shrx (F2) r, r/m, vvvv
        (0xF7, true) => {
            let (reg, rm) = c.modrm()?;
            let v = c.rd_rm(rm, sz)?;
            let control = c.greg(c.vvvv, sz);
            let count = control & (bits - 1);
            let r = match (c.opsz16, c.rep) {
                (false, 0) => {
                    let (start, len) = (control & 0xFF, (control >> 8) & 0xFF);
                    let r = if start < bits { v >> start } else { 0 };
                    let r = if len < 64 { r & ((1u64 << len) - 1) } else { r } & m;
                    c.cpu.rflags &= !(CF | OF | ZF);
                    c.cpu.set_flag(ZF, r == 0);
                    r
                }
                (true, _) => (v << count) & m,
                (_, 0xF3) => (crate::exec::sx(v, sz) as i64 >> count) as u64 & m,
                _ => v >> count,
            };
            c.sreg(reg, sz, r);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// The instructions of map 0F 3A that work lane by lane (an 8-bit immediate follows).
fn lanewise_3a(c: &mut Ctx, op: u8) -> R<bool> {
    let (reg, rm) = c.modrm()?;
    let imm = c.fetch8()?;
    let b = c.rd_x(rm, 16)?;
    let a = c.src1(reg);
    let half = c.half as u32;
    // select elements of b where the mask (one bit per element of `w` bytes) says so
    let blend = |mask: u32, w: u32| (0..16 / w).fold(0u128, |r, i| {
        let m: u128 = if w == 8 { LOW64 } else if w == 4 { LOW32 } else { 0xFFFF };
        let src = if mask >> i & 1 != 0 { b } else { a };
        r | ((src >> (8 * w * i)) & m) << (8 * w * i)
    });
    let v: u128 = match op {
        // vpblendd: 8 mask bits for the 8 dwords of a YMM; blendps likewise; blendpd 4 for 4
        0x02 if c.vex => blend(imm as u32 >> (4 * half), 4),
        0x0C => blend(imm as u32 >> (4 * half), 4),
        0x0D => blend(imm as u32 >> (2 * half), 8),
        // pblendw: the same 8 bits for both halves
        0x0E => blend(imm as u32, 2),
        // vpermilps / vpermilpd by an immediate
        0x04 if c.vex => (0..4).fold(0, |r, i| r | ((b >> (32 * ((imm >> (2 * i)) as u32 & 3))) & LOW32) << (32 * i)),
        0x05 if c.vex => (0..2).fold(0, |r, i| r | ((b >> (64 * ((imm as u32 >> (i + 2 * half)) & 1))) & LOW64) << (64 * i)),
        // roundps / roundpd
        0x08 => from_ps(ps(b).map(|x| round_f32(x, imm))),
        0x09 => from_pd(pd(b).map(|x| round_f64(x, imm))),
        // roundss / roundsd: the low element of b, the rest of a
        0x0A => (a & !LOW32) | round_f32(f32::from_bits(b as u32), imm).to_bits() as u128,
        0x0B => (a & !LOW64) | round_f64(f64::from_bits(b as u64), imm).to_bits() as u128,
        // palignr: the bytes of a:b from byte `imm` on
        0x0F => {
            let n = imm as u32;
            match n {
                0 => b,
                1..=15 => b >> (8 * n) | a << (8 * (16 - n)),
                16 => a,
                17..=31 => a >> (8 * (n - 16)),
                _ => 0,
            }
        }
        // dpps: the products the high nibble selects, summed into the elements the low one names
        0x40 => {
            let (x, y) = (ps(a), ps(b));
            let p: [f32; 4] = std::array::from_fn(|i| if imm >> (4 + i) & 1 != 0 { x[i] * y[i] } else { 0.0 });
            let sum = (p[0] + p[1]) + (p[2] + p[3]);
            from_ps(std::array::from_fn(|i| if imm >> i & 1 != 0 { sum } else { 0.0 }))
        }
        0x41 => {
            let (x, y) = (pd(a), pd(b));
            let p: [f64; 2] = std::array::from_fn(|i| if imm >> (4 + i) & 1 != 0 { x[i] * y[i] } else { 0.0 });
            let sum = p[0] + p[1];
            from_pd(std::array::from_fn(|i| if imm >> i & 1 != 0 { sum } else { 0.0 }))
        }
        // vblendvps / vblendvpd / vpblendvb: by the sign bits of a fourth register
        0x4A..=0x4C if c.vex => {
            let w = [4, 8, 1][(op - 0x4A) as usize];
            let m = lanes(c.x(imm >> 4), 0, w, |x, _| if sxl(x, w) < 0 { !0 } else { 0 });
            (a & !m) | (b & m)
        }
        _ => return Ok(false),
    };
    c.set_x(reg, v);
    Ok(true)
}

fn map_0f3a(c: &mut Ctx, op: u8) -> R<bool> {
    let vector = c.opsz16 && c.rep == 0;
    match op {
        0x02 | 0x04 | 0x05 | 0x08..=0x0F | 0x40 | 0x41 | 0x4A..=0x4C if vector => {
            if matches!(op, 0x0A | 0x0B) {
                c.l = false;
            }
            return per_half(c, |c| lanewise_3a(c, op));
        }
        0x60..=0x63 if vector && !c.l => return string_compare(c, op),
        // vpermq / vpermpd ymm, ymm/m256, imm8
        0x00 | 0x01 if vector && c.vex && c.l => {
            let (reg, rm) = c.modrm()?;
            let imm = c.fetch8()? as usize;
            let src = bytes_of(c.rd_y(rm)?);
            let mut out = [0u8; 32];
            for i in 0..4 {
                let k = (imm >> (2 * i)) & 3;
                out[8 * i..8 * i + 8].copy_from_slice(&src[8 * k..8 * k + 8]);
            }
            c.set_y(reg, halves_of(&out));
        }
        // vperm2f128 / vperm2i128: each half of the result is a half of either source, or zero
        0x06 | 0x46 if vector && c.vex && c.l => {
            let (reg, rm) = c.modrm()?;
            let imm = c.fetch8()?;
            let (a, b) = (c.y(c.vvvv), c.rd_y(rm)?);
            let pick = |sel: u8| if sel & 8 != 0 { 0 } else { [a[0], a[1], b[0], b[1]][(sel & 3) as usize] };
            c.set_y(reg, [pick(imm), pick(imm >> 4)]);
        }
        // vinsertf128 / vinserti128 ymm, ymm, xmm/m128, imm8
        0x18 | 0x38 if vector && c.vex && c.l => {
            let (reg, rm) = c.modrm()?;
            let imm = c.fetch8()?;
            let b = c.rd_low(rm, 16)?;
            let mut out = c.y(c.vvvv);
            out[(imm & 1) as usize] = b;
            c.set_y(reg, out);
        }
        // vextractf128 / vextracti128 xmm/m128, ymm, imm8
        0x19 | 0x39 if vector && c.vex && c.l => {
            let (reg, rm) = c.modrm()?;
            let imm = c.fetch8()?;
            let v = c.y(reg)[(imm & 1) as usize];
            match rm {
                Rm::Reg(r) => c.set_y(r, [v, 0]),
                _ => c.mem.write(c.ea(rm), &v.to_le_bytes())?,
            }
        }
        // pextrb / pextrw / pextrd / pextrq, extractps: an element to a register or memory
        0x14..=0x17 if vector && !c.l => {
            let (reg, rm) = c.modrm()?;
            let imm = c.fetch8()? as u32;
            let (w, sz): (u32, u8) = match op {
                0x14 => (1, 1),
                0x15 => (2, 2),
                0x16 if c.rex & 8 != 0 => (8, 8),
                _ => (4, 4),
            };
            let v = (c.x(reg) >> (8 * w * (imm % (16 / w)))) as u64 & mask(sz);
            match rm {
                // a register gets the element zero-extended
                Rm::Reg(r) => c.sreg(r, 8, v),
                _ => c.wr_rm(rm, sz, v)?,
            }
        }
        // pinsrb / pinsrd / pinsrq
        0x20 | 0x22 if vector && !c.l => {
            let (reg, rm) = c.modrm()?;
            let imm = c.fetch8()? as u32;
            let (w, sz): (u32, u8) = match op {
                0x20 => (1, 1),
                _ if c.rex & 8 != 0 => (8, 8),
                _ => (4, 4),
            };
            let v = c.rd_rm(rm, sz)? as u128;
            let shift = 8 * w * (imm % (16 / w));
            let m = (mask(sz) as u128) << shift;
            let out = (c.src1(reg) & !m) | v << shift;
            c.set_x(reg, out);
        }
        // insertps: one element of the source into one of the destination, then zero some
        0x21 if vector && !c.l => {
            let (reg, rm) = c.modrm()?;
            let imm = c.fetch8()? as u32;
            let element = match rm {
                Rm::Reg(r) => (c.x(r) >> (32 * (imm >> 6))) & LOW32,
                _ => c.rd_x(rm, 4)?,
            };
            let shift = 32 * ((imm >> 4) & 3);
            let mut out = (c.src1(reg) & !(LOW32 << shift)) | element << shift;
            for i in (0..4).filter(|i| imm >> i & 1 != 0) {
                out &= !(LOW32 << (32 * i));
            }
            c.set_x(reg, out);
        }
        // vcvtps2ph xmm/m, xmm/ymm, imm8 (rounding: nearest)
        0x1D if vector && c.vex => {
            let (reg, rm) = c.modrm()?;
            c.fetch8()?;
            let src = c.y(reg);
            let n = if c.l { 8 } else { 4 };
            let mut out = 0u128;
            for i in 0..n {
                let x = ps(src[i / 4])[i % 4];
                out |= (f32_to_f16(x) as u128) << (16 * i);
            }
            match rm {
                Rm::Reg(r) => c.set_y(r, [out, 0]),
                _ => c.mem.write(c.ea(rm), &out.to_le_bytes()[..2 * n])?,
            }
        }
        // rorx r, r/m, imm8
        0xF0 if c.vex && c.rep == 0xF2 => {
            let sz = if c.rex & 8 != 0 { 8 } else { 4 };
            let (reg, rm) = c.modrm()?;
            let n = c.fetch8()? as u32 & (sz as u32 * 8 - 1);
            let v = c.rd_rm(rm, sz)?;
            let r = if sz == 8 { v.rotate_right(n) } else { (v as u32).rotate_right(n) as u64 };
            c.sreg(reg, sz, r);
        }
        _ => return Ok(false),
    }
    Ok(true)
}
