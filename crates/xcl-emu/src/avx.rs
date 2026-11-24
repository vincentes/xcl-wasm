//! VEX-encoded instructions and the three-byte opcode maps: AVX / AVX2 forms of the SSE
//! instructions (run by `sse`, once per 128-bit half), and BMI1, BMI2, MOVBE, CRC32, ADX.

use crate::cpu::*;
use crate::exec::{mask, sign_bit, Ctx, Rm, R};

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

fn map_0f38(c: &mut Ctx, op: u8) -> R<bool> {
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

fn map_0f3a(c: &mut Ctx, op: u8) -> R<bool> {
    match (op, c.vex) {
        // rorx r, r/m, imm8
        (0xF0, true) if c.rep == 0xF2 => {
            let sz = if c.rex & 8 != 0 { 8 } else { 4 };
            let (reg, rm) = c.modrm()?;
            let n = c.fetch8()? as u32 & (sz as u32 * 8 - 1);
            let v = c.rd_rm(rm, sz)?;
            let r = if sz == 8 { v.rotate_right(n) } else { (v as u32).rotate_right(n) as u64 };
            c.sreg(reg, sz, r);
        }
        _ => return Ok(false),
    }
    let _ = Rm::Reg(0);
    Ok(true)
}
