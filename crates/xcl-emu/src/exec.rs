//! The interpreter: decodes and executes one instruction at a time. Integer instructions are
//! here; SSE is in `sse`.

use crate::cpu::*;
use crate::mem::Memory;

pub(crate) type R<T> = Result<T, Exit>;

/// A ModRM operand.
#[derive(Clone, Copy)]
pub(crate) enum Rm {
    Reg(u8),
    Mem(u64),
    /// RIP-relative: displacement from the end of the instruction (resolved once all of the
    /// instruction's bytes are fetched)
    Rip(i64),
}

pub(crate) struct Ctx<'a> {
    pub cpu: &'a mut Cpu,
    pub mem: &'a mut Memory,
    /// where the next instruction byte comes from
    pub pc: u64,
    /// the REX prefix (0: none)
    pub rex: u8,
    pub opsz16: bool,
    /// 0, 0xF2 or 0xF3
    pub rep: u8,
    /// segment base added to memory operands (FS / GS)
    pub seg: u64,
    pub addr32: bool,
}

#[inline]
pub(crate) fn mask(sz: u8) -> u64 {
    if sz >= 8 {
        !0
    } else {
        (1u64 << (sz * 8)) - 1
    }
}

#[inline]
pub(crate) fn sign_bit(sz: u8) -> u64 {
    1u64 << (sz * 8 - 1)
}

/// Sign-extend the low `sz` bytes of `v`.
#[inline]
pub(crate) fn sx(v: u64, sz: u8) -> u64 {
    match sz {
        1 => v as i8 as i64 as u64,
        2 => v as i16 as i64 as u64,
        4 => v as i32 as i64 as u64,
        _ => v,
    }
}

pub(crate) fn run(cpu: &mut Cpu, mem: &mut Memory, max: u64) -> Exit {
    let mut n = 0u64;
    loop {
        if cpu.rip >= cpu.trap_base && cpu.rip < cpu.trap_end {
            return Exit::Import(((cpu.rip - cpu.trap_base) / TRAP_STRIDE) as u32);
        }
        if n >= max {
            return Exit::Limit;
        }
        let start = cpu.rip;
        let mut c = Ctx { cpu, mem, pc: start, rex: 0, opsz16: false, rep: 0, seg: 0, addr32: false };
        match c.step() {
            Ok(()) => {
                let pc = c.pc;
                cpu.rip = pc;
                cpu.icount += 1;
                n += 1;
            }
            Err(Exit::Breakpoint) => {
                let pc = c.pc;
                cpu.rip = pc;
                return Exit::Breakpoint;
            }
            Err(e) => {
                cpu.rip = start;
                return e;
            }
        }
    }
}

impl Ctx<'_> {
    #[inline]
    pub(crate) fn fetch8(&mut self) -> R<u8> {
        let b = self.mem.read_u8(self.pc)?;
        self.pc = self.pc.wrapping_add(1);
        Ok(b)
    }
    #[inline]
    fn fetch16(&mut self) -> R<u16> {
        let v = self.mem.read_u16(self.pc)?;
        self.pc = self.pc.wrapping_add(2);
        Ok(v)
    }
    #[inline]
    pub(crate) fn fetch32(&mut self) -> R<u32> {
        let v = self.mem.read_u32(self.pc)?;
        self.pc = self.pc.wrapping_add(4);
        Ok(v)
    }
    #[inline]
    fn fetch64(&mut self) -> R<u64> {
        let v = self.mem.read_u64(self.pc)?;
        self.pc = self.pc.wrapping_add(8);
        Ok(v)
    }

    /// Immediate of an instruction with operand size `sz`: 8 / 16 / 32 bits, sign-extended.
    fn imm(&mut self, sz: u8) -> R<u64> {
        Ok(match sz {
            1 => self.fetch8()? as i8 as i64 as u64,
            2 => self.fetch16()? as i16 as i64 as u64,
            _ => self.fetch32()? as i32 as i64 as u64,
        })
    }

    pub(crate) fn undefined(&self) -> Exit {
        let rip = self.cpu.rip;
        let mut bytes = [0u8; 8];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = self.mem.read_u8(rip + i as u64).unwrap_or(0);
        }
        Exit::Undefined { rip, bytes }
    }

    /// Operand size of a "v" operand: 64 with REX.W, 16 with the 66 prefix, else 32.
    #[inline]
    pub(crate) fn opsize(&self) -> u8 {
        if self.rex & 8 != 0 {
            8
        } else if self.opsz16 {
            2
        } else {
            4
        }
    }

    #[inline]
    pub(crate) fn greg(&self, r: u8, sz: u8) -> u64 {
        let v = self.cpu.r[r as usize];
        match sz {
            1 => {
                if self.rex == 0 && (4..8).contains(&r) {
                    (self.cpu.r[r as usize - 4] >> 8) & 0xFF
                } else {
                    v & 0xFF
                }
            }
            2 => v & 0xFFFF,
            4 => v & 0xFFFF_FFFF,
            _ => v,
        }
    }

    #[inline]
    pub(crate) fn sreg(&mut self, r: u8, sz: u8, v: u64) {
        match sz {
            1 => {
                if self.rex == 0 && (4..8).contains(&r) {
                    let x = &mut self.cpu.r[r as usize - 4];
                    *x = (*x & !0xFF00) | ((v & 0xFF) << 8);
                } else {
                    let x = &mut self.cpu.r[r as usize];
                    *x = (*x & !0xFF) | (v & 0xFF);
                }
            }
            2 => {
                let x = &mut self.cpu.r[r as usize];
                *x = (*x & !0xFFFF) | (v & 0xFFFF);
            }
            4 => self.cpu.r[r as usize] = v & 0xFFFF_FFFF,
            _ => self.cpu.r[r as usize] = v,
        }
    }

    /// Decode ModRM (+ SIB, displacement): the `reg` field and the r/m operand.
    pub(crate) fn modrm(&mut self) -> R<(u8, Rm)> {
        let m = self.fetch8()?;
        let md = m >> 6;
        let reg = ((m >> 3) & 7) | ((self.rex & 4) << 1);
        let rm = m & 7;
        if md == 3 {
            return Ok((reg, Rm::Reg(rm | ((self.rex & 1) << 3))));
        }
        let mut addr: u64;
        if rm == 4 {
            let sib = self.fetch8()?;
            let idx = ((sib >> 3) & 7) | ((self.rex & 2) << 2);
            let base = sib & 7;
            addr = if base == 5 && md == 0 {
                self.fetch32()? as i32 as i64 as u64
            } else {
                self.cpu.r[(base | ((self.rex & 1) << 3)) as usize]
            };
            if idx != 4 {
                addr = addr.wrapping_add(self.cpu.r[idx as usize] << (sib >> 6));
            }
        } else if rm == 5 && md == 0 {
            let d = self.fetch32()? as i32 as i64;
            return Ok((reg, Rm::Rip(d)));
        } else {
            addr = self.cpu.r[(rm | ((self.rex & 1) << 3)) as usize];
        }
        if md == 1 {
            addr = addr.wrapping_add(self.fetch8()? as i8 as i64 as u64);
        } else if md == 2 {
            addr = addr.wrapping_add(self.fetch32()? as i32 as i64 as u64);
        }
        if self.addr32 {
            addr &= 0xFFFF_FFFF;
        }
        Ok((reg, Rm::Mem(addr.wrapping_add(self.seg))))
    }

    /// Address of a memory operand. Call it after the instruction's immediate is fetched.
    #[inline]
    pub(crate) fn ea(&self, rm: Rm) -> u64 {
        match rm {
            Rm::Mem(a) => a,
            Rm::Rip(d) => self.pc.wrapping_add(d as u64).wrapping_add(self.seg),
            Rm::Reg(_) => 0,
        }
    }

    #[inline]
    pub(crate) fn rd_rm(&self, rm: Rm, sz: u8) -> R<u64> {
        Ok(match rm {
            Rm::Reg(r) => self.greg(r, sz),
            _ => self.mem.read_sized(self.ea(rm), sz)?,
        })
    }

    #[inline]
    pub(crate) fn wr_rm(&mut self, rm: Rm, sz: u8, v: u64) -> R<()> {
        match rm {
            Rm::Reg(r) => self.sreg(r, sz, v),
            _ => self.mem.write_sized(self.ea(rm), sz, v)?,
        }
        Ok(())
    }

    fn set_szp(&mut self, r: u64, sz: u8) {
        let r = r & mask(sz);
        self.cpu.set_flag(ZF, r == 0);
        self.cpu.set_flag(SF, r & sign_bit(sz) != 0);
        self.cpu.set_flag(PF, (r as u8).count_ones() % 2 == 0);
    }

    fn add(&mut self, a: u64, b: u64, carry: bool, sz: u8) -> u64 {
        let m = mask(sz);
        let (a, b) = (a & m, b & m);
        let wide = a as u128 + b as u128 + carry as u128;
        let r = wide as u64 & m;
        self.cpu.set_flag(CF, wide >> (sz * 8) != 0);
        self.cpu.set_flag(OF, (a ^ r) & (b ^ r) & sign_bit(sz) != 0);
        self.cpu.set_flag(AF, (a ^ b ^ r) & 0x10 != 0);
        self.set_szp(r, sz);
        r
    }

    fn sub(&mut self, a: u64, b: u64, borrow: bool, sz: u8) -> u64 {
        let m = mask(sz);
        let (a, b) = (a & m, b & m);
        let r = a.wrapping_sub(b).wrapping_sub(borrow as u64) & m;
        self.cpu.set_flag(CF, (a as u128) < b as u128 + borrow as u128);
        self.cpu.set_flag(OF, (a ^ b) & (a ^ r) & sign_bit(sz) != 0);
        self.cpu.set_flag(AF, (a ^ b ^ r) & 0x10 != 0);
        self.set_szp(r, sz);
        r
    }

    fn logic(&mut self, r: u64, sz: u8) -> u64 {
        let r = r & mask(sz);
        self.cpu.rflags &= !(CF | OF | AF);
        self.set_szp(r, sz);
        r
    }

    /// ADD OR ADC SBB AND SUB XOR CMP (`op` 0..7). The result; CMP's is not to be written.
    fn alu(&mut self, op: u8, a: u64, b: u64, sz: u8) -> u64 {
        let cf = self.cpu.flag(CF);
        match op {
            0 => self.add(a, b, false, sz),
            1 => self.logic(a | b, sz),
            2 => self.add(a, b, cf, sz),
            3 => self.sub(a, b, cf, sz),
            4 => self.logic(a & b, sz),
            5 | 7 => self.sub(a, b, false, sz),
            _ => self.logic(a ^ b, sz),
        }
    }

    pub(crate) fn cond(&self, cc: u8) -> bool {
        let f = self.cpu.rflags;
        let (cf, zf, sf, of, pf) = (f & CF != 0, f & ZF != 0, f & SF != 0, f & OF != 0, f & PF != 0);
        let r = match cc >> 1 {
            0 => of,
            1 => cf,
            2 => zf,
            3 => cf || zf,
            4 => sf,
            5 => pf,
            6 => sf != of,
            _ => zf || sf != of,
        };
        r != (cc & 1 != 0)
    }

    /// ROL ROR RCL RCR SHL SHR SAL SAR (`kind` 0..7) by `count`.
    fn shift(&mut self, kind: u8, a: u64, count: u8, sz: u8) -> u64 {
        let bits = sz as u32 * 8;
        let m = mask(sz);
        let a = a & m;
        let c = (count & if sz == 8 { 63 } else { 31 }) as u32;
        if c == 0 {
            return a;
        }
        let msb = |v: u64| v & sign_bit(sz) != 0;
        match kind {
            0 | 1 => {
                let n = c % bits;
                let r = if n == 0 {
                    a
                } else if kind == 0 {
                    ((a << n) | (a >> (bits - n))) & m
                } else {
                    ((a >> n) | (a << (bits - n))) & m
                };
                if kind == 0 {
                    let cf = r & 1 != 0;
                    self.cpu.set_flag(CF, cf);
                    self.cpu.set_flag(OF, msb(r) != cf);
                } else {
                    self.cpu.set_flag(CF, msb(r));
                    self.cpu.set_flag(OF, msb(r) != (r & (sign_bit(sz) >> 1) != 0));
                }
                r
            }
            2 | 3 => {
                let mut r = a;
                let mut cf = self.cpu.flag(CF);
                for _ in 0..c % (bits + 1) {
                    if kind == 2 {
                        let out = msb(r);
                        r = ((r << 1) | cf as u64) & m;
                        cf = out;
                    } else {
                        let out = r & 1 != 0;
                        r = (r >> 1) | if cf { sign_bit(sz) } else { 0 };
                        cf = out;
                    }
                }
                self.cpu.set_flag(CF, cf);
                if kind == 2 {
                    self.cpu.set_flag(OF, msb(r) != cf);
                } else {
                    self.cpu.set_flag(OF, msb(r) != (r & (sign_bit(sz) >> 1) != 0));
                }
                r
            }
            4 | 6 => {
                let r = if c < bits { (a << c) & m } else { 0 };
                let cf = c <= bits && (a >> (bits - c)) & 1 != 0;
                self.cpu.set_flag(CF, cf);
                self.cpu.set_flag(OF, msb(r) != cf);
                self.cpu.set_flag(AF, false);
                self.set_szp(r, sz);
                r
            }
            5 => {
                let r = if c < bits { a >> c } else { 0 };
                self.cpu.set_flag(CF, c <= bits && (a >> (c - 1)) & 1 != 0);
                self.cpu.set_flag(OF, msb(a));
                self.cpu.set_flag(AF, false);
                self.set_szp(r, sz);
                r
            }
            _ => {
                let s = sx(a, sz) as i64;
                let r = (s >> c.min(63)) as u64 & m;
                self.cpu.set_flag(CF, (s >> (c - 1).min(63)) & 1 != 0);
                self.cpu.set_flag(OF, false);
                self.cpu.set_flag(AF, false);
                self.set_szp(r, sz);
                r
            }
        }
    }

    fn push(&mut self, v: u64) -> R<()> {
        Ok(self.cpu.push(self.mem, v)?)
    }

    fn pop(&mut self) -> R<u64> {
        Ok(self.cpu.pop(self.mem)?)
    }

    /// MOVS CMPS STOS LODS SCAS, with the REP prefixes.
    fn string(&mut self, op: u8, sz: u8) -> R<()> {
        let step = if self.cpu.flag(DF) { (sz as u64).wrapping_neg() } else { sz as u64 };
        let rep = self.rep;
        loop {
            if rep != 0 && self.cpu.r[RCX] == 0 {
                break;
            }
            let (si, di) = (self.cpu.r[RSI].wrapping_add(self.seg), self.cpu.r[RDI]);
            let mut compared = false;
            match op {
                // movs
                0xA4 => {
                    let v = self.mem.read_sized(si, sz)?;
                    self.mem.write_sized(di, sz, v)?;
                    self.cpu.r[RSI] = self.cpu.r[RSI].wrapping_add(step);
                    self.cpu.r[RDI] = di.wrapping_add(step);
                }
                // cmps
                0xA6 => {
                    let a = self.mem.read_sized(si, sz)?;
                    let b = self.mem.read_sized(di, sz)?;
                    self.sub(a, b, false, sz);
                    self.cpu.r[RSI] = self.cpu.r[RSI].wrapping_add(step);
                    self.cpu.r[RDI] = di.wrapping_add(step);
                    compared = true;
                }
                // stos
                0xAA => {
                    self.mem.write_sized(di, sz, self.cpu.r[RAX])?;
                    self.cpu.r[RDI] = di.wrapping_add(step);
                }
                // lods
                0xAC => {
                    let v = self.mem.read_sized(si, sz)?;
                    self.sreg(RAX as u8, sz, v);
                    self.cpu.r[RSI] = self.cpu.r[RSI].wrapping_add(step);
                }
                // scas
                _ => {
                    let b = self.mem.read_sized(di, sz)?;
                    let a = self.cpu.r[RAX];
                    self.sub(a, b, false, sz);
                    self.cpu.r[RDI] = di.wrapping_add(step);
                    compared = true;
                }
            }
            if rep == 0 {
                break;
            }
            self.cpu.r[RCX] = self.cpu.r[RCX].wrapping_sub(1);
            if compared && self.cpu.flag(ZF) != (rep == 0xF3) {
                break;
            }
        }
        Ok(())
    }

    fn cpuid(&mut self) {
        let leaf = self.cpu.r[RAX] as u32;
        let f = self.cpu.features;
        let (a, b, c, d): (u32, u32, u32, u32) = match leaf {
            // "AuthenticAMD"
            0 => (7, 0x6874_7541, 0x444D_4163, 0x6974_6E65),
            1 => {
                let c = (f.sse3 as u32) | (f.ssse3 as u32) << 9 | 1 << 13 | (f.sse41 as u32) << 19 | (f.sse42 as u32) << 20 | (f.popcnt as u32) << 23;
                // fpu tsc cx8 cmov clfsh mmx fxsr sse sse2
                let d = 1 | 1 << 4 | 1 << 8 | 1 << 15 | 1 << 19 | 1 << 23 | 1 << 24 | 1 << 25 | 1 << 26;
                (0x0087_0F10, 0x0000_0800, c, d)
            }
            0x8000_0000 => (0x8000_0008, 0x6874_7541, 0x444D_4163, 0x6974_6E65),
            // lahf/sahf; syscall, nx, long mode
            0x8000_0001 => (0x0087_0F10, 0, 1, 1 << 11 | 1 << 20 | 1 << 29),
            0x8000_0008 => (0x3030, 0, 0, 0),
            _ => (0, 0, 0, 0),
        };
        self.cpu.r[RAX] = a as u64;
        self.cpu.r[RBX] = b as u64;
        self.cpu.r[RCX] = c as u64;
        self.cpu.r[RDX] = d as u64;
    }

    /// Group 3: TEST NOT NEG MUL IMUL DIV IDIV on r/m.
    fn group3(&mut self, sz: u8) -> R<()> {
        let (reg, rm) = self.modrm()?;
        let m = mask(sz);
        match reg & 7 {
            0 | 1 => {
                let i = self.imm(sz)?;
                let a = self.rd_rm(rm, sz)?;
                self.logic(a & i, sz);
            }
            2 => {
                let a = self.rd_rm(rm, sz)?;
                self.wr_rm(rm, sz, !a & m)?;
            }
            3 => {
                let a = self.rd_rm(rm, sz)?;
                let r = self.sub(0, a, false, sz);
                self.wr_rm(rm, sz, r)?;
            }
            4 | 5 => {
                let signed = reg & 7 == 5;
                let b = self.rd_rm(rm, sz)?;
                let a = self.greg(RAX as u8, sz);
                let (lo, hi, over) = if signed {
                    let p = (sx(a, sz) as i64 as i128) * (sx(b, sz) as i64 as i128);
                    let lo = p as u64 & m;
                    (lo, (p >> (sz * 8)) as u64 & m, p != sx(lo, sz) as i64 as i128)
                } else {
                    let p = a as u128 * b as u128;
                    let hi = (p >> (sz * 8)) as u64 & m;
                    (p as u64 & m, hi, hi != 0)
                };
                if sz == 1 {
                    self.sreg(RAX as u8, 2, lo | hi << 8);
                } else {
                    self.sreg(RAX as u8, sz, lo);
                    self.sreg(RDX as u8, sz, hi);
                }
                self.cpu.set_flag(CF, over);
                self.cpu.set_flag(OF, over);
            }
            _ => {
                let signed = reg & 7 == 7;
                let b = self.rd_rm(rm, sz)?;
                if b == 0 {
                    return Err(Exit::DivideError);
                }
                let bits = sz as u32 * 8;
                let (lo, hi) = if sz == 1 { (self.greg(RAX as u8, 2), 0) } else { (self.greg(RAX as u8, sz), self.greg(RDX as u8, sz)) };
                let (q, r) = if signed {
                    let n: i128 = if sz == 1 { lo as i16 as i128 } else { (((hi as u128) << bits | lo as u128) << (128 - 2 * bits)) as i128 >> (128 - 2 * bits) };
                    let d = sx(b, sz) as i64 as i128;
                    let (q, r) = (n.wrapping_div(d), n.wrapping_rem(d));
                    if q > (m >> 1) as i128 || q < -((m >> 1) as i128) - 1 {
                        return Err(Exit::DivideError);
                    }
                    (q as u64 & m, r as u64 & m)
                } else {
                    let n: u128 = if sz == 1 { lo as u128 } else { (hi as u128) << bits | lo as u128 };
                    let (q, r) = (n / b as u128, n % b as u128);
                    if q > m as u128 {
                        return Err(Exit::DivideError);
                    }
                    (q as u64, r as u64)
                };
                if sz == 1 {
                    self.sreg(RAX as u8, 2, q | r << 8);
                } else {
                    self.sreg(RAX as u8, sz, q);
                    self.sreg(RDX as u8, sz, r);
                }
            }
        }
        Ok(())
    }

    /// Truncating signed multiply (IMUL r, r/m [, imm]) with its flags.
    fn imul(&mut self, a: u64, b: u64, sz: u8) -> u64 {
        let p = (sx(a, sz) as i64 as i128) * (sx(b, sz) as i64 as i128);
        let r = p as u64 & mask(sz);
        let over = p != sx(r, sz) as i64 as i128;
        self.cpu.set_flag(CF, over);
        self.cpu.set_flag(OF, over);
        r
    }

    /// BT BTS BTR BTC (`kind` 0..3) of bit `bit` in r/m.
    fn bit_test(&mut self, kind: u8, rm: Rm, bit: u64, sz: u8, from_reg: bool) -> R<()> {
        let bits = sz as u64 * 8;
        let rm = match rm {
            // a register bit offset reaches beyond the operand
            Rm::Reg(_) => rm,
            _ if from_reg => {
                let off = (sx(bit, sz) as i64).div_euclid(bits as i64).wrapping_mul(sz as i64);
                Rm::Mem(self.ea(rm).wrapping_add(off as u64))
            }
            _ => Rm::Mem(self.ea(rm)),
        };
        let b = bit % bits;
        let v = self.rd_rm(rm, sz)?;
        self.cpu.set_flag(CF, (v >> b) & 1 != 0);
        let r = match kind {
            0 => return Ok(()),
            1 => v | 1 << b,
            2 => v & !(1 << b),
            _ => v ^ 1 << b,
        };
        self.wr_rm(rm, sz, r)
    }

    fn step(&mut self) -> R<()> {
        let mut op;
        loop {
            op = self.fetch8()?;
            match op {
                0x66 => self.opsz16 = true,
                0x67 => self.addr32 = true,
                0xF2 | 0xF3 => self.rep = op,
                0xF0 | 0x2E | 0x36 | 0x3E | 0x26 => {}
                0x64 => self.seg = self.cpu.fs_base,
                0x65 => self.seg = self.cpu.gs_base,
                _ => break,
            }
        }
        if op & 0xF0 == 0x40 {
            self.rex = op;
            op = self.fetch8()?;
        }
        let osz = self.opsize();
        match op {
            // ALU: Eb,Gb / Ev,Gv / Gb,Eb / Gv,Ev / AL,ib / eAX,iz
            0x00..=0x3F if op & 7 < 6 => {
                let kind = op >> 3;
                let sz = if op & 1 == 0 { 1 } else { osz };
                match op & 7 {
                    0 | 1 => {
                        let (reg, rm) = self.modrm()?;
                        let a = self.rd_rm(rm, sz)?;
                        let b = self.greg(reg, sz);
                        let r = self.alu(kind, a, b, sz);
                        if kind != 7 {
                            self.wr_rm(rm, sz, r)?;
                        }
                    }
                    2 | 3 => {
                        let (reg, rm) = self.modrm()?;
                        let a = self.greg(reg, sz);
                        let b = self.rd_rm(rm, sz)?;
                        let r = self.alu(kind, a, b, sz);
                        if kind != 7 {
                            self.sreg(reg, sz, r);
                        }
                    }
                    _ => {
                        let i = self.imm(sz)?;
                        let a = self.greg(RAX as u8, sz);
                        let r = self.alu(kind, a, i, sz);
                        if kind != 7 {
                            self.sreg(RAX as u8, sz, r);
                        }
                    }
                }
            }
            0x0F => return self.two_byte(),
            0x50..=0x57 => {
                let v = self.cpu.r[((op & 7) | ((self.rex & 1) << 3)) as usize];
                self.push(v)?;
            }
            0x58..=0x5F => {
                let v = self.pop()?;
                self.cpu.r[((op & 7) | ((self.rex & 1) << 3)) as usize] = v;
            }
            // movsxd
            0x63 => {
                let (reg, rm) = self.modrm()?;
                let v = self.rd_rm(rm, 4)?;
                self.sreg(reg, osz, if osz == 8 { sx(v, 4) } else { v });
            }
            0x68 => {
                let i = self.imm(4)?;
                self.push(i)?;
            }
            0x6A => {
                let i = self.imm(1)?;
                self.push(i)?;
            }
            // imul Gv, Ev, imm
            0x69 | 0x6B => {
                let (reg, rm) = self.modrm()?;
                let i = self.imm(if op == 0x6B { 1 } else { osz })?;
                let a = self.rd_rm(rm, osz)?;
                let r = self.imul(a, i, osz);
                self.sreg(reg, osz, r);
            }
            0x70..=0x7F => {
                let d = self.fetch8()? as i8 as i64 as u64;
                if self.cond(op & 0xF) {
                    self.pc = self.pc.wrapping_add(d);
                }
            }
            // group 1: ALU r/m, imm
            0x80 | 0x81 | 0x83 => {
                let sz = if op == 0x80 { 1 } else { osz };
                let (reg, rm) = self.modrm()?;
                let i = self.imm(if op == 0x81 { sz } else { 1 })?;
                let a = self.rd_rm(rm, sz)?;
                let r = self.alu(reg & 7, a, i, sz);
                if reg & 7 != 7 {
                    self.wr_rm(rm, sz, r)?;
                }
            }
            0x84 | 0x85 => {
                let sz = if op == 0x84 { 1 } else { osz };
                let (reg, rm) = self.modrm()?;
                let a = self.rd_rm(rm, sz)?;
                let b = self.greg(reg, sz);
                self.logic(a & b, sz);
            }
            0x86 | 0x87 => {
                let sz = if op == 0x86 { 1 } else { osz };
                let (reg, rm) = self.modrm()?;
                let a = self.rd_rm(rm, sz)?;
                let b = self.greg(reg, sz);
                self.wr_rm(rm, sz, b)?;
                self.sreg(reg, sz, a);
            }
            0x88 | 0x89 => {
                let sz = if op == 0x88 { 1 } else { osz };
                let (reg, rm) = self.modrm()?;
                let v = self.greg(reg, sz);
                self.wr_rm(rm, sz, v)?;
            }
            0x8A | 0x8B => {
                let sz = if op == 0x8A { 1 } else { osz };
                let (reg, rm) = self.modrm()?;
                let v = self.rd_rm(rm, sz)?;
                self.sreg(reg, sz, v);
            }
            // lea (no segment base)
            0x8D => {
                let seg = std::mem::take(&mut self.seg);
                let (reg, rm) = self.modrm()?;
                let a = self.ea(rm);
                self.seg = seg;
                self.sreg(reg, osz, a);
            }
            0x8F => {
                let (_, rm) = self.modrm()?;
                let v = self.pop()?;
                self.wr_rm(rm, 8, v)?;
            }
            // nop / pause
            0x90 if self.rex & 1 == 0 => {}
            0x90..=0x97 => {
                let r = (op & 7) | ((self.rex & 1) << 3);
                let (a, b) = (self.greg(RAX as u8, osz), self.greg(r, osz));
                self.sreg(RAX as u8, osz, b);
                self.sreg(r, osz, a);
            }
            // cbw / cwde / cdqe
            0x98 => {
                let v = self.cpu.r[RAX];
                match osz {
                    2 => self.sreg(RAX as u8, 2, sx(v, 1)),
                    4 => self.sreg(RAX as u8, 4, sx(v, 2)),
                    _ => self.cpu.r[RAX] = sx(v, 4),
                }
            }
            // cwd / cdq / cqo
            0x99 => {
                let neg = self.cpu.r[RAX] & sign_bit(osz) != 0;
                self.sreg(RDX as u8, osz, if neg { !0 } else { 0 });
            }
            0x9C => {
                let f = self.cpu.rflags;
                self.push(f)?;
            }
            0x9D => {
                let f = self.pop()?;
                self.cpu.rflags = (f & 0x0000_0000_0024_0DD5) | 0x202;
            }
            // sahf / lahf
            0x9E => {
                let ah = (self.cpu.r[RAX] >> 8) & 0xD5;
                self.cpu.rflags = (self.cpu.rflags & !0xD5) | ah;
            }
            0x9F => {
                let f = (self.cpu.rflags & 0xD5) | 2;
                self.cpu.r[RAX] = (self.cpu.r[RAX] & !0xFF00) | f << 8;
            }
            0xA4 | 0xA6 | 0xAA | 0xAC | 0xAE => self.string(op, 1)?,
            0xA5 | 0xA7 | 0xAB | 0xAD | 0xAF => self.string(op & !1, osz)?,
            0xA8 | 0xA9 => {
                let sz = if op == 0xA8 { 1 } else { osz };
                let i = self.imm(sz)?;
                let a = self.greg(RAX as u8, sz);
                self.logic(a & i, sz);
            }
            0xB0..=0xB7 => {
                let i = self.fetch8()? as u64;
                self.sreg((op & 7) | ((self.rex & 1) << 3), 1, i);
            }
            0xB8..=0xBF => {
                let i = match osz {
                    8 => self.fetch64()?,
                    2 => self.fetch16()? as u64,
                    _ => self.fetch32()? as u64,
                };
                self.sreg((op & 7) | ((self.rex & 1) << 3), osz, i);
            }
            // group 2: shifts and rotates by imm8 / 1 / CL
            0xC0 | 0xC1 | 0xD0..=0xD3 => {
                let sz = if op & 1 == 0 { 1 } else { osz };
                let (reg, rm) = self.modrm()?;
                let count = match op {
                    0xC0 | 0xC1 => self.fetch8()?,
                    0xD0 | 0xD1 => 1,
                    _ => self.cpu.r[RCX] as u8,
                };
                let a = self.rd_rm(rm, sz)?;
                let r = self.shift(reg & 7, a, count, sz);
                self.wr_rm(rm, sz, r)?;
            }
            0xC2 => {
                let n = self.fetch16()? as u64;
                self.pc = self.pop()?;
                self.cpu.r[RSP] = self.cpu.r[RSP].wrapping_add(n);
            }
            0xC3 => self.pc = self.pop()?,
            0xC6 | 0xC7 => {
                let sz = if op == 0xC6 { 1 } else { osz };
                let (_, rm) = self.modrm()?;
                let i = self.imm(sz)?;
                self.wr_rm(rm, sz, i)?;
            }
            // leave
            0xC9 => {
                self.cpu.r[RSP] = self.cpu.r[RBP];
                self.cpu.r[RBP] = self.pop()?;
            }
            0xCC => return Err(Exit::Breakpoint),
            // fwait
            0x9B => {}
            // x87: only its control word (fninit, fnclex, fldcw, fnstcw); no x87 stack yet
            0xD9 | 0xDB => {
                let (reg, rm) = self.modrm()?;
                match (op, reg & 7, rm) {
                    // fninit (DB E3), fnclex (DB E2)
                    (0xDB, 4, Rm::Reg(3)) => self.cpu.fcw = 0x037F,
                    (0xDB, 4, Rm::Reg(2)) => {}
                    (0xD9, 5, Rm::Mem(_) | Rm::Rip(_)) => self.cpu.fcw = self.rd_rm(rm, 2)? as u16,
                    (0xD9, 7, Rm::Mem(_) | Rm::Rip(_)) => {
                        let v = self.cpu.fcw as u64;
                        self.wr_rm(rm, 2, v)?;
                    }
                    _ => return Err(self.undefined()),
                }
            }
            // jrcxz
            0xE3 => {
                let d = self.fetch8()? as i8 as i64 as u64;
                if self.cpu.r[RCX] == 0 {
                    self.pc = self.pc.wrapping_add(d);
                }
            }
            0xE8 => {
                let d = self.fetch32()? as i32 as i64 as u64;
                let ret = self.pc;
                self.push(ret)?;
                self.pc = ret.wrapping_add(d);
            }
            0xE9 => {
                let d = self.fetch32()? as i32 as i64 as u64;
                self.pc = self.pc.wrapping_add(d);
            }
            0xEB => {
                let d = self.fetch8()? as i8 as i64 as u64;
                self.pc = self.pc.wrapping_add(d);
            }
            0xF4 => return Err(Exit::Halt),
            0xF5 => self.cpu.rflags ^= CF,
            0xF6 => self.group3(1)?,
            0xF7 => self.group3(osz)?,
            0xF8 => self.cpu.rflags &= !CF,
            0xF9 => self.cpu.rflags |= CF,
            0xFC => self.cpu.rflags &= !DF,
            0xFD => self.cpu.rflags |= DF,
            // group 4 / 5: INC DEC [CALL JMP PUSH] r/m
            0xFE | 0xFF => {
                let sz = if op == 0xFE { 1 } else { osz };
                let (reg, rm) = self.modrm()?;
                match reg & 7 {
                    0 | 1 => {
                        let a = self.rd_rm(rm, sz)?;
                        let cf = self.cpu.flag(CF);
                        let r = if reg & 7 == 0 { self.add(a, 1, false, sz) } else { self.sub(a, 1, false, sz) };
                        self.cpu.set_flag(CF, cf);
                        self.wr_rm(rm, sz, r)?;
                    }
                    2 if op == 0xFF => {
                        let to = self.rd_rm(rm, 8)?;
                        let ret = self.pc;
                        self.push(ret)?;
                        self.pc = to;
                    }
                    4 if op == 0xFF => self.pc = self.rd_rm(rm, 8)?,
                    6 if op == 0xFF => {
                        let v = self.rd_rm(rm, 8)?;
                        self.push(v)?;
                    }
                    _ => return Err(self.undefined()),
                }
            }
            _ => return Err(self.undefined()),
        }
        Ok(())
    }

    fn two_byte(&mut self) -> R<()> {
        let op = self.fetch8()?;
        let osz = self.opsize();
        match op {
            // xgetbv: XCR0 = x87 | SSE
            0x01 => {
                if self.fetch8()? != 0xD0 {
                    return Err(self.undefined());
                }
                self.cpu.r[RAX] = 3;
                self.cpu.r[RDX] = 0;
            }
            0x0B => return Err(Exit::Halt),
            // prefetch / hint nops
            0x0D | 0x18..=0x1F => {
                self.modrm()?;
            }
            // rdtsc: the instruction count stands in for time
            0x31 => {
                let t = self.cpu.icount.wrapping_mul(16);
                self.cpu.r[RAX] = t & 0xFFFF_FFFF;
                self.cpu.r[RDX] = t >> 32;
            }
            0x40..=0x4F => {
                let (reg, rm) = self.modrm()?;
                let v = self.rd_rm(rm, osz)?;
                if self.cond(op & 0xF) {
                    self.sreg(reg, osz, v);
                } else if osz == 4 {
                    // a 32-bit cmov always zero-extends its destination
                    let keep = self.greg(reg, 4);
                    self.sreg(reg, 4, keep);
                }
            }
            0x80..=0x8F => {
                let d = self.fetch32()? as i32 as i64 as u64;
                if self.cond(op & 0xF) {
                    self.pc = self.pc.wrapping_add(d);
                }
            }
            0x90..=0x9F => {
                let (_, rm) = self.modrm()?;
                let v = self.cond(op & 0xF) as u64;
                self.wr_rm(rm, 1, v)?;
            }
            0xA2 => self.cpuid(),
            // bt / bts / btr / btc Ev, Gv
            0xA3 | 0xAB | 0xB3 | 0xBB => {
                let (reg, rm) = self.modrm()?;
                let bit = self.greg(reg, osz);
                self.bit_test((op >> 3) & 3, rm, bit, osz, true)?;
            }
            // group 8: bt* Ev, ib
            0xBA => {
                let (reg, rm) = self.modrm()?;
                let bit = self.fetch8()? as u64;
                if reg & 7 < 4 {
                    return Err(self.undefined());
                }
                self.bit_test(reg & 3, rm, bit, osz, false)?;
            }
            // shld / shrd
            0xA4 | 0xA5 | 0xAC | 0xAD => {
                let (reg, rm) = self.modrm()?;
                let count = if op & 1 == 0 { self.fetch8()? } else { self.cpu.r[RCX] as u8 };
                let bits = osz as u32 * 8;
                let c = (count & if osz == 8 { 63 } else { 31 }) as u32;
                let a = self.rd_rm(rm, osz)?;
                let b = self.greg(reg, osz);
                if c != 0 && c < bits {
                    let m = mask(osz);
                    let (r, cf) = if op < 0xA8 {
                        (((a << c) | (b >> (bits - c))) & m, (a >> (bits - c)) & 1 != 0)
                    } else {
                        (((a >> c) | (b << (bits - c))) & m, (a >> (c - 1)) & 1 != 0)
                    };
                    self.cpu.set_flag(CF, cf);
                    self.cpu.set_flag(OF, (a ^ r) & sign_bit(osz) != 0);
                    self.cpu.set_flag(AF, false);
                    self.set_szp(r, osz);
                    self.wr_rm(rm, osz, r)?;
                } else if osz == 4 {
                    self.wr_rm(rm, osz, a)?;
                }
            }
            // group 15: fxsave fxrstor ldmxcsr stmxcsr …; lfence / mfence / sfence
            0xAE => {
                let (reg, rm) = self.modrm()?;
                match (reg & 7, rm) {
                    (5..=7, Rm::Reg(_)) => {}
                    (2, _) if !matches!(rm, Rm::Reg(_)) => self.cpu.mxcsr = self.rd_rm(rm, 4)? as u32,
                    (3, _) if !matches!(rm, Rm::Reg(_)) => {
                        let v = self.cpu.mxcsr as u64;
                        self.wr_rm(rm, 4, v)?;
                    }
                    // clflush
                    (7, _) => {}
                    _ => return Err(self.undefined()),
                }
            }
            0xAF => {
                let (reg, rm) = self.modrm()?;
                let b = self.rd_rm(rm, osz)?;
                let a = self.greg(reg, osz);
                let r = self.imul(a, b, osz);
                self.sreg(reg, osz, r);
            }
            // cmpxchg
            0xB0 | 0xB1 => {
                let sz = if op == 0xB0 { 1 } else { osz };
                let (reg, rm) = self.modrm()?;
                let dst = self.rd_rm(rm, sz)?;
                let acc = self.greg(RAX as u8, sz);
                self.sub(acc, dst, false, sz);
                if acc == dst {
                    let v = self.greg(reg, sz);
                    self.wr_rm(rm, sz, v)?;
                } else {
                    self.sreg(RAX as u8, sz, dst);
                }
            }
            0xB6 | 0xB7 => {
                let (reg, rm) = self.modrm()?;
                let v = self.rd_rm(rm, if op == 0xB6 { 1 } else { 2 })?;
                self.sreg(reg, osz, v);
            }
            0xBE | 0xBF => {
                let from = if op == 0xBE { 1 } else { 2 };
                let (reg, rm) = self.modrm()?;
                let v = self.rd_rm(rm, from)?;
                self.sreg(reg, osz, sx(v, from));
            }
            // popcnt
            0xB8 if self.rep == 0xF3 => {
                let (reg, rm) = self.modrm()?;
                let v = self.rd_rm(rm, osz)?;
                self.cpu.rflags &= !(CF | OF | AF | SF | PF | ZF);
                self.cpu.set_flag(ZF, v == 0);
                self.sreg(reg, osz, v.count_ones() as u64);
            }
            // bsf / tzcnt, bsr / lzcnt
            0xBC | 0xBD => {
                let (reg, rm) = self.modrm()?;
                let v = self.rd_rm(rm, osz)?;
                let bits = osz as u64 * 8;
                if self.rep == 0xF3 {
                    let n = if v == 0 {
                        bits
                    } else if op == 0xBC {
                        v.trailing_zeros() as u64
                    } else {
                        v.leading_zeros() as u64 - (64 - bits)
                    };
                    self.cpu.set_flag(CF, v == 0);
                    self.cpu.set_flag(ZF, n == 0);
                    self.sreg(reg, osz, n);
                } else {
                    self.cpu.set_flag(ZF, v == 0);
                    if v != 0 {
                        let n = if op == 0xBC { v.trailing_zeros() } else { 63 - v.leading_zeros() };
                        self.sreg(reg, osz, n as u64);
                    }
                }
            }
            // xadd
            0xC0 | 0xC1 => {
                let sz = if op == 0xC0 { 1 } else { osz };
                let (reg, rm) = self.modrm()?;
                let a = self.rd_rm(rm, sz)?;
                let b = self.greg(reg, sz);
                let r = self.add(a, b, false, sz);
                self.sreg(reg, sz, a);
                self.wr_rm(rm, sz, r)?;
            }
            // movnti
            0xC3 => {
                let (reg, rm) = self.modrm()?;
                let v = self.greg(reg, osz);
                self.wr_rm(rm, osz, v)?;
            }
            // group 9: cmpxchg8b / cmpxchg16b
            0xC7 => {
                let (reg, rm) = self.modrm()?;
                if reg & 7 != 1 || matches!(rm, Rm::Reg(_)) {
                    return Err(self.undefined());
                }
                let sz = if osz == 8 { 8 } else { 4 };
                let a = self.ea(rm);
                let lo = self.mem.read_sized(a, sz)?;
                let hi = self.mem.read_sized(a + sz as u64, sz)?;
                if lo == self.greg(RAX as u8, sz) && hi == self.greg(RDX as u8, sz) {
                    let (b, c) = (self.greg(RBX as u8, sz), self.greg(RCX as u8, sz));
                    self.mem.write_sized(a, sz, b)?;
                    self.mem.write_sized(a + sz as u64, sz, c)?;
                    self.cpu.set_flag(ZF, true);
                } else {
                    self.sreg(RAX as u8, sz, lo);
                    self.sreg(RDX as u8, sz, hi);
                    self.cpu.set_flag(ZF, false);
                }
            }
            0xC8..=0xCF => {
                let r = (op & 7) | ((self.rex & 1) << 3);
                let v = self.cpu.r[r as usize];
                if osz == 8 {
                    self.cpu.r[r as usize] = v.swap_bytes();
                } else {
                    self.cpu.r[r as usize] = (v as u32).swap_bytes() as u64;
                }
            }
            _ => {
                if !crate::sse::exec(self, op)? {
                    return Err(self.undefined());
                }
            }
        }
        Ok(())
    }
}
