//! The Win64 calling convention, as the host sees an import call: `Cpu::run` has returned
//! `Exit::Import`, the return address is on top of the guest stack, the first four arguments are
//! in RCX, RDX, R8 and R9 and the rest above the 32-byte shadow space.

use crate::cpu::{Cpu, RAX, RCX, RDX, RSP};
use crate::mem::{Fault, Memory};

/// Integer / pointer arguments of the import call the guest is in.
pub struct Args<'a> {
    cpu: &'a Cpu,
    mem: &'a Memory,
}

impl<'a> Args<'a> {
    pub fn new(cpu: &'a Cpu, mem: &'a Memory) -> Self {
        Args { cpu, mem }
    }

    /// Argument `i` (0-based).
    pub fn get(&self, i: usize) -> Result<u64, Fault> {
        Ok(match i {
            0 => self.cpu.r[RCX],
            1 => self.cpu.r[RDX],
            2 => self.cpu.r[8],
            3 => self.cpu.r[9],
            _ => self.mem.read_u64(self.cpu.r[RSP] + 8 + 8 * i as u64)?,
        })
    }

    /// Floating-point argument `i` of the first four (XMM0..XMM3).
    pub fn float(&self, i: usize) -> f64 {
        f64::from_bits(self.cpu.v[i][0])
    }

    /// Where the call returns to.
    pub fn return_address(&self) -> Result<u64, Fault> {
        self.mem.read_u64(self.cpu.r[RSP])
    }
}

/// Return from the import call with `value` in RAX: pops the return address and continues there.
pub fn ret(cpu: &mut Cpu, mem: &Memory, value: u64) -> Result<(), Fault> {
    let to = mem.read_u64(cpu.r[RSP])?;
    cpu.r[RSP] = cpu.r[RSP].wrapping_add(8);
    cpu.r[RAX] = value;
    cpu.rip = to;
    Ok(())
}

/// Set up a call of the guest function at `entry` with up to four integer arguments, on the
/// current stack, returning to `return_to` (usually a trap slot, so the host sees it finish).
pub fn call(cpu: &mut Cpu, mem: &mut Memory, entry: u64, args: &[u64], return_to: u64) -> Result<(), Fault> {
    assert!(args.len() <= 4, "stack arguments are not set up");
    // 16-byte alignment at the call, shadow space, then the return address
    let sp = (cpu.r[RSP] & !0xF).wrapping_sub(0x20 + 8);
    mem.write_u64(sp, return_to)?;
    cpu.r[RSP] = sp;
    for (reg, &a) in [RCX, RDX, 8, 9].into_iter().zip(args) {
        cpu.r[reg] = a;
    }
    cpu.rip = entry;
    Ok(())
}
