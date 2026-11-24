//! Register state of one guest thread.

use crate::mem::{Fault, Memory};

pub const RAX: usize = 0;
pub const RCX: usize = 1;
pub const RDX: usize = 2;
pub const RBX: usize = 3;
pub const RSP: usize = 4;
pub const RBP: usize = 5;
pub const RSI: usize = 6;
pub const RDI: usize = 7;

pub const CF: u64 = 1 << 0;
pub const PF: u64 = 1 << 2;
pub const AF: u64 = 1 << 4;
pub const ZF: u64 = 1 << 6;
pub const SF: u64 = 1 << 7;
pub const DF: u64 = 1 << 10;
pub const OF: u64 = 1 << 11;

/// Why `Cpu::run` returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exit {
    /// The guest jumped to trap slot `n` (an import call, or a return the host asked to see).
    /// RIP is the slot's address; for a call the return address is on top of the stack.
    Import(u32),
    /// `max_instructions` ran; call `run` again to go on.
    Limit,
    /// An access to unmapped memory. RIP is at the instruction that faulted.
    Fault(Fault),
    /// An instruction the interpreter does not implement (or an invalid one). RIP is at it.
    Undefined { rip: u64, bytes: [u8; 8] },
    /// Division by zero or a quotient that does not fit (#DE). RIP is at the instruction.
    DivideError,
    /// `int3`. RIP is after it.
    Breakpoint,
    /// `int 0x29` (`__fastfail`): the program ends itself at once; the code is in RCX. RIP is
    /// at the instruction.
    FailFast,
    /// `hlt` / `ud2`-style stop the guest asked for. RIP is at the instruction.
    Halt,
}

impl From<Fault> for Exit {
    fn from(f: Fault) -> Self {
        Exit::Fault(f)
    }
}

/// What `cpuid` reports: the instruction sets the guest may use. The interpreter executes all of
/// them either way; a title that asks takes the paths for what it is told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Features {
    pub sse3: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub popcnt: bool,
    pub lzcnt: bool,
    pub movbe: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub bmi1: bool,
    pub bmi2: bool,
    pub adx: bool,
}

impl Features {
    /// Everything the interpreter implements: what a console title (built for Zen 2) expects,
    /// short of AES, PCLMULQDQ, SHA and x87 arithmetic, which it does not have.
    pub const IMPLEMENTED: Features =
        Features { sse3: true, ssse3: true, sse41: true, sse42: true, popcnt: true, lzcnt: true, movbe: true, avx: true, avx2: true, fma: true, f16c: true, bmi1: true, bmi2: true, adx: true };

    /// x86-64 as it started: SSE2 (and POPCNT).
    pub const BASELINE: Features =
        Features { sse3: false, ssse3: false, sse41: false, sse42: false, popcnt: true, lzcnt: false, movbe: false, avx: false, avx2: false, fma: false, f16c: false, bmi1: false, bmi2: false, adx: false };
}

#[derive(Clone)]
pub struct Cpu {
    /// RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI, R8..R15
    pub r: [u64; 16],
    pub rip: u64,
    pub rflags: u64,
    /// YMM0..15 as four little-endian quadwords; XMM is the low two
    pub v: [[u64; 4]; 16],
    pub fs_base: u64,
    pub gs_base: u64,
    pub mxcsr: u32,
    /// x87 control word (the x87 stack itself is not implemented)
    pub fcw: u16,
    /// instructions executed
    pub icount: u64,
    pub features: Features,
    pub(crate) trap_base: u64,
    pub(crate) trap_end: u64,
}

/// Bytes between two trap slots.
pub const TRAP_STRIDE: u64 = 16;

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        Cpu {
            r: [0; 16],
            rip: 0,
            rflags: 0x202,
            v: [[0; 4]; 16],
            fs_base: 0,
            gs_base: 0,
            mxcsr: 0x1F80,
            fcw: 0x027F,
            icount: 0,
            features: Features::BASELINE,
            trap_base: 0,
            trap_end: 0,
        }
    }

    /// The trap range: `count` slots from `base`. Nothing needs to be mapped there.
    pub fn set_traps(&mut self, base: u64, count: u32) {
        self.trap_base = base;
        self.trap_end = base + count as u64 * TRAP_STRIDE;
    }

    /// Address of trap slot `n`.
    pub fn trap_address(&self, n: u32) -> u64 {
        self.trap_base + n as u64 * TRAP_STRIDE
    }

    /// Execute until the guest leaves, at most `max_instructions`.
    pub fn run(&mut self, mem: &mut Memory, max_instructions: u64) -> Exit {
        crate::exec::run(self, mem, max_instructions)
    }

    #[inline]
    pub fn flag(&self, f: u64) -> bool {
        self.rflags & f != 0
    }

    #[inline]
    pub(crate) fn set_flag(&mut self, f: u64, on: bool) {
        if on {
            self.rflags |= f;
        } else {
            self.rflags &= !f;
        }
    }

    pub fn push(&mut self, mem: &mut Memory, v: u64) -> Result<(), Fault> {
        let sp = self.r[RSP].wrapping_sub(8);
        mem.write_u64(sp, v)?;
        self.r[RSP] = sp;
        Ok(())
    }

    pub fn pop(&mut self, mem: &Memory) -> Result<u64, Fault> {
        let v = mem.read_u64(self.r[RSP])?;
        self.r[RSP] = self.r[RSP].wrapping_add(8);
        Ok(v)
    }
}
