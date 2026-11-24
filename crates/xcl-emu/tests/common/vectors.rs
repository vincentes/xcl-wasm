//! Test vectors for the interpreter, shared by the test (`tests/oracle.rs`) and the program
//! that records what a real x86-64 CPU does (`scripts/x64-oracle`).
//!
//! A case is some machine code (a `ret` is appended) run from a pseudo-random state: registers,
//! flags, vector registers and a scratch block of memory, all derived from the case's number
//! and a seed. RDI and RSI point into the scratch block, RSP at a stack. What a run leaves is
//! written as the differences from the state it started with.

#![allow(dead_code)]

pub const CODE: u64 = 0x1_2000_0000;
pub const DATA: u64 = 0x1_2001_0000;
pub const STACK: u64 = 0x1_2002_0000;
pub const REGION: u64 = 0x1_0000;
/// bytes of the scratch block that are randomised and compared
pub const SCRATCH: usize = 512;
pub const SEEDS: u64 = 6;

pub const CF: u64 = 1;
pub const PF: u64 = 4;
pub const AF: u64 = 0x10;
pub const ZF: u64 = 0x40;
pub const SF: u64 = 0x80;
pub const OF: u64 = 0x800;
pub const STATUS: u64 = CF | PF | AF | ZF | SF | OF;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct State {
    /// RAX RCX RDX RBX RSP RBP RSI RDI R8..R15
    pub r: [u64; 16],
    pub flags: u64,
    pub ymm: [[u64; 4]; 16],
    pub mem: Vec<u8>,
}

pub struct Case {
    pub code: Vec<u8>,
    /// status flags the instruction leaves undefined (not compared)
    pub undefined: u64,
    /// registers set to fixed values: (index, value)
    pub init: Vec<(usize, u64)>,
    pub text: String,
}

const NAMES: [&str; 16] = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];

/// The cases of a specification: lines `hex bytes | undefined flags | reg=value,… | text`
/// (`-` for none; `#` starts a comment line).
pub fn parse(spec: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    for line in spec.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = line.splitn(4, '|').map(str::trim).collect();
        assert!(f.len() == 4, "four fields: {line}");
        let code = f[0].split_whitespace().map(|b| u8::from_str_radix(b, 16).unwrap_or_else(|_| panic!("hex byte in: {line}"))).collect();
        let undefined = f[1].chars().fold(0, |m, c| {
            m | match c {
                'c' => CF,
                'p' => PF,
                'a' => AF,
                'z' => ZF,
                's' => SF,
                'o' => OF,
                '-' => 0,
                _ => panic!("flag letter in: {line}"),
            }
        });
        let mut init = Vec::new();
        for item in f[2].split(',').map(str::trim).filter(|i| !i.is_empty() && *i != "-") {
            let (reg, value) = item.split_once('=').unwrap_or_else(|| panic!("reg=value in: {line}"));
            let index = NAMES.iter().position(|n| *n == reg).unwrap_or_else(|| panic!("register in: {line}"));
            let value = value.trim_start_matches("0x");
            init.push((index, u64::from_str_radix(value, 16).unwrap_or_else(|_| panic!("value in: {line}"))));
        }
        cases.push(Case { code, undefined, init, text: f[3].to_string() });
    }
    cases
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A value for a register: random, with the interesting ones more often than chance.
fn value(x: &mut u64) -> u64 {
    const EDGES: [u64; 12] = [0, 1, 2, u64::MAX, 0x7F, 0x80, 0xFF, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000];
    let r = splitmix(x);
    match r % 4 {
        0 => EDGES[(r >> 8) as usize % EDGES.len()],
        // small numbers (shift counts, indexes)
        1 => (r >> 8) % 70,
        _ => splitmix(x),
    }
}

/// The state run `seed` of case `index` starts from.
pub fn initial(case: &Case, index: usize, seed: u64) -> State {
    let mut x = (index as u64) << 32 | seed;
    let mut s = State { r: [0; 16], flags: 0x202, ymm: [[0; 4]; 16], mem: vec![0; SCRATCH] };
    for r in s.r.iter_mut() {
        *r = value(&mut x);
    }
    s.flags |= splitmix(&mut x) & STATUS;
    for y in s.ymm.iter_mut().flatten() {
        *y = value(&mut x);
    }
    for b in s.mem.chunks_mut(8) {
        b.copy_from_slice(&value(&mut x).to_le_bytes());
    }
    // the stack pointer as it is at the code's first instruction (the return address pushed)
    s.r[4] = STACK + REGION - 0x100 - 8;
    s.r[6] = DATA + 0x80;
    s.r[7] = DATA + 0x100;
    for &(reg, v) in &case.init {
        s.r[reg] = v;
    }
    s
}

/// What `after` differs in from `before`, as text; flags in `undefined` are left out.
pub fn difference(before: &State, after: &State, undefined: u64) -> String {
    let mut out = Vec::new();
    for i in 0..16 {
        // RSP: compared after the `ret`, where it is 8 above the start
        let was = if i == 4 { before.r[4] + 8 } else { before.r[i] };
        if after.r[i] != was {
            out.push(format!("{}={:x}", NAMES[i], after.r[i]));
        }
    }
    let keep = STATUS & !undefined;
    if (after.flags ^ before.flags) & keep != 0 {
        out.push(format!("f={:x}", after.flags & keep));
    }
    for i in 0..16 {
        if after.ymm[i] != before.ymm[i] {
            let y = after.ymm[i];
            out.push(format!("y{i}={:016x}{:016x}{:016x}{:016x}", y[3], y[2], y[1], y[0]));
        }
    }
    for (i, (a, b)) in after.mem.chunks(8).zip(before.mem.chunks(8)).enumerate() {
        if a != b {
            out.push(format!("m{i}={:x}", u64::from_le_bytes(a.try_into().unwrap())));
        }
    }
    if out.is_empty() {
        "-".into()
    } else {
        out.join(" ")
    }
}
