//! The interpreter against a real CPU: every case of `data/x64_cases.txt` is run from the same
//! states `scripts/x64-oracle` ran it from natively, and has to leave what the CPU left
//! (`data/x64_expected.txt`).

mod common {
    pub mod vectors;
}

use common::vectors::*;
use xcl_emu::cpu::{Cpu, Exit};
use xcl_emu::mem::{Memory, PROT_R, PROT_W, PROT_X};

const TRAPS: u64 = 0x7FFF_0000_0000;

fn run(case: &Case, before: &State) -> Result<State, String> {
    let mut mem = Memory::new();
    mem.map(CODE, REGION, PROT_R | PROT_X).unwrap();
    mem.map(DATA, REGION, PROT_R | PROT_W).unwrap();
    mem.map(STACK, REGION, PROT_R | PROT_W).unwrap();
    mem.write(CODE, &case.code).unwrap();
    mem.write_u8(CODE + case.code.len() as u64, 0xC3).unwrap();
    mem.write(DATA, &before.mem).unwrap();
    let mut cpu = Cpu::new();
    cpu.set_traps(TRAPS, 1);
    cpu.r = before.r;
    cpu.rflags = before.flags;
    cpu.v = before.ymm;
    cpu.rip = CODE;
    mem.write_u64(before.r[4], TRAPS).unwrap();
    match cpu.run(&mut mem, 100_000) {
        Exit::Import(0) => {}
        other => return Err(format!("{other:x?} at rip {:#x}", cpu.rip)),
    }
    let mut after = State { r: cpu.r, flags: cpu.rflags, ymm: cpu.v, mem: vec![0; SCRATCH] };
    mem.read(DATA, &mut after.mem).unwrap();
    Ok(after)
}

#[test]
fn the_interpreter_leaves_what_the_cpu_leaves() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let cases = parse(&std::fs::read_to_string(dir.join("x64_cases.txt")).unwrap());
    let expected = std::fs::read_to_string(dir.join("x64_expected.txt")).unwrap();
    let mut lines = expected.lines().filter(|l| !l.starts_with('#'));
    let mut wrong = Vec::new();
    let mut runs = 0;
    for (index, case) in cases.iter().enumerate() {
        for seed in 0..SEEDS {
            let line = lines.next().unwrap_or_else(|| panic!("x64_expected.txt ends before case {index}: record it again (scripts/x64-oracle)"));
            let want = line.strip_prefix(&format!("{index} {seed} ")).unwrap_or_else(|| panic!("x64_expected.txt is not in step at case {index}: record it again"));
            let before = initial(case, index, seed);
            let got = match run(case, &before) {
                Ok(after) => difference(&before, &after, case.undefined),
                Err(e) => e,
            };
            runs += 1;
            if got != want && wrong.len() < 40 {
                wrong.push(format!("{} (seed {seed})\n    cpu:         {want}\n    interpreter: {got}", case.text));
            }
        }
    }
    assert!(lines.next().is_none(), "x64_expected.txt has more runs than x64_cases.txt: record it again");
    assert!(wrong.is_empty(), "{} of {runs} runs differ from the CPU (the first of them):\n{}", wrong.len(), wrong.join("\n"));
    println!("{} cases, {runs} runs agree with the CPU", cases.len());
}
