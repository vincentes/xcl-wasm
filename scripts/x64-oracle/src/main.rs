//! Runs the interpreter's test vectors on the real CPU and prints what each run leaves: one
//! line per case and seed, `<case> <seed> <differences>` (see `vectors.rs`).
//!
//! The code, the scratch block and the stack are mapped at the addresses the interpreter's test
//! maps them at, so pointers in registers are the same numbers on both sides.

#[cfg(not(all(target_arch = "x86_64", unix)))]
compile_error!("x64-oracle runs test vectors natively: build it for x86-64 macOS or Linux");

#[path = "../../../crates/xcl-emu/tests/common/vectors.rs"]
mod vectors;

use std::arch::global_asm;

use vectors::*;

/// What the trampoline loads and stores: the general registers, the flags, YMM0..15.
#[repr(C)]
struct Native {
    r: [u64; 16],
    flags: u64,
    ymm: [[u64; 4]; 16],
}

#[no_mangle]
static mut XCL_SAVED_RSP: u64 = 0;
#[no_mangle]
static mut XCL_STATE: u64 = 0;
#[no_mangle]
static mut XCL_CODE: u64 = 0;

extern "C" {
    /// Run the code at `code` from `state` on the stack ending at `stack_top`; `state` is what
    /// it leaves.
    fn xcl_tramp(state: *mut Native, code: u64, stack_top: u64);
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
}

#[cfg(target_os = "macos")]
macro_rules! sym {
    ($s:literal) => {
        concat!("_", $s)
    };
}
#[cfg(not(target_os = "macos"))]
macro_rules! sym {
    ($s:literal) => {
        $s
    };
}

global_asm!(
    concat!(".globl ", sym!("xcl_tramp")),
    concat!(sym!("xcl_tramp"), ":"),
    "push rbx",
    "push rbp",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    concat!("mov [rip + ", sym!("XCL_SAVED_RSP"), "], rsp"),
    concat!("mov [rip + ", sym!("XCL_STATE"), "], rdi"),
    concat!("mov [rip + ", sym!("XCL_CODE"), "], rsi"),
    "mov rsp, rdx",
    "vmovdqu ymm0, [rdi + 136]",
    "vmovdqu ymm1, [rdi + 168]",
    "vmovdqu ymm2, [rdi + 200]",
    "vmovdqu ymm3, [rdi + 232]",
    "vmovdqu ymm4, [rdi + 264]",
    "vmovdqu ymm5, [rdi + 296]",
    "vmovdqu ymm6, [rdi + 328]",
    "vmovdqu ymm7, [rdi + 360]",
    "vmovdqu ymm8, [rdi + 392]",
    "vmovdqu ymm9, [rdi + 424]",
    "vmovdqu ymm10, [rdi + 456]",
    "vmovdqu ymm11, [rdi + 488]",
    "vmovdqu ymm12, [rdi + 520]",
    "vmovdqu ymm13, [rdi + 552]",
    "vmovdqu ymm14, [rdi + 584]",
    "vmovdqu ymm15, [rdi + 616]",
    "push qword ptr [rdi + 128]",
    "popfq",
    "mov rax, [rdi]",
    "mov rcx, [rdi + 8]",
    "mov rdx, [rdi + 16]",
    "mov rbx, [rdi + 24]",
    "mov rbp, [rdi + 40]",
    "mov rsi, [rdi + 48]",
    "mov r8, [rdi + 64]",
    "mov r9, [rdi + 72]",
    "mov r10, [rdi + 80]",
    "mov r11, [rdi + 88]",
    "mov r12, [rdi + 96]",
    "mov r13, [rdi + 104]",
    "mov r14, [rdi + 112]",
    "mov r15, [rdi + 120]",
    "mov rdi, [rdi + 56]",
    concat!("call qword ptr [rip + ", sym!("XCL_CODE"), "]"),
    // nothing below changes a flag before they are saved
    "pushfq",
    "push rdi",
    concat!("mov rdi, [rip + ", sym!("XCL_STATE"), "]"),
    "mov [rdi], rax",
    "mov [rdi + 8], rcx",
    "mov [rdi + 16], rdx",
    "mov [rdi + 24], rbx",
    "mov [rdi + 40], rbp",
    "mov [rdi + 48], rsi",
    "mov [rdi + 64], r8",
    "mov [rdi + 72], r9",
    "mov [rdi + 80], r10",
    "mov [rdi + 88], r11",
    "mov [rdi + 96], r12",
    "mov [rdi + 104], r13",
    "mov [rdi + 112], r14",
    "mov [rdi + 120], r15",
    "pop rax",
    "mov [rdi + 56], rax",
    "pop rax",
    "mov [rdi + 128], rax",
    "mov [rdi + 32], rsp",
    "vmovdqu [rdi + 136], ymm0",
    "vmovdqu [rdi + 168], ymm1",
    "vmovdqu [rdi + 200], ymm2",
    "vmovdqu [rdi + 232], ymm3",
    "vmovdqu [rdi + 264], ymm4",
    "vmovdqu [rdi + 296], ymm5",
    "vmovdqu [rdi + 328], ymm6",
    "vmovdqu [rdi + 360], ymm7",
    "vmovdqu [rdi + 392], ymm8",
    "vmovdqu [rdi + 424], ymm9",
    "vmovdqu [rdi + 456], ymm10",
    "vmovdqu [rdi + 488], ymm11",
    "vmovdqu [rdi + 520], ymm12",
    "vmovdqu [rdi + 552], ymm13",
    "vmovdqu [rdi + 584], ymm14",
    "vmovdqu [rdi + 616], ymm15",
    concat!("mov rsp, [rip + ", sym!("XCL_SAVED_RSP"), "]"),
    "vzeroupper",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbp",
    "pop rbx",
    "ret",
);

/// Map `REGION` bytes at exactly `at`.
fn map(at: u64, executable: bool) -> &'static mut [u8] {
    const MAP_PRIVATE: i32 = 2;
    #[cfg(target_os = "macos")]
    const MAP_ANON: i32 = 0x1000;
    #[cfg(not(target_os = "macos"))]
    const MAP_ANON: i32 = 0x20;
    let prot = if executable { 7 } else { 3 };
    let p = unsafe { mmap(at as *mut u8, REGION as usize, prot, MAP_PRIVATE | MAP_ANON, -1, 0) };
    assert_eq!(p as u64, at, "the vectors' addresses are not free in this process");
    unsafe { std::slice::from_raw_parts_mut(p, REGION as usize) }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: x64-oracle <x64_cases.txt> [where this is recorded]");
    let cases = parse(&std::fs::read_to_string(path).expect("read the cases"));
    assert!(std::arch::is_x86_feature_detected!("avx2"), "the trampoline saves YMM registers: AVX2 is needed");
    let code = map(CODE, true);
    let data = map(DATA, false);
    map(STACK, false);
    println!("# what the CPU leaves for each case and seed of x64_cases.txt; made by scripts/x64-oracle – do not edit");
    // where it was recorded (the second argument): silicon, or an x86-64 implementation like Rosetta
    println!("# recorded on: {}", std::env::args().nth(2).unwrap_or_else(|| "(not said)".into()));
    for (index, case) in cases.iter().enumerate() {
        code[..case.code.len()].copy_from_slice(&case.code);
        code[case.code.len()] = 0xC3;
        for seed in 0..SEEDS {
            let before = initial(case, index, seed);
            data.fill(0);
            data[..SCRATCH].copy_from_slice(&before.mem);
            let mut n = Native { r: before.r, flags: before.flags, ymm: before.ymm };
            unsafe { xcl_tramp(&mut n, CODE, STACK + REGION - 0x100) };
            let after = State { r: n.r, flags: n.flags, ymm: n.ymm, mem: data[..SCRATCH].to_vec() };
            println!("{index} {seed} {}", difference(&before, &after, case.undefined));
        }
    }
}
