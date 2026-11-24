//! Interpreter tests on hand-assembled code (the bytes are in the comments' order).

use xcl_emu::abi;
use xcl_emu::cpu::*;
use xcl_emu::mem::*;

const CODE: u64 = 0x40_0000;
const DATA: u64 = 0x50_0000;
const STACK_TOP: u64 = 0x70_0000;
const TRAPS: u64 = 0x7FFF_0000_0000;

fn machine(code: &[u8]) -> (Cpu, Memory) {
    let mut mem = Memory::new();
    mem.map(CODE, 0x1000, PROT_R | PROT_X).unwrap();
    mem.map(DATA, 0x1000, PROT_R | PROT_W).unwrap();
    mem.map(STACK_TOP - 0x1_0000, 0x1_0000, PROT_R | PROT_W).unwrap();
    mem.write(CODE, code).unwrap();
    let mut cpu = Cpu::new();
    cpu.set_traps(TRAPS, 16);
    cpu.r[RSP] = STACK_TOP - 0x100;
    // the code's final `ret` lands in trap slot 0
    abi::call(&mut cpu, &mut mem, CODE, &[], TRAPS).unwrap();
    (cpu, mem)
}

/// Run `code` until its final `ret`.
fn run(code: &[u8]) -> (Cpu, Memory) {
    let (mut cpu, mut mem) = machine(code);
    assert_eq!(cpu.run(&mut mem, 100_000), Exit::Import(0), "rip {:#x}", cpu.rip);
    (cpu, mem)
}

#[test]
fn arithmetic_and_a_loop() {
    // mov eax,5; add eax,7; ret
    assert_eq!(run(&[0xB8, 5, 0, 0, 0, 0x83, 0xC0, 7, 0xC3]).0.r[RAX], 12);
    // xor eax,eax; mov ecx,10; l: add eax,ecx; dec ecx; jnz l; ret
    assert_eq!(run(&[0x31, 0xC0, 0xB9, 10, 0, 0, 0, 0x01, 0xC8, 0xFF, 0xC9, 0x75, 0xFA, 0xC3]).0.r[RAX], 55);
    // mov rax,-3; imul rax,rax,5; ret
    assert_eq!(run(&[0x48, 0xC7, 0xC0, 0xFD, 0xFF, 0xFF, 0xFF, 0x48, 0x6B, 0xC0, 5, 0xC3]).0.r[RAX] as i64, -15);
    // mov eax,100; mov ecx,7; xor edx,edx; div ecx; ret
    let (cpu, _) = run(&[0xB8, 100, 0, 0, 0, 0xB9, 7, 0, 0, 0, 0x31, 0xD2, 0xF7, 0xF1, 0xC3]);
    assert_eq!((cpu.r[RAX], cpu.r[RDX]), (14, 2));
    // mov rax,-100; cqo; mov ecx,7; idiv rcx; ret
    let (cpu, _) = run(&[0x48, 0xC7, 0xC0, 0x9C, 0xFF, 0xFF, 0xFF, 0x48, 0x99, 0xB9, 7, 0, 0, 0, 0x48, 0xF7, 0xF9, 0xC3]);
    assert_eq!((cpu.r[RAX] as i64, cpu.r[RDX] as i64), (-14, -2));
}

#[test]
fn flags_of_add_sub_and_shifts() {
    // mov al,0x7f; add al,1; ret
    let (cpu, _) = run(&[0xB0, 0x7F, 0x04, 0x01, 0xC3]);
    assert_eq!(cpu.r[RAX] & 0xFF, 0x80);
    assert!(cpu.flag(OF) && cpu.flag(SF) && !cpu.flag(CF) && !cpu.flag(ZF) && cpu.flag(AF));
    // xor eax,eax; sub eax,1; ret
    let (cpu, _) = run(&[0x31, 0xC0, 0x83, 0xE8, 0x01, 0xC3]);
    assert_eq!(cpu.r[RAX], 0xFFFF_FFFF);
    assert!(cpu.flag(CF) && cpu.flag(SF) && !cpu.flag(OF));
    // mov eax,0x80000001; rol eax,1; ret
    let (cpu, _) = run(&[0xB8, 1, 0, 0, 0x80, 0xD1, 0xC0, 0xC3]);
    assert_eq!(cpu.r[RAX], 3);
    assert!(cpu.flag(CF));
    // mov eax,0x80000000; sar eax,4; ret
    assert_eq!(run(&[0xB8, 0, 0, 0, 0x80, 0xC1, 0xF8, 4, 0xC3]).0.r[RAX], 0xF800_0000);
    // mov rax,-1; shr rax,60; ret
    assert_eq!(run(&[0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, 0x48, 0xC1, 0xE8, 60, 0xC3]).0.r[RAX], 0xF);
    // mov eax,1; shl eax,31; setc cl; seto dl; ret  (shl by 31: CF = bit 1 → 0)
    let (cpu, _) = run(&[0xB8, 1, 0, 0, 0, 0xC1, 0xE0, 31, 0x0F, 0x92, 0xC1, 0xC3]);
    assert_eq!((cpu.r[RAX], cpu.r[RCX] & 0xFF), (0x8000_0000, 0));
    // stc; mov eax,0xffffffff; adc eax,0; ret
    let (cpu, _) = run(&[0xF9, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0x83, 0xD0, 0x00, 0xC3]);
    assert_eq!(cpu.r[RAX], 0);
    assert!(cpu.flag(CF) && cpu.flag(ZF));
}

#[test]
fn registers_of_every_width() {
    // mov rax,-1; mov eax,0x1234; mov bl,ah; ret   (a 32-bit write clears the upper half)
    let (cpu, _) = run(&[0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, 0xB8, 0x34, 0x12, 0, 0, 0x88, 0xE3, 0xC3]);
    assert_eq!((cpu.r[RAX], cpu.r[RBX] & 0xFF), (0x1234, 0x12));
    // mov rax,-1; mov ax,0x1234; mov al,0x56; ret   (16- and 8-bit writes keep the rest)
    let (cpu, _) = run(&[0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, 0x66, 0xB8, 0x34, 0x12, 0xB0, 0x56, 0xC3]);
    assert_eq!(cpu.r[RAX], 0xFFFF_FFFF_FFFF_1256);
    // mov rsi,0x7788; mov bl,sil (REX: sil, not dh); movsx eax,bl; movzx ecx,bl; ret
    let (cpu, _) = run(&[0x48, 0xC7, 0xC6, 0x88, 0x77, 0, 0, 0x40, 0x88, 0xF3, 0x0F, 0xBE, 0xC3, 0x0F, 0xB6, 0xCB, 0xC3]);
    assert_eq!((cpu.r[RAX], cpu.r[RCX]), (0xFFFF_FF88, 0x88));
    // mov r10,0x1122334455667788; bswap r10; movsxd rax,r10d... : mov rax,r10; ret
    let (cpu, _) = run(&[0x49, 0xBA, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x49, 0x0F, 0xCA, 0x4C, 0x89, 0xD0, 0xC3]);
    assert_eq!(cpu.r[RAX], 0x8877_6655_4433_2211);
}

#[test]
fn calls_the_stack_and_rip_relative_operands() {
    // call f; ret; f: mov eax,42; ret
    assert_eq!(run(&[0xE8, 1, 0, 0, 0, 0xC3, 0xB8, 42, 0, 0, 0, 0xC3]).0.r[RAX], 42);
    // push 7; pop rcx; lea rax,[rcx+rcx*4+3]; ret
    assert_eq!(run(&[0x6A, 7, 0x59, 0x48, 0x8D, 0x44, 0x89, 0x03, 0xC3]).0.r[RAX], 38);
    // mov dword [rip+0xf6],0x12345678  (an immediate after the displacement: next insn is at
    // +10, so the operand is CODE+0x100); cmp dword [rip+0xec],0x12345678; sete al; ret
    let code = [0xC7, 0x05, 0xF6, 0, 0, 0, 0x78, 0x56, 0x34, 0x12, 0x81, 0x3D, 0xEC, 0, 0, 0, 0x78, 0x56, 0x34, 0x12, 0x0F, 0x94, 0xC0, 0xC3];
    let (cpu, mem) = run(&code);
    assert_eq!(mem.read_u32(CODE + 0x100).unwrap(), 0x1234_5678);
    assert_eq!(cpu.r[RAX] & 0xFF, 1);
}

#[test]
fn string_instructions_and_atomics() {
    // mov rdi,DATA; mov ecx,16; mov al,0xAB; rep stosb; mov rsi,DATA; lea rdi,[rsi+32];
    // mov ecx,2; rep movsq; ret
    let mut code = vec![0x48, 0xBF];
    code.extend_from_slice(&DATA.to_le_bytes());
    code.extend_from_slice(&[0xB9, 16, 0, 0, 0, 0xB0, 0xAB, 0xF3, 0xAA, 0x48, 0xBE]);
    code.extend_from_slice(&DATA.to_le_bytes());
    code.extend_from_slice(&[0x48, 0x8D, 0x7E, 0x20, 0xB9, 2, 0, 0, 0, 0xF3, 0x48, 0xA5, 0xC3]);
    let (cpu, mem) = run(&code);
    assert_eq!(mem.read_u64(DATA + 40).unwrap(), 0xABAB_ABAB_ABAB_ABAB);
    assert_eq!(mem.read_u8(DATA + 48).unwrap(), 0);
    assert_eq!((cpu.r[RCX], cpu.r[RDI]), (0, DATA + 48));
    // mov rbx,DATA; mov dword [rbx],5; mov eax,5; mov ecx,9; lock cmpxchg [rbx],ecx;
    // lock xadd [rbx],eax; bts dword [rbx],4; ret        → [rbx] = 9+5 | 16 = 30, eax = 9
    let mut code = vec![0x48, 0xBB];
    code.extend_from_slice(&DATA.to_le_bytes());
    code.extend_from_slice(&[0xC7, 0x03, 5, 0, 0, 0, 0xB8, 5, 0, 0, 0, 0xB9, 9, 0, 0, 0, 0xF0, 0x0F, 0xB1, 0x0B, 0xF0, 0x0F, 0xC1, 0x03, 0x0F, 0xBA, 0x2B, 4, 0xC3]);
    let (cpu, mem) = run(&code);
    assert_eq!((mem.read_u32(DATA).unwrap(), cpu.r[RAX]), (30, 9));
    assert!(!cpu.flag(CF));
}

#[test]
fn sse2_integer_and_floating_point() {
    // mov eax,3; movd xmm0,eax; paddd xmm0,xmm0; movd eax,xmm0; ret
    assert_eq!(run(&[0xB8, 3, 0, 0, 0, 0x66, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0xFE, 0xC0, 0x66, 0x0F, 0x7E, 0xC0, 0xC3]).0.r[RAX], 6);
    // mov eax,7; cvtsi2sd xmm0,eax; addsd xmm0,xmm0; mulsd xmm0,xmm0; cvttsd2si eax,xmm0; ret
    let code = [0xB8, 7, 0, 0, 0, 0xF2, 0x0F, 0x2A, 0xC0, 0xF2, 0x0F, 0x58, 0xC0, 0xF2, 0x0F, 0x59, 0xC0, 0xF2, 0x0F, 0x2C, 0xC0, 0xC3];
    assert_eq!(run(&code).0.r[RAX], 196);
    // pxor xmm1,xmm1; pcmpeqb xmm1,xmm1; pmovmskb eax,xmm1; ret
    assert_eq!(run(&[0x66, 0x0F, 0xEF, 0xC9, 0x66, 0x0F, 0x74, 0xC9, 0x66, 0x0F, 0xD7, 0xC1, 0xC3]).0.r[RAX], 0xFFFF);
    // mov eax,1; cvtsi2ss xmm0,eax; mov eax,2; cvtsi2ss xmm1,eax; ucomiss xmm0,xmm1; setb al; ret
    let code = [0xB8, 1, 0, 0, 0, 0xF3, 0x0F, 0x2A, 0xC0, 0xB8, 2, 0, 0, 0, 0xF3, 0x0F, 0x2A, 0xC8, 0x0F, 0x2E, 0xC1, 0x0F, 0x92, 0xC0, 0xC3];
    assert_eq!(run(&code).0.r[RAX] & 0xFF, 1);
    // movups [DATA] round trip: mov rbx,DATA; mov rax,0x0102030405060708; movq xmm2,rax;
    // punpcklqdq xmm2,xmm2; movups [rbx],xmm2; movdqu xmm3,[rbx]; pshufd xmm3,xmm3,0x1B;
    // movq rax,xmm3; ret                                  → the high dword pair swapped
    let mut code = vec![0x48, 0xBB];
    code.extend_from_slice(&DATA.to_le_bytes());
    code.extend_from_slice(&[0x48, 0xB8, 8, 7, 6, 5, 4, 3, 2, 1, 0x66, 0x48, 0x0F, 0x6E, 0xD0, 0x66, 0x0F, 0x6C, 0xD2, 0x0F, 0x11, 0x13]);
    code.extend_from_slice(&[0xF3, 0x0F, 0x6F, 0x1B, 0x66, 0x0F, 0x70, 0xDB, 0x1B, 0x66, 0x48, 0x0F, 0x7E, 0xD8, 0xC3]);
    let (cpu, mem) = run(&code);
    assert_eq!(mem.read_u64(DATA + 8).unwrap(), 0x0102_0304_0506_0708);
    assert_eq!(cpu.r[RAX], 0x0506_0708_0102_0304);
}

#[test]
fn the_guest_leaves_through_traps_faults_and_limits() {
    // mov ecx,11; mov edx,22; call [rip+0]; (slot:) dq TRAPS+3*16 … the host answers, then:
    // add eax,1; ret
    let mut code = vec![0xB9, 11, 0, 0, 0, 0xBA, 22, 0, 0, 0, 0xFF, 0x15, 4, 0, 0, 0, 0x83, 0xC0, 1, 0xC3];
    code.extend_from_slice(&(TRAPS + 3 * TRAP_STRIDE).to_le_bytes());
    let (mut cpu, mut mem) = machine(&code);
    assert_eq!(cpu.run(&mut mem, 1000), Exit::Import(3));
    let args = abi::Args::new(&cpu, &mem);
    assert_eq!((args.get(0).unwrap(), args.get(1).unwrap(), args.return_address().unwrap()), (11, 22, CODE + 16));
    abi::ret(&mut cpu, &mem, 100).unwrap();
    assert_eq!(cpu.run(&mut mem, 1000), Exit::Import(0));
    assert_eq!(cpu.r[RAX], 101);

    // mov eax,[0x1000] (absolute, unmapped)
    let (mut cpu, mut mem) = machine(&[0x8B, 0x04, 0x25, 0x00, 0x10, 0, 0, 0xC3]);
    assert_eq!(cpu.run(&mut mem, 1000), Exit::Fault(Fault { addr: 0x1000, write: false }));
    assert_eq!(cpu.rip, CODE);

    // xor ecx,ecx; div ecx
    let (mut cpu, mut mem) = machine(&[0x31, 0xC9, 0xF7, 0xF1, 0xC3]);
    assert_eq!(cpu.run(&mut mem, 1000), Exit::DivideError);
    assert_eq!(cpu.rip, CODE + 2);

    // an x87 instruction (not implemented): fld1
    let (mut cpu, mut mem) = machine(&[0xD9, 0xE8, 0xC3]);
    assert!(matches!(cpu.run(&mut mem, 1000), Exit::Undefined { rip: CODE, .. }));

    // jmp $
    let (mut cpu, mut mem) = machine(&[0xEB, 0xFE]);
    assert_eq!(cpu.run(&mut mem, 1000), Exit::Limit);
    assert_eq!(cpu.icount, 1000);
}
