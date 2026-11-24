//! Structured exception handling for the guest: the x64 unwinder (`.pdata` / UNWIND_INFO) and
//! exception dispatch, as RaiseException, RtlUnwindEx, RtlVirtualUnwind and friends. C++ and
//! Rust exceptions are built on these (the title's own personality routines run as guest code).
//!
//! A language handler is called from the host (`Guest::call`), so a guest stack walk that
//! starts inside one ends at the host's return slot. What the walk continues with is kept in
//! the thread's `seh` list (`Boundary`): the exception's original context for a handler called
//! while dispatching, the frame being unwound for one called while unwinding (Windows'
//! "collided unwind").

#![allow(non_snake_case)]

use std::collections::HashMap;

use xcl_emu::cpu::{RAX, RSP};
use xcl_emu::mem::Fault;

use super::register;
use crate::{Guest, Handler, Stop, SLOT_CALL_RETURN};

type H = Result<u64, Stop>;

const CONTEXT_SIZE: u64 = 0x4D0;
const EXCEPTION_RECORD_SIZE: u64 = 152;
const UNW_FLAG_EHANDLER: u8 = 1;
const UNW_FLAG_UHANDLER: u8 = 2;
const UNW_FLAG_CHAININFO: u8 = 4;
const EXCEPTION_NONCONTINUABLE: u32 = 0x01;
const EXCEPTION_UNWINDING: u32 = 0x02;
const EXCEPTION_EXIT_UNWIND: u32 = 0x04;
const EXCEPTION_TARGET_UNWIND: u32 = 0x20;
const STATUS_UNWIND: u32 = 0xC000_0027;

/// The registers of a CONTEXT record the unwinder works with.
#[derive(Debug, Clone)]
pub struct Context {
    pub r: [u64; 16],
    pub rip: u64,
    pub eflags: u32,
    pub mxcsr: u32,
    pub xmm: [[u64; 2]; 16],
}

/// Where a stack walk goes on when it reaches a handler call made by the host.
#[derive(Debug, Clone)]
pub enum Boundary {
    /// the handler was called while dispatching: the walk restarts at the exception's context
    Dispatch { original: u64, depth: u32, rsp: u64 },
    /// the handler was called while unwinding `frame`: the walk goes on behind that frame
    Unwind { frame_context: Box<Context>, caller_context: Box<Context>, frame: u64, depth: u32, rsp: u64 },
}

impl Boundary {
    fn rsp(&self) -> u64 {
        match self {
            Boundary::Dispatch { rsp, .. } | Boundary::Unwind { rsp, .. } => *rsp,
        }
    }
}

fn read_context(g: &Guest, at: u64) -> Result<Context, Fault> {
    let mut c = Context { r: [0; 16], rip: 0, eflags: 0, mxcsr: 0, xmm: [[0; 2]; 16] };
    for (i, r) in c.r.iter_mut().enumerate() {
        *r = g.mem.read_u64(at + 0x78 + 8 * i as u64)?;
    }
    c.rip = g.mem.read_u64(at + 0xF8)?;
    c.eflags = g.mem.read_u32(at + 0x44)?;
    c.mxcsr = g.mem.read_u32(at + 0x34)?;
    for (i, x) in c.xmm.iter_mut().enumerate() {
        *x = [g.mem.read_u64(at + 0x1A0 + 16 * i as u64)?, g.mem.read_u64(at + 0x1A8 + 16 * i as u64)?];
    }
    Ok(c)
}

fn write_context(g: &mut Guest, at: u64, c: &Context) -> Result<(), Fault> {
    // CONTEXT_AMD64 | CONTROL | INTEGER | FLOATING_POINT
    g.mem.write_u32(at + 0x30, 0x0010_000B)?;
    g.mem.write_u32(at + 0x34, c.mxcsr)?;
    // cs ds es fs gs ss
    for (i, seg) in [0x33u16, 0x2B, 0x2B, 0x53, 0x2B, 0x2B].into_iter().enumerate() {
        g.mem.write_u16(at + 0x38 + 2 * i as u64, seg)?;
    }
    g.mem.write_u32(at + 0x44, c.eflags)?;
    for (i, r) in c.r.iter().enumerate() {
        g.mem.write_u64(at + 0x78 + 8 * i as u64, *r)?;
    }
    g.mem.write_u64(at + 0xF8, c.rip)?;
    g.mem.write_u32(at + 0x118, c.mxcsr)?;
    for (i, x) in c.xmm.iter().enumerate() {
        g.mem.write_u64(at + 0x1A0 + 16 * i as u64, x[0])?;
        g.mem.write_u64(at + 0x1A8 + 16 * i as u64, x[1])?;
    }
    Ok(())
}

/// The context the running thread has once the import call it is in has returned.
fn capture(g: &Guest) -> Result<Context, Fault> {
    let mut c = Context { r: g.cpu.r, rip: g.mem.read_u64(g.cpu.r[RSP])?, eflags: g.cpu.rflags as u32, mxcsr: g.cpu.mxcsr, xmm: [[0; 2]; 16] };
    c.r[RSP] += 8;
    for (i, x) in c.xmm.iter_mut().enumerate() {
        *x = [g.cpu.v[i][0], g.cpu.v[i][1]];
    }
    Ok(c)
}

/// Continue the running thread with `c`.
fn apply(g: &mut Guest, c: &Context) {
    g.cpu.r = c.r;
    g.cpu.rip = c.rip;
    g.cpu.rflags = (c.eflags as u64 & 0x0CD5) | 0x202;
    g.cpu.mxcsr = c.mxcsr;
    for (i, x) in c.xmm.iter().enumerate() {
        g.cpu.v[i][0] = x[0];
        g.cpu.v[i][1] = x[1];
    }
}

/// RUNTIME_FUNCTION of the function around `pc`: (image base, address of the entry).
fn lookup(g: &Guest, pc: u64) -> Option<(u64, u64)> {
    let m = g.modules.iter().find(|m| pc >= m.base && pc < m.base + m.size)?;
    let rva = (pc - m.base) as u32;
    let (mut lo, mut hi) = (0u64, m.pdata_count as u64);
    while lo < hi {
        let mid = (lo + hi) / 2;
        let at = m.pdata + 12 * mid;
        let (begin, end) = (g.mem.read_u32(at).ok()?, g.mem.read_u32(at + 4).ok()?);
        if rva < begin {
            hi = mid;
        } else if rva >= end {
            lo = mid + 1;
        } else {
            return Some((m.base, at));
        }
    }
    None
}

/// One frame of a stack walk.
struct Frame {
    image_base: u64,
    entry: u64,
    control_pc: u64,
    /// the language handler (0: none of the kind asked for) and its data
    handler: u64,
    handler_data: u64,
    /// the establisher frame
    frame: u64,
}

/// RtlVirtualUnwind: undo the function's prologue in `c` (which becomes its caller's context).
fn virtual_unwind(g: &Guest, kind: u8, image_base: u64, pc: u64, entry: u64, c: &mut Context) -> Result<(u64, u64, u64), Fault> {
    let m = &g.mem;
    let begin = m.read_u32(entry)?;
    let mut unwind = m.read_u32(entry + 8)?;
    let offset_in_function = (pc - image_base) as u32 - begin;
    let (mut handler, mut handler_data, mut frame) = (0, 0, 0);
    let mut chained = false;
    let mut machine_frame = false;
    loop {
        let info = image_base + (unwind & !1) as u64;
        let flags = m.read_u8(info)? >> 3;
        let prolog = m.read_u8(info + 1)?;
        let count = m.read_u8(info + 2)? as u64;
        let fr = m.read_u8(info + 3)?;
        let (frame_reg, frame_offset) = ((fr & 0xF) as usize, (fr >> 4) as u64 * 16);
        let in_prolog = !chained && offset_in_function < prolog as u32;
        let base = if frame_reg != 0 && !in_prolog { c.r[frame_reg].wrapping_sub(frame_offset) } else { c.r[RSP] };
        if !chained {
            frame = base;
            if flags & kind & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) != 0 {
                let at = info + 4 + ((count + 1) & !1) * 2;
                handler = image_base + m.read_u32(at)? as u64;
                handler_data = at + 4;
            }
        }
        let codes = info + 4;
        let mut i = 0;
        while i < count {
            let code_offset = m.read_u8(codes + 2 * i)?;
            let op = m.read_u8(codes + 2 * i + 1)?;
            let (op, op_info) = (op & 0xF, (op >> 4) as usize);
            let next16 = |k: u64| m.read_u16(codes + 2 * (i + k));
            let next32 = |k: u64| m.read_u32(codes + 2 * (i + k));
            // only what the prologue had done when the function was left
            let done = chained || offset_in_function >= code_offset as u32;
            let slots = match op {
                // UWOP_PUSH_NONVOL
                0 => {
                    if done {
                        c.r[op_info] = m.read_u64(c.r[RSP])?;
                        c.r[RSP] += 8;
                    }
                    1
                }
                // UWOP_ALLOC_LARGE
                1 => {
                    let (size, slots) = if op_info == 0 { (next16(1)? as u64 * 8, 2) } else { (next32(1)? as u64, 3) };
                    if done {
                        c.r[RSP] += size;
                    }
                    slots
                }
                // UWOP_ALLOC_SMALL
                2 => {
                    if done {
                        c.r[RSP] += op_info as u64 * 8 + 8;
                    }
                    1
                }
                // UWOP_SET_FPREG
                3 => {
                    if done {
                        c.r[RSP] = c.r[frame_reg].wrapping_sub(frame_offset);
                    }
                    1
                }
                // UWOP_SAVE_NONVOL (_FAR)
                4 | 5 => {
                    let (off, slots) = if op == 4 { (next16(1)? as u64 * 8, 2) } else { (next32(1)? as u64, 3) };
                    if done {
                        c.r[op_info] = m.read_u64(base + off)?;
                    }
                    slots
                }
                // UWOP_EPILOG (version 2): describes the epilogue, nothing to undo
                6 => 2,
                7 => 3,
                // UWOP_SAVE_XMM128 (_FAR)
                8 | 9 => {
                    let (off, slots) = if op == 8 { (next16(1)? as u64 * 16, 2) } else { (next32(1)? as u64, 3) };
                    if done {
                        c.xmm[op_info] = [m.read_u64(base + off)?, m.read_u64(base + off + 8)?];
                    }
                    slots
                }
                // UWOP_PUSH_MACHFRAME: [error code,] rip, cs, eflags, rsp, ss
                _ => {
                    if done {
                        let at = c.r[RSP] + if op_info != 0 { 8 } else { 0 };
                        c.rip = m.read_u64(at)?;
                        c.r[RSP] = m.read_u64(at + 24)?;
                        machine_frame = true;
                    }
                    1
                }
            };
            i += slots;
        }
        if flags & UNW_FLAG_CHAININFO == 0 {
            break;
        }
        // the parent function's RUNTIME_FUNCTION follows the codes
        unwind = m.read_u32(codes + ((count + 1) & !1) * 2 + 8)?;
        chained = true;
    }
    if !machine_frame {
        c.rip = m.read_u64(c.r[RSP])?;
        c.r[RSP] += 8;
    }
    Ok((handler, handler_data, frame))
}

/// Step from the frame `c` is in to its caller; a function without unwind data is a leaf.
fn step(g: &Guest, kind: u8, c: &mut Context) -> Result<Frame, Fault> {
    let pc = c.rip;
    match lookup(g, pc) {
        Some((image_base, entry)) => {
            let (handler, handler_data, frame) = virtual_unwind(g, kind, image_base, pc, entry, c)?;
            Ok(Frame { image_base, entry, control_pc: pc, handler, handler_data, frame })
        }
        None => {
            let frame = c.r[RSP];
            c.rip = g.mem.read_u64(frame)?;
            c.r[RSP] += 8;
            Ok(Frame { image_base: 0, entry: 0, control_pc: pc, handler: 0, handler_data: 0, frame })
        }
    }
}

/// DISPATCHER_CONTEXT for a handler call.
fn write_dispatcher(g: &mut Guest, at: u64, f: &Frame, target_ip: u64, context: u64) -> Result<(), Fault> {
    g.mem.write(at, &[0u8; 0x60])?;
    g.mem.write_u64(at, f.control_pc)?;
    g.mem.write_u64(at + 8, f.image_base)?;
    g.mem.write_u64(at + 16, f.entry)?;
    g.mem.write_u64(at + 24, f.frame)?;
    g.mem.write_u64(at + 32, target_ip)?;
    g.mem.write_u64(at + 40, context)?;
    g.mem.write_u64(at + 48, f.handler)?;
    g.mem.write_u64(at + 56, f.handler_data)?;
    Ok(())
}

/// Which trap slot `rip` is (a stack walk has left guest code), if any.
fn trap_slot(g: &Guest, rip: u64) -> Option<u32> {
    (rip >= crate::TRAP_BASE && rip < g.cpu.trap_address(crate::TRAP_SLOTS)).then(|| ((rip - crate::TRAP_BASE) / xcl_emu::cpu::TRAP_STRIDE) as u32)
}

/// Offer the exception to the handlers of the frames from `original` (a CONTEXT) outwards.
/// Returns when no frame handled it.
fn dispatch(g: &mut Guest, record: u64, original: u64) -> Result<(), Stop> {
    let depth = g.win.call_depth;
    let mut c = read_context(g, original)?;
    let work = g.heap_alloc(CONTEXT_SIZE)?;
    let dispatcher = g.heap_alloc(0x60)?;
    loop {
        if trap_slot(g, c.rip).is_some() || !g.mem.is_mapped(c.rip, 1) {
            return Ok(());
        }
        let f = step(g, UNW_FLAG_EHANDLER, &mut c)?;
        if f.handler == 0 {
            continue;
        }
        write_context(g, work, &c)?;
        write_dispatcher(g, dispatcher, &f, 0, work)?;
        let rsp = g.cpu.r[RSP];
        g.thread_mut().seh.push(Boundary::Dispatch { original, depth, rsp });
        let disposition = g.call(f.handler, &[record, f.frame, original, dispatcher])?;
        g.thread_mut().seh.pop();
        // ExceptionContinueExecution
        if disposition as u32 == 0 {
            let c = read_context(g, original)?;
            apply(g, &c);
            return Err(Stop::Resume { depth });
        }
    }
}

/// RaiseException(code, flags, argument count, arguments)
fn RaiseException(g: &mut Guest) -> H {
    let (code, flags, count, arguments) = (g.arg(0)? as u32, g.arg(1)? as u32, g.arg(2)?.min(15), g.arg(3)?);
    let original = capture(g)?;
    let record = g.heap_alloc(EXCEPTION_RECORD_SIZE)?;
    g.mem.write_u32(record, code)?;
    g.mem.write_u32(record + 4, flags & EXCEPTION_NONCONTINUABLE)?;
    g.mem.write_u64(record + 16, original.rip)?;
    g.mem.write_u32(record + 24, if arguments == 0 { 0 } else { count as u32 })?;
    for i in 0..count {
        if arguments != 0 {
            let v = g.mem.read_u64(arguments + 8 * i)?;
            g.mem.write_u64(record + 32 + 8 * i, v)?;
        }
    }
    let at = g.heap_alloc(CONTEXT_SIZE)?;
    write_context(g, at, &original)?;
    dispatch(g, record, at)?;
    g.log.push(format!("exception {code:#x} raised at {:#x}: no handler took it", original.rip));
    Ok(0)
}

/// RtlUnwindEx(target frame, target ip, exception record, return value, context, history):
/// run the unwind handlers of the frames up to the target frame and continue there.
fn RtlUnwindEx(g: &mut Guest) -> H {
    let (target_frame, target_ip, mut record, value, mut context_at) = (g.arg(0)?, g.arg(1)?, g.arg(2)?, g.arg(3)?, g.arg(4)?);
    let mut context = capture(g)?;
    if record == 0 {
        record = g.heap_alloc(EXCEPTION_RECORD_SIZE)?;
        g.mem.write_u32(record, STATUS_UNWIND)?;
        g.mem.write_u64(record + 16, context.rip)?;
    }
    if context_at == 0 {
        context_at = g.heap_alloc(CONTEXT_SIZE)?;
    }
    let flags = g.mem.read_u32(record + 4)? | EXCEPTION_UNWINDING | if target_frame == 0 { EXCEPTION_EXIT_UNWIND } else { 0 };
    g.mem.write_u32(record + 4, flags)?;
    let dispatcher = g.heap_alloc(0x60)?;
    let mut caller = context.clone();
    // where the thread goes on: this level, unless the walk leaves handler calls of outer ones
    let mut depth = g.win.call_depth;
    loop {
        match trap_slot(g, caller.rip) {
            Some(SLOT_CALL_RETURN) => match g.thread_mut().seh.pop() {
                Some(Boundary::Dispatch { original, depth: d, .. }) => {
                    caller = read_context(g, original)?;
                    context = caller.clone();
                    depth = d;
                    continue;
                }
                Some(Boundary::Unwind { frame_context, caller_context, frame, depth: d, .. }) => {
                    // collided unwind: that frame's handler is running, go on behind it
                    context = *frame_context;
                    caller = *caller_context;
                    depth = d;
                    if frame == target_frame {
                        break;
                    }
                    context = caller.clone();
                    continue;
                }
                None => return Err(Stop::Cpu(xcl_emu::cpu::Exit::Fault(Fault { addr: caller.rip, write: false }))),
            },
            // the end of the stack: nothing to continue with
            Some(_) => return Err(Stop::Cpu(xcl_emu::cpu::Exit::Fault(Fault { addr: caller.rip, write: false }))),
            None => {}
        }
        let f = step(g, UNW_FLAG_UHANDLER, &mut caller)?;
        if f.handler != 0 {
            if f.frame == target_frame {
                let flags = g.mem.read_u32(record + 4)?;
                g.mem.write_u32(record + 4, flags | EXCEPTION_TARGET_UNWIND)?;
            }
            write_context(g, context_at, &context)?;
            write_dispatcher(g, dispatcher, &f, target_ip, context_at)?;
            let rsp = g.cpu.r[RSP];
            g.thread_mut().seh.push(Boundary::Unwind { frame_context: Box::new(context.clone()), caller_context: Box::new(caller.clone()), frame: f.frame, depth, rsp });
            g.call(f.handler, &[record, f.frame, context_at, dispatcher])?;
            g.thread_mut().seh.pop();
            // the handler may have set registers for the landing pad
            context = read_context(g, context_at)?;
        }
        if f.frame == target_frame {
            break;
        }
        context = caller.clone();
    }
    context.r[RAX] = value;
    context.rip = target_ip;
    // handler calls below the frame the thread continues in are gone
    let sp = context.r[RSP];
    g.thread_mut().seh.retain(|b| b.rsp() >= sp);
    apply(g, &context);
    Err(Stop::Resume { depth })
}

fn RtlCaptureContext(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    let c = capture(g)?;
    write_context(g, at, &c)?;
    Ok(0)
}

fn RtlRestoreContext(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    let c = read_context(g, at)?;
    apply(g, &c);
    Err(Stop::Resume { depth: g.win.call_depth })
}

/// RtlLookupFunctionEntry(pc, image base out, history)
fn RtlLookupFunctionEntry(g: &mut Guest) -> H {
    let (pc, base_out) = (g.arg(0)?, g.arg(1)?);
    let found = lookup(g, pc);
    if base_out != 0 {
        g.mem.write_u64(base_out, found.map_or(0, |f| f.0))?;
    }
    Ok(found.map_or(0, |f| f.1))
}

/// RtlVirtualUnwind(handler type, image base, pc, function entry, context, handler data out,
/// establisher frame out, context pointers)
fn RtlVirtualUnwind(g: &mut Guest) -> H {
    let (kind, image_base, pc, entry, context_at, data_out, frame_out) = (g.arg(0)? as u8, g.arg(1)?, g.arg(2)?, g.arg(3)?, g.arg(4)?, g.arg(5)?, g.arg(6)?);
    let mut c = read_context(g, context_at)?;
    let (handler, data, frame) = virtual_unwind(g, kind, image_base, pc, entry, &mut c)?;
    write_context(g, context_at, &c)?;
    if data_out != 0 {
        g.mem.write_u64(data_out, data)?;
    }
    if frame_out != 0 {
        g.mem.write_u64(frame_out, frame)?;
    }
    Ok(handler)
}

/// RtlPcToFileHeader(pc, base out): the module an address is in.
fn RtlPcToFileHeader(g: &mut Guest) -> H {
    let (pc, out) = (g.arg(0)?, g.arg(1)?);
    let base = g.modules.iter().find(|m| pc >= m.base && pc < m.base + m.size).map_or(0, |m| m.base);
    if out != 0 {
        g.mem.write_u64(out, base)?;
    }
    Ok(base)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; RaiseException, RtlUnwindEx, RtlCaptureContext, RtlRestoreContext, RtlLookupFunctionEntry, RtlVirtualUnwind, RtlPcToFileHeader);
}
