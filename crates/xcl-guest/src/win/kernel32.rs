//! kernel32 / ntdll functions: the process, memory, time, handles and the console.

#![allow(non_snake_case)]

use std::collections::HashMap;

use xcl_emu::mem::{PROT_R, PROT_W, PROT_X};

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

fn ok(_: &mut Guest) -> H {
    Ok(0)
}

fn yes(_: &mut Guest) -> H {
    Ok(1)
}

// --- process and thread ---------------------------------------------------------------------

fn ExitProcess(g: &mut Guest) -> H {
    let code = g.arg(0)? as u32;
    Err(g.exit(code))
}

fn TerminateProcess(g: &mut Guest) -> H {
    let code = g.arg(1)? as u32;
    Err(g.exit(code))
}

fn GetCurrentProcess(_: &mut Guest) -> H {
    Ok(u64::MAX)
}

fn GetCurrentThread(_: &mut Guest) -> H {
    Ok(u64::MAX - 1)
}

fn GetCurrentProcessId(_: &mut Guest) -> H {
    Ok(0x1000)
}

fn GetCurrentThreadId(g: &mut Guest) -> H {
    Ok(g.thread().id as u64)
}

fn GetLastError(g: &mut Guest) -> H {
    Ok(g.thread().last_error as u64)
}

fn SetLastError(g: &mut Guest) -> H {
    let code = g.arg(0)? as u32;
    g.set_last_error(code);
    Ok(0)
}

fn GetCommandLineW(g: &mut Guest) -> H {
    g.static_block("command line", |g| {
        let line = g.command_line.clone();
        g.alloc_wstr(&line)
    })
}

fn GetEnvironmentStringsW(g: &mut Guest) -> H {
    g.heap_alloc(8)
}

fn GetEnvironmentVariableW(g: &mut Guest) -> H {
    g.set_last_error(ERROR_ENVVAR_NOT_FOUND);
    Ok(0)
}

// --- thread-local and fiber-local storage -----------------------------------------------------

fn TlsAlloc(g: &mut Guest) -> H {
    g.win.tls_count += 1;
    Ok(g.win.tls_count as u64 - 1)
}

fn TlsGetValue(g: &mut Guest) -> H {
    let i = g.arg(0)? as usize;
    g.set_last_error(0);
    Ok(g.thread().tls_slots.get(i).copied().unwrap_or(0))
}

fn TlsSetValue(g: &mut Guest) -> H {
    let (i, v) = (g.arg(0)? as usize, g.arg(1)?);
    if i >= g.win.tls_count {
        return Ok(0);
    }
    let slots = &mut g.thread_mut().tls_slots;
    if slots.len() <= i {
        slots.resize(i + 1, 0);
    }
    slots[i] = v;
    Ok(1)
}

// --- threads and waiting --------------------------------------------------------------------

const INFINITE: u32 = u32::MAX;

/// CreateThread(security, stack size, start, parameter, flags, id out)
fn CreateThread(g: &mut Guest) -> H {
    let (stack_size, start, parameter, id_out) = (g.arg(1)?, g.arg(2)?, g.arg(3)?, g.arg(5)?);
    let size = xcl_emu::mem::page_up(if stack_size == 0 { 1 << 20 } else { stack_size.max(0x1_0000) });
    let Ok(stack) = g.mem.alloc(size, PROT_R | PROT_W) else { return Ok(0) };
    let index = g.add_thread(stack, stack + size, start, parameter, crate::SLOT_THREAD_RETURN)?;
    if id_out != 0 {
        g.mem.write_u32(id_out, g.threads[index].id)?;
    }
    Ok(g.win.new_handle(Object::Thread(index)))
}

/// The wait of the running thread: `ready` says whether what it waits for has happened. `Ok`:
/// it has (true) or the timeout passed (false); `Err(Stop::Wait)`: ask again at the next turn.
fn wait(g: &mut Guest, ready: bool, timeout_ms: u32) -> Result<bool, Stop> {
    let now = g.now_ns();
    if ready {
        g.thread_mut().wait = None;
        return Ok(true);
    }
    match &g.thread().wait {
        None if timeout_ms == 0 => Ok(false),
        None => {
            let deadline = (timeout_ms != INFINITE).then(|| now + timeout_ms as u64 * 1_000_000);
            g.thread_mut().wait = Some(crate::Wait { deadline, address: 0, woken: false });
            Err(Stop::Wait)
        }
        Some(w) if w.deadline.is_some_and(|d| now >= d) => {
            g.thread_mut().wait = None;
            Ok(false)
        }
        Some(_) => Err(Stop::Wait),
    }
}

fn Sleep(g: &mut Guest) -> H {
    let ms = g.arg(0)? as u32;
    if g.thread().wait.is_none() && ms == 0 {
        // give the other threads a turn, once
        g.thread_mut().wait = Some(crate::Wait { deadline: Some(g.now_ns()), address: 0, woken: false });
        return Err(Stop::Wait);
    }
    wait(g, false, ms)?;
    Ok(0)
}

fn SwitchToThread(g: &mut Guest) -> H {
    if g.thread().wait.is_none() {
        g.thread_mut().wait = Some(crate::Wait { deadline: Some(g.now_ns()), address: 0, woken: false });
        return Err(Stop::Wait);
    }
    g.thread_mut().wait = None;
    Ok(1)
}

/// Whether the object behind `handle` is signaled; an auto-reset one is taken by this.
fn signaled(g: &mut Guest, handle: u64) -> Option<bool> {
    let now = g.now_ns();
    Some(match g.win.handles.get_mut(&handle)? {
        Object::Thread(i) => {
            let i = *i;
            g.threads[i].exit_code.is_some()
        }
        Object::Event { signaled, manual } => {
            let was = *signaled;
            if !*manual {
                *signaled = false;
            }
            was
        }
        Object::Timer { due, manual } => {
            let fired = due.is_some_and(|d| now >= d);
            if fired && !*manual {
                *due = None;
            }
            fired
        }
    })
}

/// Guest time at which the object behind `handle` becomes signaled by itself (a timer).
fn due(g: &Guest, handle: u64) -> Option<u64> {
    match g.win.handles.get(&handle)? {
        Object::Timer { due, .. } => *due,
        _ => None,
    }
}

fn WaitForSingleObject(g: &mut Guest) -> H {
    let (handle, ms) = (g.arg(0)?, g.arg(1)? as u32);
    let Some(ready) = signaled(g, handle) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(u32::MAX as u64);
    };
    let waited = wait(g, ready, ms);
    if matches!(waited, Err(Stop::Wait)) {
        // a timer ends the wait by itself
        if let (Some(d), Some(w)) = (due(g, handle), g.thread_mut().wait.as_mut()) {
            w.deadline = Some(w.deadline.map_or(d, |own| own.min(d)));
        }
    }
    Ok(if waited? { 0 } else { WAIT_TIMEOUT })
}

fn CreateEventW(g: &mut Guest) -> H {
    let (manual, initial) = (g.arg(1)? as u32 != 0, g.arg(2)? as u32 != 0);
    Ok(g.win.new_handle(Object::Event { signaled: initial, manual }))
}

fn set_event(g: &mut Guest, to: bool) -> H {
    let handle = g.arg(0)?;
    match g.win.handles.get_mut(&handle) {
        Some(Object::Event { signaled, .. }) => {
            *signaled = to;
            Ok(1)
        }
        _ => {
            g.set_last_error(ERROR_INVALID_HANDLE);
            Ok(0)
        }
    }
}

fn SetEvent(g: &mut Guest) -> H {
    set_event(g, true)
}

fn ResetEvent(g: &mut Guest) -> H {
    set_event(g, false)
}

/// CreateWaitableTimerExW(security, name, flags, access); flag 1: manual reset
fn CreateWaitableTimerExW(g: &mut Guest) -> H {
    let manual = g.arg(2)? & 1 != 0;
    Ok(g.win.new_handle(Object::Timer { due: None, manual }))
}

/// SetWaitableTimer(timer, due time *, period, …): a negative due time is relative, in 100 ns.
fn SetWaitableTimer(g: &mut Guest) -> H {
    let (handle, due_at) = (g.arg(0)?, g.arg(1)?);
    let t = g.mem.read_u64(due_at)? as i64;
    let now = g.now_ns();
    // an absolute time is a FILETIME: taken as "now" (the guest clock has no calendar)
    let at = if t < 0 { now + t.unsigned_abs() * 100 } else { now };
    match g.win.handles.get_mut(&handle) {
        Some(Object::Timer { due, .. }) => {
            *due = Some(at);
            Ok(1)
        }
        _ => {
            g.set_last_error(ERROR_INVALID_HANDLE);
            Ok(0)
        }
    }
}

// CRITICAL_SECTION in guest memory: RecursionCount at +12, OwningThread at +16.

fn InitializeCriticalSection(g: &mut Guest) -> H {
    let cs = g.arg(0)?;
    g.mem.write(cs, &[0u8; 40])?;
    Ok(1)
}

fn enter_critical_section(g: &mut Guest, block: bool) -> H {
    let cs = g.arg(0)?;
    let me = g.thread().id as u64;
    let owner = g.mem.read_u64(cs + 16)?;
    if owner != 0 && owner != me {
        if !block {
            return Ok(0);
        }
        // asked again at this thread's next turn
        g.thread_mut().wait = Some(crate::Wait::default());
        return Err(Stop::Wait);
    }
    g.thread_mut().wait = None;
    let count = g.mem.read_u32(cs + 12)?;
    g.mem.write_u32(cs + 12, count + 1)?;
    g.mem.write_u64(cs + 16, me)?;
    Ok(1)
}

fn EnterCriticalSection(g: &mut Guest) -> H {
    enter_critical_section(g, true)
}

fn TryEnterCriticalSection(g: &mut Guest) -> H {
    enter_critical_section(g, false)
}

fn LeaveCriticalSection(g: &mut Guest) -> H {
    let cs = g.arg(0)?;
    let count = g.mem.read_u32(cs + 12)?.saturating_sub(1);
    g.mem.write_u32(cs + 12, count)?;
    if count == 0 {
        g.mem.write_u64(cs + 16, 0)?;
    }
    Ok(0)
}

fn CloseHandle(g: &mut Guest) -> H {
    let handle = g.arg(0)?;
    g.win.handles.remove(&handle);
    Ok(1)
}

/// WaitOnAddress(address, compare address, size, timeout): returns once the value differs from
/// the one compared and the thread was woken, or at the timeout.
fn WaitOnAddress(g: &mut Guest) -> H {
    let (address, compare, size, ms) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u8, g.arg(3)? as u32);
    let woken = g.thread().wait.as_ref().is_some_and(|w| w.woken);
    let differs = g.mem.read_sized(address, size)? != g.mem.read_sized(compare, size)?;
    match wait(g, woken || (differs && g.thread().wait.is_none()), ms) {
        Ok(true) => Ok(1),
        Ok(false) => {
            g.set_last_error(ERROR_TIMEOUT);
            Ok(0)
        }
        Err(stop) => {
            if let Some(w) = g.thread_mut().wait.as_mut() {
                w.address = address;
            }
            Err(stop)
        }
    }
}

fn wake(g: &mut Guest, all: bool) -> H {
    let address = g.arg(0)?;
    for t in g.threads.iter_mut().filter(|t| t.exit_code.is_none()) {
        if let Some(w) = t.wait.as_mut().filter(|w| w.address == address && !w.woken) {
            w.woken = true;
            if !all {
                break;
            }
        }
    }
    Ok(0)
}

fn WakeByAddressSingle(g: &mut Guest) -> H {
    wake(g, false)
}

fn WakeByAddressAll(g: &mut Guest) -> H {
    wake(g, true)
}

// --- memory -------------------------------------------------------------------------------

fn GetProcessHeap(_: &mut Guest) -> H {
    Ok(0x0000_0001_0000_0000)
}

fn HeapAlloc(g: &mut Guest) -> H {
    let size = g.arg(2)?;
    Ok(g.win.heap.alloc(&mut g.mem, size).unwrap_or(0))
}

fn HeapReAlloc(g: &mut Guest) -> H {
    let (at, size) = (g.arg(2)?, g.arg(3)?);
    Ok(g.win.heap.realloc(&mut g.mem, at, size).unwrap_or(0))
}

fn HeapFree(g: &mut Guest) -> H {
    let at = g.arg(2)?;
    Ok((at == 0 || g.win.heap.free(&mut g.mem, at)) as u64)
}

const PAGE_NOACCESS: u32 = 0x01;
const PAGE_READONLY: u32 = 0x02;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_EXECUTE_READ: u32 = 0x20;
const PAGE_EXECUTE_READWRITE: u32 = 0x40;
const MEM_COMMIT: u32 = 0x1000;
const MEM_RELEASE: u32 = 0x8000;
const MEM_FREE: u32 = 0x1_0000;
const MEM_PRIVATE: u32 = 0x2_0000;

fn page_protect(prot: u8) -> u32 {
    match (prot & PROT_X != 0, prot & PROT_W != 0, prot & PROT_R != 0) {
        (true, true, _) => PAGE_EXECUTE_READWRITE,
        (true, false, _) => PAGE_EXECUTE_READ,
        (false, true, _) => PAGE_READWRITE,
        (false, false, true) => PAGE_READONLY,
        _ => PAGE_NOACCESS,
    }
}

fn prot_of(page: u32) -> u8 {
    match page & 0xFF {
        PAGE_NOACCESS => 0,
        PAGE_READONLY => PROT_R,
        PAGE_READWRITE | 0x08 => PROT_R | PROT_W,
        0x10 | PAGE_EXECUTE_READ => PROT_R | PROT_X,
        _ => PROT_R | PROT_W | PROT_X,
    }
}

/// VirtualAlloc(address, size, type, protect): reserving and committing are one thing here.
pub(super) fn virtual_alloc(g: &mut Guest, address: u64, size: u64, protect: u32) -> u64 {
    let size = xcl_emu::mem::page_up(size.max(1));
    if address == 0 {
        return g.mem.alloc(size, prot_of(protect)).unwrap_or(0);
    }
    let base = address & !0xFFF;
    let size = xcl_emu::mem::page_up(address + size - base);
    if g.mem.is_mapped(base, size) {
        // committing pages of a reservation made earlier
        return base;
    }
    match g.mem.map(base, size, prot_of(protect)) {
        Ok(()) => base,
        Err(_) => 0,
    }
}

fn VirtualAlloc(g: &mut Guest) -> H {
    let (address, size, protect) = (g.arg(0)?, g.arg(1)?, g.arg(3)? as u32);
    Ok(virtual_alloc(g, address, size, protect))
}

fn VirtualFree(g: &mut Guest) -> H {
    let (address, kind) = (g.arg(0)?, g.arg(2)? as u32);
    if kind & MEM_RELEASE != 0 {
        return Ok(g.mem.unmap(address) as u64);
    }
    // decommit: the pages stay mapped
    Ok(1)
}

fn VirtualProtect(g: &mut Guest) -> H {
    let (address, protect, old) = (g.arg(0)?, g.arg(2)? as u32, g.arg(3)?);
    let Some(prot) = g.mem.regions().iter().find(|r| address >= r.base && address < r.end()).map(|r| r.prot) else { return Ok(0) };
    let before = page_protect(prot);
    if old != 0 {
        g.mem.write_u32(old, before)?;
    }
    // protection is per region here: never take access away from the rest of it
    let merged = prot | prot_of(protect);
    g.mem.protect(address, merged);
    Ok(1)
}

/// MEMORY_BASIC_INFORMATION (48 bytes) of the region around an address.
fn VirtualQuery(g: &mut Guest) -> H {
    let (address, info, len) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    if len < 48 {
        return Ok(0);
    }
    let page = address & !0xFFF;
    let regions = g.mem.regions();
    let (base, alloc_base, size, state, protect) = match regions.iter().find(|r| address >= r.base && address < r.end()) {
        Some(r) => (page, r.base, r.end() - page, MEM_COMMIT, page_protect(r.prot)),
        None => {
            let next = regions.iter().map(|r| r.base).find(|&b| b > address).unwrap_or(0x0000_7FFF_FFFF_0000);
            (page, 0, next.saturating_sub(page), MEM_FREE, PAGE_NOACCESS)
        }
    };
    g.mem.write_u64(info, base)?;
    g.mem.write_u64(info + 8, alloc_base)?;
    g.mem.write_u64(info + 16, if state == MEM_FREE { 0 } else { protect } as u64)?;
    g.mem.write_u64(info + 24, size)?;
    g.mem.write_u32(info + 32, state)?;
    g.mem.write_u32(info + 36, protect)?;
    g.mem.write_u64(info + 40, if state == MEM_FREE { 0 } else { MEM_PRIVATE } as u64)?;
    Ok(48)
}

/// SYSTEM_INFO: an x64 machine with one processor (the guest has one thread so far).
fn GetSystemInfo(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    g.mem.write(at, &[0u8; 48])?;
    g.mem.write_u16(at, 9)?; // PROCESSOR_ARCHITECTURE_AMD64
    g.mem.write_u32(at + 4, 0x1000)?; // dwPageSize
    g.mem.write_u64(at + 8, 0x1_0000)?; // lpMinimumApplicationAddress
    g.mem.write_u64(at + 16, 0x0000_7FFF_FFFE_FFFF)?;
    g.mem.write_u64(at + 24, 1)?; // dwActiveProcessorMask
    g.mem.write_u32(at + 32, 1)?; // dwNumberOfProcessors
    g.mem.write_u32(at + 36, 8664)?; // dwProcessorType
    g.mem.write_u32(at + 40, 0x1_0000)?; // dwAllocationGranularity
    g.mem.write_u16(at + 44, 0x17)?; // wProcessorLevel: family 17h (Zen 2)
    g.mem.write_u16(at + 46, 0x7100)?;
    Ok(0)
}

// --- time -----------------------------------------------------------------------------------

fn QueryPerformanceCounter(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    // 100 ns units, as QueryPerformanceFrequency says
    g.mem.write_u64(at, g.now_ns() / 100)?;
    Ok(1)
}

fn QueryPerformanceFrequency(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    g.mem.write_u64(at, 10_000_000)?;
    Ok(1)
}

/// FILETIME: guest time counted from 2026-01-01 (no host clock yet).
fn GetSystemTimePreciseAsFileTime(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    const JAN_2026: u64 = 134_116_992_000_000_000;
    g.mem.write_u64(at, JAN_2026 + g.now_ns() / 100)?;
    Ok(0)
}

// --- handles and the console ------------------------------------------------------------------

fn GetStdHandle(g: &mut Guest) -> H {
    Ok(match g.arg(0)? as u32 as i32 {
        -10 => STDIN,
        -11 => STDOUT,
        -12 => STDERR,
        _ => {
            g.set_last_error(ERROR_INVALID_HANDLE);
            u64::MAX
        }
    })
}

/// GetConsoleMode: the standard handles are not consoles (the title then writes bytes).
fn GetConsoleMode(g: &mut Guest) -> H {
    g.set_last_error(ERROR_INVALID_HANDLE);
    Ok(0)
}

fn WriteConsoleW(g: &mut Guest) -> H {
    let (at, units, written) = (g.arg(1)?, g.arg(2)?, g.arg(3)?);
    let mut text = Vec::with_capacity(units as usize);
    for i in 0..units {
        text.push(g.mem.read_u16(at + 2 * i)?);
    }
    g.write_std(String::from_utf16_lossy(&text).as_bytes());
    if written != 0 {
        g.mem.write_u32(written, units as u32)?;
    }
    Ok(1)
}

fn WriteFile(g: &mut Guest) -> H {
    let (handle, at, len, written) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32, g.arg(3)?);
    if handle != STDOUT && handle != STDERR {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    }
    let mut bytes = vec![0u8; len as usize];
    g.mem.read(at, &mut bytes)?;
    g.write_std(&bytes);
    if written != 0 {
        g.mem.write_u32(written, len)?;
    }
    Ok(1)
}

/// NtWriteFile(handle, event, apc, apc context, io status block, buffer, length, offset, key)
fn NtWriteFile(g: &mut Guest) -> H {
    let (handle, iosb, at, len) = (g.arg(0)?, g.arg(4)?, g.arg(5)?, g.arg(6)? as u32);
    if handle != STDOUT && handle != STDERR {
        return Ok(0xC000_0008); // STATUS_INVALID_HANDLE
    }
    let mut bytes = vec![0u8; len as usize];
    g.mem.read(at, &mut bytes)?;
    g.write_std(&bytes);
    if iosb != 0 {
        g.mem.write_u64(iosb, 0)?;
        g.mem.write_u64(iosb + 8, len as u64)?;
    }
    Ok(0)
}

/// NtQueryVolumeInformationFile(handle, io status block, buffer, length, class): the standard
/// handles are pipes (FileFsDeviceInformation), so a title writes them bytes, not console text.
fn NtQueryVolumeInformationFile(g: &mut Guest) -> H {
    let (handle, iosb, at, len, class) = (g.arg(0)?, g.arg(1)?, g.arg(2)?, g.arg(3)?, g.arg(4)? as u32);
    if !matches!(handle, STDIN | STDOUT | STDERR) {
        return Ok(0xC000_0008);
    }
    if class != 4 || len < 8 {
        return Ok(0xC000_0003); // STATUS_INVALID_INFO_CLASS
    }
    g.mem.write_u32(at, 0x11)?; // FILE_DEVICE_NAMED_PIPE
    g.mem.write_u32(at + 4, 0)?;
    if iosb != 0 {
        g.mem.write_u64(iosb, 0)?;
        g.mem.write_u64(iosb + 8, 8)?;
    }
    Ok(0)
}

fn RtlNtStatusToDosError(g: &mut Guest) -> H {
    Ok(match g.arg(0)? as u32 {
        0 => 0,
        0xC000_0008 => ERROR_INVALID_HANDLE as u64,
        _ => 317, // ERROR_MR_MID_NOT_FOUND
    })
}

// --- modules --------------------------------------------------------------------------------

/// The base of the module named `name` (the executable for an empty name), or 0.
fn module_base(g: &Guest, name: &str) -> u64 {
    let file = name.rsplit(['\\', '/']).next().unwrap_or(name).to_ascii_lowercase();
    if file.is_empty() {
        return g.modules.first().map_or(0, |m| m.base);
    }
    let with_ext = if file.contains('.') { file.clone() } else { format!("{file}.dll") };
    g.modules.iter().find(|m| m.name.eq_ignore_ascii_case(&with_ext)).map_or(0, |m| m.base)
}

/// A handle for a system DLL the host stands in for: no image, its functions are host functions.
const SYSTEM_MODULE: u64 = 0x0000_7FFD_0000_0000;

fn module_handle(g: &mut Guest, name: Option<String>) -> u64 {
    let Some(name) = name else { return module_base(g, "") };
    let base = module_base(g, &name);
    if base != 0 {
        return base;
    }
    let file = name.rsplit(['\\', '/']).next().unwrap_or(&name).to_ascii_lowercase();
    let known = ["kernel32", "kernelbase", "ntdll", "kernelx", "api-ms-win-", "ext-ms-win-", "xgameruntime", "ucrtbase", "msvcrt"];
    if known.iter().any(|k| file.starts_with(k)) {
        return SYSTEM_MODULE;
    }
    g.set_last_error(ERROR_MOD_NOT_FOUND);
    0
}

fn GetModuleHandleW(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    let name = if at == 0 { None } else { Some(g.mem.wstr(at, 520)?) };
    Ok(module_handle(g, name))
}

fn GetModuleHandleA(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    let name = if at == 0 { None } else { Some(g.mem.cstr(at, 520)?) };
    Ok(module_handle(g, name))
}

/// GetModuleHandleExW(flags, name or address, module out); flag 4: by an address inside it
fn GetModuleHandleExW(g: &mut Guest) -> H {
    let (flags, what, out) = (g.arg(0)? as u32, g.arg(1)?, g.arg(2)?);
    let base = if flags & 4 != 0 {
        g.modules.iter().find(|m| what >= m.base && what < m.base + m.size).map_or(0, |m| m.base)
    } else {
        let name = if what == 0 { None } else { Some(g.mem.wstr(what, 520)?) };
        module_handle(g, name)
    };
    if base == 0 {
        g.set_last_error(ERROR_MOD_NOT_FOUND);
    }
    if out != 0 {
        g.mem.write_u64(out, base)?;
    }
    Ok((base != 0) as u64)
}

fn LoadLibraryW(g: &mut Guest) -> H {
    GetModuleHandleW(g)
}

fn LoadLibraryExA(g: &mut Guest) -> H {
    GetModuleHandleA(g)
}

/// GetProcAddress: a host function when the host has one by that name, else not found (a title
/// probing for a newer API then takes its fallback).
fn GetProcAddress(g: &mut Guest) -> H {
    let (module, at) = (g.arg(0)?, g.arg(1)?);
    if at < 0x1_0000 {
        g.set_last_error(ERROR_PROC_NOT_FOUND);
        return Ok(0);
    }
    let name = g.mem.cstr(at, 256)?;
    if module == SYSTEM_MODULE && g.has_handler(&name) {
        return Ok(g.import_address("@GetProcAddress", &name));
    }
    g.set_last_error(ERROR_PROC_NOT_FOUND);
    Ok(0)
}

fn GetModuleFileNameW(g: &mut Guest) -> H {
    let (at, size) = (g.arg(1)?, g.arg(2)?);
    let path: Vec<u16> = format!("G:\\{}", g.modules.first().map_or("title.exe", |m| m.name.as_str())).encode_utf16().collect();
    if (path.len() as u64) >= size {
        g.set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return Ok(size);
    }
    for (i, &c) in path.iter().chain(&[0]).enumerate() {
        g.mem.write_u16(at + 2 * i as u64, c)?;
    }
    Ok(path.len() as u64)
}

// --- strings and randomness -------------------------------------------------------------------

fn lstrlenW(g: &mut Guest) -> H {
    let s = g.arg(0)?;
    let mut n = 0;
    while s != 0 && g.mem.read_u16(s + 2 * n)? != 0 {
        n += 1;
    }
    Ok(n)
}

/// ProcessPrng(buffer, length): xorshift, the same sequence every run.
fn ProcessPrng(g: &mut Guest) -> H {
    let (at, len) = (g.arg(0)?, g.arg(1)?);
    for i in 0..len {
        let mut x = g.win.prng ^ 0x9E37_79B9_7F4A_7C15;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        g.win.prng = x;
        g.mem.write_u8(at + i, (x >> 32) as u8)?;
    }
    Ok(1)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; ExitProcess, TerminateProcess, GetCurrentProcess, GetCurrentThread, GetCurrentProcessId, GetCurrentThreadId);
    register!(map; GetLastError, SetLastError, GetCommandLineW, GetEnvironmentStringsW, GetEnvironmentVariableW);
    register!(map; TlsAlloc, TlsGetValue, TlsSetValue, GetProcessHeap, HeapAlloc, HeapReAlloc, HeapFree);
    register!(map; VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery, GetSystemInfo);
    register!(map; QueryPerformanceCounter, QueryPerformanceFrequency, GetSystemTimePreciseAsFileTime, Sleep, SwitchToThread);
    register!(map; CreateThread, WaitForSingleObject, CreateEventW, SetEvent, ResetEvent, CreateWaitableTimerExW, SetWaitableTimer, CloseHandle);
    register!(map; WaitOnAddress, WakeByAddressSingle, WakeByAddressAll);
    register!(map; GetStdHandle, GetConsoleMode, WriteConsoleW, WriteFile, NtWriteFile, NtQueryVolumeInformationFile, RtlNtStatusToDosError);
    register!(map; GetModuleHandleW, GetModuleHandleA, GetModuleHandleExW, LoadLibraryW, LoadLibraryExA, GetProcAddress, GetModuleFileNameW, lstrlenW, ProcessPrng);
    map.insert("FlsAlloc", TlsAlloc as Handler);
    map.insert("FlsGetValue", TlsGetValue as Handler);
    map.insert("FlsSetValue", TlsSetValue as Handler);
    map.insert("SleepEx", Sleep as Handler);
    register!(map; InitializeCriticalSection, EnterCriticalSection, TryEnterCriticalSection, LeaveCriticalSection);
    map.insert("InitializeCriticalSectionEx", InitializeCriticalSection as Handler);
    map.insert("InitializeCriticalSectionAndSpinCount", InitializeCriticalSection as Handler);
    for name in ["DeleteCriticalSection", "SetUnhandledExceptionFilter", "IsThreadAFiber"] {
        map.insert(name, ok as Handler);
    }
    for name in ["AddVectoredExceptionHandler", "SetThreadStackGuarantee", "FreeEnvironmentStringsW", "SetEnvironmentVariableW", "FlsFree", "TlsFree", "FreeLibrary"] {
        map.insert(name, yes as Handler);
    }
}
