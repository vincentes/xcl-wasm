//! What an MSVC-built image calls from its start-up code on: slim reader/writer locks,
//! condition variables, interlocked singly-linked lists, and the small process questions around
//! them. The lock words live in the title's memory and only these functions read them, so their
//! contents are the host's own:
//!
//! - SRWLOCK (8 bytes, 0 when free): bit 0 an exclusive owner, the shared owners counted from bit 4
//! - CONDITION_VARIABLE (8 bytes): not written; a sleeping thread waits on its address
//! - SLIST_HEADER (16 bytes): the depth at +0, the first entry at +8
//!
//! … and the kernel objects next to kernel32's events and timers: semaphores, mutexes, waits
//! for several objects, and threads made suspended.

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::kernel32::{due, leave_cs, signaled, try_enter_cs, wait, wake};
use super::*;
use crate::{Guest, Handler, Stop, Wait};

type H = Result<u64, Stop>;

const SRW_EXCLUSIVE: u64 = 1;
const SRW_SHARED: u64 = 16;

// --- slim reader/writer locks -----------------------------------------------------------------

fn InitializeSRWLock(g: &mut Guest) -> H {
    let lock = g.arg(0)?;
    g.mem.write_u64(lock, 0)?;
    Ok(0)
}

fn try_acquire_srw(g: &mut Guest, lock: u64, shared: bool) -> Result<bool, Stop> {
    let word = g.mem.read_u64(lock)?;
    let free = if shared { word & SRW_EXCLUSIVE == 0 } else { word == 0 };
    if free {
        g.mem.write_u64(lock, if shared { word + SRW_SHARED } else { SRW_EXCLUSIVE })?;
    }
    Ok(free)
}

fn release_srw(g: &mut Guest, lock: u64, shared: bool) -> Result<(), Stop> {
    let word = g.mem.read_u64(lock)?;
    g.mem.write_u64(lock, if shared { word.saturating_sub(SRW_SHARED) } else { word & !SRW_EXCLUSIVE })?;
    Ok(())
}

fn acquire_srw(g: &mut Guest, shared: bool) -> H {
    let lock = g.arg(0)?;
    if !try_acquire_srw(g, lock, shared)? {
        // asked again at this thread's next turn
        g.thread_mut().wait = Some(Wait::default());
        return Err(Stop::Wait);
    }
    g.thread_mut().wait = None;
    Ok(0)
}

fn AcquireSRWLockExclusive(g: &mut Guest) -> H {
    acquire_srw(g, false)
}

fn AcquireSRWLockShared(g: &mut Guest) -> H {
    acquire_srw(g, true)
}

fn TryAcquireSRWLockExclusive(g: &mut Guest) -> H {
    let lock = g.arg(0)?;
    Ok(try_acquire_srw(g, lock, false)? as u64)
}

fn TryAcquireSRWLockShared(g: &mut Guest) -> H {
    let lock = g.arg(0)?;
    Ok(try_acquire_srw(g, lock, true)? as u64)
}

fn ReleaseSRWLockExclusive(g: &mut Guest) -> H {
    let lock = g.arg(0)?;
    release_srw(g, lock, false)?;
    Ok(0)
}

fn ReleaseSRWLockShared(g: &mut Guest) -> H {
    let lock = g.arg(0)?;
    release_srw(g, lock, true)?;
    Ok(0)
}

// --- condition variables ----------------------------------------------------------------------

/// The lock a thread gives up while it sleeps on a condition variable.
#[derive(Clone, Copy)]
enum Lock {
    CriticalSection,
    Srw { shared: bool },
}

/// A sleep has two parts, told apart by the address the thread waits on: the variable's own
/// until it is woken or the time is up, then the variable's with a low bit set (never woken)
/// until it has the lock back: bit 0 after a wake, bit 1 after the timeout.
const REACQUIRE_WOKEN: u64 = 1;
const REACQUIRE_TIMED_OUT: u64 = 2;

fn sleep_condition_variable(g: &mut Guest, lock_kind: Lock, timeout_ms: u32) -> H {
    let (cv, lock) = (g.arg(0)?, g.arg(1)?);
    let now = g.now_ns();
    let outcome = match g.thread().wait.clone() {
        None => {
            match lock_kind {
                Lock::CriticalSection => leave_cs(g, lock)?,
                Lock::Srw { shared } => release_srw(g, lock, shared)?,
            }
            let deadline = (timeout_ms != u32::MAX).then(|| now + timeout_ms as u64 * 1_000_000);
            g.thread_mut().wait = Some(Wait { deadline, address: cv, woken: false });
            return Err(Stop::Wait);
        }
        Some(w) if w.address == cv => {
            if w.woken {
                REACQUIRE_WOKEN
            } else if w.deadline.is_some_and(|d| now >= d) {
                REACQUIRE_TIMED_OUT
            } else {
                return Err(Stop::Wait);
            }
        }
        Some(w) => w.address & 3,
    };
    let taken = match lock_kind {
        Lock::CriticalSection => try_enter_cs(g, lock)?,
        Lock::Srw { shared } => try_acquire_srw(g, lock, shared)?,
    };
    if !taken {
        g.thread_mut().wait = Some(Wait { deadline: None, address: cv | outcome, woken: false });
        return Err(Stop::Wait);
    }
    g.thread_mut().wait = None;
    if outcome == REACQUIRE_TIMED_OUT {
        g.set_last_error(ERROR_TIMEOUT);
        return Ok(0);
    }
    Ok(1)
}

/// SleepConditionVariableCS(variable, critical section, milliseconds)
fn SleepConditionVariableCS(g: &mut Guest) -> H {
    let ms = g.arg(2)? as u32;
    sleep_condition_variable(g, Lock::CriticalSection, ms)
}

/// SleepConditionVariableSRW(variable, lock, milliseconds, flags); flag 1: the lock is held shared
fn SleepConditionVariableSRW(g: &mut Guest) -> H {
    let (ms, flags) = (g.arg(2)? as u32, g.arg(3)?);
    sleep_condition_variable(g, Lock::Srw { shared: flags & 1 != 0 }, ms)
}

fn WakeConditionVariable(g: &mut Guest) -> H {
    wake(g, false)
}

fn WakeAllConditionVariable(g: &mut Guest) -> H {
    wake(g, true)
}

// --- semaphores, mutexes and waits for several objects -------------------------------------------

/// CreateSemaphoreExW(security, initial count, maximum count, name, flags, access); the first
/// three arguments are CreateSemaphoreW's too
fn CreateSemaphoreExW(g: &mut Guest) -> H {
    let (count, max) = (g.arg(1)? as u32, g.arg(2)? as u32);
    if max == 0 || count > max {
        g.set_last_error(ERROR_INVALID_PARAMETER);
        return Ok(0);
    }
    Ok(g.win.new_handle(Object::Semaphore { count, max }))
}

/// ReleaseSemaphore(semaphore, count, previous count out)
fn ReleaseSemaphore(g: &mut Guest) -> H {
    let (handle, by, previous) = (g.arg(0)?, g.arg(1)? as u32, g.arg(2)?);
    let Some(Object::Semaphore { count, max }) = g.win.handles.get_mut(&handle) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    let before = *count;
    if by == 0 || by > *max - before {
        g.set_last_error(ERROR_TOO_MANY_POSTS);
        return Ok(0);
    }
    *count = before + by;
    if previous != 0 {
        g.mem.write_u32(previous, before)?;
    }
    Ok(1)
}

fn create_mutex(g: &mut Guest, owned: bool) -> H {
    let owner = if owned { g.thread().id } else { 0 };
    Ok(g.win.new_handle(Object::Mutex { owner, count: owned as u32 }))
}

/// CreateMutexW(security, initial owner, name)
fn CreateMutexW(g: &mut Guest) -> H {
    let owned = g.arg(1)? as u32 != 0;
    create_mutex(g, owned)
}

/// CreateMutexExW(security, name, flags, access); flag 1: the caller owns it
fn CreateMutexExW(g: &mut Guest) -> H {
    let owned = g.arg(2)? & 1 != 0;
    create_mutex(g, owned)
}

fn ReleaseMutex(g: &mut Guest) -> H {
    let handle = g.arg(0)?;
    let me = g.thread().id;
    match g.win.handles.get_mut(&handle) {
        Some(Object::Mutex { owner, count }) if *owner == me => {
            *count -= 1;
            if *count == 0 {
                *owner = 0;
            }
            Ok(1)
        }
        Some(Object::Mutex { .. }) => {
            g.set_last_error(ERROR_NOT_OWNER);
            Ok(0)
        }
        _ => {
            g.set_last_error(ERROR_INVALID_HANDLE);
            Ok(0)
        }
    }
}

/// CreateEventExW(security, name, flags, access); flag 1: manual reset, flag 2: set
fn CreateEventExW(g: &mut Guest) -> H {
    let flags = g.arg(2)?;
    Ok(g.win.new_handle(Object::Event { signaled: flags & 2 != 0, manual: flags & 1 != 0 }))
}

const MAXIMUM_WAIT_OBJECTS: u64 = 64;
const WAIT_FAILED: u64 = 0xFFFF_FFFF;

/// WaitForMultipleObjects(count, handles, wait for all, milliseconds); …Ex takes the same four
/// first. The index of the object that ended the wait (0 for all of them), or WAIT_TIMEOUT.
fn WaitForMultipleObjects(g: &mut Guest) -> H {
    let (count, at, all, ms) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32 != 0, g.arg(3)? as u32);
    if count == 0 || count > MAXIMUM_WAIT_OBJECTS {
        g.set_last_error(ERROR_INVALID_PARAMETER);
        return Ok(WAIT_FAILED);
    }
    let mut handles = Vec::with_capacity(count as usize);
    for i in 0..count {
        handles.push(g.mem.read_u64(at + 8 * i)?);
    }
    let mut ready = Vec::with_capacity(handles.len());
    for &h in &handles {
        let Some(is) = signaled(g, h, false) else {
            g.set_last_error(ERROR_INVALID_HANDLE);
            return Ok(WAIT_FAILED);
        };
        ready.push(is);
    }
    // nothing is taken before the wait can end: all of them, or the first that is signaled
    let ends = if all { ready.iter().all(|&r| r).then_some(0) } else { ready.iter().position(|&r| r) };
    if let Some(first) = ends {
        for &h in if all { &handles[..] } else { &handles[first..first + 1] } {
            signaled(g, h, true);
        }
    }
    let waited = wait(g, ends.is_some(), ms);
    if matches!(waited, Err(Stop::Wait)) {
        // a timer among them ends the wait by itself
        let timers = handles.iter().filter_map(|&h| due(g, h));
        if let Some(d) = if all { timers.max() } else { timers.min() } {
            if let Some(w) = g.thread_mut().wait.as_mut() {
                w.deadline = Some(w.deadline.map_or(d, |own| own.min(d)));
            }
        }
    }
    Ok(if waited? { ends.unwrap_or(0) as u64 } else { WAIT_TIMEOUT })
}

// --- threads --------------------------------------------------------------------------------

/// ResumeThread(thread): how often it was suspended before (1 for a thread made suspended).
fn ResumeThread(g: &mut Guest) -> H {
    let handle = g.arg(0)?;
    let Some(&Object::Thread(index)) = g.win.handles.get(&handle) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(u32::MAX as u64);
    };
    Ok(std::mem::take(&mut g.threads[index].suspended) as u64)
}

// --- interlocked singly-linked lists ------------------------------------------------------------

fn InitializeSListHead(g: &mut Guest) -> H {
    let head = g.arg(0)?;
    g.mem.write(head, &[0u8; 16])?;
    Ok(0)
}

/// InterlockedPushEntrySList(head, entry): the entry that was first before.
fn InterlockedPushEntrySList(g: &mut Guest) -> H {
    let (head, entry) = (g.arg(0)?, g.arg(1)?);
    let first = g.mem.read_u64(head + 8)?;
    g.mem.write_u64(entry, first)?;
    g.mem.write_u64(head + 8, entry)?;
    let depth = g.mem.read_u16(head)?;
    g.mem.write_u16(head, depth.wrapping_add(1))?;
    Ok(first)
}

fn InterlockedPopEntrySList(g: &mut Guest) -> H {
    let head = g.arg(0)?;
    let first = g.mem.read_u64(head + 8)?;
    if first != 0 {
        let next = g.mem.read_u64(first)?;
        g.mem.write_u64(head + 8, next)?;
        let depth = g.mem.read_u16(head)?;
        g.mem.write_u16(head, depth.wrapping_sub(1))?;
    }
    Ok(first)
}

/// InterlockedFlushSList(head): the whole list, which the head no longer has.
fn InterlockedFlushSList(g: &mut Guest) -> H {
    let head = g.arg(0)?;
    let first = g.mem.read_u64(head + 8)?;
    g.mem.write(head, &[0u8; 16])?;
    Ok(first)
}

fn QueryDepthSList(g: &mut Guest) -> H {
    let head = g.arg(0)?;
    Ok(g.mem.read_u16(head)? as u64)
}

// --- the process ----------------------------------------------------------------------------

/// IsProcessorFeaturePresent: what the interpreter executes, of the features a title asks for.
fn IsProcessorFeaturePresent(g: &mut Guest) -> H {
    const PF_COMPARE_EXCHANGE_DOUBLE: u32 = 2;
    const PF_XMMI_INSTRUCTIONS_AVAILABLE: u32 = 6;
    const PF_RDTSC_INSTRUCTION_AVAILABLE: u32 = 8;
    const PF_XMMI64_INSTRUCTIONS_AVAILABLE: u32 = 10;
    const PF_NX_ENABLED: u32 = 12;
    const PF_SSE3_INSTRUCTIONS_AVAILABLE: u32 = 13;
    const PF_SSSE3_INSTRUCTIONS_AVAILABLE: u32 = 36;
    const PF_SSE4_1_INSTRUCTIONS_AVAILABLE: u32 = 37;
    const PF_SSE4_2_INSTRUCTIONS_AVAILABLE: u32 = 38;
    let f = g.cpu.features;
    Ok(match g.arg(0)? as u32 {
        PF_COMPARE_EXCHANGE_DOUBLE | PF_XMMI_INSTRUCTIONS_AVAILABLE | PF_RDTSC_INSTRUCTION_AVAILABLE | PF_XMMI64_INSTRUCTIONS_AVAILABLE | PF_NX_ENABLED => true,
        PF_SSE3_INSTRUCTIONS_AVAILABLE => f.sse3,
        PF_SSSE3_INSTRUCTIONS_AVAILABLE => f.ssse3,
        PF_SSE4_1_INSTRUCTIONS_AVAILABLE => f.sse41,
        PF_SSE4_2_INSTRUCTIONS_AVAILABLE => f.sse42,
        _ => false,
    } as u64)
}

/// GetLogicalProcessorInformation(buffer, length in/out): the one processor GetSystemInfo
/// reports, as 32-byte records: its core, its caches (those of the console's Zen 2 core), its
/// package and its NUMA node.
fn GetLogicalProcessorInformation(g: &mut Guest) -> H {
    const CORE: u32 = 0;
    const NUMA_NODE: u32 = 1;
    const CACHE: u32 = 2;
    const PACKAGE: u32 = 3;
    // (relationship, cache: level, associativity, size, type – 0 unified, 1 instruction, 2 data)
    let records: [(u32, [u32; 4]); 7] = [
        (CORE, [0; 4]),
        (CACHE, [1, 8, 32 << 10, 2]),
        (CACHE, [1, 8, 32 << 10, 1]),
        (CACHE, [2, 8, 512 << 10, 0]),
        (CACHE, [3, 16, 8 << 20, 0]),
        (PACKAGE, [0; 4]),
        (NUMA_NODE, [0; 4]),
    ];
    let (at, length) = (g.arg(0)?, g.arg(1)?);
    if length == 0 {
        g.set_last_error(ERROR_INVALID_PARAMETER);
        return Ok(0);
    }
    let needed = 32 * records.len() as u32;
    let size = g.mem.read_u32(length)?;
    g.mem.write_u32(length, needed)?;
    if at == 0 || size < needed {
        g.set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return Ok(0);
    }
    for (i, (relationship, [level, associativity, cache_size, kind])) in records.into_iter().enumerate() {
        let record = at + 32 * i as u64;
        g.mem.write(record, &[0u8; 32])?;
        // the processors it is about: processor 0
        g.mem.write_u64(record, 1)?;
        g.mem.write_u32(record + 8, relationship)?;
        if relationship == CACHE {
            g.mem.write_u8(record + 16, level as u8)?;
            g.mem.write_u8(record + 17, associativity as u8)?;
            g.mem.write_u16(record + 18, 64)?;
            g.mem.write_u32(record + 20, cache_size)?;
            g.mem.write_u32(record + 24, kind)?;
        }
    }
    Ok(1)
}

fn GetTickCount64(g: &mut Guest) -> H {
    Ok(g.now_ns() / 1_000_000)
}

fn GetTickCount(g: &mut Guest) -> H {
    Ok(g.now_ns() / 1_000_000 & 0xFFFF_FFFF)
}

fn debug_string(g: &mut Guest, text: String) -> H {
    g.log.push(format!("[debug] {}", text.trim_end()));
    Ok(0)
}

fn OutputDebugStringA(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    let text = g.mem.cstr(at, 4096)?;
    debug_string(g, text)
}

fn OutputDebugStringW(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    let text = g.mem.wstr(at, 4096)?;
    debug_string(g, text)
}

/// EncodePointer / DecodePointer: the pointer itself (there is no process cookie).
fn pointer(g: &mut Guest) -> H {
    g.arg(0)
}

fn no(_: &mut Guest) -> H {
    Ok(0)
}

fn yes(_: &mut Guest) -> H {
    Ok(1)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; InitializeSRWLock, AcquireSRWLockExclusive, AcquireSRWLockShared, TryAcquireSRWLockExclusive, TryAcquireSRWLockShared);
    register!(map; ReleaseSRWLockExclusive, ReleaseSRWLockShared);
    register!(map; SleepConditionVariableCS, SleepConditionVariableSRW, WakeConditionVariable, WakeAllConditionVariable);
    // a condition variable is its address: nothing to set up
    map.insert("InitializeConditionVariable", no as Handler);
    register!(map; InitializeSListHead, InterlockedPushEntrySList, InterlockedPopEntrySList, InterlockedFlushSList, QueryDepthSList);
    register!(map; GetLogicalProcessorInformation);
    register!(map; IsProcessorFeaturePresent, GetTickCount64, GetTickCount, OutputDebugStringA, OutputDebugStringW);
    map.insert("EncodePointer", pointer as Handler);
    map.insert("DecodePointer", pointer as Handler);
    map.insert("IsDebuggerPresent", no as Handler);
    register!(map; CreateSemaphoreExW, ReleaseSemaphore, CreateMutexW, CreateMutexExW, ReleaseMutex, CreateEventExW);
    register!(map; WaitForMultipleObjects, ResumeThread);
    map.insert("CreateSemaphoreW", CreateSemaphoreExW as Handler);
    map.insert("WaitForMultipleObjectsEx", WaitForMultipleObjects as Handler);
    // the guest's threads take turns: priorities and processors change nothing
    for name in ["SetThreadPriority", "SetThreadAffinityMask", "SetThreadIdealProcessorEx", "SetThreadPriorityBoost"] {
        map.insert(name, yes as Handler);
    }
    // THREAD_PRIORITY_NORMAL
    map.insert("GetThreadPriority", no as Handler);
    if let Some(&single) = map.get("WaitForSingleObject") {
        map.insert("WaitForSingleObjectEx", single);
    }
    map.insert("DisableThreadLibraryCalls", yes as Handler);
    // the same clock, as kernel32 has it for GetSystemTimePreciseAsFileTime
    if let Some(&precise) = map.get("GetSystemTimePreciseAsFileTime") {
        map.insert("GetSystemTimeAsFileTime", precise);
    }
    if let Some(&info) = map.get("GetSystemInfo") {
        map.insert("GetNativeSystemInfo", info);
    }
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX};
    use xcl_emu::mem::{PROT_R, PROT_W};

    use super::*;

    const R8: usize = 8;
    const R9: usize = 9;

    /// A guest with `threads` threads and a zeroed page for the title's locks.
    fn guest(threads: usize) -> (Guest, u64) {
        let mut g = Guest::new();
        for _ in 0..threads {
            let stack = g.mem.alloc(0x4000, PROT_R | PROT_W).unwrap();
            g.add_thread(stack, stack + 0x4000, 0, 0, crate::SLOT_THREAD_RETURN).unwrap();
        }
        g.cpu = g.threads[0].cpu.clone();
        let page = g.mem.alloc(0x1000, PROT_R | PROT_W).unwrap();
        (g, page)
    }

    /// Thread `thread` calls `f(args)`.
    fn call(g: &mut Guest, thread: usize, f: Handler, args: [u64; 4]) -> H {
        g.switch_to(thread);
        for (register, value) in [RCX, RDX, R8, R9].into_iter().zip(args) {
            g.cpu.r[register] = value;
        }
        f(g)
    }

    #[test]
    fn an_srw_lock_admits_one_writer_or_many_readers() {
        let (mut g, lock) = guest(2);
        assert_eq!(call(&mut g, 0, AcquireSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 1, TryAcquireSRWLockShared, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 1, AcquireSRWLockExclusive, [lock, 0, 0, 0]), Err(Stop::Wait));
        assert_eq!(call(&mut g, 0, ReleaseSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
        // the waiting thread is asked again and gets it
        assert_eq!(call(&mut g, 1, AcquireSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
        assert!(g.threads[1].wait.is_none());
        assert_eq!(call(&mut g, 1, ReleaseSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, AcquireSRWLockShared, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 1, AcquireSRWLockShared, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, TryAcquireSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, ReleaseSRWLockShared, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, TryAcquireSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 1, ReleaseSRWLockShared, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, TryAcquireSRWLockExclusive, [lock, 0, 0, 0]), Ok(1));
    }

    #[test]
    fn a_condition_variable_sleep_gives_the_lock_up_and_takes_it_back_once_woken() {
        let (mut g, page) = guest(2);
        let (lock, cv) = (page, page + 64);
        let sleep = [cv, lock, u32::MAX as u64, 0];
        assert_eq!(call(&mut g, 0, AcquireSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, SleepConditionVariableSRW, sleep), Err(Stop::Wait));
        // the sleeper released the lock: the other thread takes it, and a wake alone does not
        // let the sleeper return while the lock is held
        assert_eq!(call(&mut g, 1, AcquireSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, SleepConditionVariableSRW, sleep), Err(Stop::Wait));
        assert_eq!(call(&mut g, 1, WakeConditionVariable, [cv, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, SleepConditionVariableSRW, sleep), Err(Stop::Wait));
        assert_eq!(call(&mut g, 1, ReleaseSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, SleepConditionVariableSRW, sleep), Ok(1));
        assert!(g.threads[0].wait.is_none());
        assert_eq!(call(&mut g, 1, TryAcquireSRWLockExclusive, [lock, 0, 0, 0]), Ok(0));
    }

    #[test]
    fn a_condition_variable_sleep_on_a_critical_section_ends_at_its_timeout() {
        let (mut g, page) = guest(1);
        let (cs, cv) = (page, page + 64);
        assert!(try_enter_cs(&mut g, cs).unwrap());
        let sleep = [cv, cs, 5, 0];
        assert_eq!(call(&mut g, 0, SleepConditionVariableCS, sleep), Err(Stop::Wait));
        assert_eq!(g.mem.read_u64(cs + 16).unwrap(), 0);
        assert_eq!(call(&mut g, 0, SleepConditionVariableCS, sleep), Err(Stop::Wait));
        g.win.time_ns += 5_000_000;
        assert_eq!(call(&mut g, 0, SleepConditionVariableCS, sleep), Ok(0));
        assert_eq!(g.threads[0].last_error, ERROR_TIMEOUT);
        // … with the critical section owned again
        assert_eq!(g.mem.read_u64(cs + 16).unwrap(), g.threads[0].id as u64);
    }

    #[test]
    fn a_semaphore_counts_and_a_mutex_belongs_to_its_thread() {
        let (mut g, page) = guest(2);
        let semaphore = call(&mut g, 0, CreateSemaphoreExW, [0, 1, 2, 0]).unwrap();
        assert_eq!(call(&mut g, 0, CreateSemaphoreExW, [0, 3, 2, 0]), Ok(0));
        assert_eq!(signaled(&mut g, semaphore, true), Some(true));
        assert_eq!(signaled(&mut g, semaphore, true), Some(false));
        assert_eq!(call(&mut g, 0, ReleaseSemaphore, [semaphore, 3, 0, 0]), Ok(0));
        assert_eq!(g.threads[0].last_error, ERROR_TOO_MANY_POSTS);
        assert_eq!(call(&mut g, 0, ReleaseSemaphore, [semaphore, 2, page, 0]), Ok(1));
        assert_eq!(g.mem.read_u32(page).unwrap(), 0);
        assert_eq!(signaled(&mut g, semaphore, false), Some(true));

        let mutex = call(&mut g, 0, CreateMutexW, [0, 1, 0, 0]).unwrap();
        // its owner takes it again; the other thread neither takes nor releases it
        assert_eq!(signaled(&mut g, mutex, true), Some(true));
        g.switch_to(1);
        assert_eq!(signaled(&mut g, mutex, true), Some(false));
        assert_eq!(call(&mut g, 1, ReleaseMutex, [mutex, 0, 0, 0]), Ok(0));
        assert_eq!(g.threads[1].last_error, ERROR_NOT_OWNER);
        assert_eq!(call(&mut g, 0, ReleaseMutex, [mutex, 0, 0, 0]), Ok(1));
        g.switch_to(1);
        assert_eq!(signaled(&mut g, mutex, false), Some(false));
        assert_eq!(call(&mut g, 0, ReleaseMutex, [mutex, 0, 0, 0]), Ok(1));
        g.switch_to(1);
        assert_eq!(signaled(&mut g, mutex, true), Some(true));
    }

    #[test]
    fn a_wait_for_several_objects_takes_only_what_ends_it() {
        let (mut g, page) = guest(1);
        let auto = g.win.new_handle(Object::Event { signaled: false, manual: false });
        let semaphore = g.win.new_handle(Object::Semaphore { count: 1, max: 1 });
        g.mem.write_u64(page, auto).unwrap();
        g.mem.write_u64(page + 8, semaphore).unwrap();
        // all: the semaphore alone is not enough, and is not counted down by the try
        assert_eq!(call(&mut g, 0, WaitForMultipleObjects, [2, page, 1, 0]), Ok(WAIT_TIMEOUT));
        assert_eq!(signaled(&mut g, semaphore, false), Some(true));
        // any: the second object ends it
        assert_eq!(call(&mut g, 0, WaitForMultipleObjects, [2, page, 0, 0]), Ok(1));
        assert_eq!(signaled(&mut g, semaphore, false), Some(false));
        // a wait that has to wait is asked again, and ends once the event is set
        assert_eq!(call(&mut g, 0, WaitForMultipleObjects, [2, page, 0, 100]), Err(Stop::Wait));
        g.win.handles.insert(auto, Object::Event { signaled: true, manual: false });
        assert_eq!(call(&mut g, 0, WaitForMultipleObjects, [2, page, 0, 100]), Ok(0));
        assert_eq!(signaled(&mut g, auto, false), Some(false));
        assert!(g.threads[0].wait.is_none());
        assert_eq!(call(&mut g, 0, WaitForMultipleObjects, [2, page, 0, 5]), Err(Stop::Wait));
        g.win.time_ns += 5_000_000;
        assert_eq!(call(&mut g, 0, WaitForMultipleObjects, [2, page, 0, 5]), Ok(WAIT_TIMEOUT));
        assert_eq!(call(&mut g, 0, WaitForMultipleObjects, [1, page + 16, 0, 0]), Ok(WAIT_FAILED));
    }

    #[test]
    fn a_suspended_thread_gets_no_turn_until_it_is_resumed() {
        let (mut g, _) = guest(2);
        g.threads[1].suspended = true;
        let handle = g.win.new_handle(Object::Thread(1));
        assert_eq!(g.next_thread(), Some(0));
        assert_eq!(call(&mut g, 0, ResumeThread, [handle, 0, 0, 0]), Ok(1));
        assert_eq!(g.next_thread(), Some(1));
        assert_eq!(call(&mut g, 0, ResumeThread, [handle, 0, 0, 0]), Ok(0));
    }

    #[test]
    fn an_slist_is_last_in_first_out_and_counts_its_entries() {
        let (mut g, page) = guest(1);
        let (head, a, b) = (page, page + 0x100, page + 0x200);
        g.mem.write(head, &[0xFF; 16]).unwrap();
        call(&mut g, 0, InitializeSListHead, [head, 0, 0, 0]).unwrap();
        assert_eq!(call(&mut g, 0, InterlockedPopEntrySList, [head, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, InterlockedPushEntrySList, [head, a, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, InterlockedPushEntrySList, [head, b, 0, 0]), Ok(a));
        assert_eq!(call(&mut g, 0, QueryDepthSList, [head, 0, 0, 0]), Ok(2));
        assert_eq!(call(&mut g, 0, InterlockedPopEntrySList, [head, 0, 0, 0]), Ok(b));
        assert_eq!(call(&mut g, 0, QueryDepthSList, [head, 0, 0, 0]), Ok(1));
        assert_eq!(call(&mut g, 0, InterlockedPushEntrySList, [head, b, 0, 0]), Ok(a));
        // flushing hands the chain over: b → a
        assert_eq!(call(&mut g, 0, InterlockedFlushSList, [head, 0, 0, 0]), Ok(b));
        assert_eq!(g.mem.read_u64(b).unwrap(), a);
        assert_eq!(call(&mut g, 0, QueryDepthSList, [head, 0, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, 0, InterlockedPopEntrySList, [head, 0, 0, 0]), Ok(0));
    }

    #[test]
    fn the_processor_layout_is_asked_for_its_size_first() {
        let (mut g, page) = guest(1);
        let (length, buffer) = (page, page + 0x100);
        g.mem.write_u32(length, 0).unwrap();
        assert_eq!(call(&mut g, 0, GetLogicalProcessorInformation, [0, length, 0, 0]), Ok(0));
        assert_eq!(g.threads[0].last_error, ERROR_INSUFFICIENT_BUFFER);
        let needed = g.mem.read_u32(length).unwrap();
        assert_eq!(needed % 32, 0);
        assert_eq!(call(&mut g, 0, GetLogicalProcessorInformation, [buffer, length, 0, 0]), Ok(1));
        let mut cores = 0;
        let mut caches = Vec::new();
        for record in (0..needed as u64).step_by(32).map(|o| buffer + o) {
            assert_eq!(g.mem.read_u64(record).unwrap(), 1);
            match g.mem.read_u32(record + 8).unwrap() {
                0 => cores += 1,
                2 => caches.push((g.mem.read_u8(record + 16).unwrap(), g.mem.read_u16(record + 18).unwrap(), g.mem.read_u32(record + 20).unwrap())),
                _ => {}
            }
        }
        assert_eq!(cores, 1);
        assert_eq!(caches, [(1, 64, 32 << 10), (1, 64, 32 << 10), (2, 64, 512 << 10), (3, 64, 8 << 20)]);
    }

    #[test]
    fn processor_features_follow_the_interpreter() {
        let (mut g, _) = guest(1);
        assert_eq!(call(&mut g, 0, IsProcessorFeaturePresent, [10, 0, 0, 0]), Ok(1));
        let sse3 = g.cpu.features.sse3 as u64;
        assert_eq!(call(&mut g, 0, IsProcessorFeaturePresent, [13, 0, 0, 0]), Ok(sse3));
        // PF_FASTFAIL_AVAILABLE and whatever else is not known: no
        assert_eq!(call(&mut g, 0, IsProcessorFeaturePresent, [23, 0, 0, 0]), Ok(0));
    }
}
