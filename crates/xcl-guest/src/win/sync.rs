//! What an MSVC-built image calls from its start-up code on: slim reader/writer locks,
//! condition variables, interlocked singly-linked lists, and the small process questions around
//! them. The lock words live in the title's memory and only these functions read them, so their
//! contents are the host's own:
//!
//! - SRWLOCK (8 bytes, 0 when free): bit 0 an exclusive owner, the shared owners counted from bit 4
//! - CONDITION_VARIABLE (8 bytes): not written; a sleeping thread waits on its address
//! - SLIST_HEADER (16 bytes): the depth at +0, the first entry at +8

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::kernel32::{leave_cs, try_enter_cs, wake};
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
    register!(map; IsProcessorFeaturePresent, GetTickCount64, GetTickCount, OutputDebugStringA, OutputDebugStringW);
    map.insert("EncodePointer", pointer as Handler);
    map.insert("DecodePointer", pointer as Handler);
    map.insert("IsDebuggerPresent", no as Handler);
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
    fn processor_features_follow_the_interpreter() {
        let (mut g, _) = guest(1);
        assert_eq!(call(&mut g, 0, IsProcessorFeaturePresent, [10, 0, 0, 0]), Ok(1));
        let sse3 = g.cpu.features.sse3 as u64;
        assert_eq!(call(&mut g, 0, IsProcessorFeaturePresent, [13, 0, 0, 0]), Ok(sse3));
        // PF_FASTFAIL_AVAILABLE and whatever else is not known: no
        assert_eq!(call(&mut g, 0, IsProcessorFeaturePresent, [23, 0, 0, 0]), Ok(0));
    }
}
