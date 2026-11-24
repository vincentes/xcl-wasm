//! The Win32 thread pool: work items, timers and waits whose callbacks run on the host's
//! worker thread (the one XGameRuntime's task queues use), one at a time.
//!
//! A PTP_WORK / PTP_TIMER / PTP_WAIT is an opaque value to the title: here a number.

#![allow(non_snake_case)]

use std::collections::{BTreeMap, HashMap};

use super::kernel32::signaled;
use super::*;
use crate::{Guest, Handler, Stop, Wait};

type H = Result<u64, Stop>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// callback(instance, context, work)
    Work,
    /// callback(instance, context, timer)
    Timer,
    /// callback(instance, context, wait, wait result)
    Wait,
    /// TrySubmitThreadpoolCallback: callback(instance, context), once
    Simple,
    /// QueueUserWorkItem: callback(context), once
    Legacy,
}

#[derive(Debug)]
struct Entry {
    kind: Kind,
    callback: u64,
    context: u64,
    /// submissions that have not run yet
    pending: u32,
    /// its callback is running on the worker
    running: bool,
    /// a timer: guest time of its next callback; a wait: when it gives up
    due: Option<u64>,
    /// a timer: nanoseconds between callbacks (0: once)
    period: u64,
    /// a wait: the handle waited for (0: not set)
    handle: u64,
}

#[derive(Debug, Default)]
pub struct Pool {
    objects: BTreeMap<u64, Entry>,
    count: u64,
    /// the object whose callback the worker is in
    current: Option<u64>,
}

fn create(g: &mut Guest, kind: Kind, callback: u64, context: u64) -> H {
    xgameruntime::ensure_worker(g)?;
    let pool = &mut g.win.pool;
    pool.count += 1;
    let id = 0x7A00_0000 + pool.count * 16;
    pool.objects.insert(id, Entry { kind, callback, context, pending: 0, running: false, due: None, period: 0, handle: 0 });
    Ok(id)
}

/// CreateThreadpoolWork(callback, context, environment)
fn CreateThreadpoolWork(g: &mut Guest) -> H {
    let (callback, context) = (g.arg(0)?, g.arg(1)?);
    create(g, Kind::Work, callback, context)
}

fn CreateThreadpoolTimer(g: &mut Guest) -> H {
    let (callback, context) = (g.arg(0)?, g.arg(1)?);
    create(g, Kind::Timer, callback, context)
}

fn CreateThreadpoolWait(g: &mut Guest) -> H {
    let (callback, context) = (g.arg(0)?, g.arg(1)?);
    create(g, Kind::Wait, callback, context)
}

fn submit(g: &mut Guest, kind: Kind, callback: u64, context: u64) -> H {
    let id = create(g, kind, callback, context)?;
    if let Some(o) = g.win.pool.objects.get_mut(&id) {
        o.pending = 1;
    }
    Ok(1)
}

/// TrySubmitThreadpoolCallback(callback, context, environment)
fn TrySubmitThreadpoolCallback(g: &mut Guest) -> H {
    let (callback, context) = (g.arg(0)?, g.arg(1)?);
    submit(g, Kind::Simple, callback, context)
}

/// QueueUserWorkItem(function, context, flags)
fn QueueUserWorkItem(g: &mut Guest) -> H {
    let (callback, context) = (g.arg(0)?, g.arg(1)?);
    submit(g, Kind::Legacy, callback, context)
}

fn SubmitThreadpoolWork(g: &mut Guest) -> H {
    let id = g.arg(0)?;
    if let Some(o) = g.win.pool.objects.get_mut(&id) {
        o.pending += 1;
    }
    Ok(0)
}

/// A FILETIME the title points at, as guest time: negative is relative, in 100 ns; an absolute
/// time is "now" (the guest clock has no calendar).
fn file_time(g: &Guest, at: u64) -> Result<u64, Stop> {
    let t = g.mem.read_u64(at)? as i64;
    Ok(g.now_ns() + if t < 0 { t.unsigned_abs() * 100 } else { 0 })
}

/// SetThreadpoolTimer(timer, due time *, period ms, window); no due time: the timer stops
fn SetThreadpoolTimer(g: &mut Guest) -> H {
    let (id, due_at, period_ms) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32);
    let due = if due_at == 0 { None } else { Some(file_time(g, due_at)?) };
    if let Some(o) = g.win.pool.objects.get_mut(&id) {
        o.due = due;
        o.period = period_ms as u64 * 1_000_000;
        o.pending = 0;
    }
    Ok(0)
}

/// SetThreadpoolWait(wait, handle, timeout *); no handle: the wait stops
fn SetThreadpoolWait(g: &mut Guest) -> H {
    let (id, handle, timeout_at) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    let due = if handle == 0 || timeout_at == 0 { None } else { Some(file_time(g, timeout_at)?) };
    if let Some(o) = g.win.pool.objects.get_mut(&id) {
        o.handle = handle;
        o.due = due;
    }
    Ok(0)
}

/// WaitForThreadpool…Callbacks(object, cancel pending): until its callbacks have run. Asked
/// from inside one of them (on the worker), it returns at once.
fn wait_for_callbacks(g: &mut Guest) -> H {
    let (id, cancel) = (g.arg(0)?, g.arg(1)? as u32 != 0);
    let on_worker = g.win.pool.current.is_some() && xgameruntime::is_worker(g);
    let Some(o) = g.win.pool.objects.get_mut(&id) else { return Ok(0) };
    if cancel {
        o.pending = 0;
    }
    if !on_worker && (o.pending != 0 || o.running) {
        g.thread_mut().wait = Some(Wait::default());
        return Err(Stop::Wait);
    }
    g.thread_mut().wait = None;
    Ok(0)
}

fn close(g: &mut Guest) -> H {
    let id = g.arg(0)?;
    g.win.pool.objects.remove(&id);
    Ok(0)
}

fn no(_: &mut Guest) -> H {
    Ok(0)
}

fn yes(_: &mut Guest) -> H {
    Ok(1)
}

/// The worker is at its slot. The callback it came back from is over; the next one that is due
/// is taken: its address and arguments, or else the guest time at which one will be.
pub(super) fn take(g: &mut Guest, now: u64) -> Result<(u64, Vec<u64>), Option<u64>> {
    if let Some(done) = g.win.pool.current.take() {
        if let Some(o) = g.win.pool.objects.get_mut(&done) {
            o.running = false;
        }
    }
    let mut next: Option<u64> = None;
    let ids: Vec<u64> = g.win.pool.objects.keys().copied().collect();
    for id in ids {
        let o = &g.win.pool.objects[&id];
        let (kind, due, handle) = (o.kind, o.due, o.handle);
        let mut result = 0;
        let fires = match kind {
            Kind::Work | Kind::Simple | Kind::Legacy => o.pending != 0,
            Kind::Timer => due.is_some_and(|d| d <= now),
            Kind::Wait if handle == 0 => false,
            Kind::Wait => {
                if signaled(g, handle, true) == Some(true) {
                    true
                } else {
                    result = WAIT_TIMEOUT;
                    due.is_some_and(|d| d <= now)
                }
            }
        };
        if !fires {
            if let Some(d) = due.filter(|_| kind == Kind::Timer || handle != 0) {
                next = Some(next.map_or(d, |n| n.min(d)));
            }
            continue;
        }
        let o = g.win.pool.objects.get_mut(&id).unwrap();
        let (callback, context) = (o.callback, o.context);
        match kind {
            Kind::Work => o.pending -= 1,
            Kind::Timer => o.due = (o.period != 0).then(|| now + o.period),
            // a wait fires once and has to be set again
            Kind::Wait => (o.handle, o.due) = (0, None),
            Kind::Simple | Kind::Legacy => {}
        }
        if matches!(kind, Kind::Simple | Kind::Legacy) {
            g.win.pool.objects.remove(&id);
        } else {
            o.running = true;
            g.win.pool.current = Some(id);
        }
        let args = match kind {
            Kind::Legacy => vec![context],
            Kind::Simple => vec![id, context],
            Kind::Wait => vec![id, context, id, result],
            Kind::Work | Kind::Timer => vec![id, context, id],
        };
        return Ok((callback, args));
    }
    Err(next)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; CreateThreadpoolWork, CreateThreadpoolTimer, CreateThreadpoolWait, TrySubmitThreadpoolCallback, QueueUserWorkItem);
    register!(map; SubmitThreadpoolWork, SetThreadpoolTimer, SetThreadpoolWait);
    for name in ["WaitForThreadpoolWorkCallbacks", "WaitForThreadpoolTimerCallbacks", "WaitForThreadpoolWaitCallbacks"] {
        map.insert(name, wait_for_callbacks as Handler);
    }
    for name in ["CloseThreadpoolWork", "CloseThreadpoolTimer", "CloseThreadpoolWait"] {
        map.insert(name, close as Handler);
    }
    // one worker: a callback that runs long holds the others up whatever it says
    map.insert("CallbackMayRunLong", yes as Handler);
    map.insert("DisassociateCurrentThreadFromCallback", no as Handler);
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX};
    use xcl_emu::mem::{PROT_R, PROT_W};

    use super::*;

    fn guest() -> (Guest, u64) {
        let mut g = Guest::new();
        let stack = g.mem.alloc(0x4000, PROT_R | PROT_W).unwrap();
        g.add_thread(stack, stack + 0x4000, 0, 0, crate::SLOT_THREAD_RETURN).unwrap();
        g.cpu = g.threads[0].cpu.clone();
        let page = g.mem.alloc(0x1000, PROT_R | PROT_W).unwrap();
        (g, page)
    }

    fn call(g: &mut Guest, f: Handler, args: [u64; 3]) -> H {
        for (register, value) in [RCX, RDX, 8].into_iter().zip(args) {
            g.cpu.r[register] = value;
        }
        f(g)
    }

    #[test]
    fn work_runs_once_per_submission_and_is_waited_for() {
        let (mut g, _) = guest();
        let work = call(&mut g, CreateThreadpoolWork, [0x1000, 7, 0]).unwrap();
        assert_eq!(take(&mut g, 0), Err(None));
        call(&mut g, SubmitThreadpoolWork, [work, 0, 0]).unwrap();
        call(&mut g, SubmitThreadpoolWork, [work, 0, 0]).unwrap();
        assert_eq!(call(&mut g, wait_for_callbacks, [work, 0, 0]), Err(Stop::Wait));
        assert_eq!(take(&mut g, 0), Ok((0x1000, vec![work, 7, work])));
        assert_eq!(take(&mut g, 0), Ok((0x1000, vec![work, 7, work])));
        // the second callback is still running
        assert_eq!(call(&mut g, wait_for_callbacks, [work, 0, 0]), Err(Stop::Wait));
        assert_eq!(take(&mut g, 0), Err(None));
        assert_eq!(call(&mut g, wait_for_callbacks, [work, 0, 0]), Ok(0));
        // cancelling drops what has not started
        call(&mut g, SubmitThreadpoolWork, [work, 0, 0]).unwrap();
        assert_eq!(call(&mut g, wait_for_callbacks, [work, 1, 0]), Ok(0));
        assert_eq!(take(&mut g, 0), Err(None));
        call(&mut g, close, [work, 0, 0]).unwrap();
        call(&mut g, SubmitThreadpoolWork, [work, 0, 0]).unwrap();
        assert_eq!(take(&mut g, 0), Err(None));
    }

    #[test]
    fn a_timer_fires_when_due_and_again_each_period() {
        let (mut g, page) = guest();
        let timer = call(&mut g, CreateThreadpoolTimer, [0x2000, 1, 0]).unwrap();
        // 2 ms from now, then every 5 ms
        g.mem.write_u64(page, (-20_000i64) as u64).unwrap();
        call(&mut g, SetThreadpoolTimer, [timer, page, 5]).unwrap();
        assert_eq!(take(&mut g, 1_999_999), Err(Some(2_000_000)));
        assert_eq!(take(&mut g, 2_000_000), Ok((0x2000, vec![timer, 1, timer])));
        assert_eq!(take(&mut g, 2_000_001), Err(Some(7_000_000)));
        assert_eq!(take(&mut g, 7_000_000), Ok((0x2000, vec![timer, 1, timer])));
        // stopped
        call(&mut g, SetThreadpoolTimer, [timer, 0, 0]).unwrap();
        assert_eq!(take(&mut g, 1 << 40), Err(None));
    }

    #[test]
    fn a_wait_fires_once_for_its_handle_or_at_its_timeout() {
        let (mut g, page) = guest();
        let event = g.win.new_handle(Object::Event { signaled: false, manual: false });
        let wait = call(&mut g, CreateThreadpoolWait, [0x3000, 2, 0]).unwrap();
        call(&mut g, SetThreadpoolWait, [wait, event, 0]).unwrap();
        assert_eq!(take(&mut g, 0), Err(None));
        g.win.handles.insert(event, Object::Event { signaled: true, manual: false });
        assert_eq!(take(&mut g, 0), Ok((0x3000, vec![wait, 2, wait, 0])));
        // the event was taken, and the wait is not set any more
        assert_eq!(signaled(&mut g, event, false), Some(false));
        g.win.handles.insert(event, Object::Event { signaled: true, manual: false });
        assert_eq!(take(&mut g, 0), Err(None));
        g.win.handles.insert(event, Object::Event { signaled: false, manual: false });
        g.mem.write_u64(page, (-10_000i64) as u64).unwrap();
        call(&mut g, SetThreadpoolWait, [wait, event, page]).unwrap();
        assert_eq!(take(&mut g, 0), Err(Some(1_000_000)));
        assert_eq!(take(&mut g, 1_000_000), Ok((0x3000, vec![wait, 2, wait, WAIT_TIMEOUT])));
    }

    #[test]
    fn one_shot_callbacks_run_once_with_their_own_arguments() {
        let (mut g, _) = guest();
        assert_eq!(call(&mut g, QueueUserWorkItem, [0x4000, 9, 0]), Ok(1));
        assert_eq!(take(&mut g, 0), Ok((0x4000, vec![9])));
        assert_eq!(take(&mut g, 0), Err(None));
        assert_eq!(call(&mut g, TrySubmitThreadpoolCallback, [0x5000, 3, 0]), Ok(1));
        let Ok((0x5000, args)) = take(&mut g, 0) else { panic!("the callback is due") };
        assert_eq!(args[1], 3);
        assert!(g.win.pool.objects.is_empty());
    }
}
