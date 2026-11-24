//! XGameRuntime on the guest: task queues, asynchronous operations, the (one, offline) user,
//! the package's identity, storage and networking. The answers follow XCL's Windows runtime
//! (`xcl::hle::xgameruntime`); an asynchronous operation completes at once here.

#![allow(non_snake_case)]

use std::collections::{HashMap, VecDeque};

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const E_INVALIDARG: u64 = 0x8007_0057;
const E_NOT_SUFFICIENT_BUFFER: u64 = 0x8007_007A;
/// WININET's "cannot connect": what a request for an Xbox Live token ends with offline
const E_CANNOT_CONNECT: u32 = 0x8007_2EFD;

/// XTaskQueueDispatchMode
const MANUAL: u32 = 0;
const IMMEDIATE: u32 = 3;

/// A callback waiting on a port.
#[derive(Debug, Clone)]
pub struct Item {
    /// the queue it was submitted through (terminating that queue cancels it)
    queue: u64,
    context: u64,
    callback: u64,
    /// guest time from which it may run
    due: u64,
}

#[derive(Debug, Default)]
pub struct Port {
    mode: u32,
    items: VecDeque<Item>,
}

/// Task queues, asynchronous operations and the user.
#[derive(Default)]
pub struct State {
    /// ports by handle
    ports: HashMap<u64, Port>,
    /// queues by handle: their work and completion ports
    queues: HashMap<u64, [u64; 2]>,
    /// finished asynchronous operations by XAsyncBlock: (status, result)
    asyncs: HashMap<u64, (u32, Vec<u8>)>,
    user: u64,
    /// the thread that runs thread-pool callbacks (index in `Guest::threads`)
    worker: Option<usize>,
}

/// Copy `text` and its terminator out (XGameRuntime's string results).
fn cstr_out(g: &mut Guest, text: &str, at: u64, size: u64, used: u64) -> H {
    let needed = text.len() as u64 + 1;
    if used != 0 {
        g.mem.write_u64(used, needed)?;
    }
    if at == 0 || size < needed {
        return Ok(E_NOT_SUFFICIENT_BUFFER);
    }
    g.mem.write(at, text.as_bytes())?;
    g.mem.write_u8(at + text.len() as u64, 0)?;
    Ok(0)
}

fn XGameRuntimeInitialize(_: &mut Guest) -> H {
    Ok(0)
}

/// The package's name: `<Identity Name="…">` of the game folder's MicrosoftGame.config.
fn identity_name(g: &Guest) -> Option<String> {
    let raw = g.vfs.game_file("MicrosoftGame.config")?;
    let xml = String::from_utf8_lossy(raw);
    let tag = &xml[xml.find("<Identity")?..];
    let tag = &tag[..tag.find('>')?];
    let value = &tag[tag.find("Name=\"")? + 6..];
    Some(value[..value.find('"')?].to_string())
}

/// XPackageGetCurrentProcessPackageIdentifier(size, buffer)
fn XPackageGetCurrentProcessPackageIdentifier(g: &mut Guest) -> H {
    let (size, at) = (g.arg(0)?, g.arg(1)?);
    let id = format!("{}_1.0.0.0_x64__xcl", identity_name(g).unwrap_or_else(|| "XCL.Title".into()));
    cstr_out(g, &id, at, size, 0)
}

// --- task queues ----------------------------------------------------------------------------

/// A handle: a small heap block, so that it is a pointer no other object has.
fn new_handle(g: &mut Guest) -> Result<u64, Stop> {
    g.heap_alloc(16)
}

/// XTaskQueueCreate(work mode, completion mode, queue out)
fn XTaskQueueCreate(g: &mut Guest) -> H {
    let (work, completion, out) = (g.arg(0)? as u32, g.arg(1)? as u32, g.arg(2)?);
    if out == 0 || work > IMMEDIATE || completion > IMMEDIATE {
        return Ok(E_INVALIDARG);
    }
    let mut ports = [0u64; 2];
    for (port, mode) in ports.iter_mut().zip([work, completion]) {
        *port = new_handle(g)?;
        g.win.xgr.ports.insert(*port, Port { mode, items: VecDeque::new() });
    }
    let queue = new_handle(g)?;
    g.win.xgr.queues.insert(queue, ports);
    if work != MANUAL && work != IMMEDIATE || completion != MANUAL && completion != IMMEDIATE {
        ensure_worker(g)?;
    }
    g.mem.write_u64(out, queue)?;
    Ok(0)
}

/// XTaskQueueGetPort(queue, port, port handle out)
fn XTaskQueueGetPort(g: &mut Guest) -> H {
    let (queue, port, out) = (g.arg(0)?, g.arg(1)? as usize, g.arg(2)?);
    match g.win.xgr.queues.get(&queue) {
        Some(ports) if port < 2 && out != 0 => {
            let handle = ports[port];
            g.mem.write_u64(out, handle)?;
            Ok(0)
        }
        _ => Ok(E_INVALIDARG),
    }
}

/// XTaskQueueCreateComposite(work port, completion port, queue out): a queue on ports of others
fn XTaskQueueCreateComposite(g: &mut Guest) -> H {
    let (work, completion, out) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    if out == 0 || !g.win.xgr.ports.contains_key(&work) || !g.win.xgr.ports.contains_key(&completion) {
        return Ok(E_INVALIDARG);
    }
    let queue = new_handle(g)?;
    g.win.xgr.queues.insert(queue, [work, completion]);
    g.mem.write_u64(out, queue)?;
    Ok(0)
}

/// Put a callback on a port of `queue` (`after_ms` from now); an immediate port runs it now.
fn submit(g: &mut Guest, queue: u64, port: usize, after_ms: u64, context: u64, callback: u64) -> H {
    let Some(handle) = g.win.xgr.queues.get(&queue).filter(|_| port < 2).map(|p| p[port]) else { return Ok(E_INVALIDARG) };
    let due = g.now_ns() + after_ms * 1_000_000;
    let Some(p) = g.win.xgr.ports.get_mut(&handle) else { return Ok(E_INVALIDARG) };
    if p.mode == IMMEDIATE && after_ms == 0 {
        g.call(callback, &[context, 0])?;
        return Ok(0);
    }
    p.items.push_back(Item { queue, context, callback, due });
    Ok(0)
}

/// XTaskQueueSubmitCallback(queue, port, context, callback)
fn XTaskQueueSubmitCallback(g: &mut Guest) -> H {
    let (queue, port, context, callback) = (g.arg(0)?, g.arg(1)? as usize, g.arg(2)?, g.arg(3)?);
    submit(g, queue, port, 0, context, callback)
}

/// XTaskQueueSubmitDelayedCallback(queue, port, delay ms, context, callback)
fn XTaskQueueSubmitDelayedCallback(g: &mut Guest) -> H {
    let (queue, port, ms, context, callback) = (g.arg(0)?, g.arg(1)? as usize, g.arg(2)? as u32 as u64, g.arg(3)?, g.arg(4)?);
    submit(g, queue, port, ms, context, callback)
}

/// The first callback of a port that is due, taken off it; else when the next one is.
fn take_due(port: &mut Port, now: u64) -> Result<Item, Option<u64>> {
    match port.items.iter().position(|i| i.due <= now) {
        Some(i) => Ok(port.items.remove(i).unwrap()),
        None => Err(port.items.iter().map(|i| i.due).min()),
    }
}

/// XTaskQueueDispatch(queue, port, timeout ms): run one callback of a manual port; waits up to
/// the timeout for one to be due.
fn XTaskQueueDispatch(g: &mut Guest) -> H {
    let (queue, port, timeout) = (g.arg(0)?, g.arg(1)? as usize, g.arg(2)? as u32);
    let Some(handle) = g.win.xgr.queues.get(&queue).filter(|_| port < 2).map(|p| p[port]) else { return Ok(0) };
    let now = g.now_ns();
    let Some(p) = g.win.xgr.ports.get_mut(&handle) else { return Ok(0) };
    match take_due(p, now) {
        Ok(item) => {
            g.thread_mut().wait = None;
            g.call(item.callback, &[item.context, 0])?;
            Ok(1)
        }
        Err(next) => {
            let waited = super::kernel32::wait(g, false, timeout);
            if let (Err(Stop::Wait), Some(due), Some(w)) = (&waited, next, g.thread_mut().wait.as_mut()) {
                // a delayed callback ends the wait when it is due
                w.deadline = Some(w.deadline.map_or(due, |d| d.min(due)));
            }
            waited.map(|_| 0)
        }
    }
}

/// XTaskQueueTerminate(queue, wait, context, callback): the callbacks submitted through this
/// queue that have not run are called as cancelled; other queues' on shared ports stay.
fn XTaskQueueTerminate(g: &mut Guest) -> H {
    let (queue, context, callback) = (g.arg(0)?, g.arg(2)?, g.arg(3)?);
    let Some(ports) = g.win.xgr.queues.get(&queue).copied() else { return Ok(E_INVALIDARG) };
    let mut cancelled = Vec::new();
    for handle in ports {
        if let Some(p) = g.win.xgr.ports.get_mut(&handle) {
            let (gone, kept): (Vec<Item>, Vec<Item>) = p.items.drain(..).partition(|i| i.queue == queue);
            p.items = kept.into();
            cancelled.extend(gone);
        }
    }
    for item in cancelled {
        g.call(item.callback, &[item.context, 1])?;
    }
    if callback != 0 {
        g.call(callback, &[context])?;
    }
    Ok(0)
}

fn XTaskQueueCloseHandle(g: &mut Guest) -> H {
    let queue = g.arg(0)?;
    g.win.xgr.queues.remove(&queue);
    Ok(0)
}

/// The thread that runs thread-pool callbacks: made when the first thread-pool port is.
pub(super) fn ensure_worker(g: &mut Guest) -> Result<(), Stop> {
    if g.win.xgr.worker.is_none() {
        let size = 1u64 << 20;
        let stack = g.mem.alloc(size, xcl_emu::mem::PROT_R | xcl_emu::mem::PROT_W).map_err(|_| Stop::Deadlock)?;
        let index = g.add_host_thread(stack, stack + size)?;
        g.threads[index].name = "XCL task queue".into();
        g.win.xgr.worker = Some(index);
    }
    Ok(())
}

/// Whether the running thread is the worker.
pub(super) fn is_worker(g: &Guest) -> bool {
    g.win.xgr.worker == Some(g.current)
}

impl Guest {
    /// The worker thread is at its slot: give it the next thread-pool callback that is due
    /// (`Ok(true)`), or say until when it has nothing to do.
    pub(crate) fn worker_turn(&mut self) -> Result<bool, Stop> {
        let now = self.now_ns();
        // the Win32 thread pool's callbacks first, then the task queues'
        let mut next = match super::threadpool::take(self, now) {
            Ok((callback, args)) => {
                self.thread_mut().wait = None;
                self.host_thread_call(callback, &args)?;
                return Ok(true);
            }
            Err(due) => due,
        };
        let mut found = None;
        for port in self.win.xgr.ports.values_mut().filter(|p| p.mode != MANUAL && p.mode != IMMEDIATE) {
            match take_due(port, now) {
                Ok(item) => {
                    found = Some(item);
                    break;
                }
                Err(Some(due)) => next = Some(next.map_or(due, |n| n.min(due))),
                Err(None) => {}
            }
        }
        match found {
            Some(item) => {
                self.thread_mut().wait = None;
                self.host_thread_call(item.callback, &[item.context, 0])?;
                Ok(true)
            }
            None => {
                self.thread_mut().wait = Some(crate::Wait { deadline: next, address: 0, woken: false });
                Ok(false)
            }
        }
    }

    /// An asynchronous operation ends: its status and result are kept for the block, and the
    /// block's completion callback goes to its queue (the process's thread pool for none).
    fn complete(&mut self, block: u64, status: u32, result: Vec<u8>) -> Result<(), Stop> {
        self.win.xgr.asyncs.insert(block, (status, result));
        let (queue, callback) = (self.mem.read_u64(block)?, self.mem.read_u64(block + 16)?);
        if callback == 0 {
            return Ok(());
        }
        match self.win.xgr.queues.get(&queue).map(|p| p[1]) {
            Some(port) => {
                let due = self.now_ns();
                if let Some(p) = self.win.xgr.ports.get_mut(&port) {
                    p.items.push_back(Item { queue, context: block, callback, due });
                }
            }
            // no queue: the callback runs now, on the caller's thread
            None => {
                self.call(callback, &[block])?;
            }
        }
        Ok(())
    }
}

// --- asynchronous operations ------------------------------------------------------------------

/// XAsyncGetStatus(block, wait)
fn XAsyncGetStatus(g: &mut Guest) -> H {
    let block = g.arg(0)?;
    Ok(g.win.xgr.asyncs.get(&block).map_or(E_INVALIDARG, |(status, _)| *status as u64))
}

/// XAsyncGetResultSize(block, size out)
fn XAsyncGetResultSize(g: &mut Guest) -> H {
    let (block, out) = (g.arg(0)?, g.arg(1)?);
    match g.win.xgr.asyncs.get(&block) {
        Some((0, result)) => {
            let n = result.len() as u64;
            g.mem.write_u64(out, n)?;
            Ok(0)
        }
        Some((status, _)) => Ok(*status as u64),
        None => Ok(E_INVALIDARG),
    }
}

/// The result of a finished operation, copied to (`at`, `size`).
fn result_out(g: &mut Guest, block: u64, at: u64, size: u64) -> H {
    match g.win.xgr.asyncs.get(&block).cloned() {
        Some((0, result)) => {
            if size < result.len() as u64 {
                return Ok(E_NOT_SUFFICIENT_BUFFER);
            }
            g.mem.write(at, &result)?;
            Ok(0)
        }
        Some((status, _)) => Ok(status as u64),
        None => Ok(E_INVALIDARG),
    }
}

// --- the user: one, signed in, offline -----------------------------------------------------------

fn user(g: &mut Guest) -> Result<u64, Stop> {
    if g.win.xgr.user == 0 {
        g.win.xgr.user = g.heap_alloc(16)?;
    }
    Ok(g.win.xgr.user)
}

/// XUserAddAsync(options, block)
fn XUserAddAsync(g: &mut Guest) -> H {
    let block = g.arg(1)?;
    let u = user(g)?;
    g.complete(block, 0, u.to_le_bytes().to_vec())?;
    Ok(0)
}

/// XUserAddResult(block, user out)
fn XUserAddResult(g: &mut Guest) -> H {
    let (block, out) = (g.arg(0)?, g.arg(1)?);
    result_out(g, block, out, 8)
}

/// XUserFindForDevice(device id, user out)
fn XUserFindForDevice(g: &mut Guest) -> H {
    let out = g.arg(1)?;
    let u = user(g)?;
    g.mem.write_u64(out, u)?;
    Ok(0)
}

/// XUserGetGamertag(user, component, size, buffer, used out)
fn XUserGetGamertag(g: &mut Guest) -> H {
    let (handle, size, at, used) = (g.arg(0)?, g.arg(2)?, g.arg(3)?, g.arg(4)?);
    if handle == 0 || handle != g.win.xgr.user {
        return Ok(E_INVALIDARG);
    }
    let tag = xcl::state::config().gamertag.clone();
    cstr_out(g, &tag, at, size, used)
}

/// XUserGetTokenAndSignatureAsync(user, options, method, url, header count, headers, body size,
/// body, block): there is no Xbox Live here, the request cannot connect.
fn XUserGetTokenAndSignatureAsync(g: &mut Guest) -> H {
    let block = g.arg(8)?;
    g.complete(block, E_CANNOT_CONNECT, Vec::new())?;
    Ok(0)
}

fn ok(_: &mut Guest) -> H {
    Ok(0)
}

// --- storage and networking -----------------------------------------------------------------------

/// XGameSaveFilesGetFolderWithUiAsync(user, service configuration id, block): the save folder
/// (`U:\saves\<id>`, made now)
fn XGameSaveFilesGetFolderWithUiAsync(g: &mut Guest) -> H {
    let (scid, block) = (g.arg(1)?, g.arg(2)?);
    let scid = if scid == 0 { String::new() } else { g.mem.cstr(scid, 256)? };
    let safe: String = scid.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
    let folder = format!("U:\\saves\\{}", if safe.is_empty() { "default" } else { &safe });
    g.vfs.mkdir("U:\\saves");
    g.vfs.mkdir(&folder);
    let mut result = folder.into_bytes();
    result.push(0);
    g.complete(block, 0, result)?;
    Ok(0)
}

/// XGameSaveFilesGetFolderWithUiResult(block, size, buffer)
fn XGameSaveFilesGetFolderWithUiResult(g: &mut Guest) -> H {
    let (block, size, at) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    result_out(g, block, at, size)
}

/// XPersistentLocalStorageGetPath(size, buffer, used out): a drive root, as on the console
fn XPersistentLocalStorageGetPath(g: &mut Guest) -> H {
    let (size, at, used) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    cstr_out(g, "S:\\", at, size, used)
}

fn XNetworkingQueryPreferredLocalUdpMultiplayerPort(g: &mut Guest) -> H {
    let out = g.arg(0)?;
    if out == 0 {
        return Ok(E_INVALIDARG);
    }
    g.mem.write_u16(out, 3074)?;
    Ok(0)
}

/// XNetworkingConnectivityHint: internet access, unrestricted, Ethernet, initialized
fn XNetworkingGetConnectivityHint(g: &mut Guest) -> H {
    let out = g.arg(0)?;
    if out == 0 {
        return Ok(E_INVALIDARG);
    }
    for (i, v) in [3u32, 1, 6, 1].into_iter().enumerate() {
        g.mem.write_u32(out + 4 * i as u64, v)?;
    }
    Ok(0)
}

/// advapi32's LSA functions, which console titles reach through advapi32legacy: no policy here.
fn LsaOpenPolicy(_: &mut Guest) -> H {
    Ok(0xC000_00BB) // STATUS_NOT_SUPPORTED
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; XGameRuntimeInitialize, XPackageGetCurrentProcessPackageIdentifier);
    register!(map; XTaskQueueCreate, XTaskQueueGetPort, XTaskQueueCreateComposite, XTaskQueueSubmitCallback, XTaskQueueSubmitDelayedCallback, XTaskQueueDispatch, XTaskQueueTerminate, XTaskQueueCloseHandle);
    register!(map; XAsyncGetStatus, XAsyncGetResultSize, XUserAddAsync, XUserAddResult, XUserFindForDevice, XUserGetGamertag, XUserGetTokenAndSignatureAsync);
    register!(map; XGameSaveFilesGetFolderWithUiAsync, XGameSaveFilesGetFolderWithUiResult, XPersistentLocalStorageGetPath);
    register!(map; XNetworkingQueryPreferredLocalUdpMultiplayerPort, XNetworkingGetConnectivityHint, LsaOpenPolicy);
    map.insert("XUserGetTokenAndSignatureResultSize", XAsyncGetResultSize as Handler);
    map.insert("XUserCloseHandle", ok as Handler);
    map.insert("XGameRuntimeUninitialize", ok as Handler);
}
