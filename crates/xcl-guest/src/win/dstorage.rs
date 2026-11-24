//! `xcrdapi.dll`, the console's DirectStorage: `DStorageGetFactory` → IDStorageFactoryX. A retail
//! title reads its archives through it. The interfaces' method orders and the structures are
//! the ones XCL's Windows runtime documents (`xcl::hle::dstorage`):
//!
//! - factory: CreateQueue(desc, riid, out), OpenFile(path, riid, out), CreateStatusArray(
//!   capacity, name, riid, out), SetCpuAffinity, SetDebugFlags, SetStagingBufferSize, one more,
//!   and the newer SDK's CreateQueue in slot 10
//! - queue: EnqueueRequest(request), EnqueueStatus(array, index), EnqueueSignal(fence, value),
//!   Submit, CancelRequestsWithTag, Close, GetErrorEvent, RetrieveErrorRecord, Query
//! - file: Close, GetHandle, two more, GetId; status array: IsComplete(index), GetHResult(index)
//! - request (0x48 bytes): options @0, destination @8, its size u32 @0x18, source @0x20 (a
//!   file object, or memory), file offset @0x28, source size u32 @0x30
//!
//! A request is carried out when it is enqueued, on the calling thread: a file read through the
//! title's file system (a range of the file, whatever its size) or a copy of memory. A status
//! entry gets the first failure of the requests since the previous one. Requests that ask for
//! decompression or the console's texture formats are refused: there is no decoder yet.

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const E_NOTIMPL: u32 = 0x8000_4001;
const E_POINTER: u64 = 0x8000_4003;
const E_PENDING: u32 = 0x8000_000A;
const E_INVALIDARG: u32 = 0x8007_0057;
const E_END_OF_FILE: u32 = 0x8924_0007;
const E_ACCESS_VIOLATION: u32 = 0x8924_0009;
const E_DESTINATION_SIZE: u32 = 0x8924_000F;
const E_FILE_HANDLE: u32 = 0x8924_0017;

/// request options: bit 0 zlib, bits 1..=4 BCPack, bits 5..=20 swizzle, bit 21 the destination
/// is physical pages, bit 22 the source is memory, bit 23 the source is physical pages
const SOURCE_MEMORY: u64 = 1 << 22;
const NOT_A_PLAIN_COPY: u64 = ((1 << 24) - 1) & !SOURCE_MEMORY;

const FACTORY: [&str; 11] = [
    "IDStorage_QueryInterface",
    "IDStorage_AddRef",
    "IDStorage_Release",
    "IDStorageFactoryX_CreateQueue",
    "IDStorageFactoryX_OpenFile",
    "IDStorageFactoryX_CreateStatusArray",
    "IDStorageFactoryX_SetCpuAffinity",
    "IDStorageFactoryX_SetDebugFlags",
    "IDStorageFactoryX_SetStagingBufferSize",
    "IDStorageFactoryX_#9",
    "IDStorageFactoryX_CreateQueue1",
];
const QUEUE: [&str; 12] = [
    "IDStorage_QueryInterface",
    "IDStorage_AddRef",
    "IDStorage_Release",
    "IDStorageQueueX_EnqueueRequest",
    "IDStorageQueueX_EnqueueStatus",
    "IDStorageQueueX_EnqueueSignal",
    "IDStorageQueueX_Submit",
    "IDStorageQueueX_CancelRequestsWithTag",
    "IDStorageQueueX_Close",
    "IDStorageQueueX_GetErrorEvent",
    "IDStorageQueueX_RetrieveErrorRecord",
    "IDStorageQueueX_Query",
];
const FILE: [&str; 8] =
    ["IDStorage_QueryInterface", "IDStorage_AddRef", "IDStorage_Release", "IDStorageFileX_Close", "IDStorageFileX_GetHandle", "IDStorageFileX_#5", "IDStorageFileX_#6", "IDStorageFileX_GetId"];
const STATUS_ARRAY: [&str; 5] = ["IDStorage_QueryInterface", "IDStorage_AddRef", "IDStorage_Release", "IDStorageStatusArrayX_IsComplete", "IDStorageStatusArrayX_GetHResult"];

#[derive(Debug, Default)]
struct Queue {
    /// the queue's description, as the title gave it (Query hands it back)
    desc: Vec<u8>,
    /// the first failure of the requests since the last status entry (0: none)
    failure: u32,
    requests: u64,
    error_event: u64,
}

#[derive(Debug, Default)]
pub struct State {
    /// by the object's address
    queues: HashMap<u64, Queue>,
    status_arrays: HashMap<u64, Vec<u32>>,
    /// requests refused for what they ask (noted once per kind)
    refused: Vec<u64>,
}

/// A COM object of `interface` with `total` vtable slots: `methods`, then functions without a
/// name of their own (the host notes a call of one). Its host word (at +8) is `word`.
fn object(g: &mut Guest, interface: &'static str, methods: &[&'static str], total: usize, word: u64) -> Result<u64, Stop> {
    let mut all: Vec<String> = methods.iter().map(|m| m.to_string()).collect();
    all.extend((methods.len()..total).map(|i| format!("{interface}_#{i}")));
    let names: Vec<&str> = all.iter().map(String::as_str).collect();
    let object = g.com_object(interface, &names)?;
    g.mem.write_u64(object + 8, word)?;
    Ok(object)
}

/// DStorageGetFactory(riid, factory out)
fn DStorageGetFactory(g: &mut Guest) -> H {
    let out = g.arg(1)?;
    if out == 0 {
        return Ok(E_POINTER);
    }
    let factory = g.static_block("IDStorageFactoryX", |g| object(g, "IDStorageFactoryX", &FACTORY, 16, 0))?;
    g.mem.write_u64(out, factory)?;
    Ok(0)
}

/// QueryInterface(this, riid, out): the object is all its interfaces.
fn IDStorage_QueryInterface(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(2)?);
    if out == 0 {
        return Ok(E_POINTER);
    }
    g.mem.write_u64(out, this)?;
    Ok(0)
}

/// CreateQueue(this, desc, riid, out). The desc is 0x18 bytes, the newer SDK's 0x20.
fn create_queue(g: &mut Guest, desc_len: usize) -> H {
    let (desc_at, out) = (g.arg(1)?, g.arg(3)?);
    if desc_at == 0 || out == 0 {
        return Ok(E_POINTER);
    }
    let mut desc = vec![0u8; desc_len];
    g.mem.read(desc_at, &mut desc)?;
    let queue = object(g, "IDStorageQueueX", &QUEUE, 18, 0)?;
    let error_event = g.win.new_handle(Object::Event { signaled: false, manual: true });
    g.win.dstorage.queues.insert(queue, Queue { desc, failure: 0, requests: 0, error_event });
    g.mem.write_u64(out, queue)?;
    Ok(0)
}

fn IDStorageFactoryX_CreateQueue(g: &mut Guest) -> H {
    create_queue(g, 0x18)
}

fn IDStorageFactoryX_CreateQueue1(g: &mut Guest) -> H {
    create_queue(g, 0x20)
}

/// OpenFile(this, path, riid, out): opened as the title's own CreateFileW would, to read.
fn IDStorageFactoryX_OpenFile(g: &mut Guest) -> H {
    let (path, out) = (g.arg(1)?, g.arg(3)?);
    if path == 0 || out == 0 {
        return Ok(E_POINTER);
    }
    g.mem.write_u64(out, 0)?;
    let name = g.mem.wstr(path, 0x8000)?;
    const GENERIC_READ: u32 = 0x8000_0000;
    const OPEN_EXISTING: u32 = 3;
    match files::open(g, &name, GENERIC_READ, OPEN_EXISTING) {
        Ok(handle) => {
            let file = object(g, "IDStorageFileX", &FILE, 12, handle)?;
            g.mem.write_u64(out, file)?;
            Ok(0)
        }
        Err(code) => {
            g.log.push(format!("DirectStorage: OpenFile {name}: error {code}"));
            // HRESULT_FROM_WIN32
            Ok(0x8007_0000 | code as u64 & 0xFFFF)
        }
    }
}

/// CreateStatusArray(this, capacity, name, riid, out): every entry pending.
fn IDStorageFactoryX_CreateStatusArray(g: &mut Guest) -> H {
    let (capacity, out) = (g.arg(1)? as u32, g.arg(4)?);
    if out == 0 {
        return Ok(E_POINTER);
    }
    let array = object(g, "IDStorageStatusArrayX", &STATUS_ARRAY, 10, 0)?;
    g.win.dstorage.status_arrays.insert(array, vec![E_PENDING; capacity as usize]);
    g.mem.write_u64(out, array)?;
    Ok(0)
}

/// The file handle behind an IDStorageFileX (its host word), when it is still open.
fn file_handle(g: &Guest, file: u64) -> Option<u64> {
    let handle = g.mem.read_u64(file.checked_add(8)?).ok()?;
    matches!(g.win.handles.get(&handle), Some(Object::File(_))).then_some(handle)
}

fn IDStorageFileX_Close(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    if let Some(handle) = file_handle(g, this) {
        g.win.handles.remove(&handle);
    }
    Ok(0)
}

fn IDStorageFileX_GetHandle(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(file_handle(g, this).unwrap_or(u64::MAX))
}

/// GetId: a number of its own per open file.
fn IDStorageFileX_GetId(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(file_handle(g, this).map_or(0, |h| h & 0xFFFF_FFFF))
}

/// Carry out the request at `request`: 0, or why not.
fn run_request(g: &mut Guest, request: u64) -> Result<u32, Stop> {
    let options = g.mem.read_u64(request)?;
    let (destination, destination_size) = (g.mem.read_u64(request + 8)?, g.mem.read_u32(request + 0x18)?);
    let (source, offset, size) = (g.mem.read_u64(request + 0x20)?, g.mem.read_u64(request + 0x28)?, g.mem.read_u32(request + 0x30)?);
    if options & NOT_A_PLAIN_COPY != 0 {
        let kind = options & NOT_A_PLAIN_COPY;
        if !g.win.dstorage.refused.contains(&kind) {
            g.win.dstorage.refused.push(kind);
            g.log.push(format!("DirectStorage: a request with options {options:#x} (decompression, texture formats or physical pages) is refused"));
        }
        return Ok(E_NOTIMPL);
    }
    if size > destination_size {
        return Ok(E_DESTINATION_SIZE);
    }
    let path = if options & SOURCE_MEMORY != 0 {
        None
    } else {
        let Some(handle) = file_handle(g, source) else { return Ok(E_FILE_HANDLE) };
        match g.win.handles.get(&handle) {
            Some(Object::File(f)) => Some(f.path.clone()),
            _ => return Ok(E_FILE_HANDLE),
        }
    };
    // in pieces: a request may be gigabytes
    let mut buffer = vec![0u8; (size as usize).min(1 << 20)];
    let mut done = 0u64;
    while done < size as u64 {
        let n = buffer.len().min((size as u64 - done) as usize);
        match &path {
            Some(path) => match g.vfs.read_at(path, offset + done, &mut buffer[..n]) {
                Some(read) if read == n => {}
                Some(_) => return Ok(E_END_OF_FILE),
                None => return Ok(E_FILE_HANDLE),
            },
            None => {
                if g.mem.read(source + done, &mut buffer[..n]).is_err() {
                    return Ok(E_ACCESS_VIOLATION);
                }
            }
        }
        if g.mem.write(destination + done, &buffer[..n]).is_err() {
            return Ok(E_ACCESS_VIOLATION);
        }
        done += n as u64;
    }
    Ok(0)
}

/// EnqueueRequest(this, request)
fn IDStorageQueueX_EnqueueRequest(g: &mut Guest) -> H {
    let (this, request) = (g.arg(0)?, g.arg(1)?);
    let result = if request == 0 { E_INVALIDARG } else { run_request(g, request)? };
    if let Some(q) = g.win.dstorage.queues.get_mut(&this) {
        q.requests += 1;
        if result != 0 && q.failure == 0 {
            q.failure = result;
            let event = q.error_event;
            g.win.handles.insert(event, Object::Event { signaled: true, manual: true });
        }
    }
    Ok(0)
}

/// EnqueueStatus(this, array, index): the entry gets the result of the requests before it.
fn IDStorageQueueX_EnqueueStatus(g: &mut Guest) -> H {
    let (this, array, index) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32);
    let failure = g.win.dstorage.queues.get_mut(&this).map_or(0, |q| std::mem::take(&mut q.failure));
    if let Some(entry) = g.win.dstorage.status_arrays.get_mut(&array).and_then(|a| a.get_mut(index as usize)) {
        *entry = failure;
    }
    Ok(0)
}

/// EnqueueSignal(this, fence, value): the requests before it are done, so the fence is
/// signaled now (ID3D12Fence::Signal, slot 10 of its vtable).
fn IDStorageQueueX_EnqueueSignal(g: &mut Guest) -> H {
    let (fence, value) = (g.arg(1)?, g.arg(2)?);
    if fence == 0 {
        return Ok(0);
    }
    let Ok(signal) = g.mem.read_u64(fence).and_then(|vtable| g.mem.read_u64(vtable + 8 * 10)) else { return Ok(0) };
    if signal != 0 {
        g.call(signal, &[fence, value])?;
    }
    Ok(0)
}

fn IDStorageQueueX_GetErrorEvent(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(g.win.dstorage.queues.get(&this).map_or(0, |q| q.error_event))
}

/// RetrieveErrorRecord(this, record out): 0x290 bytes, the failure count first.
fn IDStorageQueueX_RetrieveErrorRecord(g: &mut Guest) -> H {
    let out = g.arg(1)?;
    if out != 0 {
        g.mem.write(out, &[0u8; 0x290])?;
    }
    Ok(0)
}

/// Query(this, info out): the description, then the empty slots (u16 @0x18) and the requests
/// until the queue submits by itself (u16 @0x1A). Nothing is ever waiting here.
fn IDStorageQueueX_Query(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(1)?);
    let Some(q) = g.win.dstorage.queues.get(&this) else { return Ok(0) };
    let mut info = [0u8; 0x1C];
    info[..0x18].copy_from_slice(&q.desc[..0x18]);
    let capacity = [q.desc[8], q.desc[9]];
    info[0x18..0x1A].copy_from_slice(&capacity);
    info[0x1A..0x1C].copy_from_slice(&capacity);
    g.mem.write(out, &info)?;
    Ok(0)
}

fn IDStorageQueueX_Close(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    g.win.dstorage.queues.remove(&this);
    Ok(0)
}

fn IDStorageStatusArrayX_IsComplete(g: &mut Guest) -> H {
    let (this, index) = (g.arg(0)?, g.arg(1)? as u32);
    let entry = g.win.dstorage.status_arrays.get(&this).and_then(|a| a.get(index as usize));
    Ok(entry.is_some_and(|&e| e != E_PENDING) as u64)
}

fn IDStorageStatusArrayX_GetHResult(g: &mut Guest) -> H {
    let (this, index) = (g.arg(0)?, g.arg(1)? as u32);
    let entry = g.win.dstorage.status_arrays.get(&this).and_then(|a| a.get(index as usize));
    Ok(entry.copied().unwrap_or(E_INVALIDARG) as u64)
}

fn ok(_: &mut Guest) -> H {
    Ok(0)
}

fn one(_: &mut Guest) -> H {
    Ok(1)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; DStorageGetFactory, IDStorage_QueryInterface, IDStorageFactoryX_CreateQueue, IDStorageFactoryX_CreateQueue1, IDStorageFactoryX_OpenFile, IDStorageFactoryX_CreateStatusArray);
    register!(map; IDStorageFileX_Close, IDStorageFileX_GetHandle, IDStorageFileX_GetId);
    register!(map; IDStorageQueueX_EnqueueRequest, IDStorageQueueX_EnqueueStatus, IDStorageQueueX_EnqueueSignal, IDStorageQueueX_GetErrorEvent, IDStorageQueueX_RetrieveErrorRecord);
    register!(map; IDStorageQueueX_Query, IDStorageQueueX_Close, IDStorageStatusArrayX_IsComplete, IDStorageStatusArrayX_GetHResult);
    // the objects live as long as the title; requests are done when enqueued, so there is
    // nothing to submit or to cancel
    for name in ["IDStorage_AddRef", "IDStorage_Release"] {
        map.insert(name, one as Handler);
    }
    for name in ["IDStorageFactoryX_SetCpuAffinity", "IDStorageFactoryX_SetDebugFlags", "IDStorageFactoryX_SetStagingBufferSize", "IDStorageFactoryX_#9", "IDStorageQueueX_Submit", "IDStorageQueueX_CancelRequestsWithTag"] {
        map.insert(name, ok as Handler);
    }
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX, RSP};
    use xcl_emu::mem::{PROT_R, PROT_W};

    use super::*;
    use crate::vfs::Source;

    /// A guest whose game folder has `pack\levels.rpf`: 1 TB read on demand, byte i = i % 251.
    fn guest() -> (Guest, u64) {
        let mut g = Guest::new();
        let stack = g.mem.alloc(0x4000, PROT_R | PROT_W).unwrap();
        g.add_thread(stack, stack + 0x4000, 0, 0, crate::SLOT_THREAD_RETURN).unwrap();
        g.cpu = g.threads[0].cpu.clone();
        let read = |offset: u64, buffer: &mut [u8]| {
            for (i, b) in buffer.iter_mut().enumerate() {
                *b = ((offset + i as u64) % 251) as u8;
            }
            true
        };
        g.vfs.add_game_source("pack\\levels.rpf", Source { size: 1 << 40, read: Box::new(read) });
        let page = g.mem.alloc(0x4000, PROT_R | PROT_W).unwrap();
        (g, page)
    }

    fn call(g: &mut Guest, f: Handler, args: &[u64]) -> u64 {
        for (i, &value) in args.iter().enumerate() {
            match i {
                0..4 => g.cpu.r[[RCX, RDX, 8, 9][i]] = value,
                _ => g.mem.write_u64(g.cpu.r[RSP] + 8 + 8 * i as u64, value).unwrap(),
            }
        }
        f(g).unwrap()
    }

    /// The factory, a queue, a status array of 4 and the open archive.
    fn set_up(g: &mut Guest, page: u64) -> (u64, u64, u64) {
        assert_eq!(call(g, DStorageGetFactory, &[0, page]), 0);
        let factory = g.mem.read_u64(page).unwrap();
        // every slot of the factory is a function
        let vtable = g.mem.read_u64(factory).unwrap();
        assert!((0..16).all(|i| g.mem.read_u64(vtable + 8 * i).unwrap() != 0));
        g.mem.write(page + 0x100, &[0u8; 0x20]).unwrap();
        g.mem.write_u16(page + 0x108, 64).unwrap();
        assert_eq!(call(g, IDStorageFactoryX_CreateQueue1, &[factory, page + 0x100, 0, page]), 0);
        let queue = g.mem.read_u64(page).unwrap();
        assert_eq!(call(g, IDStorageFactoryX_CreateStatusArray, &[factory, 4, 0, 0, page]), 0);
        let array = g.mem.read_u64(page).unwrap();
        let name: Vec<u8> = "G:\\pack\\levels.rpf".encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        g.mem.write(page + 0x200, &name).unwrap();
        assert_eq!(call(g, IDStorageFactoryX_OpenFile, &[factory, page + 0x200, 0, page]), 0);
        let file = g.mem.read_u64(page).unwrap();
        (queue, array, file)
    }

    fn request(g: &mut Guest, at: u64, options: u64, destination: (u64, u32), source: u64, offset: u64, size: u32) {
        g.mem.write(at, &[0u8; 0x48]).unwrap();
        g.mem.write_u64(at, options).unwrap();
        g.mem.write_u64(at + 8, destination.0).unwrap();
        g.mem.write_u32(at + 0x18, destination.1).unwrap();
        g.mem.write_u64(at + 0x20, source).unwrap();
        g.mem.write_u64(at + 0x28, offset).unwrap();
        g.mem.write_u32(at + 0x30, size).unwrap();
    }

    #[test]
    fn a_request_reads_a_range_of_an_archive_into_the_titles_memory() {
        let (mut g, page) = guest();
        let (queue, array, file) = set_up(&mut g, page);
        let (req, to) = (page + 0x400, page + 0x1000);
        // 600 GB into the file
        let far = 600u64 << 30;
        request(&mut g, req, 0, (to, 0x100), file, far, 0x100);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_IsComplete, &[array, 0]), 0);
        call(&mut g, IDStorageQueueX_EnqueueRequest, &[queue, req]);
        call(&mut g, IDStorageQueueX_EnqueueStatus, &[queue, array, 0]);
        // Submit: nothing is left to do
        call(&mut g, ok, &[queue]);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_IsComplete, &[array, 0]), 1);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_GetHResult, &[array, 0]), 0);
        let got: [u8; 0x100] = g.mem.read_n(to).unwrap();
        assert!(got.iter().enumerate().all(|(i, &b)| b == ((far + i as u64) % 251) as u8));
        // the other entries are still pending, and one past the array is an error
        assert_eq!(call(&mut g, IDStorageStatusArrayX_GetHResult, &[array, 1]), E_PENDING as u64);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_GetHResult, &[array, 9]), E_INVALIDARG as u64);
    }

    #[test]
    fn a_status_entry_gets_the_first_failure_since_the_previous_one() {
        let (mut g, page) = guest();
        let (queue, array, file) = set_up(&mut g, page);
        let (req, to) = (page + 0x400, page + 0x1000);
        // past the end of the file, then a destination that is too small, then a good one
        request(&mut g, req, 0, (to, 16), file, (1 << 40) - 8, 16);
        call(&mut g, IDStorageQueueX_EnqueueRequest, &[queue, req]);
        request(&mut g, req, 0, (to, 8), file, 0, 16);
        call(&mut g, IDStorageQueueX_EnqueueRequest, &[queue, req]);
        request(&mut g, req, 0, (to, 16), file, 0, 16);
        call(&mut g, IDStorageQueueX_EnqueueRequest, &[queue, req]);
        call(&mut g, IDStorageQueueX_EnqueueStatus, &[queue, array, 2]);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_GetHResult, &[array, 2]), E_END_OF_FILE as u64);
        let event = call(&mut g, IDStorageQueueX_GetErrorEvent, &[queue]);
        assert!(matches!(g.win.handles.get(&event), Some(Object::Event { signaled: true, .. })));
        // … and the next entry starts clean
        call(&mut g, IDStorageQueueX_EnqueueRequest, &[queue, req]);
        call(&mut g, IDStorageQueueX_EnqueueStatus, &[queue, array, 3]);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_GetHResult, &[array, 3]), 0);
        // a destination outside the title's memory, a closed file, a compressed request
        request(&mut g, req, 0, (0x10, 16), file, 0, 16);
        call(&mut g, IDStorageQueueX_EnqueueRequest, &[queue, req]);
        call(&mut g, IDStorageQueueX_EnqueueStatus, &[queue, array, 0]);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_GetHResult, &[array, 0]), E_ACCESS_VIOLATION as u64);
        request(&mut g, req, 1, (to, 16), file, 0, 16);
        call(&mut g, IDStorageQueueX_EnqueueRequest, &[queue, req]);
        call(&mut g, IDStorageQueueX_EnqueueStatus, &[queue, array, 0]);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_GetHResult, &[array, 0]), E_NOTIMPL as u64);
        assert_eq!(g.log.iter().filter(|l| l.contains("is refused")).count(), 1);
        call(&mut g, IDStorageFileX_Close, &[file]);
        request(&mut g, req, 0, (to, 16), file, 0, 16);
        call(&mut g, IDStorageQueueX_EnqueueRequest, &[queue, req]);
        call(&mut g, IDStorageQueueX_EnqueueStatus, &[queue, array, 0]);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_GetHResult, &[array, 0]), E_FILE_HANDLE as u64);
    }

    #[test]
    fn a_memory_request_copies_and_a_missing_file_is_an_error_code() {
        let (mut g, page) = guest();
        let (queue, array, _) = set_up(&mut g, page);
        let (req, from, to) = (page + 0x400, page + 0x800, page + 0x1000);
        g.mem.write(from, b"from memory").unwrap();
        request(&mut g, req, SOURCE_MEMORY, (to, 11), from, 0, 11);
        call(&mut g, IDStorageQueueX_EnqueueRequest, &[queue, req]);
        call(&mut g, IDStorageQueueX_EnqueueStatus, &[queue, array, 0]);
        assert_eq!(call(&mut g, IDStorageStatusArrayX_GetHResult, &[array, 0]), 0);
        assert_eq!(&g.mem.read_n::<11>(to).unwrap(), b"from memory");
        let factory = call(&mut g, DStorageGetFactory, &[0, page]);
        assert_eq!(factory, 0);
        let factory = g.mem.read_u64(page).unwrap();
        let name: Vec<u8> = "G:\\pack\\nothing.rpf".encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        g.mem.write(page + 0x200, &name).unwrap();
        // HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND)
        assert_eq!(call(&mut g, IDStorageFactoryX_OpenFile, &[factory, page + 0x200, 0, page]), 0x8007_0002);
        assert_eq!(g.mem.read_u64(page).unwrap(), 0);
        // Query: the description back, and all its slots free
        let info = page + 0x600;
        call(&mut g, IDStorageQueueX_Query, &[queue, info]);
        assert_eq!((g.mem.read_u16(info + 8).unwrap(), g.mem.read_u16(info + 0x18).unwrap()), (64, 64));
    }
}
