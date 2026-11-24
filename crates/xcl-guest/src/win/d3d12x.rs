//! D3D12.X on the guest, in software: the console's device as far as it is about **memory**.
//! Nothing is rendered – drawing needs a GPU back end (WebGPU) that does not exist yet – but
//! what a title sees in its own memory is real:
//!
//! * the console has unified memory: a resource's GPU address is its address in the title's
//!   address space. Committed resources and heaps get graphics memory (`kernelx`'s region),
//!   placed resources lie in their heap or, with CreatePlacedResourceX, in memory the title
//!   allocated itself; Map returns that address.
//! * command lists record what they are told; a queue runs the lists it is given in order,
//!   behind the title's back as a GPU would (`pump`, at every turn of the scheduler): copies
//!   between buffers, the console's memory commands (writes, fills, copies, timestamps) and
//!   waits for a value in memory, which hold the queue until another thread writes it. A fence
//!   is signalled when the queue gets there.
//! * draws, pipelines, views and everything else are accepted and do nothing.
//!
//! An object is a small block of guest memory whose first word points at its vtable; what the
//! host knows about it is kept here by its address. Every vtable slot is a host function named
//! `<Interface>::#<slot>`: one that is not implemented shows up under that name like any
//! missing function. Layouts follow XCL's Windows runtime (`xcl::hle::d3d12x`): the device has
//! the PC's methods at 0–43 and console extensions from 44; the graphics command list has the
//! PC's up to 27, the later ones 4 slots further, and console methods among them.

#![allow(non_snake_case)]

use std::collections::{HashMap, VecDeque};

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const VTBL_SLOTS: usize = 256;
const DLL: &str = "d3d12_xs.dll";
const E_INVALIDARG: u64 = 0x8007_0057;
const E_NOINTERFACE: u64 = 0x8000_4002;
const E_NOTIMPL: u64 = 0x8000_4001;
const E_ACCESSDENIED: u64 = 0x8007_0005;
const E_OUTOFMEMORY: u64 = 0x8007_000E;
const S_FALSE: u64 = 1;
/// the console's DMA (copy) engine as a queue / list type
const TYPE_DMA: u32 = 0x10;
const GRAIN: u64 = 0x1_0000;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
    Device,
    CommandQueue,
    CommandAllocator,
    CommandList,
    /// the DMA engine's list: a vtable of its own (Close, Reset and the copies)
    DmaCommandList,
    Fence,
    DescriptorHeap,
    Resource,
    Heap,
    PipelineState,
    RootSignature,
    QueryHeap,
    CommandSignature,
    Blob,
}

const KINDS: [Kind; 14] = [
    Kind::Device,
    Kind::CommandQueue,
    Kind::CommandAllocator,
    Kind::CommandList,
    Kind::DmaCommandList,
    Kind::Fence,
    Kind::DescriptorHeap,
    Kind::Resource,
    Kind::Heap,
    Kind::PipelineState,
    Kind::RootSignature,
    Kind::QueryHeap,
    Kind::CommandSignature,
    Kind::Blob,
];

impl Kind {
    fn iface(self) -> &'static str {
        match self {
            Kind::Device => "ID3D12Device",
            Kind::CommandQueue => "ID3D12CommandQueue",
            Kind::CommandAllocator => "ID3D12CommandAllocator",
            Kind::CommandList => "ID3D12GraphicsCommandList",
            Kind::DmaCommandList => "ID3D12XboxDmaCommandList",
            Kind::Fence => "ID3D12Fence",
            Kind::DescriptorHeap => "ID3D12DescriptorHeap",
            Kind::Resource => "ID3D12Resource",
            Kind::Heap => "ID3D12Heap",
            Kind::PipelineState => "ID3D12PipelineState",
            Kind::RootSignature => "ID3D12RootSignature",
            Kind::QueryHeap => "ID3D12QueryHeap",
            Kind::CommandSignature => "ID3D12CommandSignature",
            Kind::Blob => "ID3DBlob",
        }
    }
}

/// What the host keeps for an object.
#[derive(Debug, Clone)]
pub struct Object {
    pub kind: Kind,
    pub refs: u32,
    /// fence value / descriptor heap base / resource or heap address
    pub value: u64,
    /// the memory of a resource, heap or blob: (guest address, size)
    pub mem: (u64, u64),
    /// the description it was made from (resource 56 bytes, heap 40, descriptor heap 16)
    pub desc: [u8; 64],
}

/// What a command list records and a queue runs.
#[derive(Debug, Clone)]
pub enum Command {
    Copy { to: u64, from: u64, bytes: u64 },
    Fill { at: u64, bytes: u64, value: u32 },
    Write32 { at: u64, value: u32 },
    Write64 { at: u64, value: u64 },
    /// the queue's clock, written when the queue gets here
    Timestamp { at: u64 },
    /// hold the queue until (`size` bytes at `at`) & mask compares to value (D3D12_COMPARISON_FUNC)
    Wait { at: u64, size: u8, func: u32, value: u64, mask: u64 },
    Signal { fence: u64, value: u64 },
}

#[derive(Default)]
pub struct State {
    pub objects: HashMap<u64, Object>,
    vtables: HashMap<Kind, u64>,
    device: u64,
    pub created: u64,
    /// what each command list has recorded since its Reset (or its last KickoffX)
    lists: HashMap<u64, Vec<Command>>,
    /// what each queue still has to run, by queue (a list that was kicked off has a stream of
    /// its own, by its address)
    streams: Vec<(u64, VecDeque<Command>)>,
    /// (fence, value, event): events to signal when a fence reaches a value
    fence_events: Vec<(u64, u64, u64)>,
    /// RegisterPagePoolX: index → (address, pages)
    page_pools: Vec<Option<(u64, u32)>>,
}

/// Where XCL's Windows runtime keeps memory that the CPU reads back (its "coherent window",
/// the fourth 4 GiB of the graphics region), and up to which size a buffer goes there.
const READBACK_WINDOW: u64 = 0x7_0000_0000;
const READBACK_LIMIT: u64 = 64 << 20;

/// Graphics memory for a resource or heap: whole 64 KiB pages of the graphics region
/// (`readback`: from the read-back window, as XCL places such buffers).
fn graphics_memory(g: &mut Guest, size: u64, readback: bool) -> Option<u64> {
    let size = size.max(1).checked_next_multiple_of(GRAIN)?;
    let at = match readback && size <= READBACK_LIMIT {
        true => g.win.xmem.take_from(READBACK_WINDOW, size, GRAIN).or_else(|| g.win.xmem.take(size, GRAIN))?,
        false => g.win.xmem.take(size, GRAIN)?,
    };
    if g.mem.map(at, size, xcl_emu::mem::PROT_R | xcl_emu::mem::PROT_W).is_err() {
        g.win.xmem.give(at);
        return None;
    }
    Some(at)
}

fn new_object(g: &mut Guest, kind: Kind) -> Result<u64, Stop> {
    let vtable = match g.win.d3d.vtables.get(&kind) {
        Some(&v) => v,
        None => {
            let v = g.heap_alloc(8 * VTBL_SLOTS as u64)?;
            for i in 0..VTBL_SLOTS {
                let slot = g.import_address(DLL, &format!("{}::#{i}", kind.iface()));
                g.mem.write_u64(v + 8 * i as u64, slot)?;
            }
            g.win.d3d.vtables.insert(kind, v);
            v
        }
    };
    let at = g.heap_alloc(16)?;
    g.mem.write_u64(at, vtable)?;
    g.win.d3d.objects.insert(at, Object { kind, refs: 1, value: 0, mem: (0, 0), desc: [0; 64] });
    g.win.d3d.created += 1;
    Ok(at)
}

/// A new object of `kind` through the out pointer in argument `out`.
fn out_object(g: &mut Guest, out: usize, kind: Kind) -> H {
    let ppv = g.arg(out)?;
    if ppv == 0 {
        return Ok(S_FALSE);
    }
    let o = new_object(g, kind)?;
    g.mem.write_u64(ppv, o)?;
    Ok(0)
}

fn device(g: &mut Guest) -> Result<u64, Stop> {
    if g.win.d3d.device == 0 {
        g.win.d3d.device = new_object(g, Kind::Device)?;
        g.log.push("D3D12.X: the software device – memory only, graphics are not rendered".into());
    }
    Ok(g.win.d3d.device)
}

fn object<'a>(g: &'a mut Guest, at: u64) -> Option<&'a mut Object> {
    g.win.d3d.objects.get_mut(&at)
}

// --- common slots ---------------------------------------------------------------------------

fn QueryInterface(g: &mut Guest) -> H {
    let (this, ppv) = (g.arg(0)?, g.arg(2)?);
    if ppv == 0 {
        return Ok(E_INVALIDARG);
    }
    if let Some(o) = object(g, this) {
        o.refs += 1;
    }
    g.mem.write_u64(ppv, this)?;
    Ok(0)
}

fn AddRef(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(object(g, this).map_or(1, |o| {
        o.refs += 1;
        o.refs
    }) as u64)
}

/// Objects are never freed: a title may still reach one through a slot guessed wrong.
fn Release(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(object(g, this).map_or(0, |o| {
        o.refs = o.refs.saturating_sub(1);
        o.refs
    }) as u64)
}

fn Object_GetPrivateData(g: &mut Guest) -> H {
    let size = g.arg(2)?;
    if size != 0 {
        g.mem.write_u32(size, 0)?;
    }
    Ok(0x887A_0002) // DXGI_ERROR_NOT_FOUND
}

fn DeviceChild_GetDevice(g: &mut Guest) -> H {
    let ppv = g.arg(2)?;
    if ppv == 0 {
        return Ok(E_INVALIDARG);
    }
    let d = device(g)?;
    g.mem.write_u64(ppv, d)?;
    Ok(0)
}

fn ret_ok(_: &mut Guest) -> H {
    Ok(0)
}

fn ret_not_implemented(_: &mut Guest) -> H {
    Ok(E_NOTIMPL)
}

/// A method that returns a structure through a hidden pointer (argument 1): `size` zero bytes.
fn hidden_zero(g: &mut Guest, size: usize) -> H {
    let ret = g.arg(1)?;
    if ret != 0 {
        g.mem.write(ret, &vec![0u8; size])?;
    }
    Ok(ret)
}

fn Hidden_ZeroDesc16(g: &mut Guest) -> H {
    hidden_zero(g, 16)
}

// --- device -----------------------------------------------------------------------------------

fn Device_GetNodeCount(_: &mut Guest) -> H {
    Ok(1)
}

macro_rules! creates {
    ($name:ident, $out:expr, $kind:expr) => {
        fn $name(g: &mut Guest) -> H {
            out_object(g, $out, $kind)
        }
    };
}

creates!(Device_CreateCommandQueue, 3, Kind::CommandQueue);
creates!(Device_CreateCommandAllocator, 3, Kind::CommandAllocator);
creates!(Device_CreatePipelineState, 3, Kind::PipelineState);
// CreateGraphicsPipelineStateX / CreateDerivedPipelineStateX(…, riid, out)
creates!(Device_CreatePipelineStateX, 5, Kind::PipelineState);
creates!(Device_CreateRootSignature, 5, Kind::RootSignature);
creates!(Device_CreateQueryHeap, 3, Kind::QueryHeap);
creates!(Device_CreateCommandSignature, 4, Kind::CommandSignature);

/// CheckFeatureSupport(feature, data, size): nothing is supported – the whole structure is
/// zeroed, however long the title says it is; D3D12_OPTIONS5 (27) gets the configured
/// raytracing tier at +8.
fn Device_CheckFeatureSupport(g: &mut Guest) -> H {
    let (feature, data, size) = (g.arg(1)? as u32, g.arg(2)?, g.arg(3)? as u32);
    if data != 0 {
        g.mem.write(data, &vec![0u8; size.min(4096) as usize])?;
        if feature == 27 && size >= 12 {
            let tier = xcl::state::config().raytracing_tier_override().unwrap_or(0);
            g.mem.write_u32(data + 8, tier)?;
        }
    }
    Ok(0)
}

/// CreateDescriptorHeap(desc { type, count, flags, node mask }, riid, out): the handles are a
/// range of guest memory of its own (32 bytes a descriptor).
fn Device_CreateDescriptorHeap(g: &mut Guest) -> H {
    let (desc, ppv) = (g.arg(1)?, g.arg(3)?);
    if ppv == 0 {
        return Ok(S_FALSE);
    }
    let o = new_object(g, Kind::DescriptorHeap)?;
    if desc != 0 {
        let mut d = [0u8; 16];
        g.mem.read(desc, &mut d)?;
        let count = u32::from_le_bytes(d[4..8].try_into().unwrap()) as u64;
        let range = g.mem.alloc((count.max(1) * 32).min(64 << 20), xcl_emu::mem::PROT_R | xcl_emu::mem::PROT_W).unwrap_or(0);
        if let Some(obj) = object(g, o) {
            obj.desc[..16].copy_from_slice(&d);
            obj.value = range;
        }
    }
    g.mem.write_u64(ppv, o)?;
    Ok(0)
}

fn Device_GetDescriptorHandleIncrementSize(_: &mut Guest) -> H {
    Ok(32)
}

/// Bytes a resource of this D3D12_RESOURCE_DESC takes, roughly (dimension @0, width @16,
/// height @24, depth @28): a buffer its width, a texture 4 bytes a texel with its mips.
fn approx_size(g: &Guest, desc: u64) -> Result<u64, Stop> {
    if desc == 0 {
        return Ok(0x1000);
    }
    let width = g.mem.read_u64(desc + 16)?;
    if g.mem.read_u32(desc)? == 1 {
        return Ok(width.max(1));
    }
    let height = g.mem.read_u32(desc + 24)?.max(1) as u64;
    let depth = g.mem.read_u16(desc + 28)?.max(1) as u64;
    Ok((width.max(1).saturating_mul(height).saturating_mul(depth).saturating_mul(16) / 3).min(256 << 20))
}

/// A resource of `desc` at `address` (`None`: graphics memory of its own), through the out
/// pointer in argument `out`.
fn new_resource(g: &mut Guest, desc: u64, address: Option<u64>, readback: bool, out: usize) -> H {
    let ppv = g.arg(out)?;
    if ppv == 0 {
        return Ok(S_FALSE);
    }
    let size = approx_size(g, desc)?;
    let mut d = [0u8; 64];
    if desc != 0 {
        g.mem.read(desc, &mut d[..56])?;
    }
    let at = match address {
        Some(at) => at,
        None => match graphics_memory(g, size, readback) {
            Some(at) => at,
            None => return Ok(E_OUTOFMEMORY),
        },
    };
    let o = new_object(g, Kind::Resource)?;
    if let Some(obj) = object(g, o) {
        obj.desc = d;
        obj.mem = (at, size);
        obj.value = at;
    }
    g.mem.write_u64(ppv, o)?;
    Ok(0)
}

/// CreateCommittedResource(heap properties, heap flags, desc, state, clear value, riid, out)
fn Device_CreateCommittedResource(g: &mut Guest) -> H {
    let (properties, desc) = (g.arg(1)?, g.arg(3)?);
    // D3D12_HEAP_TYPE_READBACK
    let readback = properties != 0 && g.mem.read_u32(properties)? == 3;
    new_resource(g, desc, None, readback, 7)
}

/// CreateHeap(desc { size @0, … }, riid, out): graphics memory that resources are placed in
fn Device_CreateHeap(g: &mut Guest) -> H {
    let (desc, ppv) = (g.arg(1)?, g.arg(3)?);
    if ppv == 0 || desc == 0 {
        return Ok(S_FALSE);
    }
    let size = g.mem.read_u64(desc)?;
    let mut d = [0u8; 64];
    g.mem.read(desc, &mut d[..40])?;
    let Some(at) = graphics_memory(g, size, false) else { return Ok(E_OUTOFMEMORY) };
    let o = new_object(g, Kind::Heap)?;
    if let Some(obj) = object(g, o) {
        obj.desc = d;
        obj.mem = (at, size);
        obj.value = at;
    }
    g.mem.write_u64(ppv, o)?;
    Ok(0)
}

/// CreatePlacedResource(heap, offset, desc, state, clear value, riid, out): in the heap's memory
fn Device_CreatePlacedResource(g: &mut Guest) -> H {
    let (heap, offset, desc) = (g.arg(1)?, g.arg(2)?, g.arg(3)?);
    let Some(base) = object(g, heap).filter(|o| o.kind == Kind::Heap).map(|o| o.value) else { return Ok(E_INVALIDARG) };
    new_resource(g, desc, Some(base + offset), false, 7)
}

/// device #44 CreatePlacedResourceX(address, desc, state, clear value, riid, out): a resource
/// in memory the title allocated (which stays the title's when the resource goes)
fn Device_CreatePlacedResourceX(g: &mut Guest) -> H {
    let (address, desc) = (g.arg(1)?, g.arg(2)?);
    if address == 0 {
        return Ok(E_INVALIDARG);
    }
    new_resource(g, desc, Some(address), false, 6)
}

/// CreateReservedResource(desc, state, clear value, riid, out)
fn Device_CreateReservedResource(g: &mut Guest) -> H {
    let desc = g.arg(1)?;
    new_resource(g, desc, None, false, 5)
}

/// CreateShaderResourceView(resource, desc, descriptor): views are not used (nothing is drawn)
/// but for one whose descriptor a title reads itself: a raytracing acceleration structure
/// (view dimension 11, its address at +16) is the console's descriptor words – the address
/// from bit 8 on, bit 63 set, then ~0 and 0.
fn Device_CreateShaderResourceView(g: &mut Guest) -> H {
    let (desc, handle) = (g.arg(2)?, g.arg(3)?);
    if desc != 0 && handle != 0 && g.mem.read_u32(desc + 4)? == 11 {
        let address = g.mem.read_u64(desc + 16)?;
        let words = [(address >> 8) as u32, ((address >> 40) as u32 & 0xFF) | 1 << 31, u32::MAX, 0];
        for (i, w) in words.into_iter().enumerate() {
            g.mem.write_u32(handle + 4 * i as u64, w)?;
        }
    }
    Ok(0)
}

/// A command list of the type given: the DMA engine's has an interface of its own.
fn new_list(g: &mut Guest, kind: u32, out: usize) -> H {
    let ppv = g.arg(out)?;
    if ppv == 0 {
        return Ok(S_FALSE);
    }
    let o = new_object(g, if kind == TYPE_DMA { Kind::DmaCommandList } else { Kind::CommandList })?;
    g.win.d3d.lists.insert(o, Vec::new());
    g.mem.write_u64(ppv, o)?;
    Ok(0)
}

/// CreateCommandList(node mask, type, allocator, initial pipeline, riid, out)
fn Device_CreateCommandList(g: &mut Guest) -> H {
    let kind = g.arg(2)? as u32;
    new_list(g, kind, 6)
}

/// device #73 CreateCommandListX(params { type @0, node mask @8 }, allocator, initial pipeline,
/// riid, out)
fn Device_CreateCommandListX(g: &mut Guest) -> H {
    let params = g.arg(1)?;
    let kind = if params == 0 { 0 } else { g.mem.read_u32(params)? };
    new_list(g, kind, 5)
}

/// device #74 RegisterPagePoolX(address, pages, index out)
fn Device_RegisterPagePoolX(g: &mut Guest) -> H {
    let (address, pages, out) = (g.arg(1)?, g.arg(2)? as u32, g.arg(3)?);
    if out == 0 || pages == 0 {
        return Ok(E_INVALIDARG);
    }
    let pools = &mut g.win.d3d.page_pools;
    let index = pools.iter().position(Option::is_none).unwrap_or_else(|| {
        pools.push(None);
        pools.len() - 1
    });
    pools[index] = Some((address, pages));
    g.mem.write_u32(out, index as u32)?;
    Ok(0)
}

/// device #75 UnregisterPagePoolX(index)
fn Device_UnregisterPagePoolX(g: &mut Guest) -> H {
    let index = g.arg(1)? as usize;
    Ok(match g.win.d3d.page_pools.get_mut(index) {
        Some(pool) if pool.is_some() => {
            *pool = None;
            0
        }
        _ => E_INVALIDARG,
    })
}

fn ret_access_denied(_: &mut Guest) -> H {
    Ok(E_ACCESSDENIED)
}

/// GetResourceAllocationInfo(hidden return { size, alignment }, mask, count, descs)
fn Device_GetResourceAllocationInfo(g: &mut Guest) -> H {
    let (ret, count, descs) = (g.arg(1)?, g.arg(3)? as u32 as u64, g.arg(4)?);
    let mut total = 0u64;
    for i in 0..count {
        total += (approx_size(g, descs + 56 * i)? + 0xFFFF) & !0xFFFF;
    }
    if ret != 0 {
        g.mem.write_u64(ret, total)?;
        g.mem.write_u64(ret + 8, 0x1_0000)?;
    }
    Ok(ret)
}

/// GetCustomHeapProperties(hidden return, mask, heap type)
fn Device_GetCustomHeapProperties(g: &mut Guest) -> H {
    let (ret, kind) = (g.arg(1)?, g.arg(3)? as u32);
    if ret != 0 {
        g.mem.write(ret, &[0u8; 20])?;
        g.mem.write_u32(ret, 4)?; // CUSTOM
        g.mem.write_u32(ret + 4, if kind == 2 || kind == 3 { 3 } else { 1 })?; // CPU page property
        g.mem.write_u32(ret + 8, 1)?; // memory pool L0
    }
    Ok(ret)
}

/// CreateFence(initial value, flags, riid, out)
fn Device_CreateFence(g: &mut Guest) -> H {
    let (initial, ppv) = (g.arg(1)?, g.arg(4)?);
    if ppv == 0 {
        return Ok(S_FALSE);
    }
    let o = new_object(g, Kind::Fence)?;
    if let Some(obj) = object(g, o) {
        obj.value = initial;
    }
    g.mem.write_u64(ppv, o)?;
    Ok(0)
}

/// GetCopyableFootprints(desc, first, count, base offset, layouts, row counts, row sizes, total)
fn Device_GetCopyableFootprints(g: &mut Guest) -> H {
    let (desc, count, base) = (g.arg(1)?, g.arg(3)? as u32 as u64, g.arg(4)?);
    let (layouts, rows, row_sizes, total) = (g.arg(5)?, g.arg(6)?, g.arg(7)?, g.arg(8)?);
    let size = approx_size(g, desc)?;
    let (width, height, format) = if desc == 0 { (1, 1, 0) } else { (g.mem.read_u64(desc + 16)?.max(1), g.mem.read_u32(desc + 24)?.max(1), g.mem.read_u32(desc + 32)?) };
    let row = ((width * 4 + 255) & !255) as u32;
    for i in 0..count {
        if layouts != 0 {
            let l = layouts + 32 * i;
            g.mem.write_u64(l, base)?;
            g.mem.write_u32(l + 8, format)?;
            g.mem.write_u32(l + 12, width as u32)?;
            g.mem.write_u32(l + 16, height)?;
            g.mem.write_u32(l + 20, 1)?;
            g.mem.write_u32(l + 24, row)?;
        }
        if rows != 0 {
            g.mem.write_u32(rows + 4 * i, height)?;
        }
        if row_sizes != 0 {
            g.mem.write_u64(row_sizes + 8 * i, width * 4)?;
        }
    }
    if total != 0 {
        g.mem.write_u64(total, size.max(row as u64 * height as u64))?;
    }
    Ok(0)
}

/// GetAdapterLuid(hidden return)
fn Device_GetAdapterLuid(g: &mut Guest) -> H {
    let ret = g.arg(1)?;
    if ret != 0 {
        g.mem.write_u64(ret, 0x5843_4C00)?;
    }
    Ok(ret)
}

/// Signal the event behind a handle.
fn set_event(g: &mut Guest, handle: u64) {
    if let Some(super::Object::Event { signaled, .. }) = g.win.handles.get_mut(&handle) {
        *signaled = true;
    }
}

// --- queues: what the "GPU" does ----------------------------------------------------------------

/// The commands waiting for the queue (or kicked-off list) `owner`.
fn stream(g: &mut Guest, owner: u64) -> &mut VecDeque<Command> {
    let streams = &mut g.win.d3d.streams;
    let i = match streams.iter().position(|(o, _)| *o == owner) {
        Some(i) => i,
        None => {
            streams.push((owner, VecDeque::new()));
            streams.len() - 1
        }
    };
    &mut streams[i].1
}

/// Run one command. `Ok(false)`: a wait whose value is not there yet.
fn run(g: &mut Guest, c: &Command) -> Result<bool, xcl_emu::mem::Fault> {
    match *c {
        Command::Copy { to, from, bytes } => {
            let mut data = vec![0u8; bytes as usize];
            g.mem.read(from, &mut data)?;
            g.mem.write(to, &data)?;
        }
        Command::Fill { at, bytes, value } => {
            let data: Vec<u8> = value.to_le_bytes().into_iter().cycle().take((bytes & !3) as usize).collect();
            g.mem.write(at, &data)?;
        }
        Command::Write32 { at, value } => g.mem.write_u32(at, value)?,
        Command::Write64 { at, value } => g.mem.write_u64(at, value)?,
        // 100 ns units, never zero
        Command::Timestamp { at } => g.mem.write_u64(at, g.now_ns() / 100 + 1)?,
        Command::Wait { at, size, func, value, mask } => {
            let v = g.mem.read_sized(at, size)? & mask;
            let met = match func {
                1 => false,
                2 => v < value,
                3 => v == value,
                4 => v <= value,
                5 => v > value,
                6 => v != value,
                7 => v >= value,
                _ => true,
            };
            return Ok(met);
        }
        Command::Signal { fence, value } => {
            if let Some(f) = object(g, fence).filter(|o| o.kind == Kind::Fence) {
                f.value = f.value.max(value);
            }
        }
    }
    Ok(true)
}

/// Let every queue run what it can: up to a wait that is not met. Then the events of fences
/// that got where they were waited for are signalled.
pub(crate) fn pump(g: &mut Guest) {
    if g.win.d3d.streams.is_empty() && g.win.d3d.fence_events.is_empty() {
        return;
    }
    for i in 0..g.win.d3d.streams.len() {
        while let Some(command) = g.win.d3d.streams[i].1.front().cloned() {
            match run(g, &command) {
                Ok(true) => {}
                Ok(false) => break,
                // a command on memory that is not there: a GPU would hang or fault; say so and go on
                Err(fault) => g.log.push(format!("D3D12.X: {command:x?} faulted at {:#x}", fault.addr)),
            }
            g.win.d3d.streams[i].1.pop_front();
        }
    }
    g.win.d3d.streams.retain(|(_, s)| !s.is_empty());
    let events = std::mem::take(&mut g.win.d3d.fence_events);
    for (fence, value, event) in events {
        if g.win.d3d.objects.get(&fence).is_some_and(|f| f.value >= value) {
            set_event(g, event);
        } else {
            g.win.d3d.fence_events.push((fence, value, event));
        }
    }
}

/// ExecuteCommandLists(count, lists): what the lists recorded goes to the queue, in order
fn Queue_ExecuteCommandLists(g: &mut Guest) -> H {
    let (queue, count, lists) = (g.arg(0)?, g.arg(1)? as u32 as u64, g.arg(2)?);
    for i in 0..count {
        let list = g.mem.read_u64(lists + 8 * i)?;
        let recorded = g.win.d3d.lists.get(&list).cloned().unwrap_or_default();
        stream(g, queue).extend(recorded);
    }
    pump(g);
    Ok(0)
}

/// Signal(fence, value): the fence gets the value once the queue has run what came before
fn Queue_Signal(g: &mut Guest) -> H {
    let (queue, fence, value) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    stream(g, queue).push_back(Command::Signal { fence, value });
    pump(g);
    Ok(0)
}

fn Queue_GetTimestampFrequency(g: &mut Guest) -> H {
    let out = g.arg(1)?;
    if out == 0 {
        return Ok(E_INVALIDARG);
    }
    g.mem.write_u64(out, 10_000_000)?;
    Ok(0)
}

/// GetClockCalibration(gpu timestamp, cpu timestamp): one clock for both
fn Queue_GetClockCalibration(g: &mut Guest) -> H {
    let now = g.now_ns() / 100;
    for i in [1, 2] {
        let out = g.arg(i)?;
        if out != 0 {
            g.mem.write_u64(out, now)?;
        }
    }
    Ok(0)
}

fn Fence_GetCompletedValue(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    pump(g);
    Ok(object(g, this).map_or(0, |o| o.value))
}

/// SetEventOnCompletion(value, event): the event is signalled when the fence gets to the value
fn Fence_SetEventOnCompletion(g: &mut Guest) -> H {
    let (this, value, event) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    g.win.d3d.fence_events.push((this, value, event));
    pump(g);
    Ok(0)
}

/// device #45 SetEventOnMultipleFenceCompletion(fences, values, count, flags, event): when the
/// first of them gets there (flag 1, "any") – or, taken the same here, the last.
fn Device_SetEventOnMultipleFenceCompletion(g: &mut Guest) -> H {
    let (fences, values, count, event) = (g.arg(1)?, g.arg(2)?, g.arg(3)? as u32 as u64, g.arg(5)?);
    for i in 0..count {
        let (fence, value) = (g.mem.read_u64(fences + 8 * i)?, g.mem.read_u64(values + 8 * i)?);
        g.win.d3d.fence_events.push((fence, value, event));
    }
    pump(g);
    Ok(0)
}

fn Fence_Signal(g: &mut Guest) -> H {
    let (this, value) = (g.arg(0)?, g.arg(1)?);
    if let Some(f) = object(g, this) {
        f.value = value;
    }
    pump(g);
    Ok(0)
}

// --- command lists: recording ---------------------------------------------------------------------

fn record(g: &mut Guest, command: Command) -> H {
    let list = g.arg(0)?;
    g.win.d3d.lists.entry(list).or_default().push(command);
    Ok(0)
}

/// Reset(allocator, initial pipeline): what the list recorded is forgotten (a queue that was
/// given it has its own copy)
fn List_Reset(g: &mut Guest) -> H {
    let list = g.arg(0)?;
    g.win.d3d.lists.insert(list, Vec::new());
    Ok(0)
}

/// CopyBufferRegion(destination, offset, source, offset, bytes)
fn List_CopyBufferRegion(g: &mut Guest) -> H {
    let (dst, dst_offset, src, src_offset, bytes) = (g.arg(1)?, g.arg(2)?, g.arg(3)?, g.arg(4)?, g.arg(5)?);
    let address = |g: &mut Guest, o: u64| object(g, o).filter(|o| o.kind == Kind::Resource).map(|o| o.value);
    match (address(g, dst), address(g, src)) {
        (Some(to), Some(from)) => record(g, Command::Copy { to: to + dst_offset, from: from + src_offset, bytes }),
        _ => Ok(0),
    }
}

/// list #60 CopyMemoryToMemoryX(destination, source, bytes, flags)
fn List_CopyMemoryToMemoryX(g: &mut Guest) -> H {
    let (to, from, bytes) = (g.arg(1)?, g.arg(2)?, g.arg(3)?);
    record(g, Command::Copy { to, from, bytes })
}

/// list #61 FillMemoryWith32BitValueX(address, bytes, value)
fn List_FillMemoryWith32BitValueX(g: &mut Guest) -> H {
    let (at, bytes, value) = (g.arg(1)?, g.arg(2)?, g.arg(3)? as u32);
    record(g, Command::Fill { at, bytes, value })
}

/// list #69 Write32BitValueBottomOfPipeX(address, value, flags, kind)
fn List_Write32BitValueX(g: &mut Guest) -> H {
    let (at, value) = (g.arg(1)?, g.arg(2)? as u32);
    record(g, Command::Write32 { at, value })
}

/// list #70 Write64BitValueBottomOfPipeX(address, value, flags, kind): kind 1 is the GPU's
/// timestamp; list #89 Write64BitValueTopOfPipeX(address, value, flags)
fn List_Write64BitValueX(g: &mut Guest) -> H {
    let (at, value, kind) = (g.arg(1)?, g.arg(2)?, g.arg(4)? as u32);
    record(g, if kind == 1 { Command::Timestamp { at } } else { Command::Write64 { at, value } })
}

fn List_Write64BitValueTopOfPipeX(g: &mut Guest) -> H {
    let (at, value) = (g.arg(1)?, g.arg(2)?);
    record(g, Command::Write64 { at, value })
}

/// list #71 Wait32BitValueX(address, comparison, value, mask, flags)
fn List_Wait32BitValueX(g: &mut Guest) -> H {
    let (at, func, value, mask) = (g.arg(1)?, g.arg(2)? as u32, g.arg(3)? as u32 as u64, g.arg(4)? as u32 as u64);
    record(g, Command::Wait { at, size: 4, func, value: value & mask, mask })
}

/// list #72 Wait64BitValueX(address, comparison, value, mask, flags)
fn List_Wait64BitValueX(g: &mut Guest) -> H {
    let (at, func, value, mask) = (g.arg(1)?, g.arg(2)? as u32, g.arg(3)?, g.arg(4)?);
    record(g, Command::Wait { at, size: 8, func, value: value & mask, mask })
}

/// list #93 KickoffX: what is recorded so far starts running now, before the list is closed
fn List_KickoffX(g: &mut Guest) -> H {
    let list = g.arg(0)?;
    let recorded = g.win.d3d.lists.insert(list, Vec::new()).unwrap_or_default();
    stream(g, list).extend(recorded);
    pump(g);
    Ok(0)
}

/// list #148 AllocateUploadMemory(bytes, alignment): memory the title fills for the GPU
fn List_AllocateUploadMemory(g: &mut Guest) -> H {
    let (bytes, alignment) = (g.arg(1)?, g.arg(2)?.max(16));
    let Some(total) = bytes.checked_add(alignment) else { return Ok(0) };
    let block = g.heap_alloc(total)?;
    Ok((block + alignment - 1) & !(alignment - 1))
}

// --- heaps, resources, blobs -----------------------------------------------------------------------

/// GetDesc(hidden return): the description the heap was made from
fn Heap_GetDesc16(g: &mut Guest) -> H {
    let (this, ret) = (g.arg(0)?, g.arg(1)?);
    let d = object(g, this).map(|o| o.desc);
    if let (Some(d), true) = (d, ret != 0) {
        g.mem.write(ret, &d[..16])?;
    }
    Ok(ret)
}

/// GetCPU / GetGPUDescriptorHandleForHeapStart: console code takes the handle in RAX and passes
/// only `this` (RDX holds whatever the caller left, so nothing is written through it).
fn Heap_GetHandleStart(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(object(g, this).map_or(0, |o| o.value))
}

/// GetDesc(hidden return) of an ID3D12Heap: D3D12_HEAP_DESC (40 bytes)
fn MemoryHeap_GetDesc(g: &mut Guest) -> H {
    let (this, ret) = (g.arg(0)?, g.arg(1)?);
    let d = object(g, this).map(|o| o.desc);
    if let (Some(d), true) = (d, ret != 0) {
        g.mem.write(ret, &d[..40])?;
    }
    Ok(ret)
}

/// Map(subresource, range, data out): the resource's memory – its GPU address
fn Resource_Map(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(3)?);
    let Some(at) = object(g, this).map(|o| o.mem.0) else { return Ok(E_INVALIDARG) };
    if out != 0 {
        g.mem.write_u64(out, at)?;
    }
    Ok(0)
}

fn Resource_GetDesc(g: &mut Guest) -> H {
    let (this, ret) = (g.arg(0)?, g.arg(1)?);
    let d = object(g, this).map(|o| o.desc);
    if let (Some(d), true) = (d, ret != 0) {
        g.mem.write(ret, &d[..56])?;
    }
    Ok(ret)
}

fn Resource_GetGPUVirtualAddress(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(object(g, this).map_or(0, |o| o.value))
}

/// GetHeapProperties(properties, flags)
fn Resource_GetHeapProperties(g: &mut Guest) -> H {
    let (props, flags) = (g.arg(1)?, g.arg(2)?);
    if props != 0 {
        g.mem.write(props, &[0u8; 20])?;
        g.mem.write_u32(props, 1)?; // DEFAULT
    }
    if flags != 0 {
        g.mem.write_u32(flags, 0)?;
    }
    Ok(0)
}

fn Blob_GetBufferPointer(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(object(g, this).map_or(0, |o| o.mem.0))
}

fn Blob_GetBufferSize(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(object(g, this).map_or(0, |o| o.mem.1))
}

fn new_blob(g: &mut Guest, size: u64, out: u64) -> H {
    if out == 0 {
        return Ok(E_INVALIDARG);
    }
    let o = new_object(g, Kind::Blob)?;
    let data = g.heap_alloc(size.max(1))?;
    if let Some(obj) = object(g, o) {
        obj.mem = (data, size);
    }
    g.mem.write_u64(out, o)?;
    Ok(0)
}

// --- the DLL's exports ---------------------------------------------------------------------------

/// D3D12CreateDevice(adapter, feature level, riid, out) / D3D12XboxCreateDevice(adapter,
/// parameters, riid, out)
fn D3D12CreateDevice(g: &mut Guest) -> H {
    let ppv = g.arg(3)?;
    if ppv == 0 {
        return Ok(S_FALSE);
    }
    let d = device(g)?;
    if let Some(o) = object(g, d) {
        o.refs += 1;
    }
    g.mem.write_u64(ppv, d)?;
    Ok(0)
}

/// An out pointer in argument `out` is cleared and `code` returned.
fn none(g: &mut Guest, out: usize, code: u64) -> H {
    let ppv = g.arg(out)?;
    if ppv != 0 {
        g.mem.write_u64(ppv, 0)?;
    }
    Ok(code)
}

fn D3D12GetDebugInterface(g: &mut Guest) -> H {
    none(g, 1, E_NOINTERFACE)
}

fn D3D12GetInterface(g: &mut Guest) -> H {
    none(g, 2, E_NOINTERFACE)
}

fn D3D12CreateRootSignatureDeserializer(g: &mut Guest) -> H {
    none(g, 3, E_NOTIMPL)
}

fn D3D12EnableExperimentalFeatures(_: &mut Guest) -> H {
    Ok(E_NOINTERFACE)
}

/// D3D12SerializeRootSignature(desc, version, blob out, error blob out)
fn D3D12SerializeRootSignature(g: &mut Guest) -> H {
    let (blob, error) = (g.arg(2)?, g.arg(3)?);
    if error != 0 {
        g.mem.write_u64(error, 0)?;
    }
    new_blob(g, 16, blob)
}

/// D3D12SerializeVersionedRootSignature(desc, blob out, error blob out)
fn D3D12SerializeVersionedRootSignature(g: &mut Guest) -> H {
    let (blob, error) = (g.arg(1)?, g.arg(2)?);
    if error != 0 {
        g.mem.write_u64(error, 0)?;
    }
    new_blob(g, 16, blob)
}

/// D3D10CreateBlob(size, out)
fn D3D10CreateBlob(g: &mut Guest) -> H {
    let (size, out) = (g.arg(0)?, g.arg(1)?);
    new_blob(g, size, out)
}

/// D3DCompile(source, …, code blob out, error blob out): there is no HLSL compiler here. (A
/// console title does not compile HLSL: its shaders are console GPU code.)
fn D3DCompile(g: &mut Guest) -> H {
    for i in [9, 10] {
        let out = g.arg(i)?;
        if out != 0 {
            g.mem.write_u64(out, 0)?;
        }
    }
    Ok(E_NOTIMPL)
}

/// The implemented slots of each interface.
fn slots(kind: Kind) -> Vec<(usize, Handler)> {
    let mut v: Vec<(usize, Handler)> = vec![(0, QueryInterface), (1, AddRef), (2, Release)];
    if kind == Kind::Blob {
        v.extend([(3, Blob_GetBufferPointer as Handler), (4, Blob_GetBufferSize)]);
        return v;
    }
    // ID3D12Object: GetPrivateData, SetPrivateData, SetPrivateDataInterface, SetName
    v.extend([(3, Object_GetPrivateData as Handler), (4, ret_ok), (5, ret_ok), (6, ret_ok)]);
    if kind != Kind::Device {
        v.push((7, DeviceChild_GetDevice));
    }
    let own: &[(usize, Handler)] = match kind {
        Kind::Device => &[
            // the PC's methods (0–43)
            (7, Device_GetNodeCount),
            (8, Device_CreateCommandQueue),
            (9, Device_CreateCommandAllocator),
            (10, Device_CreatePipelineState), // CreateGraphicsPipelineState
            (11, Device_CreatePipelineState), // CreateComputePipelineState
            (12, Device_CreateCommandList),
            (13, Device_CheckFeatureSupport),
            (14, Device_CreateDescriptorHeap),
            (15, Device_GetDescriptorHandleIncrementSize),
            (16, Device_CreateRootSignature),
            // views and descriptor copies (17–24): accepted, nothing is drawn with them
            (17, ret_ok),
            (18, Device_CreateShaderResourceView),
            (19, ret_ok),
            (20, ret_ok),
            (21, ret_ok),
            (22, ret_ok),
            (23, ret_ok),
            (24, ret_ok),
            (25, Device_GetResourceAllocationInfo),
            (26, Device_GetCustomHeapProperties),
            (27, Device_CreateCommittedResource),
            (28, Device_CreateHeap),
            (29, Device_CreatePlacedResource),
            (30, Device_CreateReservedResource),
            (31, ret_not_implemented), // CreateSharedHandle
            (34, ret_ok),              // MakeResident
            (35, ret_ok),              // Evict
            (36, Device_CreateFence),
            (37, ret_ok), // GetDeviceRemovedReason
            (38, Device_GetCopyableFootprints),
            (39, Device_CreateQueryHeap),
            (40, ret_ok), // SetStablePowerState
            (41, Device_CreateCommandSignature),
            (43, Device_GetAdapterLuid),
            // the console's extensions (from 44)
            (44, Device_CreatePlacedResourceX),
            (45, Device_SetEventOnMultipleFenceCompletion),
            (47, ret_ok), // an extension that is not modelled (the PC's CreatePipelineState is #86)
            (48, Device_CreateCommandQueue), // CreateCommandQueueX(desc, riid, out)
            (52, ret_ok),                    // SetDebugFlagsX
            (55, Device_CreatePipelineStateX), // CreateDerivedPipelineStateX
            (56, ret_access_denied),           // SerializePipelineStateX: not in a release title
            (60, Device_CreatePipelineStateX), // CreateGraphicsPipelineStateX
            (61, ret_ok),                      // CreateSamplerX
            (66, ret_access_denied),
            (73, Device_CreateCommandListX),
            (74, Device_RegisterPagePoolX),
            (75, Device_UnregisterPagePoolX),
            (86, Device_CreatePipelineState), // CreatePipelineState(stream desc, riid, out)
        ],
        Kind::CommandQueue => &[
            (10, Queue_ExecuteCommandLists),
            (14, Queue_Signal),
            (15, ret_ok), // Wait
            (16, Queue_GetTimestampFrequency),
            (17, Queue_GetClockCalibration),
            (18, Hidden_ZeroDesc16), // GetDesc
        ],
        Kind::CommandAllocator => &[(8, ret_ok)], // Reset
        Kind::CommandList => &[
            (8, ret_ok), // GetType (direct)
            (9, ret_ok), // Close
            (10, List_Reset),
            (15, List_CopyBufferRegion),
            (60, List_CopyMemoryToMemoryX),
            (61, List_FillMemoryWith32BitValueX),
            (69, List_Write32BitValueX),
            (70, List_Write64BitValueX),
            (71, List_Wait32BitValueX),
            (72, List_Wait64BitValueX),
            (89, List_Write64BitValueTopOfPipeX),
            (93, List_KickoffX),
            (146, ret_ok), // a barrier
            (148, List_AllocateUploadMemory),
        ],
        // Close, Reset, CopyBufferRegion (CopyTextureRegion, CopyResource and CopyTiles follow)
        Kind::DmaCommandList => &[(9, ret_ok), (10, List_Reset), (11, List_CopyBufferRegion)],
        Kind::Fence => &[(8, Fence_GetCompletedValue), (9, Fence_SetEventOnCompletion), (10, Fence_Signal)],
        Kind::DescriptorHeap => &[(8, Heap_GetDesc16), (9, Heap_GetHandleStart), (10, Heap_GetHandleStart)],
        Kind::Resource => &[
            (8, Resource_Map),
            (9, ret_ok), // Unmap
            (10, Resource_GetDesc),
            (11, Resource_GetGPUVirtualAddress),
            (12, ret_ok), // WriteToSubresource
            (13, ret_ok), // ReadFromSubresource
            (14, Resource_GetHeapProperties),
        ],
        Kind::Heap => &[(8, MemoryHeap_GetDesc)],
        Kind::PipelineState => &[(8, ret_not_implemented)], // GetCachedBlob
        _ => &[],
    };
    v.extend_from_slice(own);
    v
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    for kind in KINDS {
        for (slot, handler) in slots(kind) {
            // the names live as long as the process: one per implemented slot
            let name: &'static str = Box::leak(format!("{}::#{slot}", kind.iface()).into_boxed_str());
            map.insert(name, handler);
        }
    }
    register!(map; D3D12CreateDevice, D3D12GetDebugInterface, D3D12GetInterface, D3D12CreateRootSignatureDeserializer, D3D12EnableExperimentalFeatures);
    register!(map; D3D12SerializeRootSignature, D3D12SerializeVersionedRootSignature, D3D10CreateBlob, D3DCompile);
    map.insert("D3D12XboxCreateDevice", D3D12CreateDevice as Handler);
    map.insert("D3D12CreateVersionedRootSignatureDeserializer", D3D12CreateRootSignatureDeserializer as Handler);
    map.insert("D3D12XboxSetProcessDebugFlags", ret_ok as Handler);
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX, RSP};
    use xcl_emu::mem::{PROT_R, PROT_W};

    use super::*;

    /// A guest with one thread and nothing loaded, and a scratch block for out parameters.
    fn guest() -> (Guest, u64) {
        let mut g = Guest::new();
        let stack = g.mem.alloc(0x1_0000, PROT_R | PROT_W).unwrap();
        let t = g.add_thread(stack, stack + 0x1_0000, 0, 0, crate::SLOT_ENTRY_RETURN).unwrap();
        g.current = t;
        g.cpu = g.threads[t].cpu.clone();
        let scratch = g.mem.alloc(0x1000, PROT_R | PROT_W).unwrap();
        (g, scratch)
    }

    /// Call the host function `name` as the title would (Win64 arguments).
    fn call(g: &mut Guest, name: &str, args: &[u64]) -> u64 {
        for (i, &a) in args.iter().enumerate() {
            match i {
                0 => g.cpu.r[RCX] = a,
                1 => g.cpu.r[RDX] = a,
                2 => g.cpu.r[8] = a,
                3 => g.cpu.r[9] = a,
                _ => g.mem.write_u64(g.cpu.r[RSP] + 8 + 8 * i as u64, a).unwrap(),
            }
        }
        let handler = *g.handlers.get(name).unwrap_or_else(|| panic!("no host function {name}"));
        handler(g).unwrap()
    }

    #[test]
    fn device_queue_fence_resource_roundtrip() {
        let (mut g, out) = guest();
        assert_eq!(call(&mut g, "D3D12CreateDevice", &[0, 0xC100, 0, out]), 0);
        let dev = g.mem.read_u64(out).unwrap();
        assert_eq!(call(&mut g, "ID3D12Device::#8", &[dev, 0, 0, out]), 0);
        let queue = g.mem.read_u64(out).unwrap();
        // CreateFence(initial 5), Signal(9)
        assert_eq!(call(&mut g, "ID3D12Device::#36", &[dev, 5, 0, 0, out]), 0);
        let fence = g.mem.read_u64(out).unwrap();
        assert_eq!(call(&mut g, "ID3D12Fence::#8", &[fence]), 5);
        assert_eq!(call(&mut g, "ID3D12CommandQueue::#14", &[queue, fence, 9]), 0);
        assert_eq!(call(&mut g, "ID3D12Fence::#8", &[fence]), 9);
        // a buffer: Map gives memory of its size, and it has a GPU address
        let desc = out + 0x100;
        g.mem.write_u32(desc, 1).unwrap();
        g.mem.write_u64(desc + 16, 4096).unwrap();
        assert_eq!(call(&mut g, "ID3D12Device::#27", &[dev, 0, 0, desc, 0, 0, 0, out]), 0);
        let resource = g.mem.read_u64(out).unwrap();
        assert_eq!(call(&mut g, "ID3D12Resource::#8", &[resource, 0, 0, out]), 0);
        let data = g.mem.read_u64(out).unwrap();
        g.mem.write(data, &[0xAB; 4096]).unwrap();
        assert_ne!(call(&mut g, "ID3D12Resource::#11", &[resource]), 0);
        // the object's vtable: 256 slots, each a host function named by interface and slot
        let vtable = g.mem.read_u64(dev).unwrap();
        let slot = g.mem.read_u64(vtable + 8 * 200).unwrap();
        let index = ((slot - crate::TRAP_BASE) / xcl_emu::cpu::TRAP_STRIDE) as usize - crate::FIRST_IMPORT_SLOT as usize;
        assert_eq!(g.imports[index].name, "ID3D12Device::#200");
        assert!(!g.has_handler("ID3D12Device::#200"), "a slot that is not implemented is a missing function");
    }

    #[test]
    fn a_queue_runs_recorded_memory_commands_and_waits_for_values() {
        let (mut g, out) = guest();
        call(&mut g, "D3D12CreateDevice", &[0, 0xC100, 0, out]);
        let dev = g.mem.read_u64(out).unwrap();
        // title memory, and a buffer placed in it: its GPU address is the title's address
        let mem = graphics_memory(&mut g, 0x1_0000, false).unwrap();
        let desc = out + 0x100;
        g.mem.write_u32(desc, 1).unwrap();
        g.mem.write_u64(desc + 16, 0x1_0000).unwrap();
        assert_eq!(call(&mut g, "ID3D12Device::#44", &[dev, mem, desc, 0, 0, 0, out]), 0);
        let placed = g.mem.read_u64(out).unwrap();
        assert_eq!(call(&mut g, "ID3D12Resource::#11", &[placed]), mem);
        // a committed read-back buffer: memory of its own, in the read-back window, and Map is
        // its GPU address
        g.mem.write_u32(out + 0x200, 3).unwrap();
        assert_eq!(call(&mut g, "ID3D12Device::#27", &[dev, out + 0x200, 0, desc, 0, 0, 0, out]), 0);
        let readback = g.mem.read_u64(out).unwrap();
        let to = call(&mut g, "ID3D12Resource::#11", &[readback]);
        assert!(to >= READBACK_WINDOW && to != mem);
        call(&mut g, "ID3D12Resource::#8", &[readback, 0, 0, out]);
        assert_eq!(g.mem.read_u64(out).unwrap(), to);

        call(&mut g, "ID3D12Device::#8", &[dev, 0, 0, out]);
        let queue = g.mem.read_u64(out).unwrap();
        call(&mut g, "ID3D12Device::#36", &[dev, 0, 0, 0, out]);
        let fence = g.mem.read_u64(out).unwrap();
        call(&mut g, "ID3D12Device::#12", &[dev, 0, 0, 0, 0, 0, out]);
        let list = g.mem.read_u64(out).unwrap();
        let event = g.win.new_handle(crate::win::Object::Event { signaled: false, manual: true });
        let signaled = |g: &Guest| matches!(g.win.handles.get(&event), Some(crate::win::Object::Event { signaled: true, .. }));

        // wait for 0xAA in the low byte at +0x300, then write, fill and copy to the buffer
        g.mem.write_u64(mem + 0x1000, 0x1122_3344_5566_7788).unwrap();
        call(&mut g, "ID3D12GraphicsCommandList::#71", &[list, mem + 0x300, 3, 0xAA, 0xFF, 0]);
        call(&mut g, "ID3D12GraphicsCommandList::#69", &[list, mem, 0x1234_5678, 0, 0]);
        call(&mut g, "ID3D12GraphicsCommandList::#61", &[list, mem + 0x100, 8, 0xABCD_EF01]);
        call(&mut g, "ID3D12GraphicsCommandList::#70", &[list, mem + 0x10, 0, 0, 1]);
        call(&mut g, "ID3D12GraphicsCommandList::#15", &[list, readback, 0x20, placed, 0x1000, 8]);
        // nothing happens while the list is only recorded
        assert_eq!(g.mem.read_u32(mem).unwrap(), 0);
        g.mem.write_u64(out, list).unwrap();
        call(&mut g, "ID3D12CommandQueue::#10", &[queue, 1, out]);
        call(&mut g, "ID3D12CommandQueue::#14", &[queue, fence, 1]);
        call(&mut g, "ID3D12Fence::#9", &[fence, 1, event]);
        // the queue is held by the wait: no write yet, the fence has not moved
        assert_eq!((g.mem.read_u32(mem).unwrap(), call(&mut g, "ID3D12Fence::#8", &[fence])), (0, 0));
        assert!(!signaled(&g));
        // "another thread" writes the value: the queue goes on at its next turn
        g.mem.write_u32(mem + 0x300, 0xDEAD_00AA).unwrap();
        pump(&mut g);
        assert_eq!(g.mem.read_u32(mem).unwrap(), 0x1234_5678);
        assert_eq!(g.mem.read_u64(mem + 0x100).unwrap(), 0xABCD_EF01_ABCD_EF01);
        assert_eq!(g.mem.read_u32(mem + 0x108).unwrap(), 0);
        assert_ne!(g.mem.read_u64(mem + 0x10).unwrap(), 0, "a timestamp");
        assert_eq!(g.mem.read_u64(to + 0x20).unwrap(), 0x1122_3344_5566_7788);
        assert_eq!(call(&mut g, "ID3D12Fence::#8", &[fence]), 1);
        assert!(signaled(&g));

        // KickoffX: what is recorded runs before the list is closed or given to a queue
        call(&mut g, "ID3D12GraphicsCommandList::#10", &[list, 0, 0]);
        call(&mut g, "ID3D12GraphicsCommandList::#69", &[list, mem + 0x400, 7, 0, 0]);
        call(&mut g, "ID3D12GraphicsCommandList::#93", &[list]);
        assert_eq!(g.mem.read_u32(mem + 0x400).unwrap(), 7);
    }

    #[test]
    fn descriptor_heaps_have_handle_ranges_of_their_own() {
        let (mut g, out) = guest();
        call(&mut g, "D3D12CreateDevice", &[0, 0xC100, 0, out]);
        let dev = g.mem.read_u64(out).unwrap();
        // { CBV_SRV_UAV, 16 descriptors, shader visible, node mask }
        let desc = out + 0x100;
        for (i, v) in [0u32, 16, 1, 0].into_iter().enumerate() {
            g.mem.write_u32(desc + 4 * i as u64, v).unwrap();
        }
        let mut starts = Vec::new();
        for _ in 0..2 {
            assert_eq!(call(&mut g, "ID3D12Device::#14", &[dev, desc, 0, out]), 0);
            let heap = g.mem.read_u64(out).unwrap();
            // the start comes back in RAX for `this` alone, whatever is in RDX
            let cpu = call(&mut g, "ID3D12DescriptorHeap::#9", &[heap, 0x5A5A]);
            assert_eq!(call(&mut g, "ID3D12DescriptorHeap::#10", &[heap]), cpu);
            starts.push(cpu);
        }
        assert!(starts[0] != 0 && starts[0] != starts[1]);
        assert_eq!(call(&mut g, "ID3D12Device::#15", &[dev, 0]), 32);
    }
}
