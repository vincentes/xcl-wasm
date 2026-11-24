//! D3D12.X "null device": the first stage of graphics on the guest, as in XCL's Windows runtime
//! (`xcl::hle::d3d12x::null`, which this follows). Objects accept every call and hand back
//! plausible outputs (child objects, descriptor handles, completed fences, mappable memory), so
//! a title gets past graphics initialisation and keeps loading data. **Nothing is rendered.**
//!
//! An object is a small block of guest memory whose first word points at its vtable; what the
//! host knows about it is kept here by its address. The vtable has the PC's D3D12 order. Every
//! slot is a host function named `<Interface>::#<slot>`: one that is not implemented shows up
//! under that name like any missing function, so a title's report says where the console's
//! layout differs.

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const VTBL_SLOTS: usize = 256;
const DLL: &str = "d3d12_xs.dll";
const E_INVALIDARG: u64 = 0x8007_0057;
const E_NOINTERFACE: u64 = 0x8000_4002;
const E_NOTIMPL: u64 = 0x8000_4001;
const S_FALSE: u64 = 1;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
    Device,
    CommandQueue,
    CommandAllocator,
    CommandList,
    Fence,
    DescriptorHeap,
    Resource,
    Heap,
    PipelineState,
    RootSignature,
    QueryHeap,
    CommandSignature,
    PipelineLibrary,
    StateObject,
    Blob,
}

const KINDS: [Kind; 15] = [
    Kind::Device,
    Kind::CommandQueue,
    Kind::CommandAllocator,
    Kind::CommandList,
    Kind::Fence,
    Kind::DescriptorHeap,
    Kind::Resource,
    Kind::Heap,
    Kind::PipelineState,
    Kind::RootSignature,
    Kind::QueryHeap,
    Kind::CommandSignature,
    Kind::PipelineLibrary,
    Kind::StateObject,
    Kind::Blob,
];

impl Kind {
    fn iface(self) -> &'static str {
        match self {
            Kind::Device => "ID3D12Device",
            Kind::CommandQueue => "ID3D12CommandQueue",
            Kind::CommandAllocator => "ID3D12CommandAllocator",
            Kind::CommandList => "ID3D12GraphicsCommandList",
            Kind::Fence => "ID3D12Fence",
            Kind::DescriptorHeap => "ID3D12DescriptorHeap",
            Kind::Resource => "ID3D12Resource",
            Kind::Heap => "ID3D12Heap",
            Kind::PipelineState => "ID3D12PipelineState",
            Kind::RootSignature => "ID3D12RootSignature",
            Kind::QueryHeap => "ID3D12QueryHeap",
            Kind::CommandSignature => "ID3D12CommandSignature",
            Kind::PipelineLibrary => "ID3D12PipelineLibrary",
            Kind::StateObject => "ID3D12StateObject",
            Kind::Blob => "ID3DBlob",
        }
    }
}

/// What the host keeps for an object.
#[derive(Debug, Clone)]
pub struct Object {
    pub kind: Kind,
    pub refs: u32,
    /// fence value / descriptor heap base / resource GPU address
    pub value: u64,
    /// resource backing memory or blob data: (guest address, size)
    pub mem: (u64, u64),
    /// the description it was made from (resource 56 bytes, heap 16)
    pub desc: [u8; 64],
}

#[derive(Default)]
pub struct State {
    pub objects: HashMap<u64, Object>,
    vtables: HashMap<Kind, u64>,
    device: u64,
    next_va: u64,
    pub created: u64,
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
        g.log.push("D3D12.X: the null device – graphics are not rendered".into());
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

fn Hidden_ZeroDesc40(g: &mut Guest) -> H {
    hidden_zero(g, 40)
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
creates!(Device_CreateCommandList, 6, Kind::CommandList);
creates!(Device_CreateCommandList1, 5, Kind::CommandList);
creates!(Device_CreateRootSignature, 5, Kind::RootSignature);
creates!(Device_CreateHeap, 3, Kind::Heap);
creates!(Device_CreateHeap1, 4, Kind::Heap);
creates!(Device_CreateQueryHeap, 3, Kind::QueryHeap);
creates!(Device_CreateCommandSignature, 4, Kind::CommandSignature);
creates!(Device_CreatePipelineLibrary, 4, Kind::PipelineLibrary);

/// CheckFeatureSupport(feature, data, size): everything as zeroed; D3D12_OPTIONS5 (27) gets the
/// configured raytracing tier at +8.
fn Device_CheckFeatureSupport(g: &mut Guest) -> H {
    let (feature, data, size) = (g.arg(1)? as u32, g.arg(2)?, g.arg(3)? as u32);
    if feature == 27 && data != 0 && size >= 12 {
        let tier = xcl::state::config().raytracing_tier_override().unwrap_or(0);
        g.mem.write_u32(data + 8, tier)?;
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

fn new_resource(g: &mut Guest, desc: u64, out: usize) -> H {
    let ppv = g.arg(out)?;
    if ppv == 0 {
        return Ok(S_FALSE);
    }
    let size = approx_size(g, desc)?;
    let mut d = [0u8; 64];
    if desc != 0 {
        g.mem.read(desc, &mut d[..56])?;
    }
    let o = new_object(g, Kind::Resource)?;
    // a GPU address of its own (64 KiB steps), never a mapping
    if g.win.d3d.next_va == 0 {
        g.win.d3d.next_va = 0x1_0000_0000;
    }
    let va = g.win.d3d.next_va;
    g.win.d3d.next_va += (size + 0xFFFF) & !0xFFFF;
    if let Some(obj) = object(g, o) {
        obj.desc = d;
        obj.mem = (0, size);
        obj.value = va;
    }
    g.mem.write_u64(ppv, o)?;
    Ok(0)
}

/// CreateCommittedResource(heap properties, heap flags, desc, state, clear value, riid, out)
fn Device_CreateCommittedResource(g: &mut Guest) -> H {
    let desc = g.arg(3)?;
    new_resource(g, desc, 7)
}

/// CreateCommittedResource1(…, protected session, riid, out)
fn Device_CreateCommittedResource1(g: &mut Guest) -> H {
    let desc = g.arg(3)?;
    new_resource(g, desc, 8)
}

/// CreatePlacedResource(heap, offset, desc, state, clear value, riid, out)
fn Device_CreatePlacedResource(g: &mut Guest) -> H {
    let desc = g.arg(3)?;
    new_resource(g, desc, 7)
}

/// CreateReservedResource(desc, state, clear value, riid, out)
fn Device_CreateReservedResource(g: &mut Guest) -> H {
    let desc = g.arg(1)?;
    new_resource(g, desc, 5)
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

/// Signal the event behind a handle (the null GPU has always finished).
fn set_event(g: &mut Guest, handle: u64) {
    if let Some(super::Object::Event { signaled, .. }) = g.win.handles.get_mut(&handle) {
        *signaled = true;
    }
}

/// SetEventOnMultipleFenceCompletion(fences, values, count, flags, event)
fn Device_SetEventOnMultipleFenceCompletion(g: &mut Guest) -> H {
    let event = g.arg(5)?;
    set_event(g, event);
    Ok(0)
}

// --- queue, fence, heap, resource, blob ---------------------------------------------------------

/// Signal(fence, value)
fn Queue_Signal(g: &mut Guest) -> H {
    let (fence, value) = (g.arg(1)?, g.arg(2)?);
    if let Some(f) = object(g, fence).filter(|o| o.kind == Kind::Fence) {
        f.value = f.value.max(value);
    }
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
    Ok(object(g, this).map_or(0, |o| o.value))
}

/// SetEventOnCompletion(value, event): the null GPU finishes everything at once
fn Fence_SetEventOnCompletion(g: &mut Guest) -> H {
    let (this, value, event) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    if let Some(f) = object(g, this) {
        f.value = f.value.max(value);
    }
    set_event(g, event);
    Ok(0)
}

fn Fence_Signal(g: &mut Guest) -> H {
    let (this, value) = (g.arg(0)?, g.arg(1)?);
    if let Some(f) = object(g, this) {
        f.value = value;
    }
    Ok(0)
}

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

/// Map(subresource, range, data out): memory of the resource's size, made at the first Map
fn Resource_Map(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(3)?);
    let Some((at, size)) = object(g, this).map(|o| o.mem) else { return Ok(E_INVALIDARG) };
    let at = if at == 0 {
        let Ok(new) = g.mem.alloc(size.max(1), xcl_emu::mem::PROT_R | xcl_emu::mem::PROT_W) else { return Ok(0x8007_000E) };
        if let Some(o) = object(g, this) {
            o.mem.0 = new;
        }
        new
    } else {
        at
    };
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

/// The implemented slots of each interface (the PC's D3D12 vtable order).
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
            (44, Device_CreatePipelineLibrary),
            (45, Device_SetEventOnMultipleFenceCompletion),
            (46, ret_ok),                     // SetResidencyPriority
            (47, Device_CreatePipelineState), // CreatePipelineState (Device2)
            (51, Device_CreateCommandList1),
            (53, Device_CreateCommittedResource1),
            (54, Device_CreateHeap1),
            (62, Device_CreatePipelineState), // CreateStateObject (Device5)
        ],
        Kind::CommandQueue => &[
            (10, ret_ok), // ExecuteCommandLists
            (14, Queue_Signal),
            (15, ret_ok), // Wait
            (16, Queue_GetTimestampFrequency),
            (17, Queue_GetClockCalibration),
            (18, Hidden_ZeroDesc16), // GetDesc
        ],
        Kind::CommandAllocator => &[(8, ret_ok)], // Reset
        // GetType (direct), Close, Reset
        Kind::CommandList => &[(8, ret_ok), (9, ret_ok), (10, ret_ok)],
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
        Kind::Heap => &[(8, Hidden_ZeroDesc40)],
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
