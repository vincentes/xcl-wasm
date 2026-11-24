//! Media Foundation, as far as a title's start-up goes: MFStartup / MFShutdown and the DXGI
//! device manager a title binds its D3D12.X device to (`MFResetDXGIDeviceManagerX`, the
//! console's function for that). **Nothing is decoded**: there are no media types, samples or
//! transforms here, so a title that plays video through Media Foundation gets no further than
//! this. (Console titles usually decode their movies with a library of their own.)

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::register;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const E_INVALIDARG: u64 = 0x8007_0057;
const E_HANDLE: u64 = 0x8007_0006;
const E_NOTIMPL: u64 = 0x8000_4001;
/// MF_E_DXGI_DEVICE_NOT_INITIALIZED
const E_NO_DEVICE: u64 = 0x8004_1000;

/// A DXGI device manager: its reset token, the device it was given, the handles it gave out.
struct Manager {
    token: u32,
    device: u64,
    handles: Vec<u64>,
    next: u64,
}

#[derive(Default)]
pub struct State {
    managers: HashMap<u64, Manager>,
    tokens: u32,
}

fn ok(_: &mut Guest) -> H {
    Ok(0)
}

fn one(_: &mut Guest) -> H {
    Ok(1)
}

fn not_implemented(_: &mut Guest) -> H {
    Ok(E_NOTIMPL)
}

/// MFCreateDXGIDeviceManager(reset token out, manager out)
fn MFCreateDXGIDeviceManager(g: &mut Guest) -> H {
    let (token_at, out) = (g.arg(0)?, g.arg(1)?);
    if token_at == 0 || out == 0 {
        return Ok(E_INVALIDARG);
    }
    let methods = [
        "MFDXGI_QueryInterface",
        "MFDXGI_AddRef",
        "MFDXGI_Release",
        "MFDXGI_CloseDeviceHandle",
        "MFDXGI_GetVideoService",
        "MFDXGI_LockDevice",
        "MFDXGI_OpenDeviceHandle",
        "MFDXGI_ResetDevice",
        "MFDXGI_TestDevice",
        "MFDXGI_UnlockDevice",
    ];
    let manager = g.com_object("IMFDXGIDeviceManager", &methods)?;
    g.win.media.tokens += 1;
    let token = 0x4D46_0000 + g.win.media.tokens;
    g.win.media.managers.insert(manager, Manager { token, device: 0, handles: Vec::new(), next: 1 });
    g.mem.write_u32(token_at, token)?;
    g.mem.write_u64(out, manager)?;
    Ok(0)
}

/// Give the manager its device; the token has to be the one the manager was made with.
fn reset(g: &mut Guest, manager: u64, device: u64, token: u32) -> H {
    match g.win.media.managers.get_mut(&manager) {
        Some(m) if m.token == token && device != 0 => {
            m.device = device;
            // handles to the device before are stale
            m.handles.clear();
            Ok(0)
        }
        _ => Ok(E_INVALIDARG),
    }
}

/// MFResetDXGIDeviceManagerX(manager, device, reset token): the console's way to bind its
/// D3D12 device (a PC manager wants a D3D11 one)
fn MFResetDXGIDeviceManagerX(g: &mut Guest) -> H {
    let (manager, device, token) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32);
    reset(g, manager, device, token)
}

/// IMFDXGIDeviceManager::ResetDevice(device, token)
fn MFDXGI_ResetDevice(g: &mut Guest) -> H {
    let (manager, device, token) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32);
    reset(g, manager, device, token)
}

/// OpenDeviceHandle(handle out)
fn MFDXGI_OpenDeviceHandle(g: &mut Guest) -> H {
    let (manager, out) = (g.arg(0)?, g.arg(1)?);
    let Some(m) = g.win.media.managers.get_mut(&manager) else { return Ok(E_INVALIDARG) };
    if m.device == 0 {
        return Ok(E_NO_DEVICE);
    }
    let handle = 0x4D46_4800_0000 + m.next;
    m.next += 1;
    m.handles.push(handle);
    g.mem.write_u64(out, handle)?;
    Ok(0)
}

/// TestDevice(handle): the handle is to the manager's current device
fn MFDXGI_TestDevice(g: &mut Guest) -> H {
    let (manager, handle) = (g.arg(0)?, g.arg(1)?);
    Ok(match g.win.media.managers.get(&manager) {
        Some(m) if m.handles.contains(&handle) => 0,
        _ => E_HANDLE,
    })
}

fn MFDXGI_CloseDeviceHandle(g: &mut Guest) -> H {
    let (manager, handle) = (g.arg(0)?, g.arg(1)?);
    let Some(m) = g.win.media.managers.get_mut(&manager) else { return Ok(E_INVALIDARG) };
    let before = m.handles.len();
    m.handles.retain(|h| *h != handle);
    Ok(if m.handles.len() < before { 0 } else { E_HANDLE })
}

fn MFDXGI_QueryInterface(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(2)?);
    if out == 0 {
        return Ok(E_INVALIDARG);
    }
    g.mem.write_u64(out, this)?;
    Ok(0)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; MFCreateDXGIDeviceManager, MFResetDXGIDeviceManagerX, MFDXGI_ResetDevice, MFDXGI_OpenDeviceHandle, MFDXGI_TestDevice, MFDXGI_CloseDeviceHandle, MFDXGI_QueryInterface);
    map.insert("MFStartup", ok as Handler);
    map.insert("MFShutdown", ok as Handler);
    map.insert("MFDXGI_AddRef", one as Handler);
    map.insert("MFDXGI_Release", ok as Handler);
    // the device itself is not handed out: nothing here would use it
    for name in ["MFDXGI_GetVideoService", "MFDXGI_LockDevice", "MFDXGI_UnlockDevice"] {
        map.insert(name, not_implemented as Handler);
    }
}
