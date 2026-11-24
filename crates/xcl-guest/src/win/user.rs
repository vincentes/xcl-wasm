//! Windows and input: the console has one screen, so every top-level window a title makes is
//! that screen (a hidden, frameless 1920×1080 popup, as in XCL's Windows runtime), and
//! GameInput, a COM object whose methods are host functions.

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const WS_POPUP: u32 = 0x8000_0000;
const WS_EX_TOOLWINDOW: u32 = 0x80;
const SCREEN: (i32, i32) = (1920, 1080);

fn RegisterClassW(g: &mut Guest) -> H {
    g.win.window_classes += 1;
    // an atom
    Ok(0xC000 + g.win.window_classes as u64)
}

/// CreateWindowEx{W,A}(ex style, class, title, style, x, y, width, height, …): the console's
/// screen, whatever was asked for.
fn CreateWindowExW(g: &mut Guest) -> H {
    Ok(g.win.new_handle(Object::Window))
}

fn is_window(g: &Guest, handle: u64) -> bool {
    matches!(g.win.handles.get(&handle), Some(Object::Window))
}

/// GetWindowLongW(window, index): GWL_STYLE (-16), GWL_EXSTYLE (-20)
fn GetWindowLongW(g: &mut Guest) -> H {
    let (handle, index) = (g.arg(0)?, g.arg(1)? as u32 as i32);
    if !is_window(g, handle) {
        return Ok(0);
    }
    Ok(match index {
        -16 => WS_POPUP,
        -20 => WS_EX_TOOLWINDOW,
        _ => 0,
    } as u64)
}

fn GetClientRect(g: &mut Guest) -> H {
    let (handle, at) = (g.arg(0)?, g.arg(1)?);
    if !is_window(g, handle) {
        return Ok(0);
    }
    for (i, v) in [0, 0, SCREEN.0, SCREEN.1].into_iter().enumerate() {
        g.mem.write_u32(at + 4 * i as u64, v as u32)?;
    }
    Ok(1)
}

/// DwmGetWindowAttribute(window, attribute, value, size): DWMWA_CLOAKED (14) – the screen is
/// not shown on a desktop.
fn DwmGetWindowAttribute(g: &mut Guest) -> H {
    let (handle, attribute, at, size) = (g.arg(0)?, g.arg(1)? as u32, g.arg(2)?, g.arg(3)? as u32);
    if !is_window(g, handle) || attribute != 14 || size < 4 {
        return Ok(0x8007_0057); // E_INVALIDARG
    }
    g.mem.write_u32(at, 1)?;
    Ok(0)
}

fn DestroyWindow(g: &mut Guest) -> H {
    let handle = g.arg(0)?;
    Ok((is_window(g, handle) && g.win.handles.remove(&handle).is_some()) as u64)
}

fn DefWindowProcW(_: &mut Guest) -> H {
    Ok(0)
}

// --- GameInput ----------------------------------------------------------------------------------

fn com_ok(_: &mut Guest) -> H {
    Ok(0)
}

fn com_refcount(_: &mut Guest) -> H {
    Ok(1)
}

fn com_no_interface(g: &mut Guest) -> H {
    let out = g.arg(2)?;
    if out != 0 {
        g.mem.write_u64(out, 0)?;
    }
    Ok(0x8000_4002) // E_NOINTERFACE
}

/// IGameInput::GetCurrentTimestamp: microseconds
fn IGameInput_GetCurrentTimestamp(g: &mut Guest) -> H {
    Ok(g.now_ns() / 1000 + 1)
}

const GAMEINPUT_E_DEVICE_NOT_FOUND: u64 = 0x838A_0002;
const GAMEINPUT_E_READING_NOT_FOUND: u64 = 0x838A_0003;

/// IGameInput's methods in vtable order. There are no devices: a reading or a device is not
/// found, and a callback is registered and never called.
const IGAMEINPUT: [&str; 22] = [
    "IGameInput_QueryInterface",
    "IGameInput_AddRef",
    "IGameInput_Release",
    "IGameInput_GetCurrentTimestamp",
    "IGameInput_GetCurrentReading",
    "IGameInput_GetNextReading",
    "IGameInput_GetPreviousReading",
    "IGameInput_GetTemporalReading",
    "IGameInput_RegisterReadingCallback",
    "IGameInput_RegisterDeviceCallback",
    "IGameInput_RegisterSystemButtonCallback",
    "IGameInput_RegisterKeyboardLayoutCallback",
    "IGameInput_StopCallback",
    "IGameInput_UnregisterCallback",
    "IGameInput_CreateDispatcher",
    "IGameInput_CreateAggregateDevice",
    "IGameInput_FindDeviceFromId",
    "IGameInput_FindDeviceFromObject",
    "IGameInput_FindDeviceFromPlatformHandle",
    "IGameInput_FindDeviceFromPlatformString",
    "IGameInput_EnableOemDeviceSupport",
    "IGameInput_SetFocusPolicy",
];

/// Write 0 to the result pointer that is argument `out` and answer `result`.
fn com_nothing(g: &mut Guest, out: usize, result: u64) -> H {
    let at = g.arg(out)?;
    if at != 0 {
        g.mem.write_u64(at, 0)?;
    }
    Ok(result)
}

/// GetCurrentReading(this, kind, device, reading out)
fn IGameInput_GetCurrentReading(g: &mut Guest) -> H {
    com_nothing(g, 3, GAMEINPUT_E_READING_NOT_FOUND)
}

/// GetNextReading / GetPreviousReading(this, reference reading, kind, device, reading out)
fn IGameInput_GetNextReading(g: &mut Guest) -> H {
    com_nothing(g, 4, GAMEINPUT_E_READING_NOT_FOUND)
}

/// The callback registrations: their token (the last argument) is 1.
fn register_callback(g: &mut Guest, token: usize) -> H {
    let at = g.arg(token)?;
    if at != 0 {
        g.mem.write_u64(at, 1)?;
    }
    Ok(0)
}

/// RegisterReadingCallback(this, device, kind, threshold, context, callback, token out)
fn IGameInput_RegisterReadingCallback(g: &mut Guest) -> H {
    register_callback(g, 6)
}

/// RegisterDeviceCallback(this, device, kind, status filter, enumeration, context, callback, token out)
fn IGameInput_RegisterDeviceCallback(g: &mut Guest) -> H {
    register_callback(g, 7)
}

/// RegisterSystemButtonCallback(this, device, buttons, context, callback, token out)
fn IGameInput_RegisterSystemButtonCallback(g: &mut Guest) -> H {
    register_callback(g, 5)
}

/// RegisterKeyboardLayoutCallback(this, device, context, callback, token out)
fn IGameInput_RegisterKeyboardLayoutCallback(g: &mut Guest) -> H {
    register_callback(g, 4)
}

/// FindDeviceFrom…(this, what, device out)
fn IGameInput_FindDevice(g: &mut Guest) -> H {
    com_nothing(g, 2, GAMEINPUT_E_DEVICE_NOT_FOUND)
}

/// GameInputCreate(out): IGameInput. Its timestamps advance; there are no devices yet.
fn GameInputCreate(g: &mut Guest) -> H {
    let out = g.arg(0)?;
    let object = g.com_object("IGameInput", &IGAMEINPUT)?;
    g.mem.write_u64(out, object)?;
    Ok(0)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; RegisterClassW, CreateWindowExW, GetWindowLongW, GetClientRect, DwmGetWindowAttribute, DestroyWindow, DefWindowProcW);
    register!(map; GameInputCreate, IGameInput_GetCurrentTimestamp);
    map.insert("CreateWindowExA", CreateWindowExW as Handler);
    map.insert("IGameInput_QueryInterface", com_no_interface as Handler);
    map.insert("IGameInput_AddRef", com_refcount as Handler);
    map.insert("IGameInput_Release", com_ok as Handler);
    register!(map; IGameInput_GetCurrentReading, IGameInput_GetNextReading);
    register!(map; IGameInput_RegisterReadingCallback, IGameInput_RegisterDeviceCallback, IGameInput_RegisterSystemButtonCallback, IGameInput_RegisterKeyboardLayoutCallback);
    map.insert("IGameInput_GetPreviousReading", IGameInput_GetNextReading as Handler);
    for name in ["IGameInput_FindDeviceFromId", "IGameInput_FindDeviceFromObject", "IGameInput_FindDeviceFromPlatformHandle", "IGameInput_FindDeviceFromPlatformString"] {
        map.insert(name, IGameInput_FindDevice as Handler);
    }
    // nothing to stop: no callback is ever running (UnregisterCallback says so with `true`)
    map.insert("IGameInput_StopCallback", com_ok as Handler);
    map.insert("IGameInput_UnregisterCallback", com_refcount as Handler);
    map.insert("IGameInput_SetFocusPolicy", com_ok as Handler);
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX, RSP};
    use xcl_emu::mem::{PROT_R, PROT_W};

    use super::*;

    #[test]
    fn game_input_has_all_its_methods_and_no_devices() {
        let mut g = Guest::new();
        let stack = g.mem.alloc(0x4000, PROT_R | PROT_W).unwrap();
        g.add_thread(stack, stack + 0x4000, 0, 0, crate::SLOT_THREAD_RETURN).unwrap();
        g.cpu = g.threads[0].cpu.clone();
        let page = g.mem.alloc(0x1000, PROT_R | PROT_W).unwrap();
        g.cpu.r[RCX] = page;
        assert_eq!(GameInputCreate(&mut g), Ok(0));
        let object = g.mem.read_u64(page).unwrap();
        let vtable = g.mem.read_u64(object).unwrap();
        // every slot of the interface is a function: none is left 0 for a title to jump to
        for slot in 0..IGAMEINPUT.len() as u64 {
            assert_ne!(g.mem.read_u64(vtable + 8 * slot).unwrap(), 0, "slot {slot}");
        }
        // GetCurrentReading(this, kind, device, reading out)
        g.mem.write_u64(page + 8, 7).unwrap();
        (g.cpu.r[RCX], g.cpu.r[RDX], g.cpu.r[8], g.cpu.r[9]) = (object, 0xE, 0, page + 8);
        assert_eq!(IGameInput_GetCurrentReading(&mut g), Ok(GAMEINPUT_E_READING_NOT_FOUND));
        assert_eq!(g.mem.read_u64(page + 8).unwrap(), 0);
        // RegisterDeviceCallback: the token is the eighth argument, on the stack
        g.mem.write_u64(g.cpu.r[RSP] + 8 + 8 * 7, page + 16).unwrap();
        assert_eq!(IGameInput_RegisterDeviceCallback(&mut g), Ok(0));
        assert_eq!(g.mem.read_u64(page + 16).unwrap(), 1);
    }
}
