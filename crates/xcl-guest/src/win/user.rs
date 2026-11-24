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

/// GameInputCreate(out): IGameInput. Its timestamps advance; there are no devices yet.
fn GameInputCreate(g: &mut Guest) -> H {
    let out = g.arg(0)?;
    let object = g.com_object("IGameInput", &["IGameInput_QueryInterface", "IGameInput_AddRef", "IGameInput_Release", "IGameInput_GetCurrentTimestamp"])?;
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
}
