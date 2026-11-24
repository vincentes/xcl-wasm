//! Text between the title's two encodings, and the small kernel32 / ole32 allocators and
//! questions a library asks before it does anything of its own.

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const CP_UTF8: u32 = 65001;

/// The length argument of the conversions: the units at `at`, or (-1) up to and with the NUL.
fn units<const N: usize>(g: &Guest, at: u64, len: i32) -> Result<Vec<[u8; N]>, Stop> {
    let mut out = Vec::new();
    let mut i = 0u64;
    loop {
        if len >= 0 && i == len as u64 {
            break;
        }
        let unit: [u8; N] = g.mem.read_n(at + i * N as u64)?;
        out.push(unit);
        if len < 0 && unit == [0; N] {
            break;
        }
        i += 1;
    }
    Ok(out)
}

/// Bytes of code page `page` as text: UTF-8, or Latin-1 for every other page (the title's own
/// strings are ASCII in either).
fn decode(page: u32, bytes: &[u8]) -> String {
    match page {
        CP_UTF8 => String::from_utf8_lossy(bytes).into_owned(),
        _ => bytes.iter().map(|&b| b as char).collect(),
    }
}

fn encode(page: u32, text: &str) -> Vec<u8> {
    match page {
        CP_UTF8 => text.as_bytes().to_vec(),
        _ => text.chars().map(|c| if (c as u32) < 0x100 { c as u8 } else { b'?' }).collect(),
    }
}

/// What a conversion returns: the number of units, written when there is room for them; with
/// no room at all (size 0) only the number.
fn converted(g: &mut Guest, out: &[u8], unit: u64, at: u64, size: u64) -> H {
    let count = out.len() as u64 / unit;
    if size == 0 {
        return Ok(count);
    }
    if count > size {
        g.set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return Ok(0);
    }
    g.mem.write(at, out)?;
    Ok(count)
}

/// MultiByteToWideChar(code page, flags, bytes, byte count, wide out, wide size)
fn MultiByteToWideChar(g: &mut Guest) -> H {
    let (page, from, len, to, size) = (g.arg(0)? as u32, g.arg(2)?, g.arg(3)? as i32, g.arg(4)?, g.arg(5)? as u32);
    let bytes: Vec<u8> = units::<1>(g, from, len)?.into_iter().map(|[b]| b).collect();
    let wide: Vec<u8> = decode(page, &bytes).encode_utf16().flat_map(u16::to_le_bytes).collect();
    converted(g, &wide, 2, to, size as u64)
}

/// WideCharToMultiByte(code page, flags, wide, wide count, bytes out, byte size, default char,
/// used default out)
fn WideCharToMultiByte(g: &mut Guest) -> H {
    let (page, from, len, to, size, used) = (g.arg(0)? as u32, g.arg(2)?, g.arg(3)? as i32, g.arg(4)?, g.arg(5)? as u32, g.arg(7)?);
    let wide: Vec<u16> = units::<2>(g, from, len)?.into_iter().map(u16::from_le_bytes).collect();
    let bytes = encode(page, &String::from_utf16_lossy(&wide));
    if used != 0 {
        g.mem.write_u32(used, 0)?;
    }
    converted(g, &bytes, 1, to, size as u64)
}

fn compare_wide(g: &mut Guest, fold: bool) -> H {
    let (a, b) = (g.arg(0)?, g.arg(1)?);
    let (a, b) = (g.mem.wstr(a, 0x8000)?, g.mem.wstr(b, 0x8000)?);
    let order = if fold { a.to_lowercase().cmp(&b.to_lowercase()) } else { a.cmp(&b) };
    Ok(order as i32 as u32 as u64)
}

fn lstrcmpW(g: &mut Guest) -> H {
    compare_wide(g, false)
}

fn lstrcmpiW(g: &mut Guest) -> H {
    compare_wide(g, true)
}

/// LocalAlloc(flags, size): a block of the process heap (zeroed, as every block of it is).
fn LocalAlloc(g: &mut Guest) -> H {
    let size = g.arg(1)?;
    Ok(g.win.heap.alloc(&mut g.mem, size).unwrap_or(0))
}

/// LocalFree / GlobalFree(block): NULL once it is freed.
fn LocalFree(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    if at == 0 || g.win.heap.free(&mut g.mem, at) {
        return Ok(0);
    }
    g.set_last_error(ERROR_INVALID_HANDLE);
    Ok(at)
}

fn CoTaskMemAlloc(g: &mut Guest) -> H {
    let size = g.arg(0)?;
    Ok(g.win.heap.alloc(&mut g.mem, size).unwrap_or(0))
}

fn CoTaskMemFree(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    if at != 0 {
        g.win.heap.free(&mut g.mem, at);
    }
    Ok(0)
}

/// The command line as bytes (GetCommandLineW's, which is ASCII or is taken as UTF-8).
fn GetCommandLineA(g: &mut Guest) -> H {
    g.static_block("command line, narrow", |g| {
        let line = g.command_line.clone();
        g.alloc_cstr(&line)
    })
}

/// RegisterAppStateChangeNotification(routine, context, registration out), and the one for
/// "constrained": registered (the title is never suspended or constrained, so never called).
fn RegisterAppStateChangeNotification(g: &mut Guest) -> H {
    let out = g.arg(2)?;
    if out != 0 {
        g.mem.write_u64(out, 1)?;
    }
    Ok(0)
}

/// GetUserDefaultGeoName(name out, characters): "US" – 3 characters with the terminator.
fn GetUserDefaultGeoName(g: &mut Guest) -> H {
    let (out, size) = (g.arg(0)?, g.arg(1)? as u32);
    if size == 0 {
        return Ok(3);
    }
    if size < 3 {
        g.set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return Ok(0);
    }
    for (i, c) in "US\0".encode_utf16().enumerate() {
        g.mem.write_u16(out + 2 * i as u64, c)?;
    }
    Ok(3)
}

/// GetStartupInfoW(info): 104 bytes, of which only the size says anything.
fn GetStartupInfoW(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    g.mem.write(at, &[0u8; 104])?;
    g.mem.write_u32(at, 104)?;
    Ok(0)
}

/// en-US, the language the console is set to.
fn GetUserDefaultUILanguage(_: &mut Guest) -> H {
    Ok(0x0409)
}

const ERROR_FILE_NOT_FOUND: u64 = 2;

/// RegOpenKeyW(key, sub key, key out) / RegOpenKeyExW(key, sub key, options, access, key out):
/// there is no registry. The answer is an error code (0 would be "opened").
fn reg_open(g: &mut Guest, out: usize) -> H {
    let (name, out) = (g.arg(1)?, g.arg(out)?);
    let name = if name == 0 { String::new() } else { g.mem.wstr(name, 512)? };
    let note = format!("[registry] no key {name}");
    if !g.log.contains(&note) {
        g.log.push(note);
    }
    if out != 0 {
        g.mem.write_u64(out, 0)?;
    }
    Ok(ERROR_FILE_NOT_FOUND)
}

fn RegOpenKeyW(g: &mut Guest) -> H {
    reg_open(g, 2)
}

fn RegOpenKeyExW(g: &mut Guest) -> H {
    reg_open(g, 4)
}

/// A value of a key that could not be opened.
fn reg_not_found(_: &mut Guest) -> H {
    Ok(ERROR_FILE_NOT_FOUND)
}

/// RegCloseKey of a key nobody has: ERROR_INVALID_HANDLE.
fn RegCloseKey(_: &mut Guest) -> H {
    Ok(ERROR_INVALID_HANDLE as u64)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; RegOpenKeyW, RegOpenKeyExW, RegCloseKey);
    map.insert("RegQueryValueExW", reg_not_found as Handler);
    map.insert("RegGetValueW", reg_not_found as Handler);
    register!(map; MultiByteToWideChar, WideCharToMultiByte, lstrcmpW, lstrcmpiW, LocalAlloc, LocalFree, CoTaskMemAlloc, CoTaskMemFree);
    register!(map; GetStartupInfoW, GetUserDefaultUILanguage, GetCommandLineA, RegisterAppStateChangeNotification, GetUserDefaultGeoName);
    map.insert("RegisterAppConstrainedChangeNotification", RegisterAppStateChangeNotification as Handler);
    map.insert("GlobalFree", LocalFree as Handler);
    // the functions that differ from one the host has only in what their extra arguments
    // say, which the host does not look at
    for (name, same) in [("GetEnvironmentVariableA", "GetEnvironmentVariableW"), ("CreateEventA", "CreateEventW"), ("LoadLibraryA", "LoadLibraryExA"), ("LoadLibraryExW", "LoadLibraryW")] {
        if let Some(&f) = map.get(same) {
            map.insert(name, f);
        }
    }
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX, RSP};
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

    /// Call `f` with up to eight arguments: four in registers, the rest above the return
    /// address and the home space.
    fn call(g: &mut Guest, f: Handler, args: &[u64]) -> H {
        for (i, &value) in args.iter().enumerate() {
            match i {
                0..4 => g.cpu.r[[RCX, RDX, 8, 9][i]] = value,
                _ => g.mem.write_u64(g.cpu.r[RSP] + 8 + 8 * i as u64, value).unwrap(),
            }
        }
        f(g)
    }

    fn wide(g: &Guest, at: u64, count: u64) -> Vec<u16> {
        (0..count).map(|i| g.mem.read_u16(at + 2 * i).unwrap()).collect()
    }

    #[test]
    fn utf8_becomes_utf16_with_the_length_asked_first() {
        let (mut g, page) = guest();
        let (from, to) = (page, page + 0x100);
        g.mem.write(from, "h\u{e9}llo \u{1F600}\0".as_bytes()).unwrap();
        let minus_one = (-1i32) as u32 as u64;
        // h é l l o space, a surrogate pair, the NUL
        assert_eq!(call(&mut g, MultiByteToWideChar, &[65001, 0, from, minus_one, 0, 0]), Ok(9));
        assert_eq!(call(&mut g, MultiByteToWideChar, &[65001, 0, from, minus_one, to, 4]), Ok(0));
        assert_eq!(g.threads[0].last_error, ERROR_INSUFFICIENT_BUFFER);
        assert_eq!(call(&mut g, MultiByteToWideChar, &[65001, 0, from, minus_one, to, 9]), Ok(9));
        assert_eq!(String::from_utf16(&wide(&g, to, 9)).unwrap(), "h\u{e9}llo \u{1F600}\0");
        // a counted length converts no terminator
        assert_eq!(call(&mut g, MultiByteToWideChar, &[65001, 0, from, 1, to + 0x100, 8]), Ok(1));
        assert_eq!(wide(&g, to + 0x100, 2), [b'h' as u16, 0]);
        // … and another code page is Latin-1: the two bytes of é are two characters
        assert_eq!(call(&mut g, MultiByteToWideChar, &[0, 0, from, 3, to, 8]), Ok(3));
        assert_eq!(wide(&g, to, 3), [b'h' as u16, 0xC3, 0xA9]);
    }

    #[test]
    fn utf16_becomes_bytes_and_what_latin1_lacks_a_question_mark() {
        let (mut g, page) = guest();
        let (from, to, used) = (page, page + 0x100, page + 0x200);
        let text: Vec<u8> = "\u{e9}\u{20AC}".encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        g.mem.write(from, &text).unwrap();
        g.mem.write_u32(used, 7).unwrap();
        let minus_one = (-1i32) as u32 as u64;
        assert_eq!(call(&mut g, WideCharToMultiByte, &[65001, 0, from, minus_one, 0, 0, 0, 0]), Ok(6));
        assert_eq!(call(&mut g, WideCharToMultiByte, &[65001, 0, from, minus_one, to, 16, 0, used]), Ok(6));
        assert_eq!(g.mem.cstr(to, 16).unwrap(), "\u{e9}\u{20AC}");
        assert_eq!(g.mem.read_u32(used).unwrap(), 0);
        assert_eq!(call(&mut g, WideCharToMultiByte, &[0, 0, from, 2, to, 16, 0, 0]), Ok(2));
        assert_eq!(g.mem.read_n::<2>(to).unwrap(), [0xE9, b'?']);
    }

    #[test]
    fn wide_strings_compare_with_and_without_case() {
        let (mut g, page) = guest();
        for (at, text) in [(page, "Game.DLL"), (page + 0x100, "game.dll"), (page + 0x200, "game.dlm")] {
            let units: Vec<u8> = text.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
            g.mem.write(at, &units).unwrap();
        }
        assert_eq!(call(&mut g, lstrcmpiW, &[page, page + 0x100]), Ok(0));
        assert_eq!(call(&mut g, lstrcmpW, &[page, page + 0x100]), Ok(u32::MAX as u64));
        assert_eq!(call(&mut g, lstrcmpiW, &[page + 0x200, page]), Ok(1));
    }

    #[test]
    fn the_command_line_is_one_block_of_bytes() {
        let (mut g, _) = guest();
        g.command_line = "title.exe -level 3".into();
        let at = call(&mut g, GetCommandLineA, &[]).unwrap();
        assert_eq!(g.mem.cstr(at, 64).unwrap(), "title.exe -level 3");
        assert_eq!(call(&mut g, GetCommandLineA, &[]), Ok(at));
    }

    #[test]
    fn the_registry_has_no_keys() {
        let (mut g, page) = guest();
        let name: Vec<u8> = "SOFTWARE\\Vendor".encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        g.mem.write(page, &name).unwrap();
        g.mem.write_u64(page + 0x100, 0x1234).unwrap();
        let hklm = 0xFFFF_FFFF_8000_0002;
        assert_eq!(call(&mut g, RegOpenKeyW, &[hklm, page, page + 0x100]), Ok(2));
        assert_eq!(g.mem.read_u64(page + 0x100).unwrap(), 0);
        assert_eq!(call(&mut g, RegOpenKeyExW, &[hklm, page, 0, 0x20019, page + 0x100]), Ok(2));
        // asked twice, noted once
        assert_eq!(g.log, ["[registry] no key SOFTWARE\\Vendor"]);
        assert_eq!(call(&mut g, reg_not_found, &[0, page]), Ok(2));
        assert_eq!(call(&mut g, RegCloseKey, &[0]), Ok(6));
    }

    #[test]
    fn local_blocks_come_from_the_heap_and_free_to_null() {
        let (mut g, _) = guest();
        let block = call(&mut g, LocalAlloc, &[0x40, 24]).unwrap();
        assert_ne!(block, 0);
        g.mem.write_u64(block + 16, 1).unwrap();
        assert_eq!(call(&mut g, LocalFree, &[block]), Ok(0));
        assert_eq!(call(&mut g, LocalFree, &[0]), Ok(0));
        // freeing it twice hands the block back, with an error
        assert_eq!(call(&mut g, LocalFree, &[block]), Ok(block));
    }
}
