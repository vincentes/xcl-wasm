//! How a retail title reaches XGameRuntime. It does not import the runtime's functions: the
//! thunk library it is linked with loads `xgameruntime.dll`, calls `InitializeApiImpl(Ex/Ex2)`
//! and asks `QueryApiImpl(class, interface)` for each API family as an object; a function is
//! then a call through a slot of the object's vtable (slots 0..2 are IUnknown), with the object
//! as its first argument.
//!
//! The objects are built from XCL's table of the GDK's functions (`xcl::hle::gdk_table`:
//! function, class, interface, slot): a slot points at the host function of that name, called
//! without the object (`Guest::arg` skips it), and a function the host lacks answers E_NOTIMPL.

#![allow(non_snake_case)]

use std::collections::HashMap;

use xcl::hle::gdk_table::API;

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

/// The "DLL" of a function called through an interface (see `Guest::import_call`).
pub const INTERFACE: &str = "@gdk";

pub const E_NOTIMPL: u64 = 0x8000_4001;
const E_INVALIDARG: u64 = 0x8007_0057;
const REGDB_E_CLASSNOTREG: u64 = 0x8004_0154;

/// A GUID in the title's memory, as the table writes it (`{xxxxxxxx-xxxx-…}` read as one number).
fn guid(g: &Guest, at: u64) -> Result<u128, Stop> {
    let b: [u8; 16] = g.mem.read_n(at)?;
    let (d1, d2, d3) = (u32::from_le_bytes([b[0], b[1], b[2], b[3]]), u16::from_le_bytes([b[4], b[5]]), u16::from_le_bytes([b[6], b[7]]));
    let d4 = u64::from_be_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
    Ok((d1 as u128) << 96 | (d2 as u128) << 80 | (d3 as u128) << 64 | d4 as u128)
}

/// The functions of `class` by slot, as interface `interface` has them: its own first, then
/// those the class's other interfaces have in the slots it leaves free (later GDK versions
/// append functions under new interface ids, and a title may ask for one the table lacks).
fn slots(class: u128, interface: u128) -> Vec<Option<&'static str>> {
    let mut slots: Vec<Option<&'static str>> = Vec::new();
    let of_class = || API.iter().filter(move |e| e.1 == class);
    for &(name, _, _, slot) in of_class().filter(|e| e.2 == interface).chain(of_class().filter(|e| e.2 != interface)) {
        let slot = slot as usize;
        if slots.len() <= slot {
            slots.resize(slot + 1, None);
        }
        slots[slot].get_or_insert(name);
    }
    slots
}

/// QueryApiImpl(class id, interface id, object out)
fn QueryApiImpl(g: &mut Guest) -> H {
    let (class_at, interface_at, out) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    if class_at == 0 || interface_at == 0 || out == 0 {
        return Ok(E_INVALIDARG);
    }
    g.mem.write_u64(out, 0)?;
    let (class, interface) = (guid(g, class_at)?, guid(g, interface_at)?);
    if let Some(&object) = g.win.gdk.get(&(class, interface)) {
        g.mem.write_u64(out, object)?;
        return Ok(0);
    }
    let slots = slots(class, interface);
    if slots.is_empty() {
        g.log.push(format!("[GDK] QueryApiImpl: no class {class:032x}"));
        return Ok(REGDB_E_CLASSNOTREG);
    }
    let have = slots.iter().flatten().filter(|n| g.has_handler(n)).count();
    let family = slots.iter().flatten().next().copied().unwrap_or("?");
    g.log.push(format!("[GDK] QueryApiImpl {class:032x} / {interface:032x}: the class of {family}, {have} of its {} functions", slots.iter().flatten().count()));
    let vtable = g.heap_alloc(8 * slots.len().max(3) as u64)?;
    for (i, name) in ["GdkObject_QueryInterface", "GdkObject_AddRef", "GdkObject_Release"].into_iter().enumerate() {
        let at = g.import_address("@com", name);
        g.mem.write_u64(vtable + 8 * i as u64, at)?;
    }
    for (i, name) in slots.iter().enumerate().skip(3) {
        let at = match name {
            Some(name) => g.import_address(INTERFACE, name),
            None => g.import_address(INTERFACE, &format!("slot {i} of class {class:032x}")),
        };
        g.mem.write_u64(vtable + 8 * i as u64, at)?;
    }
    let object = g.heap_alloc(16)?;
    g.mem.write_u64(object, vtable)?;
    g.win.gdk.insert((class, interface), object);
    g.mem.write_u64(out, object)?;
    Ok(0)
}

/// QueryInterface(object, interface id, object out): the object is all its interfaces.
fn GdkObject_QueryInterface(g: &mut Guest) -> H {
    let (object, out) = (g.arg(0)?, g.arg(2)?);
    if out == 0 {
        return Ok(E_INVALIDARG);
    }
    g.mem.write_u64(out, object)?;
    Ok(0)
}

/// AddRef / Release: the objects live as long as the title.
fn GdkObject_AddRef(_: &mut Guest) -> H {
    Ok(2)
}

fn GdkObject_Release(_: &mut Guest) -> H {
    Ok(1)
}

/// InitializeApiImpl(GDK version, OS version) and …Ex(…, flags), …Ex2(…, flags, options)
fn InitializeApiImpl(g: &mut Guest) -> H {
    let (gdk, os) = (g.arg(0)?, g.arg(1)?);
    g.log.push(format!("[GDK] InitializeApiImpl: the title's GDK is {gdk:#x}, for OS {os:#x}"));
    Ok(0)
}

/// XErrorReport(result code, message)
fn XErrorReport(g: &mut Guest) -> H {
    let (code, message) = (g.arg(0)? as u32, g.arg(1)?);
    let text = if message == 0 { String::new() } else { g.mem.cstr(message, 512)? };
    g.log.push(format!("[GDK] XErrorReport {code:#010x} {text}"));
    Ok(0)
}

fn ok(_: &mut Guest) -> H {
    Ok(0)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; QueryApiImpl, InitializeApiImpl, XErrorReport, GdkObject_QueryInterface, GdkObject_AddRef, GdkObject_Release);
    map.insert("InitializeApiImplEx", InitializeApiImpl as Handler);
    map.insert("InitializeApiImplEx2", InitializeApiImpl as Handler);
    map.insert("UninitializeApiImpl", ok as Handler);
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX};
    use xcl_emu::mem::{PROT_R, PROT_W};

    use super::*;

    /// XUser's class and main interface in the table, and a later interface of the class.
    const XUSER: u128 = 0x01acd17791f94763a38eccbb55ce32e0;
    const XUSER_LATER: u128 = 0x079415e36727437f8e9d8f8f9b2439f7;

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

    /// Write `id` as a GUID structure: {01acd177-91f9-4763-a38e-ccbb55ce32e0} is
    /// 77 d1 ac 01, f9 91, 63 47, a3 8e cc bb 55 ce 32 e0.
    fn put_guid(g: &mut Guest, at: u64, id: u128) {
        let b = id.to_be_bytes();
        let bytes = [b[3], b[2], b[1], b[0], b[5], b[4], b[7], b[6], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]];
        g.mem.write(at, &bytes).unwrap();
    }

    /// The function a slot of `object` points at: (its "DLL", its name).
    fn slot(g: &Guest, object: u64, slot: u64) -> (String, String) {
        let vtable = g.mem.read_u64(object).unwrap();
        let address = g.mem.read_u64(vtable + 8 * slot).unwrap();
        let import = &g.imports[((address - crate::TRAP_BASE) / 16) as usize - crate::FIRST_IMPORT_SLOT as usize];
        (import.dll.clone(), import.name.clone())
    }

    #[test]
    fn a_guid_is_read_as_the_table_writes_it() {
        let (mut g, page) = guest();
        g.mem.write(page, &[0x77, 0xd1, 0xac, 0x01, 0xf9, 0x91, 0x63, 0x47, 0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0]).unwrap();
        assert_eq!(guid(&g, page).unwrap(), XUSER);
        put_guid(&mut g, page + 16, XUSER);
        assert_eq!(g.mem.read_n::<16>(page).unwrap(), g.mem.read_n::<16>(page + 16).unwrap());
    }

    #[test]
    fn a_class_is_an_object_whose_slots_are_the_functions_by_name() {
        let (mut g, page) = guest();
        put_guid(&mut g, page, XUSER);
        let out = page + 0x100;
        assert_eq!(call(&mut g, QueryApiImpl, [page, page, out]), Ok(0));
        let object = g.mem.read_u64(out).unwrap();
        assert_eq!(slot(&g, object, 0), ("@com".into(), "GdkObject_QueryInterface".into()));
        // the table: XUserCloseHandle is slot 4, XUserAddAsync slot 7 of the class
        assert_eq!(slot(&g, object, 4), (INTERFACE.into(), "XUserCloseHandle".into()));
        assert_eq!(slot(&g, object, 7), (INTERFACE.into(), "XUserAddAsync".into()));
        // a slot the table has no function for still is one
        assert_eq!(slot(&g, object, 15).0, INTERFACE);
        // asked again: the same object; a later interface of the class has all the slots too
        assert_eq!(call(&mut g, QueryApiImpl, [page, page, out]), Ok(0));
        assert_eq!(g.mem.read_u64(out).unwrap(), object);
        put_guid(&mut g, page + 16, XUSER_LATER);
        assert_eq!(call(&mut g, QueryApiImpl, [page, page + 16, out]), Ok(0));
        let later = g.mem.read_u64(out).unwrap();
        assert_eq!(slot(&g, later, 42), (INTERFACE.into(), "XUserIsStoreUser".into()));
        assert_eq!(slot(&g, later, 7), (INTERFACE.into(), "XUserAddAsync".into()));
    }

    #[test]
    fn a_class_the_table_lacks_is_not_registered() {
        let (mut g, page) = guest();
        put_guid(&mut g, page, 0x1234);
        g.mem.write_u64(page + 0x100, 7).unwrap();
        assert_eq!(call(&mut g, QueryApiImpl, [page, page, page + 0x100]), Ok(REGDB_E_CLASSNOTREG));
        assert_eq!(g.mem.read_u64(page + 0x100).unwrap(), 0);
        assert_eq!(call(&mut g, QueryApiImpl, [page, 0, page + 0x100]), Ok(E_INVALIDARG));
    }

    #[test]
    fn a_function_called_through_its_interface_does_not_see_the_object() {
        let (mut g, page) = guest();
        put_guid(&mut g, page, XUSER);
        assert_eq!(call(&mut g, QueryApiImpl, [page, page, page + 0x100]), Ok(0));
        let object = g.mem.read_u64(page + 0x100).unwrap();
        let vtable = g.mem.read_u64(object).unwrap();
        // the title calls slot 3 (XUserDuplicateHandle, which the host lacks) and slot 0
        let stack = g.cpu.r[xcl_emu::cpu::RSP];
        for (slot, expected) in [(3u64, E_NOTIMPL), (0, 0)] {
            let target = g.mem.read_u64(vtable + 8 * slot).unwrap();
            let back = g.cpu.trap_address(crate::SLOT_CALL_RETURN);
            g.cpu.r[xcl_emu::cpu::RSP] = stack;
            xcl_emu::abi::call(&mut g.cpu, &mut g.mem, target, &[object, 0, page + 0x200], back).unwrap();
            assert_eq!(g.run_until(0, crate::SLOT_CALL_RETURN, 1000), Stop::Exit(expected as u32));
        }
        assert_eq!(g.mem.read_u64(page + 0x200).unwrap(), object);
        assert!(g.log.iter().any(|l| l == "[GDK] XUserDuplicateHandle → E_NOTIMPL"), "{:?}", g.log);
        // … and inside a function called that way, argument 0 is the function's own first
        g.cpu.r[RCX] = object;
        g.cpu.r[RDX] = 0xABC;
        g.arg_shift = 1;
        assert_eq!(g.arg(0), Ok(0xABC));
    }
}
