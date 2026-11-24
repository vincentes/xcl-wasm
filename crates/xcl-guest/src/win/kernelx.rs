//! Console functions of `kernelx` (and `xmem`, XGameRuntime's system information): title
//! memory and what the console says about itself. Same answers as XCL's Windows runtime
//! (`xcl::hle::kernelx`), on guest memory.

#![allow(non_snake_case)]

use std::collections::HashMap;

use xcl_emu::mem::{PROT_R, PROT_W};

use super::kernel32::virtual_alloc;
use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

/// The region graphics memory is in: from 16 GiB (what the GPU sees, aligned to 4 GiB). Its
/// first range is left free, as XCL keeps its own there.
const UMA_BASE: u64 = 0x4_0000_0000 + 0x1000_0000;
const UMA_END: u64 = 0x8_0000_0000;
/// 64 KiB: the page size of graphics memory and of physical pages
const GRAIN: u64 = 0x1_0000;
const MEM_COMMIT: u32 = 0x1000;
const MEM_RELEASE: u32 = 0x8000;

/// Title memory: the graphics region and the title's physical pages.
pub struct XMem {
    /// free ranges of the graphics region, by address
    free: Vec<(u64, u64)>,
    /// ranges handed out: base → size
    ranges: HashMap<u64, u64>,
    /// heap blocks with more than the heap's alignment: aligned address → block
    aligned: HashMap<u64, u64>,
    /// physical pages: number → where it is mapped, or its contents while it is not
    pages: HashMap<u64, Page>,
    /// address → the physical page mapped there
    mapped: HashMap<u64, u64>,
    next_page: u64,
}

enum Page {
    Mapped(u64),
    Away(Option<Vec<u8>>),
}

impl Default for XMem {
    fn default() -> Self {
        XMem { free: vec![(UMA_BASE, UMA_END - UMA_BASE)], ranges: HashMap::new(), aligned: HashMap::new(), pages: HashMap::new(), mapped: HashMap::new(), next_page: 0x100 }
    }
}

impl XMem {
    /// A range of the graphics region: first fit, `align` a power of two.
    fn take(&mut self, size: u64, align: u64) -> Option<u64> {
        for i in 0..self.free.len() {
            let (base, len) = self.free[i];
            let at = base.checked_add(align - 1)? & !(align - 1);
            let Some(end) = at.checked_add(size) else { continue };
            if end > base + len {
                continue;
            }
            self.free.remove(i);
            if end < base + len {
                self.free.insert(i, (end, base + len - end));
            }
            if at > base {
                self.free.insert(i, (base, at - base));
            }
            self.ranges.insert(at, size);
            return Some(at);
        }
        None
    }

    /// Give a range back; neighbouring free ranges merge.
    fn give(&mut self, base: u64) -> Option<u64> {
        let size = self.ranges.remove(&base)?;
        let i = self.free.partition_point(|&(b, _)| b < base);
        self.free.insert(i, (base, size));
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
        Some(size)
    }

    fn range_of(&self, address: u64) -> Option<(u64, u64)> {
        self.ranges.iter().map(|(&b, &s)| (b, s)).find(|&(b, s)| address >= b && address < b + s)
    }
}

/// XALLOC_ATTRIBUTES: (page size, alignment exponent, memory type).
fn attributes(a: u64) -> (u32, u32, u32) {
    (((a >> 14) & 3) as u32, ((a >> 24) & 0x1F) as u32, ((a >> 29) & 0xF) as u32)
}

/// Size and alignment of a graphics allocation: whole pages of 64 KiB, or 4 MiB (page size 2).
fn graphics_layout(size: u64, page_size: u32, alignment: u32) -> Option<(u64, u64)> {
    let grain = if page_size == 2 { 4 << 20 } else { GRAIN };
    Some((size.max(1).checked_next_multiple_of(grain)?, (1u64 << alignment.min(40)).max(grain)))
}

fn XMemAlloc(g: &mut Guest) -> H {
    let (size, attrs) = (g.arg(0)?, g.arg(1)?);
    let (page_size, alignment, memory_type) = attributes(attrs);
    if memory_type == 0 {
        let align = 1u64 << alignment.clamp(4, 30);
        if align == 16 {
            return g.heap_alloc(size);
        }
        let Some(total) = size.checked_add(align) else { return Ok(0) };
        let block = g.heap_alloc(total)?;
        let at = (block + align - 1) & !(align - 1);
        g.win.xmem.aligned.insert(at, block);
        return Ok(at);
    }
    let Some((size, align)) = graphics_layout(size, page_size, alignment) else { return Ok(0) };
    let Some(at) = g.win.xmem.take(size, align) else { return Ok(0) };
    if g.mem.map(at, size, PROT_R | PROT_W).is_err() {
        g.win.xmem.give(at);
        return Ok(0);
    }
    Ok(at)
}

/// Unmap everything in a range of the graphics region and give the range back.
fn release_range(g: &mut Guest, base: u64) -> bool {
    let Some(size) = g.win.xmem.give(base) else { return false };
    let inside: Vec<u64> = g.mem.regions().iter().map(|r| r.base).filter(|&b| b >= base && b < base + size).collect();
    for b in inside {
        g.mem.unmap(b);
        if let Some(page) = g.win.xmem.mapped.remove(&b) {
            g.win.xmem.pages.insert(page, Page::Away(None));
        }
    }
    true
}

fn XMemFree(g: &mut Guest) -> H {
    let (at, attrs) = (g.arg(0)?, g.arg(1)?);
    if at == 0 {
        return Ok(0);
    }
    if attributes(attrs).2 == 0 {
        let block = g.win.xmem.aligned.remove(&at).unwrap_or(at);
        g.win.heap.free(&mut g.mem, block);
    } else if !release_range(g, at) {
        g.mem.unmap(at);
    }
    Ok(0)
}

/// XMemVirtualAlloc(address, size, type, attributes, protect): graphics memory comes from the
/// graphics region (reserved there until it is committed or pages are mapped into it).
fn XMemVirtualAlloc(g: &mut Guest) -> H {
    let (address, size, kind, attrs, protect) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32, g.arg(3)?, g.arg(4)? as u32);
    let (page_size, alignment, memory_type) = attributes(attrs);
    if memory_type == 0 {
        return Ok(virtual_alloc(g, address, size, protect));
    }
    let Some((size, align)) = graphics_layout(size, page_size, alignment) else { return Ok(0) };
    let base = if address == 0 {
        let Some(at) = g.win.xmem.take(size, align) else { return Ok(0) };
        at
    } else {
        address & !(GRAIN - 1)
    };
    if kind & MEM_COMMIT != 0 {
        // page by page: part of a reservation may be committed (or hold mapped pages) already
        for at in (base..base + size).step_by(GRAIN as usize) {
            if !g.mem.is_mapped(at, GRAIN) && g.mem.map(at, GRAIN, PROT_R | PROT_W).is_err() {
                return Ok(0);
            }
        }
    }
    Ok(base)
}

fn VirtualFree(g: &mut Guest) -> H {
    let (address, kind) = (g.arg(0)?, g.arg(2)? as u32);
    if kind & MEM_RELEASE != 0 {
        return Ok((release_range(g, address) || g.mem.unmap(address)) as u64);
    }
    Ok(1)
}

/// XMemAllocatePhysicalPages(attributes, count in/out, page numbers out)
fn XMemAllocatePhysicalPages(g: &mut Guest) -> H {
    let (count_at, pages) = (g.arg(1)?, g.arg(2)?);
    let count = g.mem.read_u64(count_at)?;
    for i in 0..count {
        let page = g.win.xmem.next_page;
        g.win.xmem.next_page += 1;
        g.win.xmem.pages.insert(page, Page::Away(None));
        g.mem.write_u64(pages + 8 * i, page)?;
    }
    Ok(1)
}

/// Take the physical page at `address` out of the address space; its contents stay with it.
fn unmap_page(g: &mut Guest, address: u64) {
    if let Some(page) = g.win.xmem.mapped.remove(&address) {
        let data = g.mem.slice(address, GRAIN as usize).map(<[u8]>::to_vec);
        g.mem.unmap(address);
        g.win.xmem.pages.insert(page, Page::Away(data));
    }
}

/// XMemMapPhysicalPages(address, count, page numbers): map 64 KiB pages one after the other at
/// `address` (anywhere in the graphics region for none); no page numbers: unmap what is there.
/// A page mapped somewhere else moves, with its contents.
fn XMemMapPhysicalPages(g: &mut Guest) -> H {
    let (address, count, pages) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    let base = if address == 0 {
        let Some(at) = g.win.xmem.take(count * GRAIN, GRAIN) else { return Ok(0) };
        at
    } else {
        address
    };
    if base % GRAIN != 0 {
        return Ok(0);
    }
    for i in 0..count {
        let at = base + i * GRAIN;
        let page = if pages == 0 { 0 } else { g.mem.read_u64(pages + 8 * i)? };
        if g.win.xmem.mapped.get(&at) == Some(&page) {
            continue;
        }
        unmap_page(g, at);
        if page == 0 {
            continue;
        }
        let data = match g.win.xmem.pages.get(&page) {
            Some(Page::Mapped(old)) => {
                let old = *old;
                unmap_page(g, old);
                match g.win.xmem.pages.get_mut(&page) {
                    Some(Page::Away(data)) => data.take(),
                    _ => None,
                }
            }
            Some(Page::Away(_)) => match g.win.xmem.pages.get_mut(&page) {
                Some(Page::Away(data)) => data.take(),
                _ => None,
            },
            None => return Ok(0),
        };
        if g.mem.map(at, GRAIN, PROT_R | PROT_W).is_err() {
            return Ok(0);
        }
        if let Some(data) = data {
            g.mem.write(at, &data)?;
        }
        g.win.xmem.pages.insert(page, Page::Mapped(at));
        g.win.xmem.mapped.insert(at, page);
    }
    Ok(base)
}

fn XMemFreePhysicalPages(g: &mut Guest) -> H {
    let (count, pages) = (g.arg(0)?, g.arg(1)?);
    for i in 0..count {
        let page = g.mem.read_u64(pages + 8 * i)?;
        if let Some(Page::Mapped(at)) = g.win.xmem.pages.remove(&page) {
            g.win.xmem.mapped.remove(&at);
            g.mem.unmap(at);
        }
    }
    Ok(1)
}

/// The title's memory budget and what of it is in use (the guest memory that is mapped).
fn budget(g: &Guest) -> (u64, u64) {
    let total = xcl::state::config().title_memory_budget_mb() * 1024 * 1024;
    let used: u64 = g.mem.regions().iter().map(|r| r.data.len() as u64).sum();
    (total, used.min(total))
}

/// XMemGetWorkingSetStatistics(flags, statistics): the limit and the use (the rest of the
/// structure differs between GDK versions and is left alone).
fn XMemGetWorkingSetStatistics(g: &mut Guest) -> H {
    let at = g.arg(1)?;
    let (total, used) = budget(g);
    g.mem.write_u64(at, total)?;
    g.mem.write_u64(at + 8, used)?;
    Ok(1)
}

/// TITLEMEMORYSTATUS (80 bytes): total, available, three legacy fields, title used / available,
/// two page-table fields.
fn TitleMemoryStatus(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    if at == 0 || g.mem.read_u32(at)? < 80 {
        g.set_last_error(87); // ERROR_INVALID_PARAMETER
        return Ok(0);
    }
    let (total, used) = budget(g);
    for (i, v) in [total, total - used, 0, 0, 0, used, total - used, 0, 0].into_iter().enumerate() {
        g.mem.write_u64(at + 8 + 8 * i as u64, v)?;
    }
    Ok(1)
}

fn GetConsoleType(_: &mut Guest) -> H {
    Ok(xcl::state::config().console_type_raw as u64)
}

/// XSystemGetDeviceType: Xbox Series S (6) or X (7)
fn XSystemGetDeviceType(_: &mut Guest) -> H {
    Ok(if xcl::state::config().is_series_s() { 6 } else { 7 })
}

/// GetSystemOSVersion(out): major, minor (bytes), build, revision (words)
fn GetSystemOSVersion(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    if at != 0 {
        g.mem.write(at, &[10, 0])?;
        g.mem.write_u16(at + 2, 22621)?;
        g.mem.write_u16(at + 4, 0)?;
    }
    Ok(0)
}

/// The thread a handle stands for (the pseudo handle: the running one).
fn thread_of(g: &Guest, handle: u64) -> Option<usize> {
    if handle == u64::MAX - 1 {
        return Some(g.current);
    }
    match g.win.handles.get(&handle)? {
        Object::Thread(i) => Some(*i),
        _ => None,
    }
}

fn SetThreadName(g: &mut Guest) -> H {
    let (handle, name) = (g.arg(0)?, g.arg(1)?);
    let Some(thread) = thread_of(g, handle) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    g.threads[thread].name = g.mem.wstr(name, 1024)?;
    Ok(1)
}

/// GetThreadName(thread, buffer, length, length out)
fn GetThreadName(g: &mut Guest) -> H {
    let (handle, at, len, used) = (g.arg(0)?, g.arg(1)?, g.arg(2)?, g.arg(3)?);
    let Some(thread) = thread_of(g, handle) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    let name: Vec<u16> = g.threads[thread].name.encode_utf16().collect();
    if used != 0 {
        g.mem.write_u64(used, name.len() as u64)?;
    }
    if at == 0 || len <= name.len() as u64 {
        g.set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return Ok(0);
    }
    for (i, &c) in name.iter().chain(&[0]).enumerate() {
        g.mem.write_u16(at + 2 * i as u64, c)?;
    }
    Ok(1)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; XMemAlloc, XMemFree, XMemVirtualAlloc, VirtualFree, XMemAllocatePhysicalPages, XMemMapPhysicalPages, XMemFreePhysicalPages);
    register!(map; XMemGetWorkingSetStatistics, TitleMemoryStatus, GetConsoleType, XSystemGetDeviceType, GetSystemOSVersion, SetThreadName, GetThreadName);
    map.insert("JobTitleMemoryStatus", TitleMemoryStatus as Handler);
}
