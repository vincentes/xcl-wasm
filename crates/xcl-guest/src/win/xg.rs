//! `xg_xs.dll`, the console's texture layout library, on the guest: where the mips and texels
//! of a texture go in the title's own memory. The answers are XCL's (`xcl::hle::xg::Texture`,
//! its model of the console's GFX10 tiling, checked on retail textures); this module is the
//! library's interface on guest memory, and the texel copies the software D3D12.X device
//! shares (`read_level` / `write_level`).
//!
//! `XGCreateTextureComputer(D3D12_RESOURCE_DESC*, computer**)` gives an object (no
//! QueryInterface): AddRef, Release, GetResourceLayout, GetResourceSizeBytes,
//! GetResourceBaseAlignmentBytes, GetMipLevelOffsetBytes, GetTexelElementOffsetBytes, one that
//! is not implemented, CopyIntoSubresource, CopyFromSubresource. GetResourceTiling (#10) is
//! not there yet.

#![allow(non_snake_case)]

use std::collections::HashMap;

use xcl::hle::xg::{subresource, Texture, LAYOUT_BYTES};
use xcl_emu::mem::Memory;

use super::register;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const E_POINTER: u64 = 0x8000_4003;
const E_INVALIDARG: u64 = 0x8007_0057;
const E_NOTIMPL: u64 = 0x8000_4001;

#[derive(Default)]
pub struct State {
    /// the computers handed out, by their address
    computers: HashMap<u64, Texture>,
}

/// The texels of a subresource (mip + slice × mips) of plane `plane` of the texture at `base`,
/// untiled: rows of the level's width, one after the other (a volume: slice after slice).
pub(super) fn read_level(mem: &Memory, tex: &Texture, base: u64, plane: u32, sub: u32) -> Option<Vec<u8>> {
    let (s, at, i, k) = subresource(tex, plane, sub)?;
    if s.depth > 0 {
        let mut tiled = vec![0u8; s.size() as usize];
        mem.read(base + at, &mut tiled).ok()?;
        return Some(s.chain.detile_volume(&tiled, i));
    }
    let mut tiled = vec![0u8; s.chain.slice_bytes];
    mem.read(base + at + (k * s.chain.slice_bytes) as u64, &mut tiled).ok()?;
    Some(s.chain.detile_level(&tiled, i, k))
}

/// Put untiled texels (as `read_level` gives them) into the texture's memory.
pub(super) fn write_level(mem: &mut Memory, tex: &Texture, base: u64, plane: u32, sub: u32, linear: &[u8]) -> Option<()> {
    let (s, at, i, k) = subresource(tex, plane, sub)?;
    let m = s.chain.levels[i];
    if linear.len() < m.width * m.height * m.depth.max(1) * s.bpe as usize {
        return None;
    }
    let (from, len) = if s.depth > 0 { (base + at, s.size() as usize) } else { (base + at + (k * s.chain.slice_bytes) as u64, s.chain.slice_bytes) };
    let mut tiled = vec![0u8; len];
    mem.read(from, &mut tiled).ok()?;
    if s.depth > 0 {
        s.chain.tile_volume(&mut tiled, i, linear);
    } else {
        s.chain.tile_level(&mut tiled, i, k, linear);
    }
    mem.write(from, &tiled).ok()
}

/// (elements a row, rows, slices of a volume, bytes an element) of a subresource.
pub(super) fn level_shape(tex: &Texture, plane: u32, sub: u32) -> Option<(usize, usize, usize, usize)> {
    let (s, _, i, _) = subresource(tex, plane, sub)?;
    let m = s.chain.levels[i];
    Some((m.width, m.height, if s.depth > 0 { m.depth.max(1) } else { 1 }, s.bpe as usize))
}

/// XGCreateTextureComputer(desc, computer out)
fn XGCreateTextureComputer(g: &mut Guest) -> H {
    let (desc, out) = (g.arg(0)?, g.arg(1)?);
    if desc == 0 || out == 0 {
        return Ok(E_POINTER);
    }
    g.mem.write_u64(out, 0)?;
    let d: [u8; 0x38] = g.mem.read_n(desc)?;
    match Texture::new(&d) {
        Ok(texture) => {
            let methods = ["XG_AddRef", "XG_Release", "XG_GetResourceLayout", "XG_GetResourceSizeBytes", "XG_GetResourceBaseAlignmentBytes", "XG_GetMipLevelOffsetBytes", "XG_GetTexelElementOffsetBytes", "XGTextureAddressComputer::#7", "XG_CopyIntoSubresource", "XG_CopyFromSubresource", "XGTextureAddressComputer::#10", "XGTextureAddressComputer::#11"];
            let computer = g.com_object("XGTextureAddressComputer", &methods)?;
            g.win.xg.computers.insert(computer, texture);
            g.mem.write_u64(out, computer)?;
            Ok(0)
        }
        Err(hr) => {
            g.log.push(format!("XGCreateTextureComputer {:#x}: {:x?}", hr as u32, &d[..0x34]));
            Ok(hr as u32 as u64)
        }
    }
}

fn XG_AddRef(_: &mut Guest) -> H {
    Ok(2)
}

fn XG_Release(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    g.win.xg.computers.remove(&this);
    Ok(0)
}

/// GetResourceLayout(XG_RESOURCE_LAYOUT out)
fn XG_GetResourceLayout(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(1)?);
    let Some(texture) = g.win.xg.computers.get(&this) else { return Ok(E_INVALIDARG) };
    if out == 0 {
        return Ok(E_POINTER);
    }
    let mut layout = Box::new([0u8; LAYOUT_BYTES]);
    texture.write(&mut layout);
    g.mem.write(out, &layout[..])?;
    Ok(0)
}

fn XG_GetResourceSizeBytes(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(g.win.xg.computers.get(&this).map_or(0, Texture::size))
}

fn XG_GetResourceBaseAlignmentBytes(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    Ok(g.win.xg.computers.get(&this).map_or(0, Texture::alignment))
}

/// GetMipLevelOffsetBytes(plane, mip)
fn XG_GetMipLevelOffsetBytes(g: &mut Guest) -> H {
    let (this, plane, mip) = (g.arg(0)?, g.arg(1)? as u32, g.arg(2)? as usize);
    Ok(match g.win.xg.computers.get(&this).and_then(|t| t.plane_surface(plane)) {
        Some((s, at)) if mip < s.chain.levels.len() => at + s.mip_offset(mip),
        _ => 0,
    })
}

/// GetTexelElementOffsetBytes(plane, mip, x, y, slice, sample, bit offset out)
fn XG_GetTexelElementOffsetBytes(g: &mut Guest) -> H {
    let (this, plane, mip, x, y, slice, bit) = (g.arg(0)?, g.arg(1)? as u32, g.arg(2)? as usize, g.arg(3)? as usize, g.arg(4)? as u32 as usize, g.arg(5)? as u32, g.arg(7)?);
    if bit != 0 {
        g.mem.write_u32(bit, 0)?;
    }
    Ok(match g.win.xg.computers.get(&this).and_then(|t| t.plane_surface(plane)) {
        Some((s, at)) if mip < s.chain.levels.len() && slice < s.slices(mip).0 => at + s.element_offset(mip, x, y, slice as usize),
        _ => 0,
    })
}

/// CopyIntoSubresource(resource memory, plane, subresource, rows, row pitch, slice pitch):
/// rows of texels → their place in the texture; CopyFromSubresource(rows out, plane,
/// subresource, resource memory, row pitch, slice pitch): the other way.
fn copy(g: &mut Guest, into: bool) -> H {
    let (this, dst, plane, sub, src, row_pitch, slice_pitch) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32, g.arg(3)? as u32, g.arg(4)?, g.arg(5)? as u32 as u64, g.arg(6)? as u32 as u64);
    let Some(texture) = g.win.xg.computers.get(&this) else { return Ok(E_INVALIDARG) };
    let Some((width, height, depth, bytes)) = level_shape(texture, plane, sub).filter(|_| dst != 0 && src != 0) else { return Ok(E_INVALIDARG) };
    let row = width * bytes;
    // where row `n` (of all slices) is among the caller's rows
    let at = |n: usize| (n / height) as u64 * slice_pitch + (n % height) as u64 * row_pitch;
    if into {
        let mut linear = vec![0u8; row * height * depth];
        for (n, to) in linear.chunks_exact_mut(row).enumerate() {
            g.mem.read(src + at(n), to)?;
        }
        Ok(if write_level(&mut g.mem, texture, dst, plane, sub, &linear).is_some() { 0 } else { E_INVALIDARG })
    } else {
        let Some(linear) = read_level(&g.mem, texture, src, plane, sub) else { return Ok(E_INVALIDARG) };
        for (n, from) in linear.chunks_exact(row).enumerate() {
            g.mem.write(dst + at(n), from)?;
        }
        Ok(0)
    }
}

fn XG_CopyIntoSubresource(g: &mut Guest) -> H {
    copy(g, true)
}

fn XG_CopyFromSubresource(g: &mut Guest) -> H {
    copy(g, false)
}

fn not_implemented(_: &mut Guest) -> H {
    Ok(E_NOTIMPL)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; XGCreateTextureComputer, XG_AddRef, XG_Release, XG_GetResourceLayout, XG_GetResourceSizeBytes, XG_GetResourceBaseAlignmentBytes);
    register!(map; XG_GetMipLevelOffsetBytes, XG_GetTexelElementOffsetBytes, XG_CopyIntoSubresource, XG_CopyFromSubresource);
    map.insert("XGTextureAddressComputer::#7", not_implemented as Handler);
}
