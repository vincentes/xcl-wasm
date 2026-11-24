//! Root signature layouts: where each root parameter's argument sits, and which arguments a
//! console shader gets in its user SGPRs. Data only (no D3D12), so the hosts that are not
//! Windows use it too; `d3d12x::console_list` is where the Windows runtime applies it.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    /// `count` 32-bit constants at byte `at`
    Constants { at: u32, count: u32 },
    /// root CBV / SRV / UAV: 64-bit GPU address at `at`
    Descriptor { at: u32 },
    /// descriptor table: 32-bit GPU address of its first descriptor (console shaders load
    /// it with the high half 0; amd_fidelityfx_upscaler_xbox.dll writes 1 word)
    Table { at: u32 },
}

/// Where a root signature's arguments go: all of them packed in order (`at`, `size` bytes);
/// the first `inline` bytes of that are passed in user SGPRs (compute from s4, graphics
/// stages from s8: `inline_gfx` bytes), the rest is the table.
#[derive(Debug, Default)]
pub struct Layout {
    pub params: Vec<Param>,
    pub size: u32,
    pub inline: u32,
    pub inline_gfx: u32,
}

/// User SGPRs the console's compiler gives root arguments: compute s4…s15 (s0…s3 are the
/// driver's), graphics stages (VS, PS) s8…s31 (s0…s7: the driver's, s[4:5] of a VS the
/// fetch shader) – both measured on its output (`tests/console_shaders.rs`).
pub const INLINE_SGPRS: u32 = 12;
pub const INLINE_SGPRS_GFX: u32 = 24;
pub const FIRST_SGPR: usize = 4;
pub const FIRST_SGPR_GFX: usize = 8;

/// Layout from a serialized root signature: a DXBC container with RTS0 or a bare RTS0.
pub fn layout_from_blob(blob: &[u8]) -> Option<Layout> {
    let rts0 = match crate::gpu::shader::container_parts(blob) {
        Some(parts) => {
            let p = parts.into_iter().find(|p| p.fourcc == "RTS0")?;
            blob.get(p.offset..p.offset + p.size)?.to_vec()
        }
        None => blob.to_vec(),
    };
    let r = |o: usize| rts0.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let (n, poff) = (r(4)? as usize, r(8)? as usize);
    let mut l = Layout::default();
    // the parameters go into user SGPRs in order while each fits whole; from the first one
    // that does not, the rest go to the table (packed from its start) – as the console's
    // compiler expects them (HANDOFF: root argument binding)
    let (mut spilled, mut spilled_gfx) = (false, false);
    for k in 0..n {
        let (ty, pay) = (r(poff + 12 * k)?, r(poff + 12 * k + 8)? as usize);
        let p = match ty {
            1 => Param::Constants { at: l.size, count: r(pay + 8)? },
            2..=4 => Param::Descriptor { at: l.size },
            _ => Param::Table { at: l.size },
        };
        let bytes = match p {
            Param::Constants { count, .. } => 4 * count,
            Param::Table { .. } => 4,
            Param::Descriptor { .. } => 8,
        };
        spilled |= l.inline + bytes > 4 * INLINE_SGPRS;
        if !spilled {
            l.inline += bytes;
        }
        spilled_gfx |= l.inline_gfx + bytes > 4 * INLINE_SGPRS_GFX;
        if !spilled_gfx {
            l.inline_gfx += bytes;
        }
        l.size += bytes;
        l.params.push(p);
    }
    Some(l)
}

/// The user SGPRs of a dispatch: s[0:1] = `table_at` (where the bytes of `args` past the
/// inline part are), from s4 the inline part of `args` (all arguments packed by `l`).
pub fn user_data(l: &Layout, args: &[u8], table_at: u64) -> Vec<u32> {
    sgprs(FIRST_SGPR, l.inline, args, table_at)
}

/// Bytes of a compute dispatch's arguments that go in user SGPRs: the layout's rule, but
/// never more than the shader itself has registers for. `user_sgprs` is the shader's own
/// count (`real::pso_user_sgprs`; None = not known, the rule alone). The console's newer
/// compiler follows the rule; XboxBvhBuilder.dll's shaders were built with four user SGPRs
/// whatever their root signature (s4 is the group id) and take every argument from the
/// table, also where the first ones would fit. Parameters stay whole: the inline part ends
/// where the last parameter that fits the shader's registers ends.
pub fn inline_for(l: &Layout, user_sgprs: Option<u32>) -> u32 {
    let Some(user) = user_sgprs else { return l.inline };
    let has = 4 * user.saturating_sub(FIRST_SGPR as u32);
    if l.inline <= has {
        return l.inline;
    }
    let end = |p: &Param| match *p {
        Param::Constants { at, count } => at + 4 * count,
        Param::Descriptor { at } => at + 8,
        Param::Table { at } => at + 4,
    };
    l.params.iter().map(end).take_while(|&e| e <= has).last().unwrap_or(0)
}

/// `user_data` for a shader with `user_sgprs` user SGPRs (`inline_for`).
pub fn user_data_for(l: &Layout, user_sgprs: Option<u32>, args: &[u8], table_at: u64) -> Vec<u32> {
    sgprs(FIRST_SGPR, inline_for(l, user_sgprs), args, table_at)
}

/// `user_data` of a draw: the vertex and pixel shader's user SGPRs (inline from s8).
pub fn user_data_gfx(l: &Layout, args: &[u8], table_at: u64) -> Vec<u32> {
    sgprs(FIRST_SGPR_GFX, l.inline_gfx, args, table_at)
}

fn sgprs(first: usize, inline: u32, args: &[u8], table_at: u64) -> Vec<u32> {
    let mut s = vec![0u32; first];
    s[0] = table_at as u32;
    s[1] = (table_at >> 32) as u32;
    let inline = &args[..(inline as usize).min(args.len())];
    s.extend(inline.chunks(4).map(|c| {
        let mut w = [0u8; 4];
        w[..c.len()].copy_from_slice(c);
        u32::from_le_bytes(w)
    }));
    s
}
