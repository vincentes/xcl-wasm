//! RDNA2 → WGSL: the recompiler's back end for WebGPU. **A beginning**: a first set of
//! instructions – 32-bit float and integer arithmetic, compares, selects, the scalar unit,
//! branches, exports and attribute interpolation – as a kernel (the program runs on a register
//! file, as the HLSL back end's instruction tests do), a vertex stage or a pixel stage. Memory
//! is read (dword loads through a pointer or a buffer descriptor), not written. Textures: 2D
//! samples and loads (see below).
//!
//! WGSL has no wave of 64 lanes to rely on (the HLSL back end pins one with `WaveSize(64)` and
//! uses wave intrinsics). A program is written for a wave of `LANES` lanes: a vector register
//! is an array of `LANES` values and a vector instruction a loop over the lanes EXEC has on;
//! VCC and EXEC are two 32-bit halves and a compare sets the bits of its lanes. In a kernel
//! **one invocation runs the whole wave** (`LANES` = 64), which makes lane masks exact without
//! any subgroup feature. A vertex or a pixel is a wave of one lane (`LANES` = 1): the same
//! text, with EXEC's bit 0 for "this vertex / pixel is live". (What a pixel needs from its
//! neighbours – derivatives – will come from WGSL's own, not from lanes.)
//!
//! Control flow is the dispatcher loop (`switch pc` over basic blocks), exact for any branch.
//!
//! Kernel ABI (the HLSL back end's): `io` holds 128 SGPR slots (106 / 107 VCC, 126 / 127 EXEC,
//! 125 SCC) and then, per lane, 256 VGPRs; `res` gets the same back, and after it the exports
//! as raw dwords (`KERNEL_EXPORTS + (target · 64 + lane) · 4 + component`).
//!
//! Images. The HLSL back end takes a texture from the descriptor heap by the index its T#
//! holds when the instruction runs; WebGPU has no heap to index, so a texture is a binding
//! and whoever draws has to know **before the draw** which texture each one is. Every image
//! instruction therefore gets a slot (`Translation::images`; texture at group 1 binding 2·slot,
//! sampler at 2·slot + 1) with where its descriptors come from (`Origin`: registers the
//! program starts with, or a chain of loads from a pointer in them), found by following the
//! registers back through the program's loads. A descriptor the program computes any other
//! way is refused.
//!
//! Memory is one read-only array of dwords (`mem`, declared when the program loads: binding 2
//! of a kernel, 1 of a stage) addressed by the low half of a pointer; a load outside it is 0.
//!
//! Stage ABI: the SGPRs come from a uniform buffer (`sg`, group 0 binding 0: 32 × `vec4<u32>`).
//! A vertex stage gets attribute `k` at `@location(k)` (`vec4<f32>`, in four VGPRs from
//! `attr_base + 4k`), writes `exp pos0` to the position and `exp param N` to `@location(N)`.
//! A pixel stage reads attribute `N` at `@location(N)`, gets the system values
//! SPI_PS_INPUT_ENA names in its first VGPRs, writes `exp mrt N` to `@location(N)` and is
//! discarded when no export ran for it (it was killed).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use super::{ps_input_vgpr, KERNEL_EXPORTS, KERNEL_LANES, KERNEL_SCC_SLOT, KERNEL_SGPRS, KERNEL_VGPRS};
use crate::gpu::rdna2::{self, Decoded};

pub struct Translation {
    pub wgsl: String,
    /// dwords of the `io` and `res` buffers (a kernel)
    pub dwords: usize,
    /// the program reads memory: `mem` is declared
    pub memory: bool,
    /// the textures and samplers the program uses, by slot
    pub images: Vec<Image>,
}

/// Where a descriptor an image instruction uses comes from: with no `loads`, the registers
/// from `sgpr` as the program starts; else the pointer in `sgpr` (and the next register), each
/// offset but the last a pointer loaded from it, the last the descriptor's place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    pub sgpr: u64,
    pub loads: Vec<u32>,
}

/// A texture (and the sampler it is read with) of a program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub texture: Origin,
    /// None: only loaded (`image_load`)
    pub sampler: Option<Origin>,
}

/// An image instruction this back end has, taken apart.
struct ImageOp {
    /// a sample (else a load of a texel)
    sample: bool,
    /// address registers: bias, the coordinates, lod / mip
    bias: Option<u64>,
    coords: [u64; 2],
    lod: Option<u64>,
    /// `_lz`: level 0
    zero: bool,
    vdata: u64,
    dmask: u64,
    srsrc: u64,
    ssamp: u64,
}

fn image_op(d: &Decoded, name: &str) -> Result<ImageOp, String> {
    let get = |n: &str| field(d, n).unwrap_or(0);
    let (sample, bias, lod, zero) = match name {
        "image_sample" => (true, false, false, false),
        "image_sample_lz" => (true, false, false, true),
        "image_sample_l" => (true, false, true, false),
        "image_sample_b" => (true, true, false, false),
        "image_load" => (false, false, false, false),
        "image_load_mip" => (false, false, true, false),
        _ => return Err(name.into()),
    };
    if get("DIM") != 1 {
        return Err(format!("{name} of dimension {}", get("DIM")));
    }
    if ["TFE", "LWE", "A16", "D16"].iter().any(|f| get(f) != 0) {
        return Err(format!("{name} tfe / lwe / a16 / d16"));
    }
    // non-sequential addresses: VADDR, then VADDRA, VADDRB … of the longer encoding
    let nsa = get("NSA");
    let wide = rdna2::encoding(&format!("MIMG_NSA{nsa}"));
    let addr = |j: u64| match wide {
        Some(e) if nsa > 0 && j > 0 => rdna2::field_value(e, &d.words, &format!("VADDR{}", (b'A' + (j - 1) as u8) as char)).unwrap_or(0),
        _ => get("VADDR") + j,
    };
    let first = bias as u64;
    Ok(ImageOp {
        sample,
        bias: bias.then(|| addr(0)),
        coords: [addr(first), addr(first + 1)],
        lod: lod.then(|| addr(first + 2)),
        zero,
        vdata: get("VDATA"),
        dmask: get("DMASK"),
        srsrc: register(d, "SRSRC")?,
        ssamp: register(d, "SSAMP")?,
    })
}

/// Where SGPR `reg` as the instruction `insts[at]` reads it comes from (see `Origin`).
fn origin(insts: &[Decoded], at: usize, reg: u64) -> Result<Origin, String> {
    let Some(def) = (0..at).rev().find(|&k| rdna2::sgpr_uses(&insts[k]).1.contains(&reg)) else {
        return Ok(Origin { sgpr: reg, loads: Vec::new() });
    };
    let d = &insts[def];
    let name = d.inst.name.to_ascii_lowercase();
    let get = |n: &str| field(d, n).unwrap_or(0);
    if !name.starts_with("s_load_dword") || get("SOFFSET") != 125 {
        return Err(format!("a descriptor in s{reg}, which {name} at +{:#x} writes", d.pc));
    }
    let mut from = origin(insts, def, register(d, "SBASE")?)?;
    let imm = (((get("OFFSET") as u32) << 11) as i32 >> 11) as u32;
    from.loads.push(imm.wrapping_add(4 * (reg - get("SDATA")) as u32));
    Ok(from)
}

/// Dwords a load instruction reads (None: not a dword load).
fn load_count(name: &str) -> Option<u64> {
    let rest = ["s_buffer_load_", "s_load_", "buffer_load_"].iter().find_map(|p| name.strip_prefix(p))?;
    Some(match rest {
        "dword" => 1,
        "dwordx2" => 2,
        "dwordx3" => 3,
        "dwordx4" => 4,
        "dwordx8" => 8,
        "dwordx16" => 16,
        _ => return None,
    })
}

/// A register field as the register's number (`SBASE` counts pairs, `SRSRC` fours).
fn register(d: &Decoded, name: &str) -> Result<u64, String> {
    Ok(field(d, name).ok_or(name)? * rdna2::register_scale(name))
}

/// What a program is translated as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Kernel,
    /// `attributes` vertex attributes, four VGPRs each from `attr_base`
    Vertex { attr_base: u64, attributes: u32 },
    /// SPI_PS_INPUT_ENA: the system values in the first VGPRs
    Pixel { input_ena: u32 },
}

fn field(d: &Decoded, name: &str) -> Option<u64> {
    rdna2::field_value(d.enc, &d.words, name)
}

/// A 32-bit source operand by its 9-bit code, as a WGSL `u32` expression (`l`: the lane).
fn source(code: u64, literal: Option<u32>) -> Result<String, String> {
    Ok(match code {
        0..=105 | 124 => format!("s[{code}]"),
        106 => "vcc.x".into(),
        107 => "vcc.y".into(),
        125 | 128 => "0u".into(),
        126 => "exec.x".into(),
        127 => "exec.y".into(),
        129..=192 => format!("{}u", code - 128),
        193..=208 => format!("{}u", (-((code - 192) as i32)) as u32),
        // 0.5 -0.5 1 -1 2 -2 4 -4, 1 / (2π)
        240..=248 => format!("{}u", [0x3F00_0000u32, 0xBF00_0000, 0x3F80_0000, 0xBF80_0000, 0x4000_0000, 0xC000_0000, 0x4080_0000, 0xC080_0000, 0x3E22_F983][(code - 240) as usize]),
        251 => "u32(all(vcc == vec2<u32>(0u)))".into(),
        252 => "u32(all(exec == vec2<u32>(0u)))".into(),
        253 => "u32(scc)".into(),
        255 => format!("{}u", literal.ok_or("a literal operand without its literal")?),
        256..=511 => format!("v{}[l]", code - 256),
        _ => return Err(format!("source operand {code}")),
    })
}

/// A 64-bit scalar source: a register pair, VCC, EXEC or a constant, as `vec2<u32>` (low, high).
fn source64(code: u64, literal: Option<u32>) -> Result<String, String> {
    Ok(match code {
        0..=104 => format!("vec2<u32>(s[{code}], s[{}])", code + 1),
        106 => "vcc".into(),
        126 => "exec".into(),
        128..=192 => format!("vec2<u32>({}u, 0u)", code - 128),
        193..=208 => format!("vec2<u32>({}u, 0xffffffffu)", (-((code - 192) as i32)) as u32),
        255 => format!("vec2<u32>({}u, 0u)", literal.ok_or("a literal operand without its literal")?),
        _ => return Err(format!("64-bit source operand {code}")),
    })
}

/// Assign a 32-bit value to a scalar destination.
fn store(code: u64, value: &str) -> Result<String, String> {
    Ok(match code {
        0..=105 | 124 => format!("s[{code}] = {value};"),
        106 => format!("vcc.x = {value};"),
        107 => format!("vcc.y = {value};"),
        125 => format!("_ = {value};"),
        126 => format!("exec.x = {value};"),
        127 => format!("exec.y = {value};"),
        _ => return Err(format!("scalar destination {code}")),
    })
}

fn store64(code: u64, value: &str) -> Result<String, String> {
    Ok(match code {
        0..=104 => format!("{{ let t = {value}; s[{code}] = t.x; s[{}] = t.y; }}", code + 1),
        106 => format!("vcc = {value};"),
        125 => format!("_ = {value};"),
        126 => format!("exec = {value};"),
        _ => return Err(format!("64-bit scalar destination {code}")),
    })
}

/// A compare of `a` and `b` (expressions of type `ty`) by the instruction's condition name.
fn condition(op: &str, a: &str, b: &str, float: bool) -> Option<String> {
    // unordered float compares are the negation of the ordered opposite
    let nan = format!("({a} != {a} || {b} != {b})");
    Some(match (op, float) {
        ("f", _) => "false".into(),
        ("tru", _) | ("t", _) => "true".into(),
        ("lt", _) => format!("({a} < {b})"),
        ("eq", _) => format!("({a} == {b})"),
        ("le", _) => format!("({a} <= {b})"),
        ("gt", _) => format!("({a} > {b})"),
        ("lg", true) => format!("({a} < {b} || {a} > {b})"),
        ("lg", false) | ("ne", false) => format!("({a} != {b})"),
        ("ge", _) => format!("({a} >= {b})"),
        ("o", true) => format!("!{nan}"),
        ("u", true) => nan,
        ("nge", true) => format!("!({a} >= {b})"),
        ("nlg", true) => format!("!({a} < {b} || {a} > {b})"),
        ("ngt", true) => format!("!({a} > {b})"),
        ("nle", true) => format!("!({a} <= {b})"),
        ("neq", true) => format!("!({a} == {b})"),
        ("nlt", true) => format!("!({a} < {b})"),
        _ => return None,
    })
}

fn f(e: &str) -> String {
    format!("bitcast<f32>({e})")
}

fn i(e: &str) -> String {
    format!("bitcast<i32>({e})")
}

fn fr(e: String) -> String {
    format!("bitcast<u32>({e})")
}

/// Operand modifiers of a float instruction (VOP3): |x| and −x on sources, a scale and a
/// clamp to 0..1 on the result.
#[derive(Default)]
struct Mods {
    abs: u64,
    neg: u64,
    omod: u64,
    clamp: bool,
}

impl Mods {
    fn any(&self) -> bool {
        self.abs != 0 || self.neg != 0 || self.omod != 0 || self.clamp
    }

    /// Source `n` (a `u32` expression) as a float.
    fn src(&self, n: usize, e: &str) -> String {
        let mut s = f(e);
        if self.abs >> n & 1 == 1 {
            s = format!("abs({s})");
        }
        if self.neg >> n & 1 == 1 {
            s = format!("(-{s})");
        }
        s
    }

    /// A float result as the register's bits.
    fn res(&self, e: String) -> String {
        let e = match self.omod {
            1 => format!("(({e}) * 2.0)"),
            2 => format!("(({e}) * 4.0)"),
            3 => format!("(({e}) * 0.5)"),
            _ => e,
        };
        fr(if self.clamp { format!("clamp({e}, 0.0, 1.0)") } else { e })
    }
}

/// The value a vector instruction writes, from its sources `a` `b` `c` (`u32` expressions),
/// what the destination holds (`cur`) and the instruction's constant (`k`).
fn vector_value(name: &str, a: &str, b: &str, c: &str, m: &Mods, cur: &str, k: &str) -> Option<String> {
    let (fa, fb, fc, ia, ib, ic) = (m.src(0, a), m.src(1, b), m.src(2, c), i(a), i(b), i(c));
    let (fcur, fk) = (f(cur), f(k));
    let fr = |e: String| m.res(e);
    Some(match name {
        "v_mov_b32" => a.to_string(),
        "v_not_b32" => format!("~{a}"),
        "v_bfrev_b32" => format!("reverseBits({a})"),
        "v_ffbh_u32" => format!("ffbh({a})"),
        "v_ffbl_b32" => format!("ffbl({a})"),
        "v_ffbh_i32" => format!("ffbh_i({a})"),
        "v_add_f32" => fr(format!("{fa} + {fb}")),
        "v_sub_f32" => fr(format!("{fa} - {fb}")),
        "v_subrev_f32" => fr(format!("{fb} - {fa}")),
        "v_mul_f32" => fr(format!("{fa} * {fb}")),
        "v_mul_legacy_f32" => fr(format!("mul_legacy({fa}, {fb})")),
        "v_min_f32" => fr(format!("min({fa}, {fb})")),
        "v_max_f32" => fr(format!("max({fa}, {fb})")),
        "v_fmac_f32" => fr(format!("fma({fa}, {fb}, {fcur})")),
        "v_mac_f32" => fr(format!("mad_nf({fa}, {fb}, {fcur})")),
        "v_fma_f32" => fr(format!("fma({fa}, {fb}, {fc})")),
        "v_mad_f32" => fr(format!("mad_nf({fa}, {fb}, {fc})")),
        "v_fma_legacy_f32" => fr(format!("fma_legacy({fa}, {fb}, {fc})")),
        // (the constant is the second or the third operand; the register is VSRC1)
        "v_fmamk_f32" => fr(format!("fma({fa}, {fk}, {fb})")),
        "v_fmaak_f32" => fr(format!("fma({fa}, {fb}, {fk})")),
        "v_madmk_f32" => fr(format!("mad_nf({fa}, {fk}, {fb})")),
        "v_madak_f32" => fr(format!("mad_nf({fa}, {fb}, {fk})")),
        "v_min3_f32" => fr(format!("min(min({fa}, {fb}), {fc})")),
        "v_max3_f32" => fr(format!("max(max({fa}, {fb}), {fc})")),
        "v_med3_f32" => fr(format!("med3f({fa}, {fb}, {fc})")),
        "v_cubeid_f32" => fr(format!("cube_id({fa}, {fb}, {fc})")),
        "v_cubesc_f32" => fr(format!("cube_sc({fa}, {fb}, {fc})")),
        "v_cubetc_f32" => fr(format!("cube_tc({fa}, {fb}, {fc})")),
        "v_cubema_f32" => fr(format!("cube_ma({fa}, {fb}, {fc})")),
        "v_floor_f32" => fr(format!("floor({fa})")),
        "v_ceil_f32" => fr(format!("ceil({fa})")),
        "v_trunc_f32" => fr(format!("trunc({fa})")),
        "v_rndne_f32" => fr(format!("round({fa})")),
        "v_fract_f32" => fr(format!("fract_f({fa})")),
        "v_sqrt_f32" => fr(format!("sqrt({fa})")),
        "v_rcp_f32" | "v_rcp_iflag_f32" => fr(format!("1.0 / {fa}")),
        "v_rsq_f32" => fr(format!("inverseSqrt({fa})")),
        "v_exp_f32" => fr(format!("exp2({fa})")),
        "v_log_f32" => fr(format!("log2({fa})")),
        // the angle is in revolutions
        "v_sin_f32" => fr(format!("sin({fa} * 6.2831853)")),
        "v_cos_f32" => fr(format!("cos({fa} * 6.2831853)")),
        "v_cvt_f32_i32" => fr(format!("f32({ia})")),
        "v_cvt_f32_u32" => fr(format!("f32({a})")),
        "v_cvt_i32_f32" => format!("cvt_i32({fa})"),
        "v_cvt_u32_f32" => format!("cvt_u32({fa})"),
        "v_cvt_rpi_i32_f32" => format!("cvt_i32(floor({fa} + 0.5))"),
        "v_cvt_flr_i32_f32" => format!("cvt_i32(floor({fa}))"),
        "v_cvt_f32_ubyte0" => fr(format!("f32({a} & 0xffu)")),
        "v_cvt_f32_ubyte1" => fr(format!("f32(({a} >> 8u) & 0xffu)")),
        "v_cvt_f32_ubyte2" => fr(format!("f32(({a} >> 16u) & 0xffu)")),
        "v_cvt_f32_ubyte3" => fr(format!("f32({a} >> 24u)")),
        "v_cvt_f32_f16" => fr(format!("unpack2x16float({a} & 0xffffu).x")),
        "v_cvt_pkrtz_f16_f32" => format!("f16_rtz({fa}) | (f16_rtz({fb}) << 16u)"),
        "v_cvt_pk_u16_u32" => format!("min({a}, 0xffffu) | (min({b}, 0xffffu) << 16u)"),
        "v_cvt_pk_i16_i32" => format!("(bitcast<u32>(clamp({ia}, -32768, 32767)) & 0xffffu) | ((bitcast<u32>(clamp({ib}, -32768, 32767)) & 0xffffu) << 16u)"),
        "v_add_nc_u32" | "v_add_nc_i32" => format!("{a} + {b}"),
        "v_sub_nc_u32" | "v_sub_nc_i32" => format!("{a} - {b}"),
        "v_subrev_nc_u32" => format!("{b} - {a}"),
        "v_mul_lo_u32" => format!("{a} * {b}"),
        "v_mul_hi_u32" => format!("mul_hi({a}, {b})"),
        "v_mul_hi_i32" => format!("mul_hi_i({a}, {b})"),
        "v_mul_u32_u24" => format!("({a} & 0xffffffu) * ({b} & 0xffffffu)"),
        "v_mul_hi_u32_u24" => format!("mul_hi({a} & 0xffffffu, {b} & 0xffffffu)"),
        "v_mul_i32_i24" => fr(format!("i24({a}) * i24({b})")),
        "v_mad_u32_u24" => format!("({a} & 0xffffffu) * ({b} & 0xffffffu) + {c}"),
        "v_mad_i32_i24" => fr(format!("i24({a}) * i24({b}) + {ic}")),
        "v_and_b32" => format!("{a} & {b}"),
        "v_or_b32" => format!("{a} | {b}"),
        "v_xor_b32" => format!("{a} ^ {b}"),
        "v_xnor_b32" => format!("~({a} ^ {b})"),
        "v_lshlrev_b32" => format!("{b} << ({a} & 31u)"),
        "v_lshrrev_b32" => format!("{b} >> ({a} & 31u)"),
        "v_ashrrev_i32" => fr(format!("{ib} >> ({a} & 31u)")),
        "v_lshl_add_u32" => format!("({a} << ({b} & 31u)) + {c}"),
        "v_add_lshl_u32" => format!("({a} + {b}) << ({c} & 31u)"),
        "v_lshl_or_b32" => format!("({a} << ({b} & 31u)) | {c}"),
        "v_and_or_b32" => format!("({a} & {b}) | {c}"),
        "v_or3_b32" => format!("{a} | {b} | {c}"),
        "v_xor3_b32" => format!("{a} ^ {b} ^ {c}"),
        "v_add3_u32" => format!("{a} + {b} + {c}"),
        "v_xad_u32" => format!("({a} ^ {b}) + {c}"),
        "v_bfe_u32" => format!("bfe_u({a}, {b}, {c})"),
        "v_bfe_i32" => format!("bfe_i({a}, {b}, {c})"),
        "v_bfi_b32" => format!("({a} & {b}) | (~{a} & {c})"),
        "v_bfm_b32" => format!("bfm({a}, {b})"),
        "v_bcnt_u32_b32" => format!("countOneBits({a}) + {b}"),
        "v_alignbit_b32" => format!("alignbit({a}, {b}, {c})"),
        "v_alignbyte_b32" => format!("alignbit({a}, {b}, ({c} & 3u) * 8u)"),
        "v_min_u32" => format!("min({a}, {b})"),
        "v_max_u32" => format!("max({a}, {b})"),
        "v_min_i32" => fr(format!("min({ia}, {ib})")),
        "v_max_i32" => fr(format!("max({ia}, {ib})")),
        "v_min3_u32" => format!("min(min({a}, {b}), {c})"),
        "v_max3_u32" => format!("max(max({a}, {b}), {c})"),
        "v_med3_u32" => format!("max(min({a}, {b}), min(max({a}, {b}), {c}))"),
        "v_min3_i32" => fr(format!("min(min({ia}, {ib}), {ic})")),
        "v_max3_i32" => fr(format!("max(max({ia}, {ib}), {ic})")),
        "v_med3_i32" => fr(format!("max(min({ia}, {ib}), min(max({ia}, {ib}), {ic}))")),
        "v_sad_u32" => format!("(max({a}, {b}) - min({a}, {b})) + {c}"),
        "v_cndmask_b32" => format!("select({a}, {b}, on(vcc, l))"),
        _ => return None,
    })
}

/// A 32-bit scalar instruction: the result and SCC (None: left alone) as expressions over the
/// sources `a` `b`, the result `r` and what the destination holds (`cur`).
fn scalar_value(name: &str) -> Option<(String, Option<&'static str>)> {
    let nz = Some("r != 0u");
    let (res, scc): (&str, Option<&'static str>) = match name {
        "s_mov_b32" => ("a", None),
        "s_cmov_b32" => ("select(cur, a, scc)", None),
        "s_not_b32" => ("~a", nz),
        "s_brev_b32" => ("reverseBits(a)", None),
        "s_bcnt0_i32_b32" => ("32u - countOneBits(a)", nz),
        "s_bcnt1_i32_b32" => ("countOneBits(a)", nz),
        "s_ff0_i32_b32" => ("ffbl(~a)", None),
        "s_ff1_i32_b32" => ("ffbl(a)", None),
        "s_flbit_i32_b32" => ("ffbh(a)", None),
        "s_flbit_i32" => ("ffbh_i(a)", None),
        "s_abs_i32" => ("bitcast<u32>(abs(bitcast<i32>(a)))", nz),
        "s_sext_i32_i8" => ("bitcast<u32>(bitcast<i32>(a << 24u) >> 24u)", None),
        "s_sext_i32_i16" => ("bitcast<u32>(bitcast<i32>(a << 16u) >> 16u)", None),
        "s_and_b32" => ("a & b", nz),
        "s_or_b32" => ("a | b", nz),
        "s_xor_b32" => ("a ^ b", nz),
        "s_andn2_b32" => ("a & ~b", nz),
        "s_orn2_b32" => ("a | ~b", nz),
        "s_nand_b32" => ("~(a & b)", nz),
        "s_nor_b32" => ("~(a | b)", nz),
        "s_xnor_b32" => ("~(a ^ b)", nz),
        "s_add_u32" => ("a + b", Some("r < a")),
        "s_sub_u32" => ("a - b", Some("b > a")),
        "s_addc_u32" => ("a + b + u32(scc)", Some("(a + b < a) || (r < a + b)")),
        "s_subb_u32" => ("a - b - u32(scc)", Some("(b > a) || (scc && b == a)")),
        "s_add_i32" => ("a + b", Some("(((a ^ r) & (b ^ r)) >> 31u) != 0u")),
        "s_sub_i32" => ("a - b", Some("(((a ^ b) & (a ^ r)) >> 31u) != 0u")),
        "s_mul_i32" => ("a * b", None),
        "s_mul_hi_u32" => ("mul_hi(a, b)", None),
        "s_mul_hi_i32" => ("mul_hi_i(a, b)", None),
        "s_lshl_b32" => ("a << (b & 31u)", nz),
        "s_lshr_b32" => ("a >> (b & 31u)", nz),
        "s_ashr_i32" => ("bitcast<u32>(bitcast<i32>(a) >> (b & 31u))", nz),
        "s_lshl1_add_u32" => ("(a << 1u) + b", Some("(a >> 31u) != 0u || r < b")),
        "s_lshl2_add_u32" => ("(a << 2u) + b", Some("(a >> 30u) != 0u || r < b")),
        "s_lshl3_add_u32" => ("(a << 3u) + b", Some("(a >> 29u) != 0u || r < b")),
        "s_lshl4_add_u32" => ("(a << 4u) + b", Some("(a >> 28u) != 0u || r < b")),
        "s_bfm_b32" => ("bfm(a, b)", None),
        "s_bfe_u32" => ("sbfe_u(a, b)", nz),
        "s_bfe_i32" => ("sbfe_i(a, b)", nz),
        "s_min_u32" => ("min(a, b)", Some("a < b")),
        "s_max_u32" => ("max(a, b)", Some("a > b")),
        "s_min_i32" => ("bitcast<u32>(min(bitcast<i32>(a), bitcast<i32>(b)))", Some("bitcast<i32>(a) < bitcast<i32>(b)")),
        "s_max_i32" => ("bitcast<u32>(max(bitcast<i32>(a), bitcast<i32>(b)))", Some("bitcast<i32>(a) > bitcast<i32>(b)")),
        "s_cselect_b32" => ("select(b, a, scc)", None),
        "s_bitset0_b32" => ("cur & ~(1u << (a & 31u))", None),
        "s_bitset1_b32" => ("cur | (1u << (a & 31u))", None),
        "s_absdiff_i32" => ("bitcast<u32>(abs(bitcast<i32>(a) - bitcast<i32>(b)))", nz),
        "s_pack_ll_b32_b16" => ("(a & 0xffffu) | (b << 16u)", None),
        "s_pack_lh_b32_b16" => ("(a & 0xffffu) | (b & 0xffff0000u)", None),
        "s_pack_hh_b32_b16" => ("(a >> 16u) | (b & 0xffff0000u)", None),
        _ => return None,
    };
    Some((res.to_string(), scc))
}

/// The helper functions every program gets. (A float out of an integer's range is converted
/// by hand: WGSL's own conversion stops at the nearest float inside the range, 2^31 − 128.)
const HELPERS: &str = "\
fn on(m: vec2<u32>, l: u32) -> bool { if (l < 32u) { return ((m.x >> l) & 1u) != 0u; } return ((m.y >> (l - 32u)) & 1u) != 0u; }
fn put(m: vec2<u32>, l: u32) -> vec2<u32> { if (l < 32u) { return vec2<u32>(m.x | (1u << l), m.y); } return vec2<u32>(m.x, m.y | (1u << (l - 32u))); }
fn first(m: vec2<u32>) -> u32 { if (m.x != 0u) { return countTrailingZeros(m.x); } if (m.y != 0u) { return 32u + countTrailingZeros(m.y); } return 0u; }
fn ff1_64(m: vec2<u32>) -> u32 { if (all(m == vec2<u32>(0u))) { return 0xffffffffu; } return first(m); }
fn shl64(v: vec2<u32>, k: u32) -> vec2<u32> { let n = k & 63u; if (n == 0u) { return v; } if (n >= 32u) { return vec2<u32>(0u, v.x << (n - 32u)); } return vec2<u32>(v.x << n, (v.y << n) | (v.x >> (32u - n))); }
fn shr64(v: vec2<u32>, k: u32) -> vec2<u32> { let n = k & 63u; if (n == 0u) { return v; } if (n >= 32u) { return vec2<u32>(v.y >> (n - 32u), 0u); } return vec2<u32>((v.x >> n) | (v.y << (32u - n)), v.y >> n); }
fn mul_hi(a: u32, b: u32) -> u32 { let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u; let mid = ah * bl + ((al * bl) >> 16u); let mid2 = al * bh + (mid & 0xffffu); return ah * bh + (mid >> 16u) + (mid2 >> 16u); }
fn mul_hi_i(a: u32, b: u32) -> u32 { return mul_hi(a, b) - select(0u, b, bitcast<i32>(a) < 0) - select(0u, a, bitcast<i32>(b) < 0); }
fn i24(v: u32) -> i32 { return bitcast<i32>(v << 8u) >> 8u; }
fn ffbh(v: u32) -> u32 { if (v == 0u) { return 0xffffffffu; } return countLeadingZeros(v); }
fn ffbl(v: u32) -> u32 { if (v == 0u) { return 0xffffffffu; } return countTrailingZeros(v); }
fn ffbh_i(v: u32) -> u32 { return ffbh(select(v, ~v, bitcast<i32>(v) < 0)); }
fn low_bits(w: u32) -> u32 { if (w == 0u) { return 0u; } return 0xffffffffu >> (32u - w); }
fn bfm(a: u32, b: u32) -> u32 { return low_bits(a & 31u) << (b & 31u); }
fn bfe_u(v: u32, o: u32, w: u32) -> u32 { return (v >> (o & 31u)) & low_bits(w & 31u); }
fn bfe_i(v: u32, o: u32, w0: u32) -> u32 { let w = w0 & 31u; if (w == 0u) { return 0u; } let sh = 32u - w; return bitcast<u32>(bitcast<i32>(bitcast<u32>(bitcast<i32>(v) >> (o & 31u)) << sh) >> sh); }
fn sbfe_u(v: u32, c: u32) -> u32 { let o = c & 31u; let w = (c >> 16u) & 0x7fu; if (w >= 32u) { return v >> o; } return (v >> o) & low_bits(w); }
fn sbfe_i(v: u32, c: u32) -> u32 { let o = c & 31u; let w = (c >> 16u) & 0x7fu; if (w == 0u) { return 0u; } if (w >= 32u) { return bitcast<u32>(bitcast<i32>(v) >> o); } return bfe_i(v, o, w); }
fn alignbit(s0: u32, s1: u32, k: u32) -> u32 { let n = k & 31u; if (n == 0u) { return s1; } return (s1 >> n) | (s0 << (32u - n)); }
fn is_nan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn cvt_u32(x: f32) -> u32 { if (is_nan(x) || x <= 0.0) { return 0u; } if (x >= 4294967296.0) { return 0xffffffffu; } return u32(x); }
fn cvt_i32(x: f32) -> u32 { if (is_nan(x)) { return 0u; } if (x >= 2147483648.0) { return 0x7fffffffu; } if (x <= -2147483648.0) { return 0x80000000u; } return bitcast<u32>(i32(x)); }
fn mul_legacy(a: f32, b: f32) -> f32 { if (a == 0.0 || b == 0.0) { return 0.0; } return a * b; }
fn fma_legacy(a: f32, b: f32, c: f32) -> f32 { if (a == 0.0 || b == 0.0) { return c; } return fma(a, b, c); }
fn mad_nf(a: f32, b: f32, c: f32) -> f32 { let p = a * b; return p + c; }
fn med3f(a: f32, b: f32, c: f32) -> f32 { return max(min(a, b), min(max(a, b), c)); }
fn fract_f(x: f32) -> f32 { let r = x - floor(x); if (r >= 1.0) { return bitcast<f32>(0x3f7fffffu); } return r; }
fn cube_id(x: f32, y: f32, z: f32) -> f32 { if (abs(z) >= abs(x) && abs(z) >= abs(y)) { return select(4.0, 5.0, z < 0.0); } if (abs(y) >= abs(x)) { return select(2.0, 3.0, y < 0.0); } return select(0.0, 1.0, x < 0.0); }
fn cube_sc(x: f32, y: f32, z: f32) -> f32 { if (abs(z) >= abs(x) && abs(z) >= abs(y)) { return select(x, -x, z < 0.0); } if (abs(y) >= abs(x)) { return x; } return select(-z, z, x < 0.0); }
fn cube_tc(x: f32, y: f32, z: f32) -> f32 { if (abs(z) >= abs(x) && abs(z) >= abs(y)) { return -y; } if (abs(y) >= abs(x)) { return select(z, -z, y < 0.0); } return -y; }
fn cube_ma(x: f32, y: f32, z: f32) -> f32 { if (abs(z) >= abs(x) && abs(z) >= abs(y)) { return 2.0 * z; } if (abs(y) >= abs(x)) { return 2.0 * y; } return 2.0 * x; }
fn f16_rtz(f: f32) -> u32 { let b = bitcast<u32>(f); let sign = (b >> 16u) & 0x8000u; let u = b & 0x7fffffffu; let e = i32(u >> 23u) - 127; if (u > 0x7f800000u) { return sign | 0x7e00u; } if (u == 0x7f800000u) { return sign | 0x7c00u; } if (e > 15) { return sign | 0x7bffu; } if (e < -25) { return sign; } if (e < -14) { return sign | (((u & 0x7fffffu) | 0x800000u) >> u32(-1 - e)); } return sign | (u32(e + 15) << 10u) | ((u & 0x7fffffu) >> 13u); }
";

/// One instruction as WGSL statements (branches are the caller's).
fn statement(d: &Decoded, stage: Stage, slots: &BTreeMap<u64, usize>) -> Result<String, String> {
    let name = d.inst.name.to_ascii_lowercase();
    let lit = d.literal;
    let enc = d.enc.name;
    if name.starts_with("image_") {
        let op = image_op(d, &name)?;
        let k = slots.get(&d.pc).ok_or(&name)?;
        let value = if op.sample {
            let uv = format!("vec2<f32>(bitcast<f32>(v{}[l]), bitcast<f32>(v{}[l]))", op.coords[0], op.coords[1]);
            match (op.bias, op.lod, op.zero, stage) {
                (Some(b), ..) if matches!(stage, Stage::Pixel { .. }) => format!("textureSampleBias(t{k}, p{k}, {uv}, bitcast<f32>(v{b}[l]))"),
                (_, Some(lod), ..) => format!("textureSampleLevel(t{k}, p{k}, {uv}, bitcast<f32>(v{lod}[l]))"),
                // the level a pixel's neighbours give; anything else has no neighbours: level 0
                (None, None, false, Stage::Pixel { .. }) => format!("textureSample(t{k}, p{k}, {uv})"),
                _ => format!("textureSampleLevel(t{k}, p{k}, {uv}, 0.0)"),
            }
        } else {
            let level = op.lod.map(|r| format!("bitcast<i32>(v{r}[l])")).unwrap_or_else(|| "0".into());
            format!("textureLoad(t{k}, vec2<i32>(bitcast<i32>(v{}[l]), bitcast<i32>(v{}[l])), {level})", op.coords[0], op.coords[1])
        };
        // the components DMASK names, in consecutive registers
        let mut text = format!("let t = {value}; ");
        for (n, c) in (0..4).filter(|c| op.dmask >> c & 1 == 1).enumerate() {
            let _ = write!(text, "v{}[l] = bitcast<u32>(t[{c}]); ", op.vdata + n as u64);
        }
        return Ok(format!("for (var l = 0u; l < LANES; l++) {{ if (on(exec, l)) {{ {text}}} }}"));
    }
    if name == "exp" {
        let get = |n: &str| field(d, n).unwrap_or(0);
        let compr = get("COMPR") != 0;
        // the components as raw dwords: registers, or the halves of two (COMPR)
        let raw: Vec<Option<String>> = (0..4u64)
            .map(|c| {
                (get("EN") >> c & 1 == 1).then(|| match compr {
                    true => format!("((v{}[l] >> {}u) & 0xffffu)", get(&format!("VSRC{}", c / 2)), 16 * (c % 2)),
                    false => format!("v{}[l]", get(&format!("VSRC{c}"))),
                })
            })
            .collect();
        let tgt = get("TGT");
        let to = if stage == Stage::Kernel {
            let at = |c: usize| format!("res[{}u + l * 4u + {c}u]", KERNEL_EXPORTS as u64 + tgt * KERNEL_LANES as u64 * 4);
            raw.iter().enumerate().filter_map(|(c, r)| Some(format!("{} = {};", at(c), r.as_ref()?))).collect::<Vec<_>>().join(" ")
        } else {
            let value: Vec<String> = raw
                .iter()
                .map(|r| match (r, compr) {
                    (Some(r), true) => format!("unpack2x16float({r}).x"),
                    (Some(r), false) => format!("bitcast<f32>({r})"),
                    (None, _) => "0.0".into(),
                })
                .collect();
            let value = format!("vec4<f32>({})", value.join(", "));
            match (stage, tgt) {
                (Stage::Vertex { .. }, 12) => format!("o_pos = {value};"),
                (Stage::Vertex { .. }, t @ 32..=63) => format!("o_p{} = {value};", t - 32),
                (Stage::Pixel { .. }, t @ 0..=7) => format!("o_t{t} = {value}; live = true;"),
                (Stage::Pixel { .. }, 9) => "live = true;".into(),
                (_, t) => return Err(format!("exp to target {t} of this stage")),
            }
        };
        return Ok(format!("for (var l = 0u; l < LANES; l++) {{ if (on(exec, l)) {{ {to} }} }}"));
    }
    if let Some(count) = load_count(&name) {
        let get = |n: &str| field(d, n).unwrap_or(0);
        if let Some(scalar) = name.strip_prefix("s_") {
            let sdata = field(d, "SDATA").ok_or("SDATA")?;
            let sbase = register(d, "SBASE")?;
            let soff = match get("SOFFSET") {
                125 => "0u".to_string(),
                c => source(c, lit)?,
            };
            let mut text = String::from("{ ");
            if scalar.starts_with("buffer_") {
                // V#: base, stride in dw1[29:16], num_records; a dword past the end reads 0
                let _ = write!(text, "let off = {}u + {soff}; let st = (s[{}] >> 16u) & 0x3fffu; let lim = select(s[{}] * st, s[{}], st == 0u); let a = s[{sbase}] + off; ", get("OFFSET"), sbase + 1, sbase + 2, sbase + 2);
                for k in 0..count {
                    let _ = write!(text, "let t{k} = select(0u, ld(a + {}u), off + {}u < lim); ", 4 * k, 4 * k);
                }
            } else {
                // a pointer and a signed 21-bit offset
                let imm = (((get("OFFSET") as u32) << 11) as i32 >> 11) as u32;
                let _ = write!(text, "let a = s[{sbase}] + {imm}u + {soff}; ");
                for k in 0..count {
                    let _ = write!(text, "let t{k} = ld(a + {}u); ", 4 * k);
                }
            }
            for k in 0..count {
                let _ = write!(text, "{} ", store(sdata + k, &format!("t{k}"))?);
            }
            return Ok(text + "}");
        }
        if get("LDS") != 0 || get("TFE") != 0 {
            return Err(format!("{name} lds / tfe"));
        }
        let (vdata, vaddr, r) = (get("VDATA"), get("VADDR"), register(d, "SRSRC")?);
        let (offen, idxen) = (get("OFFEN") != 0, get("IDXEN") != 0);
        let index = if idxen { format!("v{vaddr}[l]") } else { "0u".into() };
        let voff = if offen { format!("v{}[l]", vaddr + idxen as u64) } else { "0u".into() };
        let soff = source(get("SOFFSET"), lit)?;
        let mut text = format!("let st = (s[{}] >> 16u) & 0x3fffu; let n = s[{}]; let idx = {index}; let off = {voff} + {}u; let a = s[{r}] + {soff} + off + st * idx; ", r + 1, r + 2, get("OFFSET"));
        // in range: by bytes without a stride, by records with one
        for k in 0..count {
            let _ = write!(text, "let t{k} = select(0u, ld(a + {}u), select(idx < n, off + {}u <= n, st == 0u)); ", 4 * k, 4 * k + 4);
        }
        for k in 0..count {
            let _ = write!(text, "v{}[l] = t{k}; ", vdata + k);
        }
        return Ok(format!("for (var l = 0u; l < LANES; l++) {{ if (on(exec, l)) {{ {text}}} }}"));
    }
    if name.starts_with("v_interp_") {
        if !matches!(stage, Stage::Pixel { .. }) || enc != "ENC_VINTRP" {
            return Err(name);
        }
        let get = |n: &str| field(d, n).unwrap_or(0);
        return Ok(match name.as_str() {
            // the attribute comes interpolated from the rasteriser: the second half has the value
            "v_interp_p1_f32" => String::new(),
            "v_interp_p2_f32" => format!("for (var l = 0u; l < LANES; l++) {{ if (on(exec, l)) {{ v{}[l] = bitcast<u32>(xi.a{}[{}]); }} }}", get("VDST"), get("ATTR"), get("ATTRCHAN")),
            _ => return Err(name),
        });
    }
    // --- the vector unit
    if name.starts_with("v_") {
        if enc.contains("SDWA") || enc.contains("DPP") {
            return Err(format!("{name} ({enc})"));
        }
        if name == "v_nop" {
            return Ok(String::new());
        }
        let get = |n: &str| field(d, n).unwrap_or(0);
        let m = Mods { abs: get("ABS"), neg: get("NEG"), omod: get("OMOD"), clamp: get("CLAMP") != 0 };
        if m.any() && !name.contains("f32") {
            return Err(format!("{name} with operand modifiers"));
        }
        let a = source(field(d, "SRC0").ok_or("SRC0")?, lit)?;
        let b = match (field(d, "VSRC1"), field(d, "SRC1")) {
            (Some(v), _) => format!("v{v}[l]"),
            (None, Some(c)) => source(c, lit)?,
            (None, None) => "0u".into(),
        };
        let c = field(d, "SRC2").map(|c| source(c, lit)).transpose()?.unwrap_or_else(|| "0u".into());
        let cmpx = name.starts_with("v_cmpx_");
        if let Some(rest) = name.strip_prefix("v_cmpx_").or_else(|| name.strip_prefix("v_cmp_")) {
            let (op, ty) = rest.rsplit_once('_').ok_or(&name)?;
            let cast = |n: usize, e: &str| match ty {
                "f32" => Ok(m.src(n, e)),
                "i32" => Ok(i(e)),
                "u32" => Ok(e.to_string()),
                _ => Err(format!("{name}")),
            };
            let test = condition(op, &cast(0, &a)?, &cast(1, &b)?, ty == "f32").ok_or(&name)?;
            // the mask goes to VCC, or to the register pair the VOP3 form names
            // (v_cmpx: to EXEC, so lanes that fail are off from here)
            let to = store64(if cmpx { 126 } else if enc.contains("VOP3") { field(d, "VDST").ok_or("VDST")? } else { 106 }, "m")?;
            return Ok(format!("{{ var m = vec2<u32>(0u); for (var l = 0u; l < LANES; l++) {{ if (on(exec, l) && {test}) {{ m = put(m, l); }} }} {to} }}"));
        }
        let dst = field(d, "VDST").ok_or("VDST")?;
        // lanes: the first one that is on to a scalar register, one by number
        match name.as_str() {
            "v_readfirstlane_b32" => return Ok(format!("{{ let l = first(exec); {} }}", store(dst, &a)?)),
            "v_readlane_b32" if stage == Stage::Kernel => return Ok(format!("{{ let l = {b} & 63u; {} }}", store(dst, &a)?)),
            // (whatever EXEC is)
            "v_writelane_b32" if stage == Stage::Kernel => return Ok(format!("v{dst}[{b} & 63u] = {a};")),
            _ => {}
        }
        let k = format!("{}u", lit.unwrap_or(0));
        let value = vector_value(&name, &a, &b, &c, &m, &format!("v{dst}[l]"), &k).ok_or(&name)?;
        return Ok(format!("for (var l = 0u; l < LANES; l++) {{ if (on(exec, l)) {{ v{dst}[l] = {value}; }} }}"));
    }
    // --- the scalar unit
    let s0 = field(d, "SSRC0");
    let s1 = field(d, "SSRC1");
    let dst = field(d, "SDST");
    let x = |c: Option<u64>| source(c.ok_or("a scalar source")?, lit);
    let x64 = |c: Option<u64>| source64(c.ok_or("a scalar source")?, lit);
    let to = |value: &str| store(dst.ok_or("SDST")?, value);
    let to64 = |value: &str| store64(dst.ok_or("SDST")?, value);
    Ok(match name.as_str() {
        "s_nop" | "s_waitcnt" => String::new(),
        "s_mov_b64" => to64(&x64(s0)?)?,
        "s_movk_i32" => to(&format!("{}u", field(d, "SIMM16").ok_or("SIMM16")? as u16 as i16 as i32 as u32))?,
        "s_not_b64" => format!("{{ let r = ~{}; scc = any(r != vec2<u32>(0u)); {} }}", x64(s0)?, to64("r")?),
        "s_cselect_b64" => to64(&format!("select({}, {}, scc)", x64(s1)?, x64(s0)?))?,
        "s_lshl_b64" => format!("{{ let r = shl64({}, {}); scc = any(r != vec2<u32>(0u)); {} }}", x64(s0)?, x(s1)?, to64("r")?),
        "s_lshr_b64" => format!("{{ let r = shr64({}, {}); scc = any(r != vec2<u32>(0u)); {} }}", x64(s0)?, x(s1)?, to64("r")?),
        "s_bcnt1_i32_b64" => format!("{{ let a = {}; let r = countOneBits(a.x) + countOneBits(a.y); scc = r != 0u; {} }}", x64(s0)?, to("r")?),
        "s_ff1_i32_b64" => to(&format!("ff1_64({})", x64(s0)?))?,
        "s_cmp_eq_u64" => format!("scc = all({} == {});", x64(s0)?, x64(s1)?),
        "s_cmp_lg_u64" => format!("scc = any({} != {});", x64(s0)?, x64(s1)?),
        "s_bitcmp0_b32" => format!("scc = (({} >> ({} & 31u)) & 1u) == 0u;", x(s0)?, x(s1)?),
        "s_bitcmp1_b32" => format!("scc = (({} >> ({} & 31u)) & 1u) != 0u;", x(s0)?, x(s1)?),
        "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64" | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" => {
            let (a, b) = (x64(s0)?, x64(s1)?);
            let r = match name.as_str() {
                "s_and_b64" => format!("{a} & {b}"),
                "s_or_b64" => format!("{a} | {b}"),
                "s_xor_b64" => format!("{a} ^ {b}"),
                "s_andn2_b64" => format!("{a} & ~{b}"),
                "s_nand_b64" => format!("~({a} & {b})"),
                "s_nor_b64" => format!("~({a} | {b})"),
                "s_xnor_b64" => format!("~({a} ^ {b})"),
                _ => format!("{a} | ~{b}"),
            };
            format!("{{ let r = {r}; scc = any(r != vec2<u32>(0u)); {} }}", to64("r")?)
        }
        // EXEC is saved, then masked with the source; SCC: lanes are left
        _ if name.ends_with("_saveexec_b64") => {
            let r = match name.trim_end_matches("_saveexec_b64") {
                "s_and" => "a & exec",
                "s_or" => "a | exec",
                "s_xor" => "a ^ exec",
                "s_andn2" => "a & ~exec",
                "s_orn2" => "a | ~exec",
                "s_nand" => "~(a & exec)",
                "s_nor" => "~(a | exec)",
                "s_xnor" => "~(a ^ exec)",
                "s_andn1" => "~a & exec",
                "s_orn1" => "~a | exec",
                _ => return Err(name),
            };
            format!("{{ let a = {}; let saved = exec; exec = {r}; scc = any(exec != vec2<u32>(0u)); {} }}", x64(s0)?, to64("saved")?)
        }
        _ => {
            if let Some(rest) = name.strip_prefix("s_cmp_") {
                let (op, ty) = rest.rsplit_once('_').ok_or(&name)?;
                let (a, b) = (x(s0)?, x(s1)?);
                let (a, b) = if ty == "i32" { (i(&a), i(&b)) } else { (a, b) };
                return Ok(format!("scc = {};", condition(op, &a, &b, false).ok_or(&name)?));
            }
            // SOPK: the register with a 16-bit constant
            if let Some(k) = field(d, "SIMM16") {
                let (signed, zero) = (format!("{}u", k as u16 as i16 as i32 as u32), format!("{}u", k as u16));
                let cur = x(dst)?;
                if let Some(rest) = name.strip_prefix("s_cmpk_") {
                    let (op, ty) = rest.rsplit_once('_').ok_or(&name)?;
                    let (a, b) = if ty == "i32" { (i(&cur), i(&signed)) } else { (cur, zero) };
                    return Ok(format!("scc = {};", condition(op, &a, &b, false).ok_or(&name)?));
                }
                return Ok(match name.as_str() {
                    "s_cmovk_i32" => to(&format!("select({cur}, {signed}, scc)"))?,
                    "s_addk_i32" => format!("{{ let a = {cur}; let b = {signed}; let r = a + b; scc = (((a ^ r) & (b ^ r)) >> 31u) != 0u; {} }}", to("r")?),
                    "s_mulk_i32" => to(&format!("{cur} * {signed}"))?,
                    _ => return Err(name),
                });
            }
            if let Some((res, scc)) = scalar_value(&name) {
                let (a, b) = (s0.map(|c| source(c, lit)).transpose()?, s1.map(|c| source(c, lit)).transpose()?);
                let cur = dst.filter(|&c| c != 125).map(|c| source(c, lit)).transpose()?;
                let scc = scc.map(|e| format!("scc = {e}; ")).unwrap_or_default();
                return Ok(format!(
                    "{{ let a = {}; let b = {}; let cur = {}; let r = {res}; {scc}{} }}",
                    a.unwrap_or("0u".into()),
                    b.unwrap_or("0u".into()),
                    cur.unwrap_or("0u".into()),
                    to("r")?
                ));
            }
            return Err(name);
        }
    })
}

/// Translate a program into a WGSL compute shader of the kernel ABI. `Err`: the instructions
/// this back end does not have yet, or a program it cannot decode.
pub fn translate_kernel(code: &[u8]) -> Result<Translation, String> {
    translate(code, Stage::Kernel)
}

/// Translate a program into a WGSL shader of `stage`'s ABI (see the module's text).
pub fn translate(code: &[u8], stage: Stage) -> Result<Translation, String> {
    let mut insts: Vec<Decoded> = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let d = rdna2::decode(&code[off..], off as u64).map_err(|e| format!("+{off:#x}: cannot decode ({e:?})"))?;
        off += d.len;
        insts.push(d);
    }
    let starts: BTreeSet<u64> = insts.iter().map(|d| d.pc).collect();
    // basic blocks start at the program's start, at branch targets and after branches
    let mut leaders: BTreeSet<u64> = BTreeSet::from([0]);
    for d in &insts {
        if d.inst.flags & rdna2::F_BRANCH == 0 {
            continue;
        }
        let target = rdna2::branch_target(d).ok_or_else(|| format!("+{:#x}: {} (an indirect branch)", d.pc, d.inst.name.to_ascii_lowercase()))?;
        if !starts.contains(&target) {
            return Err(format!("+{:#x}: branch to {target:#x}, which is not the start of an instruction", d.pc));
        }
        leaders.insert(target);
        leaders.insert(d.pc + d.len as u64);
    }
    let mut vgprs: BTreeSet<u64> = BTreeSet::new();
    let mut written: BTreeSet<u64> = BTreeSet::new();
    // what the stage's interface has: exp param N / mrt N, interpolated attributes
    let (mut params, mut targets, mut attrs) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    let mut memory = false;
    let mut images: Vec<Image> = Vec::new();
    let mut slots: BTreeMap<u64, usize> = BTreeMap::new();
    let mut refused: Vec<String> = Vec::new();
    for (at, d) in insts.iter().enumerate() {
        let name = d.inst.name.to_ascii_lowercase();
        if let Ok(op) = image_op(d, &name) {
            vgprs.extend(op.bias.iter().chain(&op.coords).chain(&op.lod).copied());
            let data = op.vdata..op.vdata + op.dmask.count_ones() as u64;
            vgprs.extend(data.clone());
            written.extend(data);
            let sampler = if op.sample { origin(&insts, at, op.ssamp).map(Some) } else { Ok(None) };
            match (origin(&insts, at, op.srsrc), sampler) {
                (Ok(texture), Ok(sampler)) => {
                    let image = Image { texture, sampler };
                    let slot = images.iter().position(|i| *i == image).unwrap_or_else(|| {
                        images.push(image);
                        images.len() - 1
                    });
                    slots.insert(d.pc, slot);
                }
                (Err(e), _) | (_, Err(e)) => refused.push(e),
            }
        }
    }
    if let Some(e) = refused.first() {
        return Err(format!("+ {} more: {e}", refused.len() - 1).trim_start_matches("+ 0 more: ").to_string());
    }
    for d in &insts {
        let (r, w) = rdna2::vgpr_uses(d);
        vgprs.extend(r.iter().copied());
        vgprs.extend(w.iter().copied());
        written.extend(w);
        let get = |n: &str| field(d, n).unwrap_or(0);
        if d.inst.name == "EXP" {
            vgprs.extend((0..4).filter(|c| get("EN") >> c & 1 == 1).map(|c| get(&format!("VSRC{c}"))));
            match get("TGT") {
                t @ 0..=7 => targets.insert(t),
                t @ 32..=63 => params.insert(t - 32),
                _ => false,
            };
        }
        if d.enc.name == "ENC_VINTRP" {
            vgprs.insert(get("VDST"));
            attrs.insert(get("ATTR"));
        }
        let name = d.inst.name.to_ascii_lowercase();
        if let Some(count) = load_count(&name) {
            memory = true;
            if !name.starts_with("s_") {
                vgprs.extend(get("VDATA")..get("VDATA") + count);
                written.extend(get("VDATA")..get("VDATA") + count);
                vgprs.extend((0..(get("IDXEN") + get("OFFEN"))).map(|k| get("VADDR") + k));
            }
        }
    }
    // registers the stage fills before the program runs
    let mut inputs: Vec<(u64, String)> = Vec::new();
    match stage {
        Stage::Kernel => {}
        Stage::Vertex { attr_base, attributes } => {
            for k in 0..attributes as u64 {
                inputs.extend((0..4).map(|c| (attr_base + 4 * k + c, format!("bitcast<u32>(a{k}[{c}])"))));
            }
        }
        Stage::Pixel { input_ena } => {
            for (bit, c) in [(8, "x"), (9, "y"), (10, "z"), (11, "w")] {
                inputs.extend(ps_input_vgpr(input_ena, bit).map(|v| (v, format!("bitcast<u32>(xi.pos.{c})"))));
            }
            inputs.extend(ps_input_vgpr(input_ena, 12).map(|v| (v, "bitcast<u32>(select(0.0, 1.0, xi.front))".to_string())));
        }
    }
    inputs.retain(|(v, _)| vgprs.contains(v));
    let mut unsupported: BTreeMap<String, usize> = BTreeMap::new();
    let mut body = String::new();
    const END: u32 = 0xFFFF_FFFF;
    let mut open = false;
    for d in &insts {
        if leaders.contains(&d.pc) {
            if open {
                // falling into the next block
                let _ = writeln!(body, "        pc = {}u;\n      }}", d.pc);
            }
            let _ = writeln!(body, "      case {}u {{", d.pc);
            open = true;
        }
        let name = d.inst.name.to_ascii_lowercase();
        let next = d.pc + d.len as u64;
        let _ = writeln!(body, "        // {}", rdna2::format(d));
        if name == "s_endpgm" {
            let _ = writeln!(body, "        pc = {END}u;\n      }}");
            open = false;
            continue;
        }
        if d.inst.flags & rdna2::F_BRANCH != 0 {
            let target = rdna2::branch_target(d).unwrap_or(0);
            let taken = match name.as_str() {
                "s_branch" => Some("true"),
                "s_cbranch_scc0" => Some("!scc"),
                "s_cbranch_scc1" => Some("scc"),
                "s_cbranch_vccz" => Some("all(vcc == vec2<u32>(0u))"),
                "s_cbranch_vccnz" => Some("any(vcc != vec2<u32>(0u))"),
                "s_cbranch_execz" => Some("all(exec == vec2<u32>(0u))"),
                "s_cbranch_execnz" => Some("any(exec != vec2<u32>(0u))"),
                _ => None,
            };
            match taken {
                Some(cond) => {
                    let _ = writeln!(body, "        if ({cond}) {{ pc = {target}u; }} else {{ pc = {next}u; }}\n      }}");
                }
                None => *unsupported.entry(name).or_default() += 1,
            }
            open = false;
            continue;
        }
        match statement(d, stage, &slots) {
            Ok(text) if text.is_empty() => {}
            Ok(text) => {
                let _ = writeln!(body, "        {text}");
            }
            Err(what) => *unsupported.entry(what).or_default() += 1,
        }
    }
    if open {
        // the code ends without s_endpgm
        let _ = writeln!(body, "        pc = {END}u;\n      }}");
    }
    if !unsupported.is_empty() {
        let list: Vec<String> = unsupported.iter().map(|(n, c)| format!("{n} ×{c}")).collect();
        return Err(format!("not in the WGSL back end yet: {}", list.join(", ")));
    }
    let kernel = stage == Stage::Kernel;
    let lanes = if kernel { KERNEL_LANES } else { 1 };
    let mut w = String::new();
    let _ = writeln!(w, "// RDNA2 program ({stage:?}), made by XCL (gpu::recomp::wgsl).");
    if !images.is_empty() {
        // the dispatcher is not uniform control flow, which a sample at an implicit level asks for
        let _ = writeln!(w, "diagnostic(off, derivative_uniformity);");
    }
    let _ = writeln!(w, "const LANES = {lanes}u;");
    for (k, image) in images.iter().enumerate() {
        let _ = writeln!(w, "// image {k}: texture {:?}, sampler {:?}", image.texture, image.sampler);
        let _ = writeln!(w, "@group(1) @binding({}) var t{k}: texture_2d<f32>;", 2 * k);
        if image.sampler.is_some() {
            let _ = writeln!(w, "@group(1) @binding({}) var p{k}: sampler;", 2 * k + 1);
        }
    }
    if kernel {
        let _ = writeln!(w, "@group(0) @binding(0) var<storage, read> io: array<u32>;");
        let _ = writeln!(w, "@group(0) @binding(1) var<storage, read_write> res: array<u32>;");
    } else {
        let _ = writeln!(w, "@group(0) @binding(0) var<uniform> sg: array<vec4<u32>, {}>;", KERNEL_SGPRS / 4);
    }
    if memory {
        let _ = writeln!(w, "@group(0) @binding({}) var<storage, read> mem: array<u32>;", if kernel { 2 } else { 1 });
        let _ = writeln!(w, "fn ld(a: u32) -> u32 {{ let i = a >> 2u; if (i < arrayLength(&mem)) {{ return mem[i]; }} return 0u; }}");
    }
    let _ = writeln!(w, "var<private> s: array<u32, {KERNEL_SGPRS}>;");
    let _ = writeln!(w, "var<private> exec: vec2<u32>;\nvar<private> vcc: vec2<u32>;\nvar<private> scc: bool;");
    for v in &vgprs {
        let _ = writeln!(w, "var<private> v{v}: array<u32, {lanes}>;");
    }
    w.push_str(HELPERS);
    match stage {
        Stage::Kernel => {
            let _ = writeln!(w, "@compute @workgroup_size(1) fn main() {{");
            let _ = writeln!(w, "  for (var i = 0u; i < {KERNEL_SGPRS}u; i++) {{ s[i] = io[i]; }}");
            let _ = writeln!(w, "  exec = vec2<u32>(io[126], io[127]);\n  vcc = vec2<u32>(io[106], io[107]);\n  scc = io[{KERNEL_SCC_SLOT}] != 0u;");
            for v in &vgprs {
                let _ = writeln!(w, "  for (var l = 0u; l < LANES; l++) {{ v{v}[l] = io[{KERNEL_SGPRS}u + l * {KERNEL_VGPRS}u + {v}u]; }}");
            }
            // (before the program: its exports go straight to `res`)
            let _ = writeln!(w, "  for (var i = 0u; i < {KERNEL_EXPORTS}u; i++) {{ res[i] = io[i]; }}");
        }
        Stage::Vertex { attributes, .. } => {
            let _ = writeln!(w, "var<private> o_pos: vec4<f32>;");
            let _ = writeln!(w, "struct VsOut {{\n  @builtin(position) pos: vec4<f32>,");
            for n in &params {
                let _ = writeln!(w, "  @location({n}) p{n}: vec4<f32>,");
            }
            let _ = writeln!(w, "}}");
            for n in &params {
                let _ = writeln!(w, "var<private> o_p{n}: vec4<f32>;");
            }
            let list: Vec<String> = (0..attributes).map(|k| format!("@location({k}) a{k}: vec4<f32>")).collect();
            let _ = writeln!(w, "@vertex fn main({}) -> VsOut {{", list.join(", "));
        }
        Stage::Pixel { input_ena } => {
            let _ = writeln!(w, "var<private> live: bool;");
            let _ = writeln!(w, "struct PsIn {{\n  @builtin(position) pos: vec4<f32>,");
            if input_ena >> 12 & 1 == 1 {
                let _ = writeln!(w, "  @builtin(front_facing) front: bool,");
            }
            for n in &attrs {
                let _ = writeln!(w, "  @location({n}) a{n}: vec4<f32>,");
            }
            let _ = writeln!(w, "}}");
            if !targets.is_empty() {
                let _ = writeln!(w, "struct PsOut {{");
                for t in &targets {
                    let _ = writeln!(w, "  @location({t}) t{t}: vec4<f32>,");
                }
                let _ = writeln!(w, "}}");
            }
            for t in &targets {
                let _ = writeln!(w, "var<private> o_t{t}: vec4<f32>;");
            }
            let _ = writeln!(w, "@fragment fn main(xi: PsIn){} {{", if targets.is_empty() { "" } else { " -> PsOut" });
        }
    }
    if !kernel {
        let _ = writeln!(w, "  for (var i = 0u; i < {KERNEL_SGPRS}u; i++) {{ s[i] = sg[i / 4u][i % 4u]; }}");
        // the one lane is on; the mask registers are the wave's, not the title's
        let _ = writeln!(w, "  exec = vec2<u32>(1u, 0u);\n  vcc = vec2<u32>(0u);\n  scc = false;");
        for (v, value) in &inputs {
            let _ = writeln!(w, "  v{v}[0] = {value};");
        }
    }
    let _ = writeln!(w, "  var pc = 0u;\n  // a program that never ends is cut off\n  for (var steps = 0u; steps < 1000000u; steps++) {{\n    switch pc {{");
    w.push_str(&body);
    let _ = writeln!(w, "      default {{ pc = {END}u; }}\n    }}\n    if (pc == {END}u) {{ break; }}\n  }}");
    let total = KERNEL_EXPORTS + 64 * KERNEL_LANES * 4;
    match stage {
        Stage::Kernel => {
            let _ = writeln!(w, "  for (var i = 0u; i < {KERNEL_SGPRS}u; i++) {{ res[i] = s[i]; }}");
            let _ = writeln!(w, "  res[106] = vcc.x; res[107] = vcc.y; res[126] = exec.x; res[127] = exec.y; res[{KERNEL_SCC_SLOT}] = u32(scc);");
            for v in &written {
                let _ = writeln!(w, "  for (var l = 0u; l < LANES; l++) {{ res[{KERNEL_SGPRS}u + l * {KERNEL_VGPRS}u + {v}u] = v{v}[l]; }}");
            }
        }
        Stage::Vertex { .. } => {
            let list: Vec<String> = std::iter::once("o_pos".to_string()).chain(params.iter().map(|n| format!("o_p{n}"))).collect();
            let _ = writeln!(w, "  return VsOut({});", list.join(", "));
        }
        Stage::Pixel { .. } => {
            let _ = writeln!(w, "  if (!live) {{ discard; }}");
            if !targets.is_empty() {
                let list: Vec<String> = targets.iter().map(|t| format!("o_t{t}")).collect();
                let _ = writeln!(w, "  return PsOut({});", list.join(", "));
            }
        }
    }
    let _ = writeln!(w, "}}");
    Ok(Translation { wgsl: w, dwords: total, memory, images })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asm(insts: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
        insts.iter().flat_map(|(name, enc, fields, lit)| rdna2::encode(name, enc, fields, *lit).unwrap()).collect()
    }

    #[test]
    fn a_program_becomes_a_dispatcher_over_its_blocks() {
        // v_add_f32 v5, v1, v2 ; s_cbranch_vccz +1 ; v_mov_b32 v6, 1.0 ; s_endpgm
        let code = asm(&[
            ("V_ADD_F32", "ENC_VOP2", &[("VDST", 5), ("SRC0", 257), ("VSRC1", 2)], None),
            ("S_CBRANCH_VCCZ", "ENC_SOPP", &[("SIMM16", 1)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", 242)], None),
            ("S_ENDPGM", "ENC_SOPP", &[], None),
        ]);
        let t = translate_kernel(&code).unwrap();
        assert!(t.wgsl.contains("v5[l] = bitcast<u32>(bitcast<f32>(v1[l]) + bitcast<f32>(v2[l]));"), "{}", t.wgsl);
        // three blocks: the start, the instruction the branch skips, the end
        assert_eq!(t.wgsl.matches("      case ").count(), 3, "{}", t.wgsl);
        assert!(t.wgsl.contains("if (all(vcc == vec2<u32>(0u))) { pc = 12u; } else { pc = 8u; }"), "{}", t.wgsl);
        assert!(t.wgsl.contains("v6[l] = 1065353216u;"));
        // only the registers the program uses are declared, only the written ones stored
        assert!(t.wgsl.contains("var<private> v1: array<u32, 64>;") && !t.wgsl.contains("var<private> v3:"));
        assert!(t.wgsl.contains("+ 5u] = v5[l];") && !t.wgsl.contains("+ 1u] = v1[l];"));
        assert_eq!(t.dwords, 128 + 64 * 256 + 64 * 64 * 4);
        assert!(!t.memory && !t.wgsl.contains("mem:"));
    }

    #[test]
    fn a_vertex_and_a_pixel_stage_have_their_interfaces() {
        // v0..v3 = attribute 0 ; exp pos0 ; exp param0 = attribute 1
        let vs = asm(&[
            ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 7), ("DONE", 1)], None),
            ("EXP", "ENC_EXP", &[("TGT", 32), ("EN", 0x7), ("VSRC0", 8), ("VSRC1", 9), ("VSRC2", 10), ("VSRC3", 11)], None),
        ]);
        let t = translate(&vs, Stage::Vertex { attr_base: 4, attributes: 2 }).unwrap();
        assert!(t.wgsl.contains("const LANES = 1u;") && t.wgsl.contains("@vertex fn main(@location(0) a0: vec4<f32>, @location(1) a1: vec4<f32>) -> VsOut"), "{}", t.wgsl);
        assert!(t.wgsl.contains("v9[0] = bitcast<u32>(a1[1]);") && t.wgsl.contains("@location(0) p0: vec4<f32>,"), "{}", t.wgsl);
        // a component the export leaves out is 0
        assert!(t.wgsl.contains("o_p0 = vec4<f32>(bitcast<f32>(v8[l]), bitcast<f32>(v9[l]), bitcast<f32>(v10[l]), 0.0);"), "{}", t.wgsl);
        // v_cmpx kills ; attribute 1 channel 2 ; exp mrt0
        let ps = asm(&[
            ("V_CMPX_LT_F32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(8.0f32.to_bits())),
            ("V_INTERP_P2_F32", "ENC_VINTRP", &[("VDST", 4), ("VSRC", 1), ("ATTR", 1), ("ATTRCHAN", 2)], None),
            ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0x1), ("VSRC0", 4), ("DONE", 1), ("VM", 1)], None),
        ]);
        let t = translate(&ps, Stage::Pixel { input_ena: 0x302 }).unwrap();
        // persp center is v0 v1, so the position's x is v2
        assert!(t.wgsl.contains("v2[0] = bitcast<u32>(xi.pos.x);") && t.wgsl.contains("exec = m;"), "{}", t.wgsl);
        assert!(t.wgsl.contains("v4[l] = bitcast<u32>(xi.a1[2]);") && t.wgsl.contains("@location(1) a1: vec4<f32>,"), "{}", t.wgsl);
        assert!(t.wgsl.contains("if (!live) { discard; }") && t.wgsl.contains("return PsOut(o_t0);"), "{}", t.wgsl);
        // an export a stage does not have is refused
        assert!(translate(&vs, Stage::Pixel { input_ena: 0 }).is_err() && translate(&ps, Stage::Vertex { attr_base: 4, attributes: 0 }).is_err());
    }

    #[test]
    fn loads_read_the_memory_array_within_a_descriptor() {
        let code = asm(&[
            ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 0), ("OFFSET", 0x10), ("SOFFSET", 125)], None),
            ("S_BUFFER_LOAD_DWORD", "ENC_SMEM", &[("SDATA", 10), ("SBASE", 6), ("OFFSET", 4), ("SOFFSET", 125)], None),
            ("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 3), ("VADDR", 0), ("SRSRC", 3), ("SOFFSET", 128), ("OFFSET", 8), ("OFFEN", 1)], None),
            ("EXP", "ENC_EXP", &[("TGT", 12), ("EN", 0x3), ("VSRC0", 3), ("VSRC1", 3), ("COMPR", 1), ("DONE", 1)], None),
        ]);
        let t = translate_kernel(&code).unwrap();
        assert!(t.memory && t.wgsl.contains("@group(0) @binding(2) var<storage, read> mem: array<u32>;"), "{}", t.wgsl);
        // SBASE counts register pairs, SRSRC fours: both descriptors are s12..
        assert!(t.wgsl.contains("let a = s[0] + 16u + 0u; let t0 = ld(a + 0u); let t1 = ld(a + 4u); s[8] = t0; s[9] = t1;"), "{}", t.wgsl);
        assert!(t.wgsl.contains("let a = s[12] + off; let t0 = select(0u, ld(a + 0u), off + 0u < lim); s[10] = t0;"), "{}", t.wgsl);
        assert!(t.wgsl.contains("let off = v0[l] + 8u; let a = s[12] + 0u + off + st * idx;") && t.wgsl.contains("v3[l] = t0;"), "{}", t.wgsl);
        // a kernel's export is raw dwords after the registers: target 12, the halves of v3
        let at = 128 + 64 * 256 + 12 * 64 * 4;
        assert!(t.wgsl.contains(&format!("res[{at}u + l * 4u + 1u] = ((v3[l] >> 16u) & 0xffffu);")), "{}", t.wgsl);
        // in a stage the array is binding 1
        let t = translate(&code, Stage::Vertex { attr_base: 4, attributes: 0 }).unwrap();
        assert!(t.wgsl.contains("@group(0) @binding(1) var<storage, read> mem") && t.wgsl.contains("unpack2x16float(((v3[l] >> 0u) & 0xffffu)).x"), "{}", t.wgsl);
    }

    #[test]
    fn an_image_instruction_gets_a_slot_and_its_descriptors_origin() {
        let ps = asm(&[
            // s8.. = the T# at [s2 + 0x20]; s4:s5 = the pointer at [s2 + 8]; s16.. = the S# at [that + 0x40]
            ("S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 1), ("OFFSET", 0x20), ("SOFFSET", 125)], None),
            ("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 4), ("SBASE", 1), ("OFFSET", 8), ("SOFFSET", 125)], None),
            ("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 2), ("OFFSET", 0x40), ("SOFFSET", 125)], None),
            ("IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 2), ("SSAMP", 4), ("DMASK", 0xB), ("DIM", 1)], None),
            // the same texture and sampler again at level 0, and a texel of the texture in s24..
            ("IMAGE_SAMPLE_LZ", "ENC_MIMG", &[("VDATA", 8), ("VADDR", 0), ("SRSRC", 2), ("SSAMP", 4), ("DMASK", 0x1), ("DIM", 1)], None),
            ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 9), ("VADDR", 2), ("SRSRC", 6), ("DMASK", 0xF), ("DIM", 1)], None),
            ("EXP", "ENC_EXP", &[("TGT", 0), ("EN", 0xF), ("VSRC0", 4), ("VSRC1", 5), ("VSRC2", 6), ("VSRC3", 8), ("DONE", 1), ("VM", 1)], None),
        ]);
        let t = translate(&ps, Stage::Pixel { input_ena: 0 }).unwrap();
        assert_eq!(
            t.images,
            [
                Image { texture: Origin { sgpr: 2, loads: vec![0x20] }, sampler: Some(Origin { sgpr: 2, loads: vec![8, 0x40] }) },
                Image { texture: Origin { sgpr: 24, loads: vec![] }, sampler: None },
            ]
        );
        assert!(t.wgsl.starts_with("// RDNA2 program") && t.wgsl.contains("diagnostic(off, derivative_uniformity);\nconst LANES"), "{}", t.wgsl);
        assert!(t.wgsl.contains("@group(1) @binding(0) var t0: texture_2d<f32>;") && t.wgsl.contains("@group(1) @binding(1) var p0: sampler;"), "{}", t.wgsl);
        assert!(t.wgsl.contains("@group(1) @binding(2) var t1: texture_2d<f32>;") && !t.wgsl.contains("var p1"), "{}", t.wgsl);
        // DMASK 0xB: x, y and w into three registers
        assert!(t.wgsl.contains("let t = textureSample(t0, p0, vec2<f32>(bitcast<f32>(v0[l]), bitcast<f32>(v1[l]))); v4[l] = bitcast<u32>(t[0]); v5[l] = bitcast<u32>(t[1]); v6[l] = bitcast<u32>(t[3]);"), "{}", t.wgsl);
        assert!(t.wgsl.contains("let t = textureSampleLevel(t0, p0, vec2<f32>(bitcast<f32>(v0[l]), bitcast<f32>(v1[l])), 0.0); v8[l]"), "{}", t.wgsl);
        assert!(t.wgsl.contains("let t = textureLoad(t1, vec2<i32>(bitcast<i32>(v2[l]), bitcast<i32>(v3[l])), 0);"), "{}", t.wgsl);
        // a vertex has no neighbours: level 0
        let t = translate(&ps[..ps.len() - 8], Stage::Vertex { attr_base: 4, attributes: 0 }).unwrap();
        assert!(!t.wgsl.contains("textureSample(") && t.wgsl.matches("textureSampleLevel(").count() == 2, "{}", t.wgsl);
        // a descriptor the program computes is refused; so is a cube for now
        let made = asm(&[
            ("S_MOVK_I32", "ENC_SOPK", &[("SDST", 8), ("SIMM16", 5)], None),
            ("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 9), ("VADDR", 2), ("SRSRC", 2), ("DMASK", 0xF), ("DIM", 1)], None),
        ]);
        assert!(translate_kernel(&made).err().unwrap().contains("a descriptor in s8, which s_movk_i32 at +0x0 writes"));
        let cube = asm(&[("IMAGE_SAMPLE", "ENC_MIMG", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 2), ("SSAMP", 4), ("DMASK", 0xF), ("DIM", 3)], None)]);
        assert!(translate_kernel(&cube).err().unwrap().contains("image_sample of dimension 3"));
    }

    #[test]
    fn what_is_missing_is_named() {
        // v_cvt_f16_f32 v0, v1 is not in this back end; an integer instruction has no modifiers
        let code = asm(&[
            ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 257)], None),
            ("V_OR3_B32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 257), ("SRC1", 257), ("SRC2", 257), ("CLAMP", 1)], None),
            ("S_ENDPGM", "ENC_SOPP", &[], None),
        ]);
        let e = translate_kernel(&code).err().unwrap();
        assert!(e.contains("v_cvt_f16_f32 ×1") && e.contains("v_or3_b32 with operand modifiers ×1"), "{e}");
    }

    #[test]
    fn modifiers_constants_and_the_scalar_table() {
        let code = asm(&[
            // v0 = clamp((|v1| + −v2) · 2)
            ("V_ADD_F32", "ENC_VOP3", &[("VDST", 0), ("SRC0", 257), ("SRC1", 258), ("ABS", 1), ("NEG", 2), ("OMOD", 1), ("CLAMP", 1)], None),
            // v3 = v1 · k + v2 ; v3 += v1 · v2
            ("V_MADMK_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 257), ("VSRC1", 2)], Some(3.0f32.to_bits())),
            ("V_FMAC_F32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 257), ("VSRC1", 2)], None),
            ("S_BFE_U32", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 0), ("SSRC1", 1)], None),
            ("S_CMPK_GT_I32", "ENC_SOPK", &[("SDST", 4), ("SIMM16", 0xFFFF)], None),
            ("S_ENDPGM", "ENC_SOPP", &[], None),
        ]);
        let t = translate_kernel(&code).unwrap();
        assert!(t.wgsl.contains("v0[l] = bitcast<u32>(clamp(((abs(bitcast<f32>(v1[l])) + (-bitcast<f32>(v2[l]))) * 2.0), 0.0, 1.0));"), "{}", t.wgsl);
        assert!(t.wgsl.contains("v3[l] = bitcast<u32>(mad_nf(bitcast<f32>(v1[l]), bitcast<f32>(1077936128u), bitcast<f32>(v2[l])));"), "{}", t.wgsl);
        assert!(t.wgsl.contains("v3[l] = bitcast<u32>(fma(bitcast<f32>(v1[l]), bitcast<f32>(v2[l]), bitcast<f32>(v3[l])));"), "{}", t.wgsl);
        assert!(t.wgsl.contains("{ let a = s[0]; let b = s[1]; let cur = s[4]; let r = sbfe_u(a, b); scc = r != 0u; s[4] = r; }"), "{}", t.wgsl);
        // the constant of a signed compare is sign-extended
        assert!(t.wgsl.contains("scc = (bitcast<i32>(s[4]) > bitcast<i32>(4294967295u));"), "{}", t.wgsl);
    }
}
