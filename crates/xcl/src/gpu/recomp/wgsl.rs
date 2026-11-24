//! RDNA2 → WGSL: the recompiler's back end for WebGPU. **A beginning**: a first set of
//! instructions – 32-bit float and integer arithmetic, compares, selects, the scalar unit,
//! branches, exports and attribute interpolation – as a kernel (the program runs on a register
//! file, as the HLSL back end's instruction tests do), a vertex stage or a pixel stage. Memory
//! is read (dword loads through a pointer or a buffer descriptor), not written; no textures.
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

/// The value a vector instruction writes, from its sources `a` `b` `c` (`u32` expressions).
fn vector_value(name: &str, a: &str, b: &str, c: &str) -> Option<String> {
    let (fa, fb, fc, ia, ib) = (f(a), f(b), f(c), i(a), i(b));
    Some(match name {
        "v_mov_b32" => a.to_string(),
        "v_not_b32" => format!("~{a}"),
        "v_add_f32" => fr(format!("{fa} + {fb}")),
        "v_sub_f32" => fr(format!("{fa} - {fb}")),
        "v_subrev_f32" => fr(format!("{fb} - {fa}")),
        "v_mul_f32" => fr(format!("{fa} * {fb}")),
        "v_min_f32" => fr(format!("min({fa}, {fb})")),
        "v_max_f32" => fr(format!("max({fa}, {fb})")),
        "v_fma_f32" => fr(format!("fma({fa}, {fb}, {fc})")),
        "v_floor_f32" => fr(format!("floor({fa})")),
        "v_ceil_f32" => fr(format!("ceil({fa})")),
        "v_trunc_f32" => fr(format!("trunc({fa})")),
        "v_rndne_f32" => fr(format!("round({fa})")),
        "v_fract_f32" => fr(format!("fract({fa})")),
        "v_sqrt_f32" => fr(format!("sqrt({fa})")),
        "v_rcp_f32" => fr(format!("1.0 / {fa}")),
        "v_rsq_f32" => fr(format!("inverseSqrt({fa})")),
        "v_cvt_f32_i32" => fr(format!("f32({ia})")),
        "v_cvt_f32_u32" => fr(format!("f32({a})")),
        "v_cvt_i32_f32" => fr(format!("i32({fa})")),
        "v_cvt_u32_f32" => format!("u32({fa})"),
        "v_cvt_pkrtz_f16_f32" => format!("f16_rtz({fa}) | (f16_rtz({fb}) << 16u)"),
        "v_add_nc_u32" => format!("{a} + {b}"),
        "v_sub_nc_u32" => format!("{a} - {b}"),
        "v_subrev_nc_u32" => format!("{b} - {a}"),
        "v_mul_lo_u32" => format!("{a} * {b}"),
        "v_mul_hi_u32" => format!("mul_hi({a}, {b})"),
        "v_and_b32" => format!("{a} & {b}"),
        "v_or_b32" => format!("{a} | {b}"),
        "v_xor_b32" => format!("{a} ^ {b}"),
        "v_xnor_b32" => format!("~({a} ^ {b})"),
        "v_lshlrev_b32" => format!("{b} << ({a} & 31u)"),
        "v_lshrrev_b32" => format!("{b} >> ({a} & 31u)"),
        "v_ashrrev_i32" => fr(format!("{ib} >> ({a} & 31u)")),
        "v_min_u32" => format!("min({a}, {b})"),
        "v_max_u32" => format!("max({a}, {b})"),
        "v_min_i32" => fr(format!("min({ia}, {ib})")),
        "v_max_i32" => fr(format!("max({ia}, {ib})")),
        "v_cndmask_b32" => format!("select({a}, {b}, on(vcc, l))"),
        _ => return None,
    })
}

/// One instruction as WGSL statements (branches are the caller's).
fn statement(d: &Decoded, stage: Stage) -> Result<String, String> {
    let name = d.inst.name.to_ascii_lowercase();
    let lit = d.literal;
    let enc = d.enc.name;
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
        if ["ABS", "NEG", "OMOD", "CLAMP"].iter().any(|m| field(d, m).unwrap_or(0) != 0) {
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
            let cast = |e: &str| match ty {
                "f32" => Ok(f(e)),
                "i32" => Ok(i(e)),
                "u32" => Ok(e.to_string()),
                _ => Err(format!("{name}")),
            };
            let test = condition(op, &cast(&a)?, &cast(&b)?, ty == "f32").ok_or(&name)?;
            // the mask goes to VCC, or to the register pair the VOP3 form names
            // (v_cmpx: to EXEC, so lanes that fail are off from here)
            let to = store64(if cmpx { 126 } else if enc.contains("VOP3") { field(d, "VDST").ok_or("VDST")? } else { 106 }, "m")?;
            return Ok(format!("{{ var m = vec2<u32>(0u); for (var l = 0u; l < LANES; l++) {{ if (on(exec, l) && {test}) {{ m = put(m, l); }} }} {to} }}"));
        }
        let value = vector_value(&name, &a, &b, &c).ok_or(&name)?;
        let dst = field(d, "VDST").ok_or("VDST")?;
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
        "s_mov_b32" => to(&x(s0)?)?,
        "s_mov_b64" => to64(&x64(s0)?)?,
        "s_movk_i32" => to(&format!("{}u", field(d, "SIMM16").ok_or("SIMM16")? as u16 as i16 as i32 as u32))?,
        "s_not_b32" => format!("{{ let r = ~{}; scc = r != 0u; {} }}", x(s0)?, to("r")?),
        "s_not_b64" => format!("{{ let r = ~{}; scc = any(r != vec2<u32>(0u)); {} }}", x64(s0)?, to64("r")?),
        "s_add_u32" => format!("{{ let a = {}; let r = a + {}; scc = r < a; {} }}", x(s0)?, x(s1)?, to("r")?),
        "s_sub_u32" => format!("{{ let a = {}; let b = {}; scc = b > a; {} }}", x(s0)?, x(s1)?, to("a - b")?),
        "s_add_i32" => format!("{{ let a = {}; let b = {}; let r = a + b; scc = ((~(a ^ b)) & (a ^ r)) >> 31u != 0u; {} }}", x(s0)?, x(s1)?, to("r")?),
        "s_sub_i32" => format!("{{ let a = {}; let b = {}; let r = a - b; scc = ((a ^ b) & (a ^ r)) >> 31u != 0u; {} }}", x(s0)?, x(s1)?, to("r")?),
        "s_mul_i32" => to(&format!("{} * {}", x(s0)?, x(s1)?))?,
        "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32" => {
            let (a, b) = (x(s0)?, x(s1)?);
            let r = match name.as_str() {
                "s_and_b32" => format!("{a} & {b}"),
                "s_or_b32" => format!("{a} | {b}"),
                "s_xor_b32" => format!("{a} ^ {b}"),
                "s_andn2_b32" => format!("{a} & ~{b}"),
                _ => format!("{a} | ~{b}"),
            };
            format!("{{ let r = {r}; scc = r != 0u; {} }}", to("r")?)
        }
        "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64" => {
            let (a, b) = (x64(s0)?, x64(s1)?);
            let r = match name.as_str() {
                "s_and_b64" => format!("{a} & {b}"),
                "s_or_b64" => format!("{a} | {b}"),
                "s_xor_b64" => format!("{a} ^ {b}"),
                "s_andn2_b64" => format!("{a} & ~{b}"),
                _ => format!("{a} | ~{b}"),
            };
            format!("{{ let r = {r}; scc = any(r != vec2<u32>(0u)); {} }}", to64("r")?)
        }
        "s_lshl_b32" => format!("{{ let r = {} << ({} & 31u); scc = r != 0u; {} }}", x(s0)?, x(s1)?, to("r")?),
        "s_lshr_b32" => format!("{{ let r = {} >> ({} & 31u); scc = r != 0u; {} }}", x(s0)?, x(s1)?, to("r")?),
        "s_ashr_i32" => format!("{{ let r = bitcast<u32>({} >> ({} & 31u)); scc = r != 0u; {} }}", i(&x(s0)?), x(s1)?, to("r")?),
        "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
            let (a, b) = (x(s0)?, x(s1)?);
            let signed = name.ends_with("i32");
            let less = if signed { format!("{} < {}", i("a"), i("b")) } else { "a < b".to_string() };
            let first = if name.starts_with("s_min") { less } else { if signed { format!("{} > {}", i("a"), i("b")) } else { "a > b".to_string() } };
            format!("{{ let a = {a}; let b = {b}; scc = {first}; {} }}", to("select(b, a, scc)")?)
        }
        "s_cselect_b32" => to(&format!("select({}, {}, scc)", x(s1)?, x(s0)?))?,
        // EXEC is saved, then masked with the source; SCC: lanes are left
        "s_and_saveexec_b64" | "s_or_saveexec_b64" | "s_andn2_saveexec_b64" | "s_andn1_saveexec_b64" => {
            let a = x64(s0)?;
            let r = match name.as_str() {
                "s_and_saveexec_b64" => "a & exec",
                "s_or_saveexec_b64" => "a | exec",
                "s_andn2_saveexec_b64" => "a & ~exec",
                _ => "~a & exec",
            };
            format!("{{ let a = {a}; let saved = exec; exec = {r}; scc = any(exec != vec2<u32>(0u)); {} }}", to64("saved")?)
        }
        _ => {
            if let Some(rest) = name.strip_prefix("s_cmp_") {
                let (op, ty) = rest.rsplit_once('_').ok_or(&name)?;
                let (a, b) = (x(s0)?, x(s1)?);
                let (a, b) = if ty == "i32" { (i(&a), i(&b)) } else { (a, b) };
                return Ok(format!("scc = {};", condition(op, &a, &b, false).ok_or(&name)?));
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
        match statement(d, stage) {
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
    let _ = writeln!(w, "const LANES = {lanes}u;");
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
    // helpers: a lane's bit in a mask, the mask with a lane's bit, the high half of a product
    let _ = writeln!(w, "fn on(m: vec2<u32>, l: u32) -> bool {{ if (l < 32u) {{ return ((m.x >> l) & 1u) != 0u; }} return ((m.y >> (l - 32u)) & 1u) != 0u; }}");
    let _ = writeln!(w, "fn put(m: vec2<u32>, l: u32) -> vec2<u32> {{ if (l < 32u) {{ return vec2<u32>(m.x | (1u << l), m.y); }} return vec2<u32>(m.x, m.y | (1u << (l - 32u))); }}");
    let _ = writeln!(w, "fn mul_hi(a: u32, b: u32) -> u32 {{ let al = a & 0xffffu; let ah = a >> 16u; let bl = b & 0xffffu; let bh = b >> 16u; let mid = ah * bl + ((al * bl) >> 16u); let mid2 = al * bh + (mid & 0xffffu); return ah * bh + (mid >> 16u) + (mid2 >> 16u); }}");
    // a float as a half, rounded toward zero (from the bits: WGSL's pack leaves the rounding open)
    let _ = writeln!(w, "fn f16_rtz(f: f32) -> u32 {{ let b = bitcast<u32>(f); let sign = (b >> 16u) & 0x8000u; let u = b & 0x7fffffffu; let e = i32(u >> 23u) - 127; if (u > 0x7f800000u) {{ return sign | 0x7e00u; }} if (u == 0x7f800000u) {{ return sign | 0x7c00u; }} if (e > 15) {{ return sign | 0x7bffu; }} if (e < -25) {{ return sign; }} if (e < -14) {{ return sign | (((u & 0x7fffffu) | 0x800000u) >> u32(-1 - e)); }} return sign | (u32(e + 15) << 10u) | ((u & 0x7fffffu) >> 13u); }}");
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
    Ok(Translation { wgsl: w, dwords: total, memory })
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
    fn what_is_missing_is_named() {
        // v_sin_f32 v0, v1 is not in this back end
        let code = asm(&[("V_SIN_F32", "ENC_VOP1", &[("VDST", 0), ("SRC0", 257)], None), ("S_ENDPGM", "ENC_SOPP", &[], None)]);
        let e = translate_kernel(&code).err().unwrap();
        assert!(e.contains("v_sin_f32 ×1"), "{e}");
    }
}
