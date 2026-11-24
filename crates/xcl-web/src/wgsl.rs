//! The shader recompiler's WGSL back end, for the page and its tests: assemble RDNA2
//! instructions and translate a program to a WGSL compute shader (kernel mode).

use super::{answer, bytes};

/// Assemble RDNA2 instructions, one a line: `NAME ENCODING FIELD=value … [LIT=value]` (values
/// decimal or 0x…; `#` starts a comment). The answer is the code as hexadecimal digits.
///
/// # Safety
/// `text` (UTF-8): a block from `xcl_alloc`.
#[no_mangle]
pub unsafe extern "C" fn xcl_rdna2_assemble(text: *const u8, text_len: usize) -> u32 {
    let text = String::from_utf8_lossy(bytes(text, text_len)).into_owned();
    let number = |v: &str| match v.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => v.parse().ok(),
    };
    let mut code = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut words = line.split_whitespace();
        let (Some(name), Some(encoding)) = (words.next(), words.next()) else { return answer(Err(format!("line {}: NAME ENCODING …", n + 1))) };
        let (mut fields, mut literal) = (Vec::new(), None);
        for w in words {
            let Some((field, value)) = w.split_once('=').and_then(|(f, v)| Some((f, number(v)?))) else { return answer(Err(format!("line {}: {w}", n + 1))) };
            if field == "LIT" {
                literal = Some(value as u32);
            } else {
                fields.push((field, value));
            }
        }
        match xcl::gpu::rdna2::encode(name, encoding, &fields, literal) {
            Ok(bytes) => code.extend(bytes),
            Err(e) => return answer(Err(format!("line {}: {e}", n + 1))),
        }
    }
    answer(Ok(code.iter().map(|b| format!("{b:02x}")).collect()))
}

/// Translate an RDNA2 program to a WGSL compute shader in kernel mode (`io` and `res`: 128
/// scalar slots, then 256 vector registers for each of 64 lanes). An error names what the back
/// end does not have yet.
///
/// # Safety
/// `code`: a block from `xcl_alloc`.
#[no_mangle]
pub unsafe extern "C" fn xcl_wgsl_kernel(code: *const u8, code_len: usize) -> u32 {
    answer(xcl::gpu::recomp::wgsl::translate_kernel(bytes(code, code_len)).map(|t| t.wgsl))
}

/// Translate an RDNA2 program to a WGSL vertex stage (`stage` 1: `a` the first attribute VGPR,
/// `b` the number of attributes, each `@location(k)` in four VGPRs) or pixel stage (`stage` 2:
/// `a` SPI_PS_INPUT_ENA). The SGPRs are a uniform buffer at group 0 binding 0.
///
/// # Safety
/// `code`: a block from `xcl_alloc`.
#[no_mangle]
pub unsafe extern "C" fn xcl_wgsl_stage(code: *const u8, code_len: usize, stage: u32, a: u32, b: u32) -> u32 {
    use xcl::gpu::recomp::wgsl::{translate, Stage};
    let stage = match stage {
        1 => Stage::Vertex { attr_base: a as u64, attributes: b },
        2 => Stage::Pixel { input_ena: a },
        _ => return answer(Err(format!("stage {stage}"))),
    };
    answer(translate(bytes(code, code_len), stage).map(|t| t.wgsl))
}
