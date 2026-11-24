//! `xcl disasm <file> [--offset N] [--length N] [--all] [--hex] [--fssm FILE]` – RDNA2
//! disassembler for captured shader blobs (or any binary with console GPU code).
//! Console containers (XDAT/XBCH/FSSM) are unpacked and their layout is printed.

use std::fmt::Write as _;
use std::path::Path;

use crate::gpu::{rdna2, shader, xbox};

/// `println!` into the listing.
macro_rules! outln {
    ($o:expr, $($a:tt)*) => {{
        let _ = writeln!($o, $($a)*);
    }};
}

fn parse_num(s: &str) -> Option<usize> {
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => usize::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

fn number_option(args: &[String], name: &str) -> Result<Option<usize>, String> {
    let Some(index) = args.iter().position(|arg| arg == name) else { return Ok(None) };
    let value = args.get(index + 1).ok_or_else(|| format!("{name} requires a byte count"))?;
    parse_num(value).map(Some).ok_or_else(|| format!("invalid value for {name}: {value}"))
}

fn print_lines(out: &mut String, code: &[u8], base: usize, stop: bool) {
    for l in rdna2::disassemble(code, base as u64, stop) {
        let hex: Vec<String> = l.bytes.iter().map(|b| format!("{b:02x}")).collect();
        let mark = if l.flags & rdna2::F_BRANCH != 0 { "  ; →" } else { "" };
        outln!(out, "  {:08x}: {:<52} {}{mark}", l.pc, l.text, hex.join(" "));
    }
}

fn hexdump(out: &mut String, data: &[u8], base: usize) {
    for (i, row) in data.chunks(16).enumerate() {
        let hex: Vec<String> = row.iter().map(|b| format!("{b:02x}")).collect();
        let txt: String = row.iter().map(|&b| if (0x20..0x7f).contains(&b) { b as char } else { '.' }).collect();
        outln!(out, "    {:06x}: {:<47}  {txt}", base + i * 16, hex.join(" "));
    }
}

fn xdat_label(id: u32) -> &'static str {
    match id {
        xbox::XDAT_HASH => "hash",
        xbox::XDAT_XBCH => "XBCH → FSSM (native code)",
        xbox::XDAT_NESTED_HEADER => "nested DXIL header (NOBC)",
        _ => "?",
    }
}

fn print_xbox(out: &mut String, sh: &xbox::XboxShader, full_hex: bool, fssm_out: Option<&str>) {
    outln!(
        out,
        "console container: stage {} (DXIL), SM {}.{}, XDAT @ +{:#x}",
        sh.stage, sh.shader_model.0, sh.shader_model.1, sh.xdat_offset
    );
    for e in &sh.entries {
        outln!(out, "  XDAT {:#04x}  +{:#x}  {} B  {}", e.id, e.offset, e.size, xdat_label(e.id));
    }
    if let Some(h) = sh.hash {
        outln!(out, "  hash: {}", h.iter().map(|b| format!("{b:02x}")).collect::<String>());
    }
    for n in &sh.notes {
        outln!(out, "  NOTE: {n}");
    }
    for (i, f) in sh.fssm.iter().enumerate() {
        outln!(
            out,
            "FSSM #{i}: XBCH v{} @ +{:#x}, zlib {} B → {} B, FSSM v{:#x}",
            f.xbch_version,
            f.xbch_offset,
            f.compressed,
            f.data.len(),
            f.version
        );
        if let Some(e) = &f.error {
            outln!(out, "  WARNING: {e}");
        }
        for s in &f.stages {
            outln!(out, "  hardware stage: {} – descriptor @ +{:#x}", s.label(), s.offset);
        }
        match f.code {
            Some((o, l)) => outln!(
                out,
                "  code: +{o:#x}, {l} B ({}), programs: {}",
                if f.code_from_header { "pointer from header +0xC0" } else { "found by scanning" },
                f.programs.len()
            ),
            None => outln!(out, "  code: not found"),
        }
        let limit = if full_hex { usize::MAX } else { 0x140 };
        outln!(out, "  header – non-zero qwords (without the 0xDEADBEEF filler){}:", if full_hex { "" } else { ", up to +0x140; --hex: all" });
        for (o, q) in xbox::header_qwords(f, limit) {
            outln!(out, "    +{o:#05x}: {q:016x}");
        }
        if full_hex {
            if let Some((o, l)) = f.code {
                outln!(out, "  data after the code (+{:#x}):", o + l);
                hexdump(out, &f.data[o + l..], o + l);
            }
        }
        if let Some(file) = fssm_out {
            let name = if i == 0 { file.to_string() } else { format!("{file}.{i}") };
            match std::fs::write(&name, &f.data) {
                Ok(()) => outln!(out, "  saved FSSM to {name}"),
                Err(e) => eprintln!("xcl disasm: {name}: {e}"),
            }
        }
    }
}

/// The listing `xcl disasm` prints for `data` (`name`: the file, for the first line). `Err`: a
/// bad argument.
pub fn render(name: &str, data: &[u8], args: &[String]) -> Result<String, String> {
    let (offset, length) = (number_option(args, "--offset")?, number_option(args, "--length")?);
    let all = args.iter().any(|a| a == "--all");
    let mut text = String::new();
    let out = &mut text;
    outln!(out, "{name} – {} B, decoder: {}", data.len(), crate::gpu::RDNA2_ARCH);

    if offset.is_some() || length.is_some() {
        let off = offset.unwrap_or(0);
        if off >= data.len() {
            return Err(format!("offset {off:#x} is past the end of the file"));
        }
        let len = length.unwrap_or(data.len() - off).min(data.len() - off);
        print_lines(out, &data[off..off + len], off, !all);
        return Ok(text);
    }

    let (kind, parts) = shader::classify(data);
    outln!(out, "format: {}", kind.label());
    if !parts.is_empty() {
        outln!(out, "container parts:");
        for p in &parts {
            outln!(out, "  {}  +{:#x}  {} B", p.fourcc, p.offset, p.size);
        }
    }
    if kind == shader::Kind::XboxNative {
        if let Some(sh) = xbox::parse(data) {
            let file = args.iter().position(|a| a == "--fssm").and_then(|i| args.get(i + 1)).map(String::as_str);
            print_xbox(out, &sh, args.iter().any(|a| a == "--hex"), file);
        }
    }
    let listing = shader::rdna2_listing(data);
    if listing.is_empty() {
        outln!(out, "no RDNA2 programs found (ending in s_endpgm); use --offset N to force decoding");
        if matches!(kind, shader::Kind::Dxil | shader::Kind::Dxbc) {
            outln!(out, "(this is a PC-format shader – no recompilation needed)");
        }
    } else {
        out.push_str(&listing);
    }
    Ok(text)
}

pub fn run(path: &Path, args: &[String]) -> i32 {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("xcl disasm: {}: {e}", path.display());
            return 2;
        }
    };
    match render(&path.display().to_string(), &data, args) {
        Ok(text) => {
            print!("{text}");
            0
        }
        Err(error) => {
            eprintln!("xcl disasm: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_forced_listing_and_rejects_bad_arguments() {
        // s_endpgm
        let code = 0xBF81_0000u32.to_le_bytes();
        let text = render("blob.bin", &code, &["--offset".into(), "0".into()]).unwrap();
        assert!(text.starts_with("blob.bin – 4 B, decoder: "), "{text}");
        assert!(text.contains("s_endpgm"), "{text}");
        assert!(render("blob.bin", &code, &["--offset".into(), "8".into()]).unwrap_err().contains("past the end"));
        assert!(render("blob.bin", &code, &["--offset".into()]).is_err());
    }
}
