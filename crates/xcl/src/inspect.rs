//! `xcl inspect <exe>` – static analysis without running anything (read-only).

use std::collections::BTreeMap;
use std::fmt::Write as _;
#[cfg(windows)]
use std::path::Path;

use serde::Serialize;

use crate::pe::{ImportName, Layout, Pe};
#[cfg(windows)]
use crate::resolver::{self, Source};

#[derive(Serialize)]
struct Entry {
    dll: String,
    function: String,
    delay: bool,
    status: String,
}

/// What `xcl inspect` reports about a PE.
#[derive(Serialize)]
pub struct Out {
    file: String,
    machine: String,
    image_base: String,
    entry_rva: String,
    size_of_image: String,
    subsystem: u16,
    is_dll: bool,
    has_relocations: bool,
    tls_template_bytes: Option<u64>,
    exception_entries: u32,
    stack_reserve: String,
    imports: Vec<Entry>,
}

/// Console-only DLLs a PE imports (kernelx.dll, xgameruntime*, d3d12_xs …).
/// `None` when the file is not a valid x64 PE. A plain Windows program yields `Some([])`.
#[cfg(windows)]
pub fn console_imports(path: &Path) -> Option<Vec<String>> {
    let data = std::fs::read(path).ok()?;
    console_imports_of(&data)
}

/// `console_imports` of a file's bytes.
pub fn console_imports_of(data: &[u8]) -> Option<Vec<String>> {
    let pe = Pe::parse(data, Layout::File).ok()?;
    let mut all = pe.imports().unwrap_or_default();
    all.extend(pe.delay_imports().unwrap_or_default());
    let reg = crate::hle::registry();
    let mut out: Vec<String> = Vec::new();
    for d in &all {
        let key = crate::util::dll_key(&d.dll);
        let console = key == crate::hle::KERNELX
            || reg
                .hle_for(&key)
                .map(|m| m.name == crate::hle::XGAMERUNTIME || m.name == crate::hle::D3D12X)
                .unwrap_or(false);
        if console && !out.contains(&key) {
            out.push(key);
        }
    }
    Some(out)
}

#[cfg(windows)]
fn status(s: &Source) -> String {
    match s {
        Source::Hle(m) => format!("HLE ({m})"),
        Source::HleOther(m) => format!("HLE ({m}, other module)"),
        Source::System(d) => format!("Windows ({d})"),
        Source::GameDll(d) => format!("game DLL ({d})"),
        Source::Stub => "STUB".into(),
    }
}

/// What the routing alone says about an import: the HLE module its DLL goes to. Whether that
/// module implements the function is known only where the HLE code is built (Windows).
pub fn route_status(dll: &str, _what: &ImportName) -> String {
    let key = crate::util::dll_key(dll);
    match crate::hle::registry().route(&key) {
        Some(crate::hle::Route::Hle(i)) => format!("HLE ({}, function not checked)", crate::hle::registry().modules[*i].name),
        Some(crate::hle::Route::System) => format!("Windows ({key})"),
        Some(crate::hle::Route::Stub) => "STUB".into(),
        None => format!("Windows or game DLL ({key}, not checked)"),
    }
}

/// The report for a PE file's bytes; `classify` gives each import its status.
pub fn describe(file: &str, data: &[u8], classify: impl Fn(&str, &ImportName) -> String) -> Result<Out, String> {
    let pe = Pe::parse(data, Layout::File).map_err(|e| e.to_string())?;
    let mut imports = Vec::new();
    let mut all = pe.imports().unwrap_or_default();
    all.extend(pe.delay_imports().unwrap_or_default());
    for d in &all {
        for e in &d.entries {
            imports.push(Entry { dll: d.dll.clone(), function: e.what.to_string(), delay: d.delay, status: classify(&d.dll, &e.what) });
        }
    }
    let tls = pe.tls().ok().flatten().map(|t| t.total_len());
    let ex = pe.dir(crate::pe::DIR_EXCEPTION);
    Ok(Out {
        file: file.to_string(),
        machine: format!("{:#06x}", pe.machine),
        image_base: format!("{:#x}", pe.image_base),
        entry_rva: format!("{:#x}", pe.entry_rva),
        size_of_image: format!("{:#x}", pe.size_of_image),
        subsystem: pe.subsystem,
        is_dll: pe.is_dll(),
        has_relocations: pe.has_relocs(),
        tls_template_bytes: tls,
        exception_entries: ex.size / 12,
        stack_reserve: format!("{:#x}", pe.stack_reserve),
        imports,
    })
}

/// The text `xcl inspect` prints for a report. `console`: `console_imports_of` the same file.
pub fn render(out: &Out, console: Option<Vec<String>>) -> String {
    let mut text = String::new();
    let w = &mut text;
    let tls = out.tls_template_bytes;
    let _ = writeln!(w, "File:          {}", out.file);
    let _ = writeln!(w, "Type:          {} x64, subsystem {}", if out.is_dll { "DLL" } else { "EXE" }, out.subsystem);
    let _ = writeln!(w, "ImageBase:     {}  (size {}, entry +{})", out.image_base, out.size_of_image, out.entry_rva);
    let _ = writeln!(w, "Relocations:   {}", if out.has_relocations { "yes" } else { "NO (the image must be mapped at ImageBase)" });
    let _ = writeln!(w, "Static TLS:    {}", tls.map(|n| format!("{n} B (handled by the TLS carrier)")).unwrap_or_else(|| "none".into()));
    let _ = writeln!(w, "Exceptions:    {} .pdata entries", out.exception_entries);
    let _ = writeln!(w, "Stack:         {}", out.stack_reserve);
    match console {
        Some(c) if !c.is_empty() => {
            let _ = writeln!(w, "Console game:  yes (imports {})", c.join(", "));
        }
        _ => {
            let _ = writeln!(w, "Console game:  NO – no kernelx / xgameruntime / d3d12_xs imports (a regular Windows program?)");
        }
    }
    let mut per_dll: BTreeMap<String, Vec<&Entry>> = BTreeMap::new();
    for e in &out.imports {
        per_dll.entry(format!("{}{}", e.dll, if e.delay { " (delay-load)" } else { "" })).or_default().push(e);
    }
    let total_stub = out.imports.iter().filter(|e| e.status == "STUB").count();
    let _ = writeln!(w, "\nImports: {} functions from {} DLLs, {} of them unimplemented (stubs):", out.imports.len(), per_dll.len(), total_stub);
    for (dll, entries) in &per_dll {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for e in entries {
            let k = if e.status.starts_with("HLE") {
                "HLE"
            } else if e.status.starts_with("Windows") {
                "Windows"
            } else if e.status.starts_with("game DLL") {
                "game DLL"
            } else {
                "stubs"
            };
            *counts.entry(k).or_default() += 1;
        }
        let c: Vec<String> = counts.iter().map(|(k, v)| format!("{v} {k}")).collect();
        let _ = writeln!(w, "  {dll}: {}", c.join(", "));
        for e in entries.iter().filter(|e| e.status == "STUB") {
            let _ = writeln!(w, "      - {}", e.function);
        }
    }
    text
}

#[cfg(windows)]
pub fn run(path: &Path, json: bool) -> i32 {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("xcl inspect: cannot read {}: {e}", path.display());
            return 2;
        }
    };
    if let Some(dir) = path.parent() {
        resolver::set_search_dirs(vec![dir.to_path_buf()]);
    }
    let out = match describe(&path.to_string_lossy(), &data, |dll, what| status(&resolver::classify(dll, what))) {
        Ok(out) => out,
        Err(e) => {
            eprintln!(
                "xcl inspect: {}: {e}\n(If this is a file from an XVC/MSIXVC package – XCL only works on already unpacked files.)",
                path.display()
            );
            return 3;
        }
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
        return 0;
    }
    print!("{}", render(&out, console_imports_of(&data)));
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal x64 PE file: headers and one section holding an import table (name → functions).
    fn pe_with_imports(imports: &[(&str, &[&str])]) -> Vec<u8> {
        const RVA: u32 = 0x1000;
        const RAW: usize = 0x200;
        // section: descriptors, then per DLL its thunks, then the strings
        let mut sec = vec![0u8; (imports.len() + 1) * 20];
        let put32 = |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        for (i, (dll, functions)) in imports.iter().enumerate() {
            let thunks = sec.len();
            sec.resize(thunks + (functions.len() + 1) * 8, 0);
            for (k, f) in functions.iter().enumerate() {
                let hint_name = sec.len() as u32 + RVA;
                sec.extend_from_slice(&[0, 0]);
                sec.extend_from_slice(f.as_bytes());
                sec.push(0);
                put32(&mut sec, thunks + k * 8, hint_name);
            }
            let name = sec.len() as u32 + RVA;
            sec.extend_from_slice(dll.as_bytes());
            sec.push(0);
            put32(&mut sec, i * 20, thunks as u32 + RVA);
            put32(&mut sec, i * 20 + 12, name);
            put32(&mut sec, i * 20 + 16, thunks as u32 + RVA);
        }
        let mut pe = vec![0u8; RAW];
        pe[0] = b'M';
        pe[1] = b'Z';
        put32(&mut pe, 0x3C, 0x40);
        pe[0x40..0x44].copy_from_slice(b"PE\0\0");
        let coff = 0x44;
        pe[coff..coff + 2].copy_from_slice(&0x8664u16.to_le_bytes());
        pe[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes());
        pe[coff + 16..coff + 18].copy_from_slice(&240u16.to_le_bytes());
        pe[coff + 18..coff + 20].copy_from_slice(&0x0022u16.to_le_bytes());
        let opt = coff + 20;
        pe[opt..opt + 2].copy_from_slice(&0x20Bu16.to_le_bytes());
        put32(&mut pe, opt + 16, 0x1000);
        pe[opt + 24..opt + 32].copy_from_slice(&0x1_4000_0000u64.to_le_bytes());
        put32(&mut pe, opt + 32, 0x1000);
        put32(&mut pe, opt + 36, 0x200);
        put32(&mut pe, opt + 56, 0x2000);
        put32(&mut pe, opt + 60, RAW as u32);
        pe[opt + 68..opt + 70].copy_from_slice(&2u16.to_le_bytes());
        pe[opt + 72..opt + 80].copy_from_slice(&0x10_0000u64.to_le_bytes());
        put32(&mut pe, opt + 108, 16);
        put32(&mut pe, opt + 112 + 8, RVA);
        put32(&mut pe, opt + 112 + 12, ((imports.len() + 1) * 20) as u32);
        let s = opt + 240;
        pe[s..s + 6].copy_from_slice(b".rdata");
        put32(&mut pe, s + 8, sec.len() as u32);
        put32(&mut pe, s + 12, RVA);
        put32(&mut pe, s + 16, sec.len() as u32);
        put32(&mut pe, s + 20, RAW as u32);
        put32(&mut pe, s + 36, 0x4000_0040);
        pe.extend_from_slice(&sec);
        pe
    }

    #[test]
    fn describes_a_pe_and_its_imports_from_bytes() {
        let pe = pe_with_imports(&[("kernelx.dll", &["XMemAlloc", "CreateFileW"]), ("d3d12_xs.dll", &["D3D12XboxCreateDevice"]), ("ucrtbase.dll", &["malloc"])]);
        let out = describe("game.exe", &pe, route_status).unwrap();
        assert_eq!(out.image_base, "0x140000000");
        assert!(!out.is_dll);
        let got: Vec<(&str, &str, &str)> = out.imports.iter().map(|e| (e.dll.as_str(), e.function.as_str(), e.status.as_str())).collect();
        assert_eq!(
            got,
            [
                ("kernelx.dll", "XMemAlloc", "HLE (kernelx.dll, function not checked)"),
                ("kernelx.dll", "CreateFileW", "HLE (kernelx.dll, function not checked)"),
                ("d3d12_xs.dll", "D3D12XboxCreateDevice", "HLE (d3d12_xs.dll, function not checked)"),
                ("ucrtbase.dll", "malloc", "Windows or game DLL (ucrtbase.dll, not checked)"),
            ]
        );
        assert_eq!(console_imports_of(&pe), Some(vec!["kernelx.dll".to_string(), "d3d12_xs.dll".to_string()]));
        let text = render(&out, console_imports_of(&pe));
        assert!(text.contains("Console game:  yes (imports kernelx.dll, d3d12_xs.dll)"), "{text}");
        assert!(text.contains("Imports: 4 functions from 3 DLLs"), "{text}");
    }

    #[test]
    fn a_file_that_is_not_a_pe_is_an_error() {
        assert!(describe("x.bin", b"not a PE", route_status).is_err());
        assert_eq!(console_imports_of(b"not a PE"), None);
    }
}
