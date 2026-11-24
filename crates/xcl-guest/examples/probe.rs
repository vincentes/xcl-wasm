//! What a game folder's images need from the guest, before the title is run:
//!   cargo run -p xcl-guest --example probe -- <game folder> [image.exe | image.dll …]
//!
//!   cargo run --release -p xcl-guest --example probe -- <game folder> --run [million instructions]
//!
//! Without image names, every `.exe` and `.dll` of the folder is probed. Each one is mapped on a
//! guest of its own (with the folder's other DLLs for its imports), the functions it imports
//! that the host does not implement are listed, and a DLL's entry point is run
//! (DLL_PROCESS_ATTACH). With `--run`, the folder's executable is started instead and run for
//! that many instructions (100 million when none is given): where it stopped, the last system
//! functions it called and the ones the host answered with a stub. Without `--run` only the
//! folder's images are read; with it the title also gets the folder's other files: those of up
//! to 16 MB whole, the larger ones (its archives) read from disk a range at a time, as it asks.

use std::collections::BTreeMap;
use std::path::Path;

use xcl::pe::{ImportName, Layout, Pe};
use xcl_guest::vfs::Source;
use xcl_guest::{Guest, Stop};

fn is_image(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".dll") || lower.ends_with(".exe")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(folder) = args.next() else {
        eprintln!("usage: probe <game folder> [image …]");
        std::process::exit(2);
    };
    let folder = Path::new(&folder);
    let mut images: Vec<(String, Vec<u8>)> = Vec::new();
    for entry in std::fs::read_dir(folder).expect("the game folder").flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.path().is_file() && is_image(&name) {
            images.push((name, std::fs::read(entry.path()).expect("an image of the game folder")));
        }
    }
    images.sort_by_key(|(name, _)| name.to_ascii_lowercase());
    let mut wanted: Vec<String> = args.collect();
    if let Some(at) = wanted.iter().position(|w| w == "--run") {
        let millions = wanted.get(at + 1).and_then(|n| n.parse::<u64>().ok()).unwrap_or(100);
        wanted.truncate(at);
        run(folder, &images, millions * 1_000_000);
        return;
    }
    // function → the images that import it, over the whole folder
    let mut missing_all: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, data) in &images {
        if !wanted.is_empty() && !wanted.iter().any(|w| w.eq_ignore_ascii_case(name)) {
            continue;
        }
        let mut g = Guest::new();
        for (other, bytes) in &images {
            g.vfs.add_game_file(other, bytes.clone());
        }
        let pe = match Pe::parse(data, Layout::File) {
            Ok(pe) => pe,
            Err(e) => {
                println!("{name}: not an image: {e}");
                continue;
            }
        };
        let mut missing = Vec::new();
        let mut total = 0;
        for dll in pe.imports().unwrap_or_default() {
            let from_folder = images.iter().any(|(n, _)| n.eq_ignore_ascii_case(&dll.dll));
            for e in &dll.entries {
                total += 1;
                let function = match &e.what {
                    ImportName::Name { name, .. } => name.clone(),
                    ImportName::Ordinal(o) => format!("#{o}"),
                };
                if !from_folder && !g.has_handler(&function) {
                    missing.push(format!("{}!{function}", dll.dll.to_ascii_lowercase()));
                }
            }
        }
        for m in &missing {
            missing_all.entry(m.clone()).or_default().push(name.clone());
        }
        let ran = match g.map_image(name, data) {
            Err(e) => format!("not mapped: {e}"),
            Ok(index) if !pe.is_dll() => format!("mapped at {:#x} (the executable: not run here)", g.modules[index].base),
            Ok(index) => match g.start(index, 1 << 20) {
                Ok(()) => "entry point ran (DLL_PROCESS_ATTACH)".to_string(),
                Err(e) => format!("entry point: {e}"),
            },
        };
        println!("{name}: {total} imports, {} without a host function; {ran}", missing.len());
        for line in g.log.iter().take(8) {
            println!("    log: {line}");
        }
    }
    println!("\n{} functions the host does not implement:", missing_all.len());
    for (function, by) in &missing_all {
        println!("  {function}  ({})", by.join(", "));
    }
}

/// The largest file the title is given whole.
const WHOLE_FILE_LIMIT: u64 = 16 << 20;

/// A file on disk that is read a range at a time.
fn on_demand(path: &Path, size: u64) -> Option<Source> {
    use std::io::{Read, Seek, SeekFrom};
    let file = std::cell::RefCell::new(std::fs::File::open(path).ok()?);
    let read = move |offset: u64, buffer: &mut [u8]| {
        let mut file = file.borrow_mut();
        file.seek(SeekFrom::Start(offset)).is_ok() && file.read_exact(buffer).is_ok()
    };
    Some(Source { size, read: Box::new(read) })
}

/// Give the title the files of `dir` (and below) that are not images, as `prefix` + their name:
/// the small ones whole, the others (counted in `large`) read on demand.
fn add_files(g: &mut Guest, dir: &Path, prefix: &str, large: &mut (usize, u64)) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
        let path = entry.path();
        if path.is_dir() {
            add_files(g, &path, &format!("{name}\\"), large);
        } else if !is_image(&name) && !name.starts_with('.') {
            match std::fs::metadata(&path).map(|m| m.len()) {
                Ok(size) if size <= WHOLE_FILE_LIMIT => {
                    if let Ok(bytes) = std::fs::read(&path) {
                        g.vfs.add_game_file(&name, bytes);
                    }
                }
                Ok(size) => {
                    if let Some(source) = on_demand(&path, size) {
                        g.vfs.add_game_source(&name, source);
                        *large = (large.0 + 1, large.1 + size);
                    }
                }
                Err(_) => {}
            }
        }
    }
}

/// Start the folder's executable and run it for `budget` instructions.
fn run(folder: &Path, images: &[(String, Vec<u8>)], budget: u64) {
    let Some((name, data)) = images.iter().find(|(n, _)| n.to_ascii_lowercase().ends_with(".exe")) else {
        println!("the folder has no executable");
        return;
    };
    let mut g = Guest::new();
    g.trace_len = 48;
    g.command_line = name.clone();
    for (other, bytes) in images {
        g.vfs.add_game_file(other, bytes.clone());
    }
    let mut large = (0, 0);
    add_files(&mut g, folder, "", &mut large);
    println!("{} files of the folder ({} GB) are read on demand", large.0, large.1 >> 30);
    let started = g.map_image(name, data).and_then(|index| g.start(index, 1 << 20));
    // XCL_PROBE_AFTER=function: from the first call of that system function on, the title is
    // run an instruction at a time and what it executes is counted by opcode
    let after = std::env::var("XCL_PROBE_AFTER").ok();
    let mut opcodes: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut path: std::collections::VecDeque<u64> = Default::default();
    let stop = match started {
        Ok(()) if after.is_some() => {
            let name = after.as_deref().unwrap_or_default();
            let mut left = budget;
            let mut stepping = false;
            loop {
                if !stepping {
                    stepping = g.imports.iter().any(|i| i.name == name && i.calls != 0);
                }
                if stepping {
                    let rip = g.cpu.rip;
                    let entry = opcodes.entry(opcode_of(&g, rip)).or_insert((0, rip));
                    entry.0 += 1;
                    if path.len() == 400_000 {
                        path.pop_front();
                    }
                    path.push_back(rip);
                }
                let step = if stepping { 1 } else { 2000.min(left) };
                match g.run(step) {
                    Stop::Limit if left > step => left -= step,
                    stop => break stop,
                }
            }
        }
        Ok(()) => g.run(budget),
        Err(e) => {
            println!("{name}: not started: {e}");
            Stop::Limit
        }
    };
    let instructions: u64 = g.cpu.icount;
    println!("{name}: {stop:x?} after {instructions} instructions on the running thread, rip {:#x}, {} threads", g.cpu.rip, g.threads.len());
    let names = ["rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"];
    let registers: Vec<String> = names.iter().zip(g.cpu.r).map(|(n, v)| format!("{n} {v:#x}")).collect();
    println!("    {}", registers.join(", "));
    for m in &g.modules {
        println!("    module {:#x}..{:#x} {}", m.base, m.base + m.size, m.name);
    }
    let teb = g.thread().teb;
    let slots = g.mem.read_u64(teb + 0x58).unwrap_or(0);
    for (i, t) in g.win.tls_templates.iter().enumerate() {
        let block = g.mem.read_u64(slots + 8 * i as u64).unwrap_or(0);
        let index = g.mem.read_u32(t.index_address).unwrap_or(u32::MAX);
        println!("    static TLS {i}: {} + {} bytes of module {:#x}, its index {index}, this thread's block {block:#x}", t.len, t.zero_fill, t.module_base);
    }
    // XCL_PROBE_PEEK=address,address: the text at those addresses of the title's memory now
    for at in std::env::var("XCL_PROBE_PEEK").unwrap_or_default().split(',').filter_map(|a| u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()) {
        let bytes: Vec<u8> = (0..48).map_while(|i| g.mem.read_u8(at + i).ok()).collect();
        println!("    at {at:#x}: {:?} {bytes:02x?}", g.mem.cstr(at, 96).unwrap_or_default());
    }
    if !opcodes.is_empty() {
        let mut by_count: Vec<_> = opcodes.iter().collect();
        by_count.sort_by_key(|(_, (count, _))| *count);
        println!("{} different opcodes since {}, {} instructions; the rarest first (count, first seen at):", by_count.len(), after.unwrap_or_default(), by_count.iter().map(|(_, (c, _))| c).sum::<u64>());
        for (opcode, (count, first)) in by_count {
            println!("    {opcode:<22} {count:>9}  {first:#x}");
        }
        // where control went last: the instructions that did not follow their predecessor
        let jumps: Vec<String> = path.iter().zip(path.iter().skip(1)).filter(|(a, b)| **b < **a || **b > **a + 15).map(|(a, b)| format!("{a:#x}→{b:#x}")).collect();
        println!("the last jumps: {}", jumps[jumps.len().saturating_sub(40)..].join(" "));
        // XCL_PROBE_PATH=file: every jump since then, one a line
        if let Ok(file) = std::env::var("XCL_PROBE_PATH") {
            let _ = std::fs::write(file, jumps.join("\n"));
        }
    }
    println!("last system calls:");
    for line in &g.trace {
        // the text a function was given, when its first argument points at some
        let first = line.split_once('(').and_then(|(_, args)| u64::from_str_radix(args.split([',', ')']).next()?.trim_start_matches("0x"), 16).ok());
        let name = line.split_once('(').map_or("", |(name, _)| name);
        let text = first.filter(|&at| at > 0x1_0000).and_then(|at| if name.ends_with('W') { g.mem.wstr(at, 260).ok() } else { g.mem.cstr(at, 260).ok() });
        match text.filter(|t| t.len() > 1 && t.chars().all(|c| !c.is_control())) {
            Some(text) => println!("    {line}  \"{text}\""),
            None => println!("    {line}"),
        }
    }
    let mut called: Vec<_> = g.imports.iter().filter(|i| i.calls != 0).collect();
    called.sort_by_key(|i| std::cmp::Reverse(i.calls));
    println!("{} system functions called; the most:", called.len());
    for i in called.iter().take(12) {
        println!("    {:>9}  {}!{}", i.calls, i.dll, i.name);
    }
    println!("host notes ({}):", g.log.len());
    for line in g.log.iter().take(60) {
        println!("    {line}");
    }
    let output = String::from_utf8_lossy(&g.output);
    if !output.is_empty() {
        println!("the title wrote:\n{output}");
    }
}

/// The opcode of the instruction at `rip`, without its operands: prefixes that change what it
/// is (66 / F2 / F3, REX.W), the opcode map and byte, and for a group the /digit.
fn opcode_of(g: &Guest, rip: u64) -> String {
    let b: Vec<u8> = (0..15).map_while(|i| g.mem.read_u8(rip + i).ok()).collect();
    let mut i = 0;
    let mut prefix = String::new();
    while i < b.len() {
        match b[i] {
            0x66 | 0xF2 | 0xF3 => prefix.push_str(&format!("{:02x} ", b[i])),
            0x2E | 0x36 | 0x3E | 0x26 | 0x64 | 0x65 | 0x67 | 0xF0 => {}
            0x40..=0x4F => {
                if b[i] & 8 != 0 {
                    prefix.push_str("W ");
                }
            }
            _ => break,
        }
        i += 1;
    }
    let at = |k: usize| b.get(i + k).copied().unwrap_or(0);
    let digit = |k: usize| (at(k) >> 3) & 7;
    match at(0) {
        0xC5 => format!("vex.{} 0f {:02x}", ["", "66", "f3", "f2"][(at(1) & 3) as usize], at(2)),
        0xC4 => format!("vex.{}{} map{} {:02x}", ["", "66", "f3", "f2"][(at(2) & 3) as usize], if at(2) & 0x80 != 0 { " W" } else { "" }, at(1) & 0x1F, at(3)),
        0x0F if at(1) == 0x38 || at(1) == 0x3A => format!("{prefix}0f {:02x} {:02x}", at(1), at(2)),
        0x0F if matches!(at(1), 0x00 | 0x01 | 0x18 | 0x71..=0x73 | 0xAE | 0xBA | 0xC7) => format!("{prefix}0f {:02x} /{}", at(1), digit(2)),
        0x0F => format!("{prefix}0f {:02x}", at(1)),
        op @ (0x80..=0x83 | 0xC0 | 0xC1 | 0xD0..=0xD3 | 0xF6 | 0xF7 | 0xFE | 0xFF | 0xD8..=0xDF) => format!("{prefix}{op:02x} /{}", digit(1)),
        op => format!("{prefix}{op:02x}"),
    }
}
