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
//! folder's images are read; with it the title also gets the folder's other files of up to 16 MB
//! (the guest's file system holds a file whole: the large archives are left out, and named).

use std::collections::BTreeMap;
use std::path::Path;

use xcl::pe::{ImportName, Layout, Pe};
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

/// Give the title the files of `dir` (and below) that are not images, as `prefix` + their name.
fn add_small_files(g: &mut Guest, dir: &Path, prefix: &str, left_out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
        let path = entry.path();
        if path.is_dir() {
            add_small_files(g, &path, &format!("{name}\\"), left_out);
        } else if !is_image(&name) && !name.starts_with('.') {
            match std::fs::metadata(&path).map(|m| m.len()) {
                Ok(size) if size <= WHOLE_FILE_LIMIT => {
                    if let Ok(bytes) = std::fs::read(&path) {
                        g.vfs.add_game_file(&name, bytes);
                    }
                }
                Ok(size) => left_out.push(format!("{name} ({} MB)", size >> 20)),
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
    g.trace_len = 24;
    g.command_line = name.clone();
    for (other, bytes) in images {
        g.vfs.add_game_file(other, bytes.clone());
    }
    let mut left_out = Vec::new();
    add_small_files(&mut g, folder, "", &mut left_out);
    println!("{} files of the folder are too large to give the title whole: {}", left_out.len(), left_out.join(", "));
    let started = g.map_image(name, data).and_then(|index| g.start(index, 1 << 20));
    let stop = match started {
        Ok(()) => g.run(budget),
        Err(e) => {
            println!("{name}: not started: {e}");
            Stop::Limit
        }
    };
    let instructions: u64 = g.cpu.icount;
    println!("{name}: {stop:x?} after {instructions} instructions on the running thread, rip {:#x}, {} threads", g.cpu.rip, g.threads.len());
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
