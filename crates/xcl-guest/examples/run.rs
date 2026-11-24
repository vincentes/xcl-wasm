//! Runs a title on the guest and says how far it got:
//!   cargo run -p xcl-guest --example run -- <title.exe> [--game <folder>] [--lenient] [--max N]
//! Strict by default: stops at the first function the host does not implement.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: run <title.exe> [--lenient] [--max N]");
        std::process::exit(2);
    };
    let max = args.iter().position(|a| a == "--max").and_then(|i| args.get(i + 1)).and_then(|n| n.parse().ok()).unwrap_or(2_000_000_000u64);
    let data = std::fs::read(path).expect("read the title");
    let mut g = xcl_guest::Guest::new();
    g.strict = !args.iter().any(|a| a == "--lenient");
    g.trace_len = 12;
    let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().into_owned();
    g.command_line = format!("{name} --hello");
    // --game <folder>: the game folder (its files are read into memory)
    if let Some(folder) = args.iter().position(|a| a == "--game").and_then(|i| args.get(i + 1)) {
        add_folder(&mut g, std::path::Path::new(folder), "");
    }
    let exe = g.map_image(&name, &data).expect("map the image");
    g.start(exe, 0x10_0000).expect("start");
    let stop = loop {
        match g.run(max) {
            // the title's own check that a missing function becomes a stub
            xcl_guest::Stop::Unimplemented { name, .. } if name == "XclDefinitelyMissingFunction" => g.answer_import(0).expect("answer"),
            stop => break stop,
        }
    };
    print!("{}", String::from_utf8_lossy(&g.output));
    eprintln!("--- {} instructions, {} imports bound, {} called", g.cpu.icount, g.imports.len(), g.imports.iter().filter(|i| i.calls != 0).count());
    for l in &g.log {
        eprintln!("{l}");
    }
    for t in &g.trace {
        eprintln!("  {t}");
    }
    eprintln!("--- stopped: {stop:x?} at rip {:#x}", g.cpu.rip);
}

fn add_folder(g: &mut xcl_guest::Guest, dir: &std::path::Path, prefix: &str) {
    for entry in std::fs::read_dir(dir).expect("read the game folder").flatten() {
        let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
        if entry.path().is_dir() {
            add_folder(g, &entry.path(), &format!("{name}\\"));
        } else {
            g.vfs.add_game_file(&name, std::fs::read(entry.path()).expect("read a game file"));
        }
    }
}
