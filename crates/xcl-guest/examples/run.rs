//! Runs a title on the guest and says how far it got:
//!   cargo run -p xcl-guest --example run -- <title.exe> [--lenient] [--max N]
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
    let exe = g.map_image(&name, &data).expect("map the image");
    g.start(exe, 0x10_0000).expect("start");
    let stop = g.run(max);
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
