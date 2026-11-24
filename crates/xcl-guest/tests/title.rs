//! XCL's test title on the guest. The title is a Windows executable that is not in the
//! repository: build it with `scripts/cross-testtitle.sh` (or point `XCL_GUEST_TITLE` at an
//! `xcl_testtitle.exe`). Without it the test does nothing and says so.

use std::path::PathBuf;

use xcl_guest::{Guest, Stop};

fn title() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("XCL_GUEST_TITLE") {
        return Some(PathBuf::from(p));
    }
    let built = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/x86_64-pc-windows-gnu/debug/xcl_testtitle.exe");
    built.exists().then_some(built)
}

fn add_folder(g: &mut Guest, dir: &std::path::Path, prefix: &str) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let name = format!("{prefix}{}", entry.file_name().to_string_lossy());
        if entry.path().is_dir() {
            add_folder(g, &entry.path(), &format!("{name}\\"));
        } else {
            g.vfs.add_game_file(&name, std::fs::read(entry.path()).unwrap());
        }
    }
}

/// The checks of the title the guest passes so far, in the title's order.
const PASSES: &[&str] = &[
    "argv: command line belongs to the game",
    "static TLS: template on main thread",
    "static TLS: every new thread gets its own initialised copy",
    "C++-style exceptions: panic is caught (RtlPcToFileHeader)",
    "kernelx: GetConsoleType / XSystemGetDeviceType",
    "kernelx: XMemAlloc alignment + zeroing",
    "kernelx: XMemVirtualAlloc – graphics memory type in UMA, others from Windows",
    "xmem: working-set statistics, physical pages mapped and freed",
    "xmem: physical pages mapped into reserved address space keep their contents",
    "kernelx: graphics memory lives in the GPU-visible 4 GiB-aligned region (UMA)",
    "user32: a top-level window made either way (CreateWindowExW / A) is the hidden 1920x1080 console screen",
    "GameInput: GameInputCreate reaches the system GameInput (timestamps advance)",
    "kernelx: TitleMemoryStatus",
    "kernelx: SetThreadName / GetThreadName",
    "kernelx: unimplemented import becomes a stub and the title keeps running",
    "modules: GetModuleHandle(NULL) is the game image",
    "modules: LoadLibrary/GetProcAddress on an HLE module",
    "game DLL: static import + DllMain + TLS + its own kernelx import",
    "files: console paths G:\\ and GAME: read from the game folder",
    "files: CRT fopen goes through the VFS too",
    "files: writes go to the overlay, game folder untouched",
    "files: delete hides the file without touching the original",
    "files: directory listing merges overlay and hides deleted files",
    "files: T:\\ title scratch storage",
    "files: T:\\ free space (the host volume behind the mount)",
];

#[test]
fn the_test_title_passes_its_checks_up_to_where_the_guest_gets() {
    let Some(path) = title() else {
        println!("SKIPPED the_test_title_passes_its_checks_up_to_where_the_guest_gets: no title (scripts/cross-testtitle.sh builds it)");
        return;
    };
    let data = std::fs::read(&path).unwrap();
    let mut g = Guest::new();
    g.strict = true;
    g.command_line = "xcl_testtitle.exe --hello".into();
    // the game folder: the repository's test data and the title's DLL, built next to the title
    let folder = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/testgame");
    add_folder(&mut g, &folder, "");
    g.vfs.add_game_file("testgame_mw.dll", std::fs::read(path.with_file_name("testgame_mw.dll")).expect("testgame_mw.dll next to the title"));
    let exe = g.map_image("xcl_testtitle.exe", &data).unwrap();
    g.start(exe, 1 << 20).unwrap();
    let stop = loop {
        match g.run(2_000_000_000) {
            // the title checks that a function nobody implements becomes a stub answering 0
            Stop::Unimplemented { name, .. } if name == "XclDefinitelyMissingFunction" => g.answer_import(0).unwrap(),
            stop => break stop,
        }
    };
    let output = String::from_utf8_lossy(&g.output).into_owned();
    println!("{output}--- stopped: {stop:x?} after {} instructions of the main thread", g.cpu.icount);
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.first(), Some(&"xcl_testtitle: start"));
    assert!(!output.contains("[FAIL]"), "a check failed");
    let passed: Vec<&str> = lines.iter().filter_map(|l| l.strip_prefix("[PASS] ")).collect();
    assert!(passed.len() >= PASSES.len(), "only {} checks passed", passed.len());
    assert_eq!(&passed[..PASSES.len()], PASSES);
    // it has not crashed: it either ran to its end or reached a function that is not there yet
    assert!(matches!(stop, Stop::Exit(_) | Stop::Unimplemented { .. }), "{stop:x?}");
    // the game folder is as it was: what the title changed is in the overlay
    assert_eq!(g.vfs.game_file("data\\hello.txt"), Some(&std::fs::read(folder.join("data/hello.txt")).unwrap()[..]));
    assert!(g.vfs.game_file("data\\delete_me.txt").is_some());
    assert!(g.vfs.overlay().any(|(p, _)| p.eq_ignore_ascii_case("G:\\data\\renamed.bin")));
}
