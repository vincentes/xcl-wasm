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

/// The checks of the title that cannot pass yet, by the start of their names, and why. Every
/// other check has to pass.
const FAILS: &[&str] = &[
    // they compile HLSL at run time (D3DCompile), render and read pixels back: no GPU back end
    "D3D12.X: shaders captured",
    "D3D12.X: retail-format console shader",
    "D3D12.X: triangle rendered",
    "D3D12.X: dynamic depth bias",
    // libraries the guest does not have
    "MFPlat",
    "XAudio2",
];

/// Checks of areas that are being worked on: they may pass or fail.
const EITHER: &[&str] = &["DirectStorage"];

#[test]
fn the_test_title_passes_its_checks_up_to_where_the_guest_gets() {
    let Some(path) = title() else {
        println!("SKIPPED the_test_title_passes_its_checks_up_to_where_the_guest_gets: no title (scripts/cross-testtitle.sh builds it)");
        return;
    };
    let data = std::fs::read(&path).unwrap();
    let mut g = Guest::new();
    g.command_line = "xcl_testtitle.exe --hello".into();
    // the game folder: the repository's test data and the title's DLL, built next to the title
    let folder = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/testgame");
    add_folder(&mut g, &folder, "");
    assert!(g.vfs.game_file("MicrosoftGame.config").is_some());
    g.vfs.add_game_file("testgame_mw.dll", std::fs::read(path.with_file_name("testgame_mw.dll")).expect("testgame_mw.dll next to the title"));
    let exe = g.map_image("xcl_testtitle.exe", &data).unwrap();
    g.start(exe, 1 << 20).unwrap();
    // functions the guest lacks answer 0: the checks that need them fail, the title goes on
    let stop = g.run(4_000_000_000);
    let output = String::from_utf8_lossy(&g.output).into_owned();
    println!("{output}--- stopped: {stop:x?} after {} instructions of the main thread", g.cpu.icount);
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.first(), Some(&"xcl_testtitle: start"));
    let checks: Vec<&str> = lines.iter().skip(1).copied().filter(|l| l.starts_with("[PASS] ") || l.starts_with("[FAIL] ")).collect();
    assert_eq!(checks.len(), 50, "the title has 50 checks");
    let mut failed = 0;
    for line in &checks {
        let name = &line[7..];
        let fails = line.starts_with("[FAIL] ");
        failed += fails as u32;
        if !EITHER.iter().any(|e| name.starts_with(e)) {
            assert_eq!(fails, FAILS.iter().any(|f| name.starts_with(f)), "{line}");
        }
    }
    // it runs to its end and exits with the number of checks that failed
    assert_eq!(lines.last(), Some(&format!("xcl_testtitle: done, failures: {failed}").as_str()));
    assert_eq!(stop, Stop::Exit(failed));
    assert!(failed <= 7, "{failed} checks failed");
    // the game folder is as it was: what the title changed is in the overlay
    assert_eq!(g.vfs.game_file("data\\hello.txt"), Some(&std::fs::read(folder.join("data/hello.txt")).unwrap()[..]));
    assert!(g.vfs.game_file("data\\delete_me.txt").is_some());
    assert!(g.vfs.overlay().any(|(p, _)| p.eq_ignore_ascii_case("G:\\data\\renamed.bin")));
}
