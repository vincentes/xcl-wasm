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
    let exe = g.map_image("xcl_testtitle.exe", &data).unwrap();
    g.start(exe, 1 << 20).unwrap();
    let stop = g.run(2_000_000_000);
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
}
