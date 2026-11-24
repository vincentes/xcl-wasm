//! XCL for the browser: the part of XCL that needs no Windows, as a WebAssembly module
//! (`wasm64-unknown-unknown`). No bindings generator: the page (`www/xcl.mjs`) calls these
//! exports directly.
//!
//! Calling convention: the page copies its input into memory it got from `xcl_alloc`, calls a
//! function, and reads the answer (UTF-8 text) from `xcl_out_ptr()` / `xcl_out_len()`. A function
//! returns 0 when the answer is its result and 1 when it is an error message.

use std::sync::Mutex;

static OUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());

fn answer(result: Result<String, String>) -> u32 {
    let (code, text) = match result {
        Ok(text) => (0, text),
        Err(text) => (1, text),
    };
    *OUT.lock().unwrap() = text.into_bytes();
    code
}

/// # Safety
/// `ptr` / `len`: a block from `xcl_alloc` the page has filled (or `len` 0).
unsafe fn bytes<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

/// `len` bytes for the page to fill; given back with `xcl_free`.
#[no_mangle]
pub extern "C" fn xcl_alloc(len: usize) -> *mut u8 {
    let mut block = vec![0u8; len.max(1)].into_boxed_slice();
    let ptr = block.as_mut_ptr();
    std::mem::forget(block);
    ptr
}

/// # Safety
/// `ptr` / `len`: exactly what `xcl_alloc(len)` returned, not freed before.
#[no_mangle]
pub unsafe extern "C" fn xcl_free(ptr: *mut u8, len: usize) {
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len.max(1))));
}

#[no_mangle]
pub extern "C" fn xcl_out_ptr() -> *const u8 {
    OUT.lock().unwrap().as_ptr()
}

#[no_mangle]
pub extern "C" fn xcl_out_len() -> usize {
    OUT.lock().unwrap().len()
}

/// `xcl --version`.
#[no_mangle]
pub extern "C" fn xcl_version() -> u32 {
    answer(Ok(format!("xcl {}", env!("CARGO_PKG_VERSION"))))
}

/// `xcl inspect <file> [--json]` of a PE file's bytes. Imports get the HLE module their DLL
/// routes to; whether XCL implements each function is known only to the Windows build.
///
/// # Safety
/// `name` (UTF-8) and `data`: blocks from `xcl_alloc`.
#[no_mangle]
pub unsafe extern "C" fn xcl_inspect(name: *const u8, name_len: usize, data: *const u8, data_len: usize, json: u32) -> u32 {
    let name = String::from_utf8_lossy(bytes(name, name_len)).into_owned();
    let data = bytes(data, data_len);
    answer(xcl::inspect::describe(&name, data, xcl::inspect::route_status).map(|out| {
        if json != 0 {
            serde_json::to_string_pretty(&out).unwrap_or_default()
        } else {
            xcl::inspect::render(&out, xcl::inspect::console_imports_of(data))
        }
    }))
}

/// `xcl disasm <file> [args]` of a file's bytes; `args`: the command's options, one per line
/// (`--offset`, `0x40`, `--all`, `--hex`).
///
/// # Safety
/// `name`, `data` and `args` (UTF-8): blocks from `xcl_alloc`.
#[no_mangle]
pub unsafe extern "C" fn xcl_disasm(name: *const u8, name_len: usize, data: *const u8, data_len: usize, args: *const u8, args_len: usize) -> u32 {
    let name = String::from_utf8_lossy(bytes(name, name_len)).into_owned();
    let args: Vec<String> = String::from_utf8_lossy(bytes(args, args_len)).lines().map(str::trim).filter(|a| !a.is_empty()).map(String::from).collect();
    if args.iter().any(|a| a == "--fssm") {
        return answer(Err("--fssm writes a file: not available in the browser".into()));
    }
    answer(xcl::disasm::render(&name, bytes(data, data_len), &args))
}

/// Run a title (a PE file's bytes) on the x86-64 guest: at most `max_instructions` (0: two
/// billion). The answer is what the title wrote to its standard output and error, then a line
/// starting with `--- ` that says why it stopped; functions the host does not implement answer 0
/// and are listed before that line.
///
/// # Safety
/// `name` (UTF-8), `data` and `command_line` (UTF-8; empty: the name): blocks from `xcl_alloc`.
#[no_mangle]
pub unsafe extern "C" fn xcl_run(name: *const u8, name_len: usize, data: *const u8, data_len: usize, command_line: *const u8, command_line_len: usize, max_instructions: u64) -> u32 {
    let name = String::from_utf8_lossy(bytes(name, name_len)).into_owned();
    let command_line = String::from_utf8_lossy(bytes(command_line, command_line_len)).into_owned();
    let mut g = xcl_guest::Guest::new();
    g.command_line = if command_line.is_empty() { name.clone() } else { command_line };
    let started = g.map_image(&name, bytes(data, data_len)).and_then(|exe| g.start(exe, 1 << 20));
    if let Err(e) = started {
        return answer(Err(e));
    }
    let stop = g.run(if max_instructions == 0 { 2_000_000_000 } else { max_instructions });
    let mut text = String::from_utf8_lossy(&g.output).into_owned();
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    for line in &g.log {
        text.push_str(line);
        text.push('\n');
    }
    let instructions: u64 = g.cpu.icount;
    text.push_str(&format!("--- stopped: {stop:x?} ({instructions} instructions on the last thread, {} threads)\n", g.threads.len()));
    answer(Ok(text))
}
