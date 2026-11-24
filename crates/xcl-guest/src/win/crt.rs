//! The C runtime (ucrt) functions a title's startup code and libraries import.

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::register;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

fn ok(_: &mut Guest) -> H {
    Ok(0)
}

/// A 16-byte zeroed block, once per name: the CRT's globals behind `__p_*`.
fn global(g: &mut Guest, name: &'static str) -> H {
    g.static_block(name, |g| g.heap_alloc(16))
}

fn __p___argc(g: &mut Guest) -> H {
    g.static_block("argc", |g| {
        let at = g.heap_alloc(16)?;
        g.mem.write_u32(at, 1)?;
        Ok(at)
    })
}

fn __p___argv(g: &mut Guest) -> H {
    g.static_block("argv", |g| {
        // char **argv = { "title.exe", NULL }, and the pointer to it
        let name = g.alloc_cstr("title.exe")?;
        let at = g.heap_alloc(32)?;
        g.mem.write_u64(at + 8, name)?;
        g.mem.write_u64(at, at + 8)?;
        Ok(at)
    })
}

fn __p__environ(g: &mut Guest) -> H {
    g.static_block("environ", |g| {
        let at = g.heap_alloc(32)?;
        g.mem.write_u64(at, at + 8)?;
        Ok(at)
    })
}

fn __p__commode(g: &mut Guest) -> H {
    global(g, "commode")
}

fn __p__fmode(g: &mut Guest) -> H {
    global(g, "fmode")
}

/// Call the functions of a table of pointers; `stop_on_error`: until one returns non-zero.
fn initterm(g: &mut Guest, stop_on_error: bool) -> H {
    let (mut at, end) = (g.arg(0)?, g.arg(1)?);
    while at < end {
        let f = g.mem.read_u64(at)?;
        if f != 0 {
            let r = g.call(f, &[])? as u32;
            if stop_on_error && r != 0 {
                return Ok(r as u64);
            }
        }
        at += 8;
    }
    Ok(0)
}

fn _initterm(g: &mut Guest) -> H {
    initterm(g, false)
}

fn _initterm_e(g: &mut Guest) -> H {
    initterm(g, true)
}

fn _crt_atexit(g: &mut Guest) -> H {
    let f = g.arg(0)?;
    g.win.atexit.push(f);
    Ok(0)
}

/// _initialize_onexit_table(table): a DLL's own atexit list (three pointers, kept empty: the
/// functions are held by the host).
fn _initialize_onexit_table(g: &mut Guest) -> H {
    let table = g.arg(0)?;
    if table == 0 {
        return Ok(u32::MAX as u64);
    }
    g.mem.write(table, &[0u8; 24])?;
    Ok(0)
}

/// _register_onexit_function(table, function)
fn _register_onexit_function(g: &mut Guest) -> H {
    let f = g.arg(1)?;
    g.win.atexit.push(f);
    Ok(0)
}

/// Run the `atexit` functions, last registered first.
fn run_atexit(g: &mut Guest) -> Result<(), Stop> {
    while let Some(f) = g.win.atexit.pop() {
        g.call(f, &[])?;
    }
    Ok(())
}

fn _cexit(g: &mut Guest) -> H {
    run_atexit(g)?;
    Ok(0)
}

fn exit(g: &mut Guest) -> H {
    let code = g.arg(0)? as u32;
    run_atexit(g)?;
    Err(g.exit(code))
}

fn _exit(g: &mut Guest) -> H {
    let code = g.arg(0)? as u32;
    Err(g.exit(code))
}

fn abort(g: &mut Guest) -> H {
    g.log.push("abort()".into());
    Err(g.exit(3))
}

fn malloc(g: &mut Guest) -> H {
    let size = g.arg(0)?;
    g.heap_alloc(size)
}

fn calloc(g: &mut Guest) -> H {
    let (n, size) = (g.arg(0)?, g.arg(1)?);
    match n.checked_mul(size) {
        Some(total) => g.heap_alloc(total),
        None => Ok(0),
    }
}

fn realloc(g: &mut Guest) -> H {
    let (at, size) = (g.arg(0)?, g.arg(1)?);
    Ok(g.win.heap.realloc(&mut g.mem, at, size).unwrap_or(0))
}

fn free(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    if at != 0 {
        g.win.heap.free(&mut g.mem, at);
    }
    Ok(0)
}

fn strlen(g: &mut Guest) -> H {
    let s = g.arg(0)?;
    let mut n = 0;
    while g.mem.read_u8(s + n)? != 0 {
        n += 1;
    }
    Ok(n)
}

fn wcslen(g: &mut Guest) -> H {
    let s = g.arg(0)?;
    let mut n = 0;
    while g.mem.read_u16(s + 2 * n)? != 0 {
        n += 1;
    }
    Ok(n)
}

fn strncmp(g: &mut Guest) -> H {
    let (a, b, n) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    for i in 0..n {
        let (x, y) = (g.mem.read_u8(a + i)?, g.mem.read_u8(b + i)?);
        if x != y {
            return Ok((x as i32 - y as i32) as u32 as u64);
        }
        if x == 0 {
            break;
        }
    }
    Ok(0)
}

/// `FILE *` of stdin / stdout / stderr: the standard handle values stand in for them.
fn __acrt_iob_func(g: &mut Guest) -> H {
    Ok(match g.arg(0)? {
        0 => super::STDIN,
        1 => super::STDOUT,
        _ => super::STDERR,
    })
}

/// printf to a `FILE *`: the format string is written as it is (its arguments are not
/// formatted) – enough for the CRT's own startup error messages to be seen.
fn __stdio_common_vfprintf(g: &mut Guest) -> H {
    let (file, format) = (g.arg(1)?, g.arg(2)?);
    let text = g.mem.cstr(format, 4096)?;
    if file == super::STDOUT || file == super::STDERR {
        g.write_std(text.as_bytes());
    }
    Ok(text.len() as u64)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; _initialize_onexit_table, _register_onexit_function);
    register!(map; __p___argc, __p___argv, __p__environ, __p__commode, __p__fmode, _initterm, _initterm_e, _crt_atexit, _cexit, exit, _exit, abort);
    register!(map; malloc, calloc, realloc, free, strlen, wcslen, strncmp, __acrt_iob_func, __stdio_common_vfprintf);
    for name in [
        "_configure_narrow_argv",
        "_initialize_narrow_environment",
        "_set_app_type",
        "__setusermatherr",
        "_set_new_mode",
        "_configthreadlocale",
        "_set_invalid_parameter_handler",
        "_execute_onexit_table",
        "fflush",
        "setvbuf",
    ] {
        map.insert(name, ok as Handler);
    }
}
