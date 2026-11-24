//! More of the C runtime (ucrt), as the title's own C++ runtime (msvcp140.dll) and a retail
//! executable import it: the "C" locale, character classes, maths and strings.
//!
//! There is one locale, "C": its tables are made once in the title's memory.

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

// --- the locale -------------------------------------------------------------------------------

const UPPER: u16 = 0x1;
const LOWER: u16 = 0x2;
const DIGIT: u16 = 0x4;
const SPACE: u16 = 0x8;
const PUNCT: u16 = 0x10;
const CONTROL: u16 = 0x20;
const BLANK: u16 = 0x40;
const HEX: u16 = 0x80;
const ALPHA: u16 = 0x100;

/// The classes of character `c` in the "C" locale, as the runtime's table has them.
fn classes(c: u8) -> u16 {
    let mut f = 0;
    if c.is_ascii_uppercase() {
        f |= UPPER | ALPHA;
    }
    if c.is_ascii_lowercase() {
        f |= LOWER | ALPHA;
    }
    if c.is_ascii_digit() {
        f |= DIGIT;
    }
    if c.is_ascii_hexdigit() {
        f |= HEX;
    }
    if c == b' ' || (9..=13).contains(&c) {
        f |= SPACE;
    }
    if c.is_ascii_punctuation() {
        f |= PUNCT;
    }
    if c.is_ascii_control() {
        f |= CONTROL;
    }
    if c == b' ' || c == b'\t' {
        f |= BLANK;
    }
    f
}

/// __pctype_func: the class table, 256 words that may also be indexed with -1 (EOF).
fn __pctype_func(g: &mut Guest) -> H {
    g.static_block("pctype", |g| {
        let at = g.heap_alloc(2 * 257)?;
        for c in 0..=255u8 {
            g.mem.write_u16(at + 2 + 2 * c as u64, classes(c))?;
        }
        Ok(at + 2)
    })
}

/// ___lc_locale_name_func: the names of the six categories, none in the "C" locale.
fn ___lc_locale_name_func(g: &mut Guest) -> H {
    g.static_block("locale names", |g| g.heap_alloc(8 * 6))
}

/// setlocale(category, name): there is the "C" locale, which is also what a query answers.
fn setlocale(g: &mut Guest) -> H {
    let name = g.arg(1)?;
    if name != 0 {
        let wanted = g.mem.cstr(name, 64)?;
        if !wanted.is_empty() && wanted != "C" {
            return Ok(0);
        }
    }
    g.static_block("locale C", |g| g.alloc_cstr("C"))
}

/// localeconv: struct lconv of the "C" locale – ten strings, eight small numbers (none:
/// CHAR_MAX), and the wide forms of the strings.
fn localeconv(g: &mut Guest) -> H {
    g.static_block("lconv", |g| {
        let at = g.heap_alloc(152)?;
        let (point, empty) = (g.alloc_cstr(".")?, g.alloc_cstr("")?);
        let (wide_point, wide_empty) = (g.alloc_wstr(".")?, g.alloc_wstr("")?);
        for i in 0..10 {
            g.mem.write_u64(at + 8 * i, if i == 0 { point } else { empty })?;
        }
        g.mem.write(at + 80, &[127u8; 8])?;
        for i in 0..8 {
            g.mem.write_u64(at + 88 + 8 * i, if i == 0 { wide_point } else { wide_empty })?;
        }
        Ok(at)
    })
}

/// _get_stream_buffer_pointers(stream, base **, pointer **, count **): where a FILE keeps its
/// buffer. The host's streams have none: the three are words of a block that stays zero.
fn _get_stream_buffer_pointers(g: &mut Guest) -> H {
    let outs = [g.arg(1)?, g.arg(2)?, g.arg(3)?];
    let block = g.static_block("stream buffer", |g| g.heap_alloc(32))?;
    for (i, out) in outs.into_iter().enumerate() {
        if out != 0 {
            g.mem.write_u64(out, block + 8 * i as u64)?;
        }
    }
    Ok(0)
}

fn _errno(g: &mut Guest) -> H {
    g.static_block("errno", |g| g.heap_alloc(8))
}

/// _get_narrow_winmain_command_line: the command line after the program's name.
fn _get_narrow_winmain_command_line(g: &mut Guest) -> H {
    g.static_block("WinMain command line", |g| {
        let line = g.command_line.clone();
        let rest = line.split_once(' ').map_or("", |(_, rest)| rest.trim_start());
        g.alloc_cstr(rest)
    })
}

/// _time64(seconds out): the guest clock, counted from 2026-01-01.
fn _time64(g: &mut Guest) -> H {
    let out = g.arg(0)?;
    let seconds = 1_767_225_600 + g.now_ns() / 1_000_000_000;
    if out != 0 {
        g.mem.write_u64(out, seconds)?;
    }
    Ok(seconds)
}

/// _aligned_malloc(size, alignment): the heap's blocks are aligned to 16; more gets pages.
fn _aligned_malloc(g: &mut Guest) -> H {
    let (size, alignment) = (g.arg(0)?, g.arg(1)?);
    if alignment <= 16 {
        return Ok(g.win.heap.alloc(&mut g.mem, size).unwrap_or(0));
    }
    Ok(g.mem.alloc(size.max(1), xcl_emu::mem::PROT_R | xcl_emu::mem::PROT_W).unwrap_or(0))
}

fn _aligned_free(g: &mut Guest) -> H {
    let at = g.arg(0)?;
    if at != 0 && !g.win.heap.free(&mut g.mem, at) {
        g.mem.unmap(at);
    }
    Ok(0)
}

fn terminate(g: &mut Guest) -> H {
    g.log.push("terminate()".into());
    Err(g.exit(3))
}

// --- characters and strings ---------------------------------------------------------------------

/// The is… functions: whether the character (0..255, or EOF) has one of `mask`'s classes.
fn is_class(g: &mut Guest, mask: u16) -> H {
    let c = g.arg(0)? as u32;
    Ok((c < 256 && classes(c as u8) & mask != 0) as u64)
}

macro_rules! class_functions {
    ($($name:ident, $wide:ident => $mask:expr;)*) => {
        $(
            fn $name(g: &mut Guest) -> H {
                is_class(g, $mask)
            }
        )*
        fn register_classes(map: &mut HashMap<&'static str, Handler>) {
            $(
                map.insert(stringify!($name), $name as Handler);
                // the wide ones answer for the same characters; no other has a class in "C"
                map.insert(stringify!($wide), $name as Handler);
            )*
        }
    };
}

class_functions! {
    isalnum, iswalnum => ALPHA | DIGIT;
    isalpha, iswalpha => ALPHA;
    isdigit, iswdigit => DIGIT;
    isxdigit, iswxdigit => HEX;
    isspace, iswspace => SPACE;
    isupper, iswupper => UPPER;
    islower, iswlower => LOWER;
    ispunct, iswpunct => PUNCT;
}

fn tolower(g: &mut Guest) -> H {
    let c = g.arg(0)? as u32;
    Ok(if c < 128 { (c as u8).to_ascii_lowercase() as u64 } else { c as u64 })
}

fn toupper(g: &mut Guest) -> H {
    let c = g.arg(0)? as u32;
    Ok(if c < 128 { (c as u8).to_ascii_uppercase() as u64 } else { c as u64 })
}

/// strcmp and its relatives: up to `limit` characters, with or without case.
fn compare(g: &mut Guest, limit: u64, fold: bool) -> H {
    let (a, b) = (g.arg(0)?, g.arg(1)?);
    for i in 0..limit {
        let (mut x, mut y) = (g.mem.read_u8(a + i)?, g.mem.read_u8(b + i)?);
        if fold {
            (x, y) = (x.to_ascii_lowercase(), y.to_ascii_lowercase());
        }
        if x != y {
            return Ok(if x < y { -1i32 } else { 1 } as u32 as u64);
        }
        if x == 0 {
            break;
        }
    }
    Ok(0)
}

fn strcmp(g: &mut Guest) -> H {
    compare(g, u64::MAX, false)
}

fn _stricmp(g: &mut Guest) -> H {
    compare(g, u64::MAX, true)
}

fn _strnicmp(g: &mut Guest) -> H {
    let n = g.arg(2)?;
    compare(g, n, true)
}

/// strcspn / strspn(text, set): the length of the start of `text` without (or of only)
/// characters of `set`.
fn span(g: &mut Guest, of_set: bool) -> H {
    let (text, set) = (g.arg(0)?, g.arg(1)?);
    let set = g.mem.cstr(set, 256)?;
    let mut n = 0;
    loop {
        let c = g.mem.read_u8(text + n)?;
        if c == 0 || set.as_bytes().contains(&c) != of_set {
            return Ok(n);
        }
        n += 1;
    }
}

fn strcspn(g: &mut Guest) -> H {
    span(g, false)
}

fn strspn(g: &mut Guest) -> H {
    span(g, true)
}

/// strtok_s(text or NULL, delimiters, context): the next token, ended in place; the context
/// keeps where the scan goes on.
fn strtok_s(g: &mut Guest) -> H {
    let (text, delimiters, context) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    let delimiters = g.mem.cstr(delimiters, 256)?;
    let is = |c: u8| delimiters.as_bytes().contains(&c);
    let mut at = if text != 0 { text } else { g.mem.read_u64(context)? };
    if at == 0 {
        return Ok(0);
    }
    while is(g.mem.read_u8(at)?) {
        at += 1;
    }
    if g.mem.read_u8(at)? == 0 {
        g.mem.write_u64(context, at)?;
        return Ok(0);
    }
    let token = at;
    loop {
        let c = g.mem.read_u8(at)?;
        if c == 0 {
            break;
        }
        if is(c) {
            g.mem.write_u8(at, 0)?;
            at += 1;
            break;
        }
        at += 1;
    }
    g.mem.write_u64(context, at)?;
    Ok(token)
}

/// rand_s(value out): the process's generator (the same sequence every run).
fn rand_s(g: &mut Guest) -> H {
    let out = g.arg(0)?;
    if out == 0 {
        return Ok(EINVAL);
    }
    let mut x = g.win.prng ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    g.win.prng = x;
    g.mem.write_u32(out, (x >> 32) as u32)?;
    Ok(0)
}

const EINVAL: u64 = 22;
const ERANGE: u64 = 34;

/// wcscpy_s / wcscat_s(destination, its size in characters, source): 0, or an error with the
/// destination made empty.
fn wide_copy(g: &mut Guest, append: bool) -> H {
    let (to, size, from) = (g.arg(0)?, g.arg(1)?, g.arg(2)?);
    if to == 0 || size == 0 {
        return Ok(EINVAL);
    }
    if from == 0 {
        g.mem.write_u16(to, 0)?;
        return Ok(EINVAL);
    }
    let mut at = 0;
    while append && at < size && g.mem.read_u16(to + 2 * at)? != 0 {
        at += 1;
    }
    let mut i = 0;
    while at < size {
        let c = g.mem.read_u16(from + 2 * i)?;
        g.mem.write_u16(to + 2 * at, c)?;
        if c == 0 {
            return Ok(0);
        }
        at += 1;
        i += 1;
    }
    g.mem.write_u16(to, 0)?;
    Ok(ERANGE)
}

fn wcscpy_s(g: &mut Guest) -> H {
    wide_copy(g, false)
}

fn wcscat_s(g: &mut Guest) -> H {
    wide_copy(g, true)
}

// --- maths --------------------------------------------------------------------------------------
// Arguments come in XMM0 (and XMM1), the result goes to XMM0.

fn f32_arg(g: &Guest, i: usize) -> f32 {
    f32::from_bits(g.cpu.v[i][0] as u32)
}

fn f64_arg(g: &Guest, i: usize) -> f64 {
    f64::from_bits(g.cpu.v[i][0])
}

fn f32_result(g: &mut Guest, x: f32) -> H {
    g.cpu.v[0] = [x.to_bits() as u64, 0, 0, 0];
    Ok(0)
}

fn f64_result(g: &mut Guest, x: f64) -> H {
    g.cpu.v[0] = [x.to_bits(), 0, 0, 0];
    Ok(0)
}

macro_rules! maths {
    ($($name:ident: $kind:ident $f:expr;)*) => {
        $(maths!(@one $name $kind $f);)*
        fn register_maths(map: &mut HashMap<&'static str, Handler>) {
            $(map.insert(stringify!($name), $name as Handler);)*
        }
    };
    (@one $name:ident single $f:expr) => {
        fn $name(g: &mut Guest) -> H {
            let f: fn(f32) -> f32 = $f;
            f32_result(g, f(f32_arg(g, 0)))
        }
    };
    (@one $name:ident single2 $f:expr) => {
        fn $name(g: &mut Guest) -> H {
            let f: fn(f32, f32) -> f32 = $f;
            f32_result(g, f(f32_arg(g, 0), f32_arg(g, 1)))
        }
    };
    (@one $name:ident double $f:expr) => {
        fn $name(g: &mut Guest) -> H {
            let f: fn(f64) -> f64 = $f;
            f64_result(g, f(f64_arg(g, 0)))
        }
    };
    (@one $name:ident double2 $f:expr) => {
        fn $name(g: &mut Guest) -> H {
            let f: fn(f64, f64) -> f64 = $f;
            f64_result(g, f(f64_arg(g, 0), f64_arg(g, 1)))
        }
    };
}

maths! {
    sinf: single f32::sin; cosf: single f32::cos; tanf: single f32::tan;
    asinf: single f32::asin; acosf: single f32::acos; atanf: single f32::atan;
    expf: single f32::exp; exp2f: single f32::exp2; logf: single f32::ln; log2f: single f32::log2; log10f: single f32::log10;
    sqrtf: single f32::sqrt; floorf: single f32::floor; ceilf: single f32::ceil;
    atan2f: single2 f32::atan2; powf: single2 f32::powf; fmodf: single2 |x, y| x % y;
    sin: double f64::sin; cos: double f64::cos; tan: double f64::tan;
    asin: double f64::asin; acos: double f64::acos; atan: double f64::atan;
    sinh: double f64::sinh; cosh: double f64::cosh; tanh: double f64::tanh;
    exp: double f64::exp; expm1: double f64::exp_m1; log: double f64::ln; log1p: double f64::ln_1p; log2: double f64::log2; log10: double f64::log10;
    sqrt: double f64::sqrt; floor: double f64::floor; ceil: double f64::ceil;
    atan2: double2 f64::atan2; pow: double2 f64::powf; fmod: double2 |x, y| x % y;
}

/// ldexp(x, exponent): x · 2^exponent (the exponent is the second argument, an integer).
fn ldexp(g: &mut Guest) -> H {
    let (x, e) = (f64_arg(g, 0), g.arg(1)? as i32);
    f64_result(g, x * 2f64.powi(e))
}

fn ok(_: &mut Guest) -> H {
    Ok(0)
}

fn one(_: &mut Guest) -> H {
    Ok(1)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; __pctype_func, ___lc_locale_name_func, setlocale, localeconv, _get_stream_buffer_pointers, _errno);
    register!(map; _get_narrow_winmain_command_line, _time64, _aligned_malloc, _aligned_free, terminate);
    register!(map; strcspn, strspn, strtok_s, rand_s);
    register!(map; tolower, toupper, strcmp, _stricmp, _strnicmp, wcscpy_s, wcscat_s, ldexp);
    register_classes(map);
    register_maths(map);
    // the "C" locale: code page 0 (none), one byte per character; nothing to lock
    for name in ["___lc_codepage_func", "___lc_collate_cp_func", "_lock_locales", "_unlock_locales", "_callnewh", "_set_new_handler", "_lock_file", "_unlock_file"] {
        map.insert(name, ok as Handler);
    }
    map.insert("___mb_cur_max_func", one as Handler);
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX};
    use xcl_emu::mem::{PROT_R, PROT_W};

    use super::*;

    fn guest() -> (Guest, u64) {
        let mut g = Guest::new();
        let stack = g.mem.alloc(0x4000, PROT_R | PROT_W).unwrap();
        g.add_thread(stack, stack + 0x4000, 0, 0, crate::SLOT_THREAD_RETURN).unwrap();
        g.cpu = g.threads[0].cpu.clone();
        let page = g.mem.alloc(0x1000, PROT_R | PROT_W).unwrap();
        (g, page)
    }

    fn call(g: &mut Guest, f: Handler, args: [u64; 3]) -> H {
        for (register, value) in [RCX, RDX, 8].into_iter().zip(args) {
            g.cpu.r[register] = value;
        }
        f(g)
    }

    fn wide(text: &str) -> Vec<u8> {
        text.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect()
    }

    #[test]
    fn the_class_table_is_the_c_locales_and_takes_eof() {
        let (mut g, _) = guest();
        let table = call(&mut g, __pctype_func, [0; 3]).unwrap();
        let class = |g: &Guest, c: i64| g.mem.read_u16((table as i64 + 2 * c) as u64).unwrap();
        assert_eq!(class(&g, -1), 0);
        // the values the runtime's own table has
        assert_eq!((class(&g, 'A' as i64), class(&g, 'a' as i64), class(&g, 'G' as i64)), (0x181, 0x182, 0x101));
        assert_eq!((class(&g, '0' as i64), class(&g, ' ' as i64), class(&g, '\t' as i64), class(&g, '\n' as i64)), (0x84, 0x48, 0x68, 0x28));
        assert_eq!((class(&g, '!' as i64), class(&g, 0x7F), class(&g, 0xE9)), (0x10, 0x20, 0));
        assert_eq!(call(&mut g, __pctype_func, [0; 3]), Ok(table));
        assert_eq!(call(&mut g, isxdigit, ['f' as u64, 0, 0]), Ok(1));
        assert_eq!(call(&mut g, isxdigit, ['g' as u64, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, isspace, [u32::MAX as u64, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, toupper, ['q' as u64, 0, 0]), Ok('Q' as u64));
        assert_eq!(call(&mut g, tolower, [0xC9, 0, 0]), Ok(0xC9));
    }

    #[test]
    fn there_is_the_c_locale_and_no_other() {
        let (mut g, page) = guest();
        let c = call(&mut g, setlocale, [0, 0, 0]).unwrap();
        assert_eq!(g.mem.cstr(c, 8).unwrap(), "C");
        g.mem.write(page, b"de-DE\0").unwrap();
        assert_eq!(call(&mut g, setlocale, [0, page, 0]), Ok(0));
        g.mem.write(page, b"C\0").unwrap();
        assert_eq!(call(&mut g, setlocale, [0, page, 0]), Ok(c));
        let lconv = call(&mut g, localeconv, [0; 3]).unwrap();
        let (point, separator) = (g.mem.read_u64(lconv).unwrap(), g.mem.read_u64(lconv + 8).unwrap());
        assert_eq!((g.mem.cstr(point, 8).unwrap().as_str(), g.mem.cstr(separator, 8).unwrap().as_str()), (".", ""));
        assert_eq!(g.mem.read_u8(lconv + 80).unwrap(), 127);
        let wide_point = g.mem.read_u64(lconv + 88).unwrap();
        assert_eq!(g.mem.wstr(wide_point, 8).unwrap(), ".");
        let names = call(&mut g, ___lc_locale_name_func, [0; 3]).unwrap();
        assert_eq!(g.mem.read_u64(names + 8).unwrap(), 0);
    }

    #[test]
    fn strings_compare_and_wide_copies_keep_to_their_size() {
        let (mut g, page) = guest();
        let (a, b, to) = (page, page + 0x100, page + 0x200);
        g.mem.write(a, b"Level\0").unwrap();
        g.mem.write(b, b"levels\0").unwrap();
        assert_eq!(call(&mut g, strcmp, [a, b, 0]), Ok(u32::MAX as u64));
        assert_eq!(call(&mut g, strcmp, [a, a, 0]), Ok(0));
        assert_eq!(call(&mut g, _stricmp, [b, a, 0]), Ok(1));
        assert_eq!(call(&mut g, _strnicmp, [a, b, 5]), Ok(0));
        g.mem.write(a, &wide("G:\\game")).unwrap();
        g.mem.write(b, &wide(".dll")).unwrap();
        assert_eq!(call(&mut g, wcscpy_s, [to, 16, a]), Ok(0));
        assert_eq!(call(&mut g, wcscat_s, [to, 16, b]), Ok(0));
        assert_eq!(g.mem.wstr(to, 32).unwrap(), "G:\\game.dll");
        // no room for the terminator: an error, and an empty destination
        assert_eq!(call(&mut g, wcscat_s, [to, 12, b]), Ok(ERANGE));
        assert_eq!(g.mem.read_u16(to).unwrap(), 0);
        assert_eq!(call(&mut g, wcscpy_s, [0, 16, a]), Ok(EINVAL));
    }

    #[test]
    fn a_string_is_scanned_and_split_at_its_delimiters() {
        let (mut g, page) = guest();
        let (text, set, context) = (page, page + 0x100, page + 0x200);
        g.mem.write(text, b"packroot:/levels.rpf\0").unwrap();
        g.mem.write(set, b":/\0").unwrap();
        assert_eq!(call(&mut g, strcspn, [text, set, 0]), Ok(8));
        assert_eq!(call(&mut g, strspn, [text + 8, set, 0]), Ok(2));
        assert_eq!(call(&mut g, strcspn, [text + 10, set, 0]), Ok(10));
        let first = call(&mut g, strtok_s, [text, set, context]).unwrap();
        assert_eq!(g.mem.cstr(first, 32).unwrap(), "packroot");
        let second = call(&mut g, strtok_s, [0, set, context]).unwrap();
        assert_eq!((second, g.mem.cstr(second, 32).unwrap().as_str()), (text + 10, "levels.rpf"));
        assert_eq!(call(&mut g, strtok_s, [0, set, context]), Ok(0));
        assert_eq!(call(&mut g, rand_s, [page + 0x300, 0, 0]), Ok(0));
        assert_eq!(call(&mut g, rand_s, [0, 0, 0]), Ok(EINVAL));
    }

    #[test]
    fn maths_take_and_return_their_values_in_xmm() {
        let (mut g, _) = guest();
        g.cpu.v[0] = [2.0f32.to_bits() as u64, 7, 7, 7];
        g.cpu.v[1][0] = 10.0f32.to_bits() as u64;
        powf(&mut g).unwrap();
        assert_eq!(g.cpu.v[0], [1024.0f32.to_bits() as u64, 0, 0, 0]);
        g.cpu.v[0][0] = 0.25f64.to_bits();
        sqrt(&mut g).unwrap();
        assert_eq!(f64::from_bits(g.cpu.v[0][0]), 0.5);
        g.cpu.v[0][0] = 7.5f64.to_bits();
        g.cpu.v[1][0] = 2.0f64.to_bits();
        fmod(&mut g).unwrap();
        assert_eq!(f64::from_bits(g.cpu.v[0][0]), 1.5);
        g.cpu.v[0][0] = 3.0f64.to_bits();
        assert_eq!(call(&mut g, ldexp, [0, 4, 0]), Ok(0));
        assert_eq!(f64::from_bits(g.cpu.v[0][0]), 48.0);
    }

    #[test]
    fn the_winmain_command_line_is_what_follows_the_program() {
        let (mut g, page) = guest();
        g.command_line = "title.exe  -level 3".into();
        let at = call(&mut g, _get_narrow_winmain_command_line, [0; 3]).unwrap();
        assert_eq!(g.mem.cstr(at, 64).unwrap(), "-level 3");
        assert_eq!(call(&mut g, _time64, [page, 0, 0]), Ok(1_767_225_600));
        assert_eq!(g.mem.read_u64(page).unwrap(), 1_767_225_600);
        let big = call(&mut g, _aligned_malloc, [100, 64, 0]).unwrap();
        assert_eq!(big % 64, 0);
        call(&mut g, _aligned_free, [big, 0, 0]).unwrap();
        assert!(g.mem.read_u8(big).is_err());
    }
}
