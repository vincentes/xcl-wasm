//! The C runtime's formatted output: the `__stdio_common_v*printf*` functions that every
//! printf of a title built with MSVC ends in.

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::*;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

/// A truncated result is its full length (C99 snprintf), not -1.
const STANDARD_SNPRINTF_BEHAVIOR: u64 = 2;

/// Pad `body` to `width`: on the right with spaces (`left`), else on the left with spaces or,
/// for a number (`zero`), with zeros after its sign.
fn pad(body: String, width: usize, left: bool, zero: bool) -> String {
    let len = body.chars().count();
    if len >= width {
        return body;
    }
    let fill = width - len;
    if left {
        format!("{body}{}", " ".repeat(fill))
    } else if zero {
        let sign = body.chars().next().filter(|c| "+- ".contains(*c)).map_or(0, char::len_utf8);
        format!("{}{}{}", &body[..sign], "0".repeat(fill), &body[sign..])
    } else {
        format!("{}{body}", " ".repeat(fill))
    }
}

/// `%g`: the shorter of `%e` and `%f` at this precision, without trailing zeros.
fn general(x: f64, precision: usize, upper: bool) -> String {
    let p = precision.max(1);
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let exponent = x.abs().log10().floor() as i32;
    let trim = |s: String| if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    if exponent < -4 || exponent >= p as i32 {
        let s = format!("{:.*e}", p - 1, x);
        let (mantissa, e) = s.split_once('e').unwrap_or((&s, "0"));
        let e: i32 = e.parse().unwrap_or(0);
        let out = format!("{}e{}{:02}", trim(mantissa.to_string()), if e < 0 { '-' } else { '+' }, e.abs());
        if upper { out.to_uppercase() } else { out }
    } else {
        trim(format!("{:.*}", (p as i32 - 1 - exponent).max(0) as usize, x))
    }
}

/// Format `format` with the arguments at `args` (a va_list: 8 bytes each). `wide`: the format's
/// own strings (`%s`) are wide, as in the wprintf family.
pub(super) fn format(g: &Guest, format: &str, mut args: u64, wide: bool) -> Result<String, Stop> {
    let mut out = String::new();
    let mut next = |g: &Guest| -> Result<u64, Stop> {
        let v = g.mem.read_u64(args)?;
        args += 8;
        Ok(v)
    };
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let (mut left, mut zero, mut plus, mut space, mut alt) = (false, false, false, false, false);
        while let Some(&f) = chars.peek() {
            match f {
                '-' => left = true,
                '0' => zero = true,
                '+' => plus = true,
                ' ' => space = true,
                '#' => alt = true,
                _ => break,
            }
            chars.next();
        }
        let mut number = |chars: &mut std::iter::Peekable<std::str::Chars>, g: &Guest| -> Result<Option<usize>, Stop> {
            if chars.peek() == Some(&'*') {
                chars.next();
                return Ok(Some(next(g)? as i32 as usize));
            }
            let mut n = None;
            while let Some(d) = chars.peek().and_then(|c| c.to_digit(10)) {
                n = Some(n.unwrap_or(0) * 10 + d as usize);
                chars.next();
            }
            Ok(n)
        };
        let width = number(&mut chars, g)?.unwrap_or(0);
        let precision = if chars.peek() == Some(&'.') {
            chars.next();
            Some(number(&mut chars, g)?.unwrap_or(0))
        } else {
            None
        };
        // length: hh h l ll z t j I64 I32 I; w for a wide string
        let (mut long, mut size64, mut short_string) = (false, false, false);
        loop {
            match chars.peek() {
                Some('l') | Some('w') => {
                    size64 |= long;
                    long = true;
                }
                Some('h') => short_string = true,
                Some('z') | Some('t') | Some('j') => size64 = true,
                Some('I') => {
                    chars.next();
                    let digits: String = [chars.peek().copied(), chars.clone().nth(1)].into_iter().flatten().collect();
                    size64 = digits != "32";
                    if digits == "64" || digits == "32" {
                        chars.next();
                    } else {
                        continue;
                    }
                }
                _ => break,
            }
            chars.next();
        }
        let Some(conversion) = chars.next() else { break };
        let signed = |v: u64| if size64 { v as i64 } else { v as i32 as i64 };
        let unsigned = |v: u64| if size64 { v } else { v as u32 as u64 };
        let body = match conversion {
            '%' => "%".to_string(),
            'd' | 'i' => {
                let v = signed(next(g)?);
                let digits = format!("{:0>1$}", v.unsigned_abs(), precision.unwrap_or(1));
                format!("{}{digits}", if v < 0 { "-" } else if plus { "+" } else if space { " " } else { "" })
            }
            'u' => format!("{:0>1$}", unsigned(next(g)?), precision.unwrap_or(1)),
            'x' => format!("{}{:0>2$x}", if alt { "0x" } else { "" }, unsigned(next(g)?), precision.unwrap_or(1)),
            'X' => format!("{}{:0>2$X}", if alt { "0X" } else { "" }, unsigned(next(g)?), precision.unwrap_or(1)),
            'o' => format!("{:0>1$o}", unsigned(next(g)?), precision.unwrap_or(1)),
            'p' => format!("{:016X}", next(g)?),
            'c' => char::from_u32(next(g)? as u32 & if long || wide { 0xFFFF } else { 0xFF }).unwrap_or('?').to_string(),
            's' | 'S' => {
                let at = next(g)?;
                // %s follows the function, %S is the other kind; %ls / %hs say which
                let is_wide = if long { true } else if short_string { false } else { wide != (conversion == 'S') };
                let text = if at == 0 {
                    "(null)".to_string()
                } else if is_wide {
                    g.mem.wstr(at, 0x1_0000)?
                } else {
                    g.mem.cstr(at, 0x1_0000)?
                };
                match precision {
                    Some(p) => text.chars().take(p).collect(),
                    None => text,
                }
            }
            'f' | 'F' | 'e' | 'E' | 'g' | 'G' => {
                let x = f64::from_bits(next(g)?);
                let p = precision.unwrap_or(6);
                let digits = match conversion {
                    'f' | 'F' => format!("{:.*}", p, x.abs()),
                    'e' | 'E' => {
                        let s = format!("{:.*e}", p, x.abs());
                        let (mantissa, e) = s.split_once('e').unwrap_or((&s, "0"));
                        let e: i32 = e.parse().unwrap_or(0);
                        let s = format!("{mantissa}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
                        if conversion == 'E' { s.to_uppercase() } else { s }
                    }
                    _ => general(x.abs(), p, conversion == 'G'),
                };
                format!("{}{digits}", if x.is_sign_negative() { "-" } else if plus { "+" } else if space { " " } else { "" })
            }
            other => format!("%{other}"),
        };
        let numeric = !matches!(conversion, 's' | 'S' | 'c' | '%');
        out.push_str(&pad(body, width, left, zero && numeric && !left));
    }
    Ok(out)
}

/// __stdio_common_vsprintf(options, buffer, count, format, locale, arguments) and its _s and
/// wide forms: the length without the terminator; when the text does not fit, what fits is
/// written (terminated) and the answer is the full length or -1, as the options say.
fn vsprintf(g: &mut Guest, wide: bool) -> H {
    let (options, buffer, count, format_at, args) = (g.arg(0)?, g.arg(1)?, g.arg(2)?, g.arg(3)?, g.arg(5)?);
    let pattern = if wide { g.mem.wstr(format_at, 0x1_0000)? } else { g.mem.cstr(format_at, 0x1_0000)? };
    let text = format(g, &pattern, args, wide)?;
    let units: Vec<u16> = if wide { text.encode_utf16().collect() } else { text.bytes().map(u16::from).collect() };
    let size = if wide { 2 } else { 1 };
    if buffer != 0 && count != 0 {
        let fits = units.len().min(count as usize - 1);
        for (i, &u) in units[..fits].iter().chain([&0]).enumerate() {
            g.mem.write_sized(buffer + size * i as u64, size as u8, u as u64)?;
        }
    }
    let truncated = buffer != 0 && units.len() as u64 >= count;
    Ok(if truncated && options & STANDARD_SNPRINTF_BEHAVIOR == 0 { -1i32 as u32 as u64 } else { units.len() as u64 })
}

fn __stdio_common_vsprintf(g: &mut Guest) -> H {
    vsprintf(g, false)
}

fn __stdio_common_vswprintf(g: &mut Guest) -> H {
    vsprintf(g, true)
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; __stdio_common_vsprintf, __stdio_common_vswprintf);
    map.insert("__stdio_common_vsprintf_s", __stdio_common_vsprintf as Handler);
    map.insert("__stdio_common_vsnprintf_s", __stdio_common_vsprintf as Handler);
    map.insert("__stdio_common_vswprintf_s", __stdio_common_vswprintf as Handler);
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX, RSP};
    use xcl_emu::mem::{PROT_R, PROT_W};

    use super::*;

    fn guest() -> (Guest, u64) {
        let mut g = Guest::new();
        let stack = g.mem.alloc(0x4000, PROT_R | PROT_W).unwrap();
        g.add_thread(stack, stack + 0x4000, 0, 0, crate::SLOT_THREAD_RETURN).unwrap();
        g.cpu = g.threads[0].cpu.clone();
        let page = g.mem.alloc(0x2000, PROT_R | PROT_W).unwrap();
        (g, page)
    }

    /// `pattern` formatted with `args` as a va_list.
    fn formatted(g: &mut Guest, page: u64, pattern: &str, args: &[u64]) -> String {
        for (i, &a) in args.iter().enumerate() {
            g.mem.write_u64(page + 0x1000 + 8 * i as u64, a).unwrap();
        }
        format(g, pattern, page + 0x1000, false).unwrap()
    }

    #[test]
    fn printf_conversions_come_out_as_the_c_runtime_writes_them() {
        let (mut g, page) = guest();
        g.mem.write(page, b"levels\0").unwrap();
        let wide: Vec<u8> = "pack".encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        g.mem.write(page + 0x100, &wide).unwrap();
        let minus = (-42i32) as u32 as u64;
        assert_eq!(formatted(&mut g, page, "%s:/%ls.rpf %S", &[page, page + 0x100, page + 0x100]), "levels:/pack.rpf pack");
        assert_eq!(formatted(&mut g, page, "[%5d] [%-5d] [%05d] [%+d] [%.3d]", &[minus, 7, minus, 7, 7]), "[  -42] [7    ] [-0042] [+7] [007]");
        assert_eq!(formatted(&mut g, page, "%u %x %#X %08x %o", &[minus, 255, 255, 0xBEEF, 8]), "4294967254 ff 0XFF 0000beef 10");
        assert_eq!(formatted(&mut g, page, "%lld %I64u %zu %llx", &[(-5i64) as u64, u64::MAX, 1 << 40, 0x1_0000_0000]), "-5 18446744073709551615 1099511627776 100000000");
        assert_eq!(formatted(&mut g, page, "%c%c %% %p", &['o' as u64, 'k' as u64, 0x1400_0000]), "ok % 0000000014000000");
        let f = |x: f64| x.to_bits();
        assert_eq!(formatted(&mut g, page, "%f %.2f %8.3f %e %g %g %g", &[f(1.5), f(-2.345), f(3.14159), f(12345.678), f(0.0001), f(1e-5), f(1234567.0)]), "1.500000 -2.35    3.142 1.234568e+04 0.0001 1e-05 1.23457e+06");
        assert_eq!(formatted(&mut g, page, "%.3s|%*d|%s", &[page, 4, 9, 0]), "lev|   9|(null)");
    }

    #[test]
    fn vsprintf_keeps_to_the_buffer_and_says_what_it_needed() {
        let (mut g, page) = guest();
        let (buffer, pattern, args) = (page, page + 0x100, page + 0x200);
        g.mem.write(pattern, b"frame %d\0").unwrap();
        g.mem.write_u64(args, 1234).unwrap();
        let call = |g: &mut Guest, f: Handler, options: u64, count: u64| {
            (g.cpu.r[RCX], g.cpu.r[RDX], g.cpu.r[8], g.cpu.r[9]) = (options, buffer, count, pattern);
            g.mem.write_u64(g.cpu.r[RSP] + 8 + 8 * 5, args).unwrap();
            f(g).unwrap()
        };
        assert_eq!(call(&mut g, __stdio_common_vsprintf, 0, 64), 10);
        assert_eq!(g.mem.cstr(buffer, 64).unwrap(), "frame 1234");
        // too small: what fits, terminated; -1, or the full length for snprintf
        assert_eq!(call(&mut g, __stdio_common_vsprintf, 0, 6), u32::MAX as u64);
        assert_eq!(g.mem.cstr(buffer, 64).unwrap(), "frame");
        assert_eq!(call(&mut g, __stdio_common_vsprintf, STANDARD_SNPRINTF_BEHAVIOR, 6), 10);
        let wide: Vec<u8> = "%d fps".encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        g.mem.write(pattern, &wide).unwrap();
        assert_eq!(call(&mut g, __stdio_common_vswprintf, 0, 64), 8);
        assert_eq!(g.mem.wstr(buffer, 64).unwrap(), "1234 fps");
    }
}
