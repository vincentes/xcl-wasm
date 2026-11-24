//! File functions (kernel32, ntdll and the C runtime's `fopen` family) on the title's file
//! system (`crate::vfs`). An open file is its contents in a kernel object; every write goes
//! back to the overlay at once.

#![allow(non_snake_case)]

use std::collections::HashMap;

use super::*;
use crate::vfs::{self, Kind};
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const ERROR_FILE_NOT_FOUND: u32 = 2;
const ERROR_PATH_NOT_FOUND: u32 = 3;
const ERROR_ACCESS_DENIED: u32 = 5;
const ERROR_NO_MORE_FILES: u32 = 18;
const ERROR_FILE_EXISTS: u32 = 80;
const ERROR_INVALID_PARAMETER: u32 = 87;
const ERROR_ALREADY_EXISTS: u32 = 183;
const ERROR_DIR_NOT_EMPTY: u32 = 145;
const INVALID_HANDLE_VALUE: u64 = u64::MAX;
const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
/// file times: 2026-01-01 (the guest has no calendar)
const FILE_TIME: u64 = 134_116_992_000_000_000;
const STATUS_END_OF_FILE: u64 = 0xC000_0011;
const STATUS_INVALID_HANDLE: u64 = 0xC000_0008;

/// An open file or folder.
#[derive(Debug, Clone)]
pub struct File {
    pub path: String,
    /// its contents (empty for a file that is read on demand)
    pub data: Vec<u8>,
    /// the size of a file of the game folder that is read on demand: `data` is not used
    pub source_size: Option<u64>,
    pub position: u64,
    pub is_dir: bool,
    pub writable: bool,
    /// every write goes to the end
    pub append: bool,
}

impl File {
    pub fn len(&self) -> u64 {
        self.source_size.unwrap_or(self.data.len() as u64)
    }
}

/// A directory listing in progress (FindFirstFile).
#[derive(Debug, Clone)]
pub struct Find {
    pub entries: Vec<(String, Kind)>,
    pub next: usize,
}

fn wide_arg(g: &Guest, at: u64) -> Result<String, Stop> {
    Ok(g.mem.wstr(at, 0x8000)?)
}

fn write_wide(g: &mut Guest, at: u64, text: &str) -> Result<(), Stop> {
    for (i, c) in text.encode_utf16().chain([0]).enumerate() {
        g.mem.write_u16(at + 2 * i as u64, c)?;
    }
    Ok(())
}

fn attributes(kind: Kind) -> u32 {
    match kind {
        Kind::Dir => FILE_ATTRIBUTE_DIRECTORY,
        Kind::File(_) => FILE_ATTRIBUTE_NORMAL,
    }
}

/// The error for a path that is not there: its folder exists (file not found) or not.
fn missing(g: &Guest, path: &str) -> u32 {
    let parent = path.rfind('\\').map_or("G:\\", |i| if i > 2 { &path[..i] } else { &path[..3] });
    if g.vfs.kind(parent) == Some(Kind::Dir) {
        ERROR_FILE_NOT_FOUND
    } else {
        ERROR_PATH_NOT_FOUND
    }
}

/// Open (or create) `path`. `disposition`: CREATE_NEW 1, CREATE_ALWAYS 2, OPEN_EXISTING 3,
/// OPEN_ALWAYS 4, TRUNCATE_EXISTING 5.
pub(super) fn open(g: &mut Guest, name: &str, access: u32, disposition: u32) -> Result<u64, u32> {
    let path = vfs::resolve(name).ok_or(ERROR_PATH_NOT_FOUND)?;
    let kind = g.vfs.kind(&path);
    // GENERIC_WRITE, GENERIC_ALL, FILE_WRITE_DATA, FILE_APPEND_DATA
    let writable = access & (0x4000_0000 | 0x1000_0000 | 0x2 | 0x4) != 0;
    let append = access & 0x4 != 0 && access & (0x4000_0000 | 0x1000_0000 | 0x2) == 0;
    let file = match (kind, disposition) {
        (Some(_), 1) => return Err(ERROR_FILE_EXISTS),
        (Some(Kind::Dir), 3 | 4) => File { path, data: Vec::new(), source_size: None, position: 0, is_dir: true, writable: false, append: false },
        (Some(Kind::Dir), _) => return Err(ERROR_ACCESS_DENIED),
        // a file read on demand is opened as it is, to be read; its size keeps it out of the overlay
        (Some(Kind::File(_)), 3 | 4) if g.vfs.source_size(&path).is_some() => {
            if writable {
                return Err(ERROR_ACCESS_DENIED);
            }
            File { source_size: g.vfs.source_size(&path), data: Vec::new(), path, position: 0, is_dir: false, writable: false, append: false }
        }
        (Some(Kind::File(_)), 3 | 4) => File { data: g.vfs.read(&path).unwrap_or(&[]).to_vec(), source_size: None, path, position: 0, is_dir: false, writable, append },
        (None, 3 | 5) => return Err(missing(g, &path)),
        // create, or truncate
        _ => {
            if !g.vfs.write(&path, Vec::new()) {
                return Err(missing(g, &path));
            }
            File { path, data: Vec::new(), source_size: None, position: 0, is_dir: false, writable: true, append }
        }
    };
    // CREATE_ALWAYS / OPEN_ALWAYS on a file that was there: Windows reports it, and succeeds
    if kind.is_some() && matches!(disposition, 2 | 4) {
        g.set_last_error(ERROR_ALREADY_EXISTS);
    } else {
        g.set_last_error(0);
    }
    Ok(g.win.new_handle(Object::File(file)))
}

/// CreateFileW(name, access, share, security, disposition, flags, template)
fn CreateFileW(g: &mut Guest) -> H {
    let (name, access, disposition) = (g.arg(0)?, g.arg(1)? as u32, g.arg(4)? as u32);
    let name = wide_arg(g, name)?;
    match open(g, &name, access, disposition) {
        Ok(handle) => Ok(handle),
        Err(code) => {
            g.set_last_error(code);
            Ok(INVALID_HANDLE_VALUE)
        }
    }
}

fn file<'a>(g: &'a mut Guest, handle: u64) -> Option<&'a mut File> {
    match g.win.handles.get_mut(&handle)? {
        Object::File(f) => Some(f),
        _ => None,
    }
}

/// Read up to `len` bytes at the file's position (or at `offset`) into guest memory.
fn read(g: &mut Guest, handle: u64, at: u64, len: u64, offset: Option<u64>) -> Result<Option<u64>, Stop> {
    let Some(f) = file(g, handle) else { return Ok(None) };
    let size = f.len();
    let from = offset.unwrap_or(f.position).min(size);
    let n = (size - from).min(len) as usize;
    f.position = from + n as u64;
    let bytes = if f.source_size.is_some() {
        let path = f.path.clone();
        let mut bytes = vec![0u8; n];
        if g.vfs.read_at(&path, from, &mut bytes) != Some(n) {
            return Ok(None);
        }
        bytes
    } else {
        f.data[from as usize..from as usize + n].to_vec()
    };
    g.mem.write(at, &bytes)?;
    Ok(Some(n as u64))
}

/// Write guest memory to the file at its position (or at `offset`, or at its end).
fn write(g: &mut Guest, handle: u64, at: u64, len: u64, offset: Option<u64>) -> Result<Option<u64>, Stop> {
    let mut bytes = vec![0u8; len as usize];
    g.mem.read(at, &mut bytes)?;
    let Some(f) = file(g, handle) else { return Ok(None) };
    if !f.writable || f.is_dir {
        return Ok(None);
    }
    let from = if f.append { f.data.len() } else { offset.unwrap_or(f.position) as usize };
    if f.data.len() < from + bytes.len() {
        f.data.resize(from + bytes.len(), 0);
    }
    f.data[from..from + bytes.len()].copy_from_slice(&bytes);
    f.position = (from + bytes.len()) as u64;
    let (path, data) = (f.path.clone(), f.data.clone());
    g.vfs.write(&path, data);
    Ok(Some(len))
}

/// The byte offset a caller gave (a LARGE_INTEGER; -1 and -2 mean "the file's position").
fn offset_arg(g: &Guest, at: u64) -> Result<Option<u64>, Stop> {
    if at == 0 {
        return Ok(None);
    }
    let v = g.mem.read_u64(at)?;
    Ok((v < 1 << 62).then_some(v))
}

/// ReadFile(handle, buffer, length, read out, overlapped)
fn ReadFile(g: &mut Guest) -> H {
    let (handle, at, len, done) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32 as u64, g.arg(3)?);
    let n = if handle == STDIN { Some(0) } else { read(g, handle, at, len, None)? };
    let Some(n) = n else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    if done != 0 {
        g.mem.write_u32(done, n as u32)?;
    }
    Ok(1)
}

/// WriteFile(handle, buffer, length, written out, overlapped)
fn WriteFile(g: &mut Guest) -> H {
    let (handle, at, len, done) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32 as u64, g.arg(3)?);
    let n = if handle == STDOUT || handle == STDERR {
        let mut bytes = vec![0u8; len as usize];
        g.mem.read(at, &mut bytes)?;
        g.write_std(&bytes);
        Some(len)
    } else {
        write(g, handle, at, len, None)?
    };
    let Some(n) = n else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    if done != 0 {
        g.mem.write_u32(done, n as u32)?;
    }
    Ok(1)
}

/// Nt{Read,Write}File(handle, event, apc, apc context, io status block, buffer, length, offset, key)
fn nt_io(g: &mut Guest, writing: bool) -> H {
    let (handle, iosb, at, len, offset) = (g.arg(0)?, g.arg(4)?, g.arg(5)?, g.arg(6)? as u32 as u64, g.arg(7)?);
    let offset = offset_arg(g, offset)?;
    let n = match (writing, handle) {
        (true, STDOUT | STDERR) => {
            let mut bytes = vec![0u8; len as usize];
            g.mem.read(at, &mut bytes)?;
            g.write_std(&bytes);
            Some(len)
        }
        (false, STDIN) => Some(0),
        (true, _) => write(g, handle, at, len, offset)?,
        (false, _) => read(g, handle, at, len, offset)?,
    };
    let Some(n) = n else { return Ok(STATUS_INVALID_HANDLE) };
    let status = if !writing && n == 0 && len != 0 { STATUS_END_OF_FILE } else { 0 };
    if iosb != 0 {
        g.mem.write_u64(iosb, status)?;
        g.mem.write_u64(iosb + 8, n)?;
    }
    Ok(status)
}

fn NtReadFile(g: &mut Guest) -> H {
    nt_io(g, false)
}

fn NtWriteFile(g: &mut Guest) -> H {
    nt_io(g, true)
}

/// SetFilePointerEx(handle, distance, new position out, method)
fn SetFilePointerEx(g: &mut Guest) -> H {
    let (handle, distance, out, method) = (g.arg(0)?, g.arg(1)? as i64, g.arg(2)?, g.arg(3)? as u32);
    let Some(f) = file(g, handle) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    let base = match method {
        0 => 0,
        1 => f.position as i64,
        _ => f.len() as i64,
    };
    let Some(to) = base.checked_add(distance).filter(|&p| p >= 0) else {
        g.set_last_error(ERROR_INVALID_PARAMETER);
        return Ok(0);
    };
    f.position = to as u64;
    if out != 0 {
        g.mem.write_u64(out, to as u64)?;
    }
    Ok(1)
}

/// GetFileTime(handle, creation out, last access out, last write out): the one time files have.
fn GetFileTime(g: &mut Guest) -> H {
    let (handle, outs) = (g.arg(0)?, [g.arg(1)?, g.arg(2)?, g.arg(3)?]);
    if file(g, handle).is_none() {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    }
    for out in outs.into_iter().filter(|&o| o != 0) {
        g.mem.write_u64(out, FILE_TIME)?;
    }
    Ok(1)
}

fn GetFileSizeEx(g: &mut Guest) -> H {
    let (handle, out) = (g.arg(0)?, g.arg(1)?);
    let Some(size) = file(g, handle).map(|f| f.len()) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    g.mem.write_u64(out, size)?;
    Ok(1)
}

/// BY_HANDLE_FILE_INFORMATION
fn GetFileInformationByHandle(g: &mut Guest) -> H {
    let (handle, at) = (g.arg(0)?, g.arg(1)?);
    let Some((size, is_dir)) = file(g, handle).map(|f| (f.len(), f.is_dir)) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    g.mem.write(at, &[0u8; 52])?;
    g.mem.write_u32(at, if is_dir { FILE_ATTRIBUTE_DIRECTORY } else { FILE_ATTRIBUTE_NORMAL })?;
    for t in 0..3 {
        g.mem.write_u64(at + 4 + 8 * t, FILE_TIME)?;
    }
    g.mem.write_u32(at + 32, (size >> 32) as u32)?;
    g.mem.write_u32(at + 36, size as u32)?;
    g.mem.write_u32(at + 40, 1)?;
    g.mem.write_u32(at + 48, handle as u32)?;
    Ok(1)
}

/// GetFileInformationByHandleEx(handle, class, buffer, size): FileBasicInfo (0),
/// FileStandardInfo (1), FileAttributeTagInfo (9)
fn GetFileInformationByHandleEx(g: &mut Guest) -> H {
    let (handle, class, at, len) = (g.arg(0)?, g.arg(1)? as u32, g.arg(2)?, g.arg(3)? as u32);
    let Some((size, is_dir)) = file(g, handle).map(|f| (f.len(), f.is_dir)) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    let attrs = if is_dir { FILE_ATTRIBUTE_DIRECTORY } else { FILE_ATTRIBUTE_NORMAL };
    match (class, len) {
        (0, 40..) => {
            for t in 0..4 {
                g.mem.write_u64(at + 8 * t, FILE_TIME)?;
            }
            g.mem.write_u64(at + 32, attrs as u64)?;
        }
        (1, 24..) => {
            g.mem.write_u64(at, size)?;
            g.mem.write_u64(at + 8, size)?;
            g.mem.write_u32(at + 16, 1)?;
            g.mem.write_u32(at + 20, (is_dir as u32) << 8)?;
        }
        (9, 8..) => {
            g.mem.write_u32(at, attrs)?;
            g.mem.write_u32(at + 4, 0)?;
        }
        _ => {
            g.set_last_error(ERROR_INVALID_PARAMETER);
            return Ok(0);
        }
    }
    Ok(1)
}

/// SetFileInformationByHandle(handle, class, buffer, size): delete (FileDispositionInfo 4,
/// …Ex 21), rename (FileRenameInfo 3, …Ex 22), truncate (FileEndOfFileInfo 6); times and
/// allocation sizes are taken and ignored.
fn SetFileInformationByHandle(g: &mut Guest) -> H {
    let (handle, class, at) = (g.arg(0)?, g.arg(1)? as u32, g.arg(2)?);
    let Some(path) = file(g, handle).map(|f| f.path.clone()) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    let done = match class {
        4 | 21 => {
            let delete = if class == 4 { g.mem.read_u8(at)? != 0 } else { g.mem.read_u32(at)? & 1 != 0 };
            !delete || g.vfs.remove(&path)
        }
        3 | 22 => {
            // { replace flags, root directory, name length in bytes, name }
            let len = g.mem.read_u32(at + 16)? as u64 / 2;
            let mut name = Vec::with_capacity(len as usize);
            for i in 0..len {
                name.push(g.mem.read_u16(at + 20 + 2 * i)?);
            }
            match vfs::resolve(&String::from_utf16_lossy(&name)) {
                Some(to) => {
                    let moved = g.vfs.rename(&path, &to);
                    if let (true, Some(f)) = (moved, file(g, handle)) {
                        f.path = to;
                    }
                    moved
                }
                None => false,
            }
        }
        6 => {
            let size = g.mem.read_u64(at)? as usize;
            let data = file(g, handle).map(|f| {
                f.data.resize(size, 0);
                f.data.clone()
            });
            data.is_some_and(|d| g.vfs.write(&path, d))
        }
        _ => true,
    };
    if !done {
        g.set_last_error(ERROR_ACCESS_DENIED);
    }
    Ok(done as u64)
}

fn GetFileAttributesW(g: &mut Guest) -> H {
    let name = g.arg(0)?;
    let name = wide_arg(g, name)?;
    let Some(path) = vfs::resolve(&name) else {
        g.set_last_error(ERROR_PATH_NOT_FOUND);
        return Ok(u32::MAX as u64);
    };
    match g.vfs.kind(&path) {
        Some(kind) => Ok(attributes(kind) as u64),
        None => {
            let code = missing(g, &path);
            g.set_last_error(code);
            Ok(u32::MAX as u64)
        }
    }
}

/// A function that takes one path and does `change` to it.
fn change(g: &mut Guest, change: impl FnOnce(&mut Guest, &str) -> Result<(), u32>) -> H {
    let name = g.arg(0)?;
    let name = wide_arg(g, name)?;
    let result = match vfs::resolve(&name) {
        Some(path) => change(g, &path),
        None => Err(ERROR_PATH_NOT_FOUND),
    };
    match result {
        Ok(()) => Ok(1),
        Err(code) => {
            g.set_last_error(code);
            Ok(0)
        }
    }
}

fn CreateDirectoryW(g: &mut Guest) -> H {
    change(g, |g, path| match g.vfs.kind(path) {
        Some(_) => Err(ERROR_ALREADY_EXISTS),
        None if g.vfs.mkdir(path) => Ok(()),
        None => Err(ERROR_PATH_NOT_FOUND),
    })
}

fn RemoveDirectoryW(g: &mut Guest) -> H {
    change(g, |g, path| match g.vfs.kind(path) {
        Some(Kind::Dir) if g.vfs.remove(path) => Ok(()),
        Some(Kind::Dir) => Err(ERROR_DIR_NOT_EMPTY),
        Some(_) => Err(267), // ERROR_DIRECTORY
        None => Err(missing(g, path)),
    })
}

fn DeleteFileW(g: &mut Guest) -> H {
    change(g, |g, path| match g.vfs.kind(path) {
        Some(Kind::File(_)) if g.vfs.remove(path) => Ok(()),
        Some(_) => Err(ERROR_ACCESS_DENIED),
        None => Err(missing(g, path)),
    })
}

/// MoveFileExW(from, to, flags); flag 1: replace an existing file
fn MoveFileExW(g: &mut Guest) -> H {
    let (to, flags) = (g.arg(1)?, g.arg(2)? as u32);
    let to = wide_arg(g, to)?;
    change(g, |g, from| {
        let to = vfs::resolve(&to).ok_or(ERROR_PATH_NOT_FOUND)?;
        if g.vfs.kind(&to).is_some() && flags & 1 == 0 {
            return Err(ERROR_ALREADY_EXISTS);
        }
        if g.vfs.rename(from, &to) {
            Ok(())
        } else {
            Err(missing(g, from))
        }
    })
}

/// `*` and `?` matching of a file name, without regard to case.
fn matches(pattern: &str, name: &str) -> bool {
    fn go(p: &[char], n: &[char]) -> bool {
        match p.split_first() {
            None => n.is_empty(),
            Some(('*', rest)) => (0..=n.len()).any(|i| go(rest, &n[i..])),
            Some((c, rest)) => n.split_first().is_some_and(|(d, more)| (*c == '?' || c.eq_ignore_ascii_case(d)) && go(rest, more)),
        }
    }
    let (p, n): (Vec<char>, Vec<char>) = (pattern.chars().collect(), name.chars().collect());
    pattern == "*.*" || go(&p, &n)
}

/// WIN32_FIND_DATAW for one entry.
fn write_find_data(g: &mut Guest, at: u64, name: &str, kind: Kind) -> Result<(), Stop> {
    g.mem.write(at, &[0u8; 592])?;
    g.mem.write_u32(at, attributes(kind))?;
    for t in 0..3 {
        g.mem.write_u64(at + 4 + 8 * t, FILE_TIME)?;
    }
    if let Kind::File(size) = kind {
        g.mem.write_u32(at + 28, (size >> 32) as u32)?;
        g.mem.write_u32(at + 32, size as u32)?;
    }
    let short: String = name.chars().take(259).collect();
    write_wide(g, at + 44, &short)
}

/// FindFirstFileExW(pattern, level, data, search op, filter, flags)
fn FindFirstFileExW(g: &mut Guest) -> H {
    let (pattern, at) = (g.arg(0)?, g.arg(2)?);
    let pattern = wide_arg(g, pattern)?;
    let Some(full) = vfs::resolve(&pattern) else {
        g.set_last_error(ERROR_PATH_NOT_FOUND);
        return Ok(INVALID_HANDLE_VALUE);
    };
    let (dir, mask) = match full.rfind('\\') {
        Some(i) if i > 2 => (&full[..i], &full[i + 1..]),
        _ => (&full[..3], &full[3..]),
    };
    if g.vfs.kind(dir) != Some(Kind::Dir) {
        g.set_last_error(ERROR_PATH_NOT_FOUND);
        return Ok(INVALID_HANDLE_VALUE);
    }
    let entries: Vec<(String, Kind)> = g.vfs.list(dir).into_iter().filter(|(n, _)| matches(mask, n)).collect();
    let Some((name, kind)) = entries.first().cloned() else {
        g.set_last_error(ERROR_FILE_NOT_FOUND);
        return Ok(INVALID_HANDLE_VALUE);
    };
    write_find_data(g, at, &name, kind)?;
    Ok(g.win.new_handle(Object::Find(Find { entries, next: 1 })))
}

fn FindFirstFileW(g: &mut Guest) -> H {
    // (pattern, data): the data is argument 1 here
    let (pattern, at) = (g.arg(0)?, g.arg(1)?);
    g.cpu.r[8] = at;
    g.cpu.r[xcl_emu::cpu::RCX] = pattern;
    FindFirstFileExW(g)
}

fn FindNextFileW(g: &mut Guest) -> H {
    let (handle, at) = (g.arg(0)?, g.arg(1)?);
    let next = match g.win.handles.get_mut(&handle) {
        Some(Object::Find(f)) => {
            let e = f.entries.get(f.next).cloned();
            f.next += 1;
            e
        }
        _ => {
            g.set_last_error(ERROR_INVALID_HANDLE);
            return Ok(0);
        }
    };
    match next {
        Some((name, kind)) => {
            write_find_data(g, at, &name, kind)?;
            Ok(1)
        }
        None => {
            g.set_last_error(ERROR_NO_MORE_FILES);
            Ok(0)
        }
    }
}

fn FindClose(g: &mut Guest) -> H {
    let handle = g.arg(0)?;
    Ok(g.win.handles.remove(&handle).is_some() as u64)
}

/// Copy `text` out as UTF-16 the way GetFullPathNameW and friends do: its length when it fits,
/// else the size needed with the terminator.
fn wide_out(g: &mut Guest, text: &str, at: u64, size: u64) -> H {
    let len = text.encode_utf16().count() as u64;
    if len >= size {
        return Ok(len + 1);
    }
    write_wide(g, at, text)?;
    Ok(len)
}

/// GetFullPathNameW(name, size, buffer, file part out)
fn GetFullPathNameW(g: &mut Guest) -> H {
    let (name, size, at, part) = (g.arg(0)?, g.arg(1)? as u32 as u64, g.arg(2)?, g.arg(3)?);
    let name = wide_arg(g, name)?;
    // a path outside the title's drives is given back as it is
    let full = vfs::resolve(&name).unwrap_or(name);
    let n = wide_out(g, &full, at, size)?;
    if part != 0 && n < size {
        let file = full.rfind('\\').map_or(0, |i| i + 1);
        let offset = full[..file].encode_utf16().count() as u64;
        g.mem.write_u64(part, if file < full.len() { at + 2 * offset } else { 0 })?;
    }
    Ok(n)
}

fn GetCurrentDirectoryW(g: &mut Guest) -> H {
    let (size, at) = (g.arg(0)? as u32 as u64, g.arg(1)?);
    wide_out(g, "G:\\", at, size)
}

/// GetFinalPathNameByHandleW(handle, buffer, size, flags)
fn GetFinalPathNameByHandleW(g: &mut Guest) -> H {
    let (handle, at, size) = (g.arg(0)?, g.arg(1)?, g.arg(2)? as u32 as u64);
    let Some(path) = file(g, handle).map(|f| format!("\\\\?\\{}", f.path)) else {
        g.set_last_error(ERROR_INVALID_HANDLE);
        return Ok(0);
    };
    wide_out(g, &path, at, size)
}

/// The space of a title drive: 64 GiB, half of it free (there is no host volume behind it).
const DISK_TOTAL: u64 = 64 << 30;
const DISK_FREE: u64 = 32 << 30;

/// GetDiskFreeSpaceExW(directory, available out, total out, free out)
fn GetDiskFreeSpaceExW(g: &mut Guest) -> H {
    let (name, available, total, free) = (g.arg(0)?, g.arg(1)?, g.arg(2)?, g.arg(3)?);
    if name != 0 && vfs::resolve(&wide_arg(g, name)?).is_none() {
        g.set_last_error(ERROR_PATH_NOT_FOUND);
        return Ok(0);
    }
    for (at, v) in [(available, DISK_FREE), (total, DISK_TOTAL), (free, DISK_FREE)] {
        if at != 0 {
            g.mem.write_u64(at, v)?;
        }
    }
    Ok(1)
}

/// GetDiskFreeSpaceW(root, sectors per cluster, bytes per sector, free clusters, clusters)
fn GetDiskFreeSpaceW(g: &mut Guest) -> H {
    let name = g.arg(0)?;
    if name != 0 && vfs::resolve(&wide_arg(g, name)?).is_none() {
        g.set_last_error(ERROR_PATH_NOT_FOUND);
        return Ok(0);
    }
    for (i, v) in [8u32, 512, (DISK_FREE / 4096) as u32, (DISK_TOTAL / 4096) as u32].into_iter().enumerate() {
        let at = g.arg(1 + i)?;
        if at != 0 {
            g.mem.write_u32(at, v)?;
        }
    }
    Ok(1)
}

// --- the C runtime: a FILE * is the file's handle ----------------------------------------------

/// fopen(name, mode)
fn fopen(g: &mut Guest) -> H {
    let (name, mode) = (g.arg(0)?, g.arg(1)?);
    let (name, mode) = (g.mem.cstr(name, 0x8000)?, g.mem.cstr(mode, 16)?);
    let (access, disposition) = match (mode.as_bytes().first(), mode.contains('+')) {
        (Some(b'r'), false) => (0x8000_0000, 3),
        (Some(b'r'), true) => (0xC000_0000, 3),
        (Some(b'w'), _) => (0xC000_0000, 2),
        (Some(b'a'), _) => (0x8000_0004, 4),
        _ => return Ok(0),
    };
    Ok(open(g, &name, access, disposition).unwrap_or(0))
}

/// fread(buffer, size, count, file) → items read
fn fread(g: &mut Guest) -> H {
    let (at, size, count, handle) = (g.arg(0)?, g.arg(1)?, g.arg(2)?, g.arg(3)?);
    if size == 0 {
        return Ok(0);
    }
    Ok(read(g, handle, at, size.saturating_mul(count), None)?.unwrap_or(0) / size)
}

/// fwrite(buffer, size, count, file) → items written
fn fwrite(g: &mut Guest) -> H {
    let (at, size, count, handle) = (g.arg(0)?, g.arg(1)?, g.arg(2)?, g.arg(3)?);
    let len = size.saturating_mul(count).min(1 << 30);
    if handle == STDOUT || handle == STDERR {
        let mut bytes = vec![0u8; len as usize];
        g.mem.read(at, &mut bytes)?;
        g.write_std(&bytes);
        return Ok(count);
    }
    if size == 0 {
        return Ok(0);
    }
    Ok(write(g, handle, at, len, None)?.unwrap_or(0) / size)
}

fn fclose(g: &mut Guest) -> H {
    let handle = g.arg(0)?;
    Ok(if g.win.handles.remove(&handle).is_some() { 0 } else { u32::MAX as u64 })
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; CreateFileW, ReadFile, WriteFile, NtReadFile, NtWriteFile, SetFilePointerEx, GetFileSizeEx);
    register!(map; GetFileInformationByHandle, GetFileInformationByHandleEx, SetFileInformationByHandle, GetFileAttributesW);
    register!(map; CreateDirectoryW, RemoveDirectoryW, DeleteFileW, MoveFileExW, FindFirstFileExW, FindFirstFileW, FindNextFileW, FindClose);
    register!(map; GetFullPathNameW, GetCurrentDirectoryW, GetFinalPathNameByHandleW, GetDiskFreeSpaceExW, GetDiskFreeSpaceW);
    register!(map; fopen, fread, fwrite, fclose, GetFileTime);
}
