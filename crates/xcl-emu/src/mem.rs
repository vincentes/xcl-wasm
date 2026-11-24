//! Guest memory: page-aligned regions of a 64-bit address space.

use std::cell::Cell;

pub const PAGE: u64 = 0x1000;

pub const PROT_R: u8 = 1;
pub const PROT_W: u8 = 2;
pub const PROT_X: u8 = 4;

/// An access to guest memory that is not mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    pub addr: u64,
    pub write: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapError {
    /// not page-aligned, empty, or wraps the address space
    BadRange,
    /// part of the range is mapped already
    Overlap,
    /// no free range of that size
    NoSpace,
}

pub struct Region {
    pub base: u64,
    pub data: Vec<u8>,
    pub prot: u8,
}

impl Region {
    pub fn end(&self) -> u64 {
        self.base + self.data.len() as u64
    }
}

/// The guest's address space. Regions are kept sorted by address.
pub struct Memory {
    regions: Vec<Region>,
    /// the region the last access hit
    last: Cell<usize>,
    /// where `alloc` looks first
    next_alloc: u64,
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

pub fn page_up(n: u64) -> u64 {
    n.wrapping_add(PAGE - 1) & !(PAGE - 1)
}

impl Memory {
    pub fn new() -> Self {
        Memory { regions: Vec::new(), last: Cell::new(0), next_alloc: 0x0000_0200_0000_0000 }
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    fn find(&self, addr: u64) -> Option<usize> {
        let i = self.last.get();
        if let Some(r) = self.regions.get(i) {
            if addr >= r.base && addr < r.end() {
                return Some(i);
            }
        }
        let i = self.regions.partition_point(|r| r.end() <= addr);
        let r = self.regions.get(i)?;
        if addr >= r.base {
            self.last.set(i);
            Some(i)
        } else {
            None
        }
    }

    pub fn is_mapped(&self, addr: u64, len: u64) -> bool {
        let Some(end) = addr.checked_add(len) else { return false };
        let mut at = addr;
        while at < end {
            match self.find(at) {
                Some(i) => at = self.regions[i].end(),
                None => return false,
            }
        }
        true
    }

    /// Map zeroed memory at `base`.
    pub fn map(&mut self, base: u64, len: u64, prot: u8) -> Result<(), MapError> {
        if len == 0 || base % PAGE != 0 || len % PAGE != 0 || base.checked_add(len).is_none() {
            return Err(MapError::BadRange);
        }
        let i = self.regions.partition_point(|r| r.end() <= base);
        if self.regions.get(i).is_some_and(|r| r.base < base + len) {
            return Err(MapError::Overlap);
        }
        self.regions.insert(i, Region { base, data: vec![0; len as usize], prot });
        Ok(())
    }

    /// Map zeroed memory wherever there is room (aligned to 64 KiB, as VirtualAlloc does).
    pub fn alloc(&mut self, len: u64, prot: u8) -> Result<u64, MapError> {
        let len = page_up(len.max(1));
        let mut base = self.next_alloc;
        loop {
            let end = base.checked_add(len).filter(|&e| e <= 0x0000_7FFF_0000_0000).ok_or(MapError::NoSpace)?;
            let i = self.regions.partition_point(|r| r.end() <= base);
            match self.regions.get(i) {
                Some(r) if r.base < end => base = (r.end() + 0xFFFF) & !0xFFFF,
                _ => break,
            }
        }
        self.map(base, len, prot)?;
        self.next_alloc = (base + len + 0xFFFF) & !0xFFFF;
        Ok(base)
    }

    /// Unmap the region that starts at `base`.
    pub fn unmap(&mut self, base: u64) -> bool {
        match self.regions.iter().position(|r| r.base == base) {
            Some(i) => {
                self.regions.remove(i);
                self.last.set(0);
                true
            }
            None => false,
        }
    }

    pub fn protect(&mut self, addr: u64, prot: u8) -> bool {
        match self.find(addr) {
            Some(i) => {
                self.regions[i].prot = prot;
                true
            }
            None => false,
        }
    }

    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), Fault> {
        let mut done = 0;
        while done < buf.len() {
            let at = addr.wrapping_add(done as u64);
            let i = self.find(at).ok_or(Fault { addr: at, write: false })?;
            let r = &self.regions[i];
            let off = (at - r.base) as usize;
            let n = (r.data.len() - off).min(buf.len() - done);
            buf[done..done + n].copy_from_slice(&r.data[off..off + n]);
            done += n;
        }
        Ok(())
    }

    pub fn write(&mut self, addr: u64, buf: &[u8]) -> Result<(), Fault> {
        let mut done = 0;
        while done < buf.len() {
            let at = addr.wrapping_add(done as u64);
            let i = self.find(at).ok_or(Fault { addr: at, write: true })?;
            let r = &mut self.regions[i];
            let off = (at - r.base) as usize;
            let n = (r.data.len() - off).min(buf.len() - done);
            r.data[off..off + n].copy_from_slice(&buf[done..done + n]);
            done += n;
        }
        Ok(())
    }

    /// The bytes at `addr` when they lie in one region.
    pub fn slice(&self, addr: u64, len: usize) -> Option<&[u8]> {
        let r = &self.regions[self.find(addr)?];
        let off = (addr - r.base) as usize;
        r.data.get(off..off.checked_add(len)?)
    }

    pub fn slice_mut(&mut self, addr: u64, len: usize) -> Option<&mut [u8]> {
        let i = self.find(addr)?;
        let r = &mut self.regions[i];
        let off = (addr - r.base) as usize;
        r.data.get_mut(off..off.checked_add(len)?)
    }

    #[inline]
    pub fn read_n<const N: usize>(&self, addr: u64) -> Result<[u8; N], Fault> {
        if let Some(i) = self.find(addr) {
            let r = &self.regions[i];
            let off = (addr - r.base) as usize;
            if let Some(s) = r.data.get(off..off + N) {
                let mut b = [0u8; N];
                b.copy_from_slice(s);
                return Ok(b);
            }
        }
        let mut b = [0u8; N];
        self.read(addr, &mut b)?;
        Ok(b)
    }

    pub fn read_u8(&self, addr: u64) -> Result<u8, Fault> {
        Ok(self.read_n::<1>(addr)?[0])
    }
    pub fn read_u16(&self, addr: u64) -> Result<u16, Fault> {
        Ok(u16::from_le_bytes(self.read_n(addr)?))
    }
    pub fn read_u32(&self, addr: u64) -> Result<u32, Fault> {
        Ok(u32::from_le_bytes(self.read_n(addr)?))
    }
    pub fn read_u64(&self, addr: u64) -> Result<u64, Fault> {
        Ok(u64::from_le_bytes(self.read_n(addr)?))
    }
    pub fn write_u8(&mut self, addr: u64, v: u8) -> Result<(), Fault> {
        self.write(addr, &[v])
    }
    pub fn write_u16(&mut self, addr: u64, v: u16) -> Result<(), Fault> {
        self.write(addr, &v.to_le_bytes())
    }
    pub fn write_u32(&mut self, addr: u64, v: u32) -> Result<(), Fault> {
        self.write(addr, &v.to_le_bytes())
    }
    pub fn write_u64(&mut self, addr: u64, v: u64) -> Result<(), Fault> {
        self.write(addr, &v.to_le_bytes())
    }

    /// Little-endian value of `size` bytes (1, 2, 4 or 8), zero-extended.
    #[inline]
    pub fn read_sized(&self, addr: u64, size: u8) -> Result<u64, Fault> {
        Ok(match size {
            1 => self.read_u8(addr)? as u64,
            2 => self.read_u16(addr)? as u64,
            4 => self.read_u32(addr)? as u64,
            _ => self.read_u64(addr)?,
        })
    }

    #[inline]
    pub fn write_sized(&mut self, addr: u64, size: u8, v: u64) -> Result<(), Fault> {
        match size {
            1 => self.write_u8(addr, v as u8),
            2 => self.write_u16(addr, v as u16),
            4 => self.write_u32(addr, v as u32),
            _ => self.write_u64(addr, v),
        }
    }

    /// NUL-terminated bytes at `addr` (at most `max`), lossily as text.
    pub fn cstr(&self, addr: u64, max: usize) -> Result<String, Fault> {
        let mut out = Vec::new();
        for i in 0..max as u64 {
            let b = self.read_u8(addr + i)?;
            if b == 0 {
                break;
            }
            out.push(b);
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// NUL-terminated UTF-16 at `addr` (at most `max` units).
    pub fn wstr(&self, addr: u64, max: usize) -> Result<String, Fault> {
        let mut out = Vec::new();
        for i in 0..max as u64 {
            let c = self.read_u16(addr + i * 2)?;
            if c == 0 {
                break;
            }
            out.push(c);
        }
        Ok(String::from_utf16_lossy(&out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_reads_and_writes_across_regions() {
        let mut m = Memory::new();
        m.map(0x1000, 0x1000, PROT_R | PROT_W).unwrap();
        m.map(0x2000, 0x1000, PROT_R | PROT_W).unwrap();
        assert_eq!(m.map(0x1000, 0x2000, PROT_R), Err(MapError::Overlap));
        assert_eq!(m.map(0x1001, 0x1000, PROT_R), Err(MapError::BadRange));
        m.write_u64(0x1FFC, 0x1122_3344_5566_7788).unwrap();
        assert_eq!(m.read_u64(0x1FFC).unwrap(), 0x1122_3344_5566_7788);
        assert_eq!(m.read_u32(0x2000).unwrap(), 0x1122_3344);
        assert_eq!(m.read_u8(0x3000), Err(Fault { addr: 0x3000, write: false }));
        assert_eq!(m.write_u32(0x2FFE, 1), Err(Fault { addr: 0x3000, write: true }));
        assert!(m.is_mapped(0x1800, 0x1000));
        assert!(!m.is_mapped(0x2800, 0x1000));
    }

    #[test]
    fn alloc_finds_free_ranges_and_unmap_frees_them() {
        let mut m = Memory::new();
        let a = m.alloc(10, PROT_R | PROT_W).unwrap();
        let b = m.alloc(0x2_0000, PROT_R | PROT_W).unwrap();
        assert_eq!(a % 0x1_0000, 0);
        assert!(b >= a + 0x1000);
        m.write_u8(b + 0x1_FFFF, 7).unwrap();
        assert!(m.unmap(a));
        assert!(!m.unmap(a));
        assert!(m.read_u8(a).is_err());
        m.write(b, b"title\0").unwrap();
        assert_eq!(m.cstr(b, 16).unwrap(), "title");
        assert!(m.cstr(b + 0x1_FFFF, 16).is_err(), "the string runs off the region");
    }
}
