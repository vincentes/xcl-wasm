//! The guest's heap (malloc, HeapAlloc, XMemAlloc …): blocks are carved from arenas in guest
//! memory; the bookkeeping stays on the host.

use std::collections::HashMap;

use xcl_emu::mem::{MapError, Memory, PROT_R, PROT_W};

const ARENA: u64 = 4 << 20;
/// blocks from this size get a mapping of their own
const LARGE: u64 = 256 << 10;

#[derive(Default)]
pub struct Heap {
    /// block address → its size (rounded)
    blocks: HashMap<u64, u64>,
    /// freed blocks by size
    free: HashMap<u64, Vec<u64>>,
    next: u64,
    end: u64,
}

impl Heap {
    /// A zeroed block of at least `size` bytes, 16-byte aligned.
    pub fn alloc(&mut self, mem: &mut Memory, size: u64) -> Result<u64, MapError> {
        let size = size.max(1).checked_add(15).ok_or(MapError::NoSpace)? & !15;
        if size >= LARGE {
            let at = mem.alloc(size, PROT_R | PROT_W)?;
            self.blocks.insert(at, size);
            return Ok(at);
        }
        if let Some(at) = self.free.get_mut(&size).and_then(Vec::pop) {
            if let Some(s) = mem.slice_mut(at, size as usize) {
                s.fill(0);
            }
            self.blocks.insert(at, size);
            return Ok(at);
        }
        if self.end - self.next < size {
            self.next = mem.alloc(ARENA, PROT_R | PROT_W)?;
            self.end = self.next + ARENA;
        }
        let at = self.next;
        self.next += size;
        self.blocks.insert(at, size);
        Ok(at)
    }

    /// Size of the block at `at` (`None`: not a live block).
    pub fn size(&self, at: u64) -> Option<u64> {
        self.blocks.get(&at).copied()
    }

    pub fn free(&mut self, mem: &mut Memory, at: u64) -> bool {
        match self.blocks.remove(&at) {
            Some(size) if size >= LARGE => mem.unmap(at),
            Some(size) => {
                self.free.entry(size).or_default().push(at);
                true
            }
            None => false,
        }
    }

    /// Move the block to one of `size` bytes, keeping its contents.
    pub fn realloc(&mut self, mem: &mut Memory, at: u64, size: u64) -> Result<u64, MapError> {
        let Some(old) = self.size(at) else { return self.alloc(mem, size) };
        if size <= old && old < LARGE {
            return Ok(at);
        }
        let to = self.alloc(mem, size)?;
        let mut data = vec![0u8; old.min(size) as usize];
        let _ = mem.read(at, &mut data);
        let _ = mem.write(to, &data);
        self.free(mem, at);
        Ok(to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_are_zeroed_reused_and_moved() {
        let mut mem = Memory::new();
        let mut h = Heap::default();
        let a = h.alloc(&mut mem, 10).unwrap();
        let b = h.alloc(&mut mem, 10).unwrap();
        assert_eq!((a % 16, b - a, h.size(a)), (0, 16, Some(16)));
        mem.write_u64(a, 0x1122_3344_5566_7788).unwrap();
        let c = h.realloc(&mut mem, a, 100).unwrap();
        assert_ne!(c, a);
        assert_eq!(mem.read_u64(c).unwrap(), 0x1122_3344_5566_7788);
        // the freed block comes back zeroed
        let d = h.alloc(&mut mem, 16).unwrap();
        assert_eq!((d, mem.read_u64(d).unwrap()), (a, 0));
        let big = h.alloc(&mut mem, 1 << 20).unwrap();
        mem.write_u8(big + (1 << 20) - 1, 1).unwrap();
        assert!(h.free(&mut mem, big));
        assert!(mem.read_u8(big).is_err());
        assert!(!h.free(&mut mem, big));
    }
}
