//! The system functions a title imports, answered by the host. A function is found by its name
//! alone (as XCL does for console functions), whichever DLL the title names for it.

use std::collections::HashMap;

use xcl_emu::cpu::Cpu;
use xcl_emu::mem::{Fault, Memory, PROT_R, PROT_W};

use crate::{Guest, Handler, Stop};

mod crt;
pub mod heap;
mod kernel32;

/// A module's static TLS, from its TLS directory.
#[derive(Debug, Clone)]
pub struct TlsTemplate {
    pub start: u64,
    pub len: u64,
    pub zero_fill: u64,
    pub index_address: u64,
    pub callbacks: u64,
    pub module_base: u64,
}

/// What the system functions keep between calls.
#[derive(Default)]
pub struct State {
    pub tls_templates: Vec<TlsTemplate>,
    /// set once the title asked to exit
    pub exit_code: Option<u32>,
    /// nesting of `Guest::call`
    pub call_depth: u32,
    pub heap: heap::Heap,
    /// TlsAlloc / FlsAlloc slots handed out
    pub tls_count: usize,
    /// guest time (see `Guest::now_ns`)
    pub time_ns: u64,
    /// kernel objects by handle
    pub handles: HashMap<u64, Object>,
    pub next_handle: u64,
    /// small guest blocks handed out once (`__p___argc`, the command line …), by name
    pub statics: HashMap<&'static str, u64>,
    /// functions registered with `_crt_atexit`
    pub atexit: Vec<u64>,
    pub prng: u64,
}

/// A kernel object behind a handle.
#[derive(Debug, Clone)]
pub enum Object {
    /// index into `Guest::threads`
    Thread(usize),
    Event { signaled: bool, manual: bool },
    /// a waitable timer: signaled from guest time `due` on
    Timer { due: Option<u64>, manual: bool },
}

impl State {
    /// A thread's TEB and static TLS blocks; GS points at the TEB. Returns the TEB.
    pub fn setup_thread(&mut self, cpu: &mut Cpu, mem: &mut Memory, stack_base: u64, stack_top: u64, id: u32) -> Result<u64, Fault> {
        let fault = |_| Fault { addr: 0, write: true };
        let teb = mem.alloc(0x2000, PROT_R | PROT_W).map_err(fault)?;
        // NT_TIB: stack base (+8) and limit (+0x10), self (+0x30); PEB (+0x60)
        mem.write_u64(teb + 0x08, stack_top)?;
        mem.write_u64(teb + 0x10, stack_base)?;
        mem.write_u64(teb + 0x30, teb)?;
        mem.write_u64(teb + 0x60, teb + 0x1000)?;
        // process and thread id
        mem.write_u64(teb + 0x40, 0x1000)?;
        mem.write_u64(teb + 0x48, id as u64)?;
        // ThreadLocalStoragePointer (+0x58): one block per module with static TLS
        let slots = mem.alloc(8 * self.tls_templates.len().max(1) as u64, PROT_R | PROT_W).map_err(fault)?;
        mem.write_u64(teb + 0x58, slots)?;
        for (i, t) in self.tls_templates.iter().enumerate() {
            let block = mem.alloc((t.len + t.zero_fill).max(1), PROT_R | PROT_W).map_err(fault)?;
            let mut template = vec![0u8; t.len as usize];
            mem.read(t.start, &mut template)?;
            mem.write(block, &template)?;
            mem.write_u64(slots + 8 * i as u64, block)?;
            mem.write_u32(t.index_address, i as u32)?;
        }
        cpu.gs_base = teb;
        Ok(teb)
    }

    pub fn new_handle(&mut self, object: Object) -> u64 {
        self.next_handle += 4;
        let h = 0x100 + self.next_handle;
        self.handles.insert(h, object);
        h
    }
}

/// `map.insert("Name", Name)` for functions named as the title imports them.
macro_rules! register {
    ($map:expr; $($name:ident),* $(,)?) => {
        $( $map.insert(stringify!($name), $name as $crate::Handler); )*
    };
}
pub(crate) use register;

pub fn handlers() -> HashMap<&'static str, Handler> {
    let mut map = HashMap::new();
    crt::register(&mut map);
    kernel32::register(&mut map);
    map
}

pub(crate) const ERROR_INVALID_HANDLE: u32 = 6;
pub(crate) const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
pub(crate) const ERROR_ENVVAR_NOT_FOUND: u32 = 203;
pub(crate) const ERROR_MOD_NOT_FOUND: u32 = 126;
pub(crate) const ERROR_PROC_NOT_FOUND: u32 = 127;
pub(crate) const ERROR_TIMEOUT: u32 = 1460;
pub(crate) const WAIT_TIMEOUT: u64 = 258;

/// The standard handles the title gets from GetStdHandle.
pub(crate) const STDIN: u64 = 0x20;
pub(crate) const STDOUT: u64 = 0x24;
pub(crate) const STDERR: u64 = 0x28;

impl Guest {
    /// Guest time in nanoseconds. Until the host supplies a clock it is the work done: an
    /// instruction of any thread is a nanosecond, and waiting skips ahead to the first deadline.
    pub fn now_ns(&self) -> u64 {
        self.win.time_ns
    }

    pub(crate) fn set_last_error(&mut self, code: u32) {
        let teb = self.thread().teb;
        self.thread_mut().last_error = code;
        // TEB.LastErrorValue
        let _ = self.mem.write_u32(teb + 0x68, code);
    }

    /// A zeroed heap block (never freed by the host).
    pub(crate) fn heap_alloc(&mut self, size: u64) -> Result<u64, Stop> {
        self.win.heap.alloc(&mut self.mem, size).map_err(|_| Stop::Cpu(xcl_emu::cpu::Exit::Fault(Fault { addr: 0, write: true })))
    }

    /// A guest block that exists once per process, made by `init` the first time.
    pub(crate) fn static_block(&mut self, name: &'static str, init: impl FnOnce(&mut Guest) -> Result<u64, Stop>) -> Result<u64, Stop> {
        if let Some(&at) = self.win.statics.get(name) {
            return Ok(at);
        }
        let at = init(self)?;
        self.win.statics.insert(name, at);
        Ok(at)
    }

    /// `text` as NUL-terminated UTF-16 in a new heap block.
    pub(crate) fn alloc_wstr(&mut self, text: &str) -> Result<u64, Stop> {
        let units: Vec<u8> = text.encode_utf16().chain([0]).flat_map(u16::to_le_bytes).collect();
        let at = self.heap_alloc(units.len() as u64)?;
        self.mem.write(at, &units)?;
        Ok(at)
    }

    /// `text` as NUL-terminated bytes in a new heap block.
    pub(crate) fn alloc_cstr(&mut self, text: &str) -> Result<u64, Stop> {
        let at = self.heap_alloc(text.len() as u64 + 1)?;
        self.mem.write(at, text.as_bytes())?;
        Ok(at)
    }

    /// Bytes the title writes to a standard handle.
    pub(crate) fn write_std(&mut self, bytes: &[u8]) {
        self.output.extend_from_slice(bytes);
    }

    /// The title asked to exit.
    pub(crate) fn exit(&mut self, code: u32) -> Stop {
        self.win.exit_code = Some(code);
        Stop::Exit(code)
    }
}
