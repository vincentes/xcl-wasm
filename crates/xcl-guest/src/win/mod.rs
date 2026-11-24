//! The system functions a title imports, answered by the host. A function is found by its name
//! alone (as XCL does for console functions), whichever DLL the title names for it.

use std::collections::HashMap;

use xcl_emu::cpu::Cpu;
use xcl_emu::mem::{Fault, Memory, PROT_R, PROT_W};

use crate::{Guest, Handler, Stop};

mod crt;
pub mod d3d12_null;
pub mod files;
pub mod gdk;
pub mod heap;
mod kernel32;
mod kernelx;
pub mod seh;
mod sync;
mod text;
mod threadpool;
mod user;
pub mod xgameruntime;

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
    pub xmem: kernelx::XMem,
    pub window_classes: u32,
    pub xgr: xgameruntime::State,
    pub d3d: d3d12_null::State,
    pub pool: threadpool::Pool,
    /// the GDK runtime's interface objects handed out, by (class, interface)
    pub gdk: HashMap<(u128, u128), u64>,
    /// COM objects the host made, by interface: one vtable each
    pub vtables: HashMap<&'static str, u64>,
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
    File(files::File),
    /// the console's screen
    Window,
    Find(files::Find),
    Semaphore { count: u32, max: u32 },
    /// owned by the thread with this id (0: nobody), `count` times
    Mutex { owner: u32, count: u32 },
}

impl State {
    /// Static TLS of a module that was mapped. The executable's is index 0, as on Windows where
    /// it is the first module loaded: code built to know that (MSVC's /GA) does not read its
    /// index. Only before the first thread: a thread's blocks are in the order of the indexes.
    pub fn add_tls_template(&mut self, template: TlsTemplate, is_dll: bool) {
        if is_dll {
            self.tls_templates.push(template);
        } else {
            self.tls_templates.insert(0, template);
        }
    }

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
    seh::register(&mut map);
    sync::register(&mut map);
    threadpool::register(&mut map);
    // after kernel32: names for functions it has
    text::register(&mut map);
    user::register(&mut map);
    xgameruntime::register(&mut map);
    gdk::register(&mut map);
    d3d12_null::register(&mut map);
    // after kernel32 and crt: the file functions also take the standard handles
    files::register(&mut map);
    // after kernel32: VirtualFree also knows the graphics region
    kernelx::register(&mut map);
    map
}

pub(crate) const ERROR_INVALID_HANDLE: u32 = 6;
pub(crate) const ERROR_INVALID_PARAMETER: u32 = 87;
pub(crate) const ERROR_NOT_OWNER: u32 = 288;
pub(crate) const ERROR_TOO_MANY_POSTS: u32 = 298;
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

    /// A COM object in guest memory whose methods are the host functions `methods` (by the
    /// names they are registered under), in vtable order. The object is 16 bytes: the vtable
    /// pointer and a word for the host.
    pub(crate) fn com_object(&mut self, interface: &'static str, methods: &[&str]) -> Result<u64, Stop> {
        let vtable = match self.win.vtables.get(interface) {
            Some(&v) => v,
            None => {
                let v = self.heap_alloc(8 * methods.len() as u64)?;
                for (i, m) in methods.iter().enumerate() {
                    let slot = self.import_address("@com", m);
                    self.mem.write_u64(v + 8 * i as u64, slot)?;
                }
                self.win.vtables.insert(interface, v);
                v
            }
        };
        let object = self.heap_alloc(16)?;
        self.mem.write_u64(object, vtable)?;
        Ok(object)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_executables_static_tls_is_index_0_whatever_was_mapped_before_it() {
        let mut mem = Memory::new();
        let data = mem.alloc(0x1000, PROT_R | PROT_W).unwrap();
        let stack = mem.alloc(0x4000, PROT_R | PROT_W).unwrap();
        mem.write_u64(data + 0x100, 0xD11).unwrap();
        mem.write_u64(data + 0x200, 0xE8E).unwrap();
        let template = |start: u64, index_address: u64, module_base: u64| TlsTemplate { start, len: 8, zero_fill: 0x2000, index_address, callbacks: 0, module_base };
        let mut state = State::default();
        // a DLL the executable imports is mapped first
        state.add_tls_template(template(data + 0x100, data, 0x1_8000_0000), true);
        state.add_tls_template(template(data + 0x200, data + 8, 0x1_4000_0000), false);
        mem.write_u32(data + 8, 77).unwrap();
        let mut cpu = Cpu::new();
        let teb = state.setup_thread(&mut cpu, &mut mem, stack, stack + 0x4000, 1).unwrap();
        let slots = mem.read_u64(teb + 0x58).unwrap();
        let (first, second) = (mem.read_u64(slots).unwrap(), mem.read_u64(slots + 8).unwrap());
        assert_eq!((mem.read_u32(data + 8).unwrap(), mem.read_u32(data).unwrap()), (0, 1));
        assert_eq!((mem.read_u64(first).unwrap(), mem.read_u64(second).unwrap()), (0xE8E, 0xD11));
        // … and slot 0 is the executable's whole block, zero fill included
        assert_eq!(mem.read_u64(first + 0x2000).unwrap(), 0);
    }
}
