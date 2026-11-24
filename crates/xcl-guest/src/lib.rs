//! A title running on the x86-64 guest (`xcl-emu`): its image loaded into guest memory, and the
//! functions it imports answered by the host (`win`).
//!
//! Trap slots (see `xcl_emu::cpu`): slot 0 is where the entry point returns, slot 1 where a
//! guest function called by the host returns (`Guest::call`), the rest are the title's imports
//! (`Guest::imports`, slot = index + `FIRST_IMPORT_SLOT`).

use std::collections::HashMap;

use xcl_emu::abi;
use xcl_emu::cpu::{Cpu, Exit};
use xcl_emu::mem::{Fault, Memory};

pub mod loader;
pub mod win;

pub const TRAP_BASE: u64 = 0x0000_7FFE_0000_0000;
pub const TRAP_SLOTS: u32 = 1 << 20;
pub const SLOT_ENTRY_RETURN: u32 = 0;
pub const SLOT_CALL_RETURN: u32 = 1;
pub const FIRST_IMPORT_SLOT: u32 = 2;

/// Why the title stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stop {
    /// It exited (ExitProcess, exit, or a return from the entry point) with this code.
    Exit(u32),
    /// It called a function the host does not implement (strict mode).
    Unimplemented { dll: String, name: String },
    /// The CPU stopped: a fault, an instruction the interpreter lacks …
    Cpu(Exit),
    /// The instruction budget of `run` is used up; `run` again to go on.
    Limit,
}

impl From<Fault> for Stop {
    fn from(f: Fault) -> Self {
        Stop::Cpu(Exit::Fault(f))
    }
}

/// An imported function: where the title's import table points.
#[derive(Debug, Clone)]
pub struct Import {
    pub dll: String,
    pub name: String,
    pub calls: u64,
}

/// A function the host implements: reads its arguments from the guest (`Guest::arg`) and
/// returns the value for RAX.
pub type Handler = fn(&mut Guest) -> Result<u64, Stop>;

/// A mapped PE image.
#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub base: u64,
    pub size: u64,
    pub entry: u64,
}

pub struct Guest {
    pub cpu: Cpu,
    pub mem: Memory,
    pub modules: Vec<Module>,
    pub imports: Vec<Import>,
    import_slots: HashMap<String, u32>,
    handlers: HashMap<&'static str, Handler>,
    /// stop at a function the host does not implement (otherwise: log it once, return 0)
    pub strict: bool,
    /// what the title wrote to its standard output and error
    pub output: Vec<u8>,
    /// the host's own notes (unimplemented functions, debug strings)
    pub log: Vec<String>,
    /// the last `trace_len` import calls, oldest first (0: off)
    pub trace: std::collections::VecDeque<String>,
    pub trace_len: usize,
    pub win: win::State,
}

impl Default for Guest {
    fn default() -> Self {
        Self::new()
    }
}

impl Guest {
    pub fn new() -> Self {
        let mut cpu = Cpu::new();
        cpu.set_traps(TRAP_BASE, TRAP_SLOTS);
        Guest {
            cpu,
            mem: Memory::new(),
            modules: Vec::new(),
            imports: Vec::new(),
            import_slots: HashMap::new(),
            handlers: win::handlers(),
            strict: false,
            output: Vec::new(),
            log: Vec::new(),
            trace: Default::default(),
            trace_len: 0,
            win: win::State::default(),
        }
    }

    /// The trap address standing for `dll!name` (one slot per function name and DLL).
    pub fn import_address(&mut self, dll: &str, name: &str) -> u64 {
        let key = format!("{}!{name}", dll.to_ascii_lowercase());
        let slot = match self.import_slots.get(&key) {
            Some(&s) => s,
            None => {
                let s = FIRST_IMPORT_SLOT + self.imports.len() as u32;
                self.imports.push(Import { dll: dll.to_string(), name: name.to_string(), calls: 0 });
                self.import_slots.insert(key, s);
                s
            }
        };
        self.cpu.trap_address(slot)
    }

    /// Whether the host implements a function of this name.
    pub fn has_handler(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// Integer argument `i` of the import call being answered.
    pub fn arg(&self, i: usize) -> Result<u64, Stop> {
        Ok(abi::Args::new(&self.cpu, &self.mem).get(i)?)
    }

    /// Run the title: at most `max_instructions`, until it stops.
    pub fn run(&mut self, max_instructions: u64) -> Stop {
        self.run_until(SLOT_ENTRY_RETURN, max_instructions)
    }

    /// Call the guest function at `entry` from the host and return its RAX.
    pub fn call(&mut self, entry: u64, args: &[u64]) -> Result<u64, Stop> {
        let saved = (self.cpu.r[xcl_emu::cpu::RSP], self.cpu.rip);
        // below the caller's frame (and its red-zone-free Win64 stack)
        self.cpu.r[xcl_emu::cpu::RSP] = saved.0.wrapping_sub(0x100);
        let ret = self.cpu.trap_address(SLOT_CALL_RETURN);
        abi::call(&mut self.cpu, &mut self.mem, entry, args, ret)?;
        let depth = self.win.call_depth;
        self.win.call_depth += 1;
        let stop = self.run_until(SLOT_CALL_RETURN, u64::MAX);
        self.win.call_depth = depth;
        match stop {
            Stop::Exit(_) if self.win.exit_code.is_none() => {
                let v = self.cpu.r[xcl_emu::cpu::RAX];
                self.cpu.r[xcl_emu::cpu::RSP] = saved.0;
                self.cpu.rip = saved.1;
                Ok(v)
            }
            other => Err(other),
        }
    }

    /// Run until the guest returns to `until` (reported as `Stop::Exit(eax)`) or stops.
    fn run_until(&mut self, until: u32, max_instructions: u64) -> Stop {
        let mut left = max_instructions;
        loop {
            let before = self.cpu.icount;
            let exit = self.cpu.run(&mut self.mem, left.min(1 << 24));
            left = left.saturating_sub(self.cpu.icount - before);
            match exit {
                Exit::Limit => {
                    if left == 0 {
                        return Stop::Limit;
                    }
                }
                Exit::Import(slot) if slot == until => return Stop::Exit(self.cpu.r[xcl_emu::cpu::RAX] as u32),
                Exit::Import(slot) if slot >= FIRST_IMPORT_SLOT && ((slot - FIRST_IMPORT_SLOT) as usize) < self.imports.len() => {
                    if let Err(stop) = self.import_call((slot - FIRST_IMPORT_SLOT) as usize) {
                        return stop;
                    }
                }
                other => return Stop::Cpu(other),
            }
        }
    }

    fn import_call(&mut self, index: usize) -> Result<(), Stop> {
        self.imports[index].calls += 1;
        let (dll, name) = (self.imports[index].dll.clone(), self.imports[index].name.clone());
        if self.trace_len != 0 {
            if self.trace.len() == self.trace_len {
                self.trace.pop_front();
            }
            let a = |i| self.arg(i).unwrap_or(0);
            self.trace.push_back(format!("{dll}!{name}({:#x}, {:#x}, {:#x}, {:#x})", a(0), a(1), a(2), a(3)));
        }
        let value = match self.handlers.get(name.as_str()) {
            Some(&handler) => handler(self)?,
            None if self.strict => return Err(Stop::Unimplemented { dll, name }),
            None => {
                if self.imports[index].calls == 1 {
                    self.log.push(format!("[STUB] {dll}!{name} → 0"));
                }
                0
            }
        };
        abi::ret(&mut self.cpu, &self.mem, value)?;
        Ok(())
    }
}
