//! A title running on the x86-64 guest (`xcl-emu`): its image loaded into guest memory, and the
//! functions it imports answered by the host (`win`).
//!
//! Trap slots (see `xcl_emu::cpu`): slot 0 is where the entry point returns, slot 1 where a
//! guest function called by the host returns (`Guest::call`), slot 2 where a thread's start
//! function returns, the rest are the title's imports (`Guest::imports`, slot = index +
//! `FIRST_IMPORT_SLOT`).
//!
//! Threads are the guest's own (`Thread`): one runs at a time, for a slice of instructions or
//! until it waits. A system function that has to wait returns `Stop::Wait` without finishing the
//! call; the thread stays at the import and the function is asked again at its next turn.

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
/// where a thread's start function returns
pub const SLOT_THREAD_RETURN: u32 = 2;
pub const FIRST_IMPORT_SLOT: u32 = 3;

/// Instructions a thread runs before the next one gets its turn.
const SLICE: u64 = 20_000;

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
    /// Every thread waits for something that cannot happen any more.
    Deadlock,
    /// (between a system function and the scheduler) the calling thread has to wait: ask again
    /// at its next turn
    Wait,
    /// (between a system function and the scheduler) the function has set the running thread's
    /// registers, an unwind for one: go on from there, in the `run` loop of this nesting depth
    Resume { depth: u32 },
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
    /// its function table (`.pdata`): address and number of RUNTIME_FUNCTION entries
    pub pdata: u64,
    pub pdata_count: u32,
}

/// What a thread waits for (kept while its system function answers `Stop::Wait`).
#[derive(Debug, Clone, Default)]
pub struct Wait {
    /// guest time (`Guest::now_ns`) at which the wait gives up; `None`: never
    pub deadline: Option<u64>,
    /// WaitOnAddress: the address waited on
    pub address: u64,
    /// WakeByAddress* reached this thread
    pub woken: bool,
}

/// A guest thread. The registers of the running one are in `Guest::cpu`.
pub struct Thread {
    pub id: u32,
    /// SetThreadName
    pub name: String,
    /// its registers while another thread runs
    cpu: Cpu,
    pub teb: u64,
    pub last_error: u32,
    /// TlsAlloc / FlsAlloc values
    pub tls_slots: Vec<u64>,
    pub wait: Option<Wait>,
    /// set once its start function returned (or it called ExitThread)
    pub exit_code: Option<u32>,
    /// handler calls in progress, innermost last (see `win::seh`)
    pub seh: Vec<win::seh::Boundary>,
}

pub struct Guest {
    /// the registers of the running thread (`threads[current]`)
    pub cpu: Cpu,
    pub threads: Vec<Thread>,
    pub current: usize,
    pub mem: Memory,
    pub modules: Vec<Module>,
    pub imports: Vec<Import>,
    import_slots: HashMap<String, u32>,
    handlers: HashMap<&'static str, Handler>,
    /// what GetCommandLineW answers
    pub command_line: String,
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
            threads: Vec::new(),
            current: 0,
            mem: Memory::new(),
            modules: Vec::new(),
            imports: Vec::new(),
            import_slots: HashMap::new(),
            handlers: win::handlers(),
            command_line: "title.exe".into(),
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

    pub fn thread(&self) -> &Thread {
        &self.threads[self.current]
    }

    pub fn thread_mut(&mut self) -> &mut Thread {
        &mut self.threads[self.current]
    }

    /// A new thread on the stack `[stack, top)`, about to call `entry(argument)`. Its index.
    pub fn add_thread(&mut self, stack: u64, top: u64, entry: u64, argument: u64, return_slot: u32) -> Result<usize, Fault> {
        let mut cpu = Cpu::new();
        cpu.features = self.cpu.features;
        cpu.set_traps(TRAP_BASE, TRAP_SLOTS);
        let id = 0x1004 + 4 * self.threads.len() as u32;
        let teb = self.win.setup_thread(&mut cpu, &mut self.mem, stack, top, id)?;
        cpu.r[xcl_emu::cpu::RSP] = top - 0x40;
        let ret = cpu.trap_address(return_slot);
        abi::call(&mut cpu, &mut self.mem, entry, &[argument], ret)?;
        self.threads.push(Thread { id, name: String::new(), cpu, teb, last_error: 0, tls_slots: Vec::new(), wait: None, exit_code: None, seh: Vec::new() });
        Ok(self.threads.len() - 1)
    }

    /// Make thread `to` the running one.
    fn switch_to(&mut self, to: usize) {
        if to != self.current {
            self.threads[self.current].cpu = self.cpu.clone();
            self.cpu = self.threads[to].cpu.clone();
            self.current = to;
        }
    }

    /// The next thread that has not ended, after the running one (itself when it is the only one).
    fn next_thread(&self) -> Option<usize> {
        let n = self.threads.len();
        (1..=n).map(|k| (self.current + k) % n).find(|&i| self.threads[i].exit_code.is_none())
    }

    /// Run the title: at most `max_instructions`, until it stops.
    pub fn run(&mut self, max_instructions: u64) -> Stop {
        self.run_until(0, SLOT_ENTRY_RETURN, max_instructions)
    }

    /// Call the guest function at `entry` from the host, on the running thread, and return its
    /// RAX. Other threads get their turns while it runs.
    pub fn call(&mut self, entry: u64, args: &[u64]) -> Result<u64, Stop> {
        let thread = self.current;
        let saved = (self.cpu.r[xcl_emu::cpu::RSP], self.cpu.rip);
        // below the frame of the function that called the host
        self.cpu.r[xcl_emu::cpu::RSP] = saved.0.wrapping_sub(0x100);
        let ret = self.cpu.trap_address(SLOT_CALL_RETURN);
        abi::call(&mut self.cpu, &mut self.mem, entry, args, ret)?;
        self.win.call_depth += 1;
        let stop = self.run_until(thread, SLOT_CALL_RETURN, u64::MAX);
        self.win.call_depth -= 1;
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

    /// Run the threads until thread `thread` returns to trap slot `until` (reported as
    /// `Stop::Exit(eax)`, with that thread running) or the title stops.
    fn run_until(&mut self, thread: usize, until: u32, max_instructions: u64) -> Stop {
        let mut left = max_instructions;
        // turns in a row in which a thread could only wait
        let mut idle = 0usize;
        loop {
            let before = self.cpu.icount;
            let exit = self.cpu.run(&mut self.mem, left.min(SLICE));
            let ran = self.cpu.icount - before;
            self.win.time_ns += ran;
            left = left.saturating_sub(ran);
            if ran != 0 {
                idle = 0;
            }
            let waits = match exit {
                Exit::Limit => {
                    if left == 0 {
                        return Stop::Limit;
                    }
                    false
                }
                Exit::Import(slot) if slot == until && self.current == thread => return Stop::Exit(self.cpu.r[xcl_emu::cpu::RAX] as u32),
                // a host call of another thread ended, inside this one's: it goes on once this
                // one's has returned
                Exit::Import(SLOT_CALL_RETURN) | Exit::Import(SLOT_ENTRY_RETURN) => true,
                Exit::Import(SLOT_THREAD_RETURN) => {
                    let code = self.cpu.r[xcl_emu::cpu::RAX] as u32;
                    self.thread_mut().exit_code = Some(code);
                    idle = 0;
                    false
                }
                Exit::Import(slot) if ((slot - FIRST_IMPORT_SLOT) as usize) < self.imports.len() => match self.import_call((slot - FIRST_IMPORT_SLOT) as usize) {
                    Ok(()) => {
                        idle = 0;
                        // keep the slice: a thread is not switched out at every call
                        continue;
                    }
                    Err(Stop::Wait) => true,
                    Err(Stop::Resume { depth }) if depth == self.win.call_depth => {
                        idle = 0;
                        continue;
                    }
                    Err(stop) => return stop,
                },
                other => return Stop::Cpu(other),
            };
            if waits {
                idle += 1;
                let live = self.threads.iter().filter(|t| t.exit_code.is_none()).count();
                if idle > live {
                    // nobody can run: time passes until the first wait gives up
                    let now = self.now_ns();
                    match self.threads.iter().filter(|t| t.exit_code.is_none()).filter_map(|t| t.wait.as_ref()?.deadline).min() {
                        Some(deadline) => self.win.time_ns += deadline.saturating_sub(now),
                        None => return Stop::Deadlock,
                    }
                    idle = 0;
                }
            }
            match self.next_thread() {
                Some(next) => self.switch_to(next),
                None => return Stop::Deadlock,
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
