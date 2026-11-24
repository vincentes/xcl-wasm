//! The x86-64 guest: where XCL cannot run the game's code natively (in WebAssembly), the code
//! runs here. Three things the rest of the port builds on:
//!
//! * [`mem::Memory`] – the guest's address space. A guest address is a `u64` and never a host
//!   pointer: everything that touches title memory goes through `Memory`.
//! * [`cpu::Cpu`] – the register state of one guest thread, and [`cpu::Cpu::run`], which executes
//!   until the guest leaves ([`cpu::Exit`]).
//! * Import calls: the loader points every import at a slot of the *trap range*
//!   ([`cpu::Cpu::set_traps`]). A call that lands there ends `run` with [`cpu::Exit::Import`]; the
//!   host reads the arguments ([`abi::Args`], the Win64 convention), does the work, and returns
//!   with [`abi::ret`].

pub mod abi;
mod avx;
pub mod cpu;
mod exec;
pub mod mem;
mod sse;
