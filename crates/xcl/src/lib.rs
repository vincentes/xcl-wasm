//! XCL — Xbox Series X|S → Windows compatibility layer.
//!
//! See `docs/specs/2026-11-20-xcl-design.md` for the design.

// Function → address casts are the whole point of an import resolver.
#![allow(function_casts_as_integer)]
// Off Windows only the platform-independent part is built (for the WebAssembly target): what is
// left of a module without its Windows half has helpers nothing calls.
#![cfg_attr(not(windows), allow(dead_code))]

#[macro_use]
pub mod log;
pub mod manifest;
pub mod config;
pub mod disasm;
#[cfg(windows)]
pub mod eh;
pub mod gpu;
pub mod hle;
#[cfg(windows)]
pub mod hooks;
#[cfg(windows)]
pub mod host;
pub mod insn;
pub mod inspect;
pub mod launcher;
#[cfg(windows)]
pub mod loader;
#[cfg(windows)]
pub mod modules;
pub mod pe;
#[cfg(windows)]
pub mod preflight;
#[cfg(windows)]
pub mod report;
#[cfg(windows)]
pub mod resolver;
#[cfg(windows)]
pub mod rpf;
#[cfg(windows)]
mod rpf_probe;
#[cfg(windows)]
pub mod shaders;
pub mod state;
#[cfg(windows)]
pub mod stubs;
#[cfg(windows)]
pub mod tls;
pub mod util;
#[cfg(windows)]
pub mod vfs;
#[cfg(windows)]
pub mod vfs_hooks;
