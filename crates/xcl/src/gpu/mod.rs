//! GPU side of XCL: console shader capture and the RDNA2 microcode toolchain
//! (decoder / disassembler – the front-end of the shader recompiler).

#[cfg(windows)]
pub mod adapter;
#[cfg(windows)]
pub mod agility;
pub mod bvh;
#[cfg(windows)]
pub mod desc;
pub mod dxbc;
#[cfg(windows)]
pub(crate) mod fence;
pub mod pack;
pub mod rdna2;
pub mod recomp;
mod rdna2_spec;
mod rdna2_xbox;
pub mod shader;
#[cfg(windows)]
pub mod texture;
pub mod tiling;
mod tiling_tables;
#[cfg(windows)]
pub mod uma;
pub mod xbox;

pub use rdna2_spec::ARCH as RDNA2_ARCH;
