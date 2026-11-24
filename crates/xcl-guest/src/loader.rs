//! Maps a PE image into guest memory: sections, relocations, imports as trap slots, static TLS
//! and the thread's stack and TEB.

use xcl::pe::{ImportName, Layout, Pe};
use xcl_emu::abi;
use xcl_emu::cpu::RSP;
use xcl_emu::mem::{page_up, PROT_R, PROT_W, PROT_X};

use crate::{Guest, Module, SLOT_ENTRY_RETURN};

const IMAGE_REL_BASED_DIR64: u16 = 10;

impl Guest {
    /// Map the image `data` (a PE file) and bind its imports. Returns its index in `modules`.
    pub fn map_image(&mut self, name: &str, data: &[u8]) -> Result<usize, String> {
        let pe = Pe::parse(data, Layout::File).map_err(|e| e.to_string())?;
        let size = page_up(pe.size_of_image as u64);
        let mut base = pe.image_base;
        if self.mem.map(base, size, PROT_R | PROT_W | PROT_X).is_err() {
            if !pe.has_relocs() {
                return Err(format!("{name}: {base:#x} is taken and the image has no relocations"));
            }
            base = self.mem.alloc(size, PROT_R | PROT_W | PROT_X).map_err(|e| format!("{name}: {e:?}"))?;
        }
        let headers = (pe.size_of_headers as usize).min(data.len());
        self.mem.write(base, &data[..headers]).map_err(|e| format!("{name}: headers: {e:?}"))?;
        for s in &pe.sections {
            let raw = s.raw_size.min(s.mapped_size()) as usize;
            let bytes = data.get(s.raw_ptr as usize..s.raw_ptr as usize + raw).ok_or_else(|| format!("{name}: section {} is truncated", s.name))?;
            self.mem.write(base + s.va as u64, bytes).map_err(|e| format!("{name}: section {}: {e:?}", s.name))?;
        }
        let delta = base.wrapping_sub(pe.image_base);
        if delta != 0 {
            for r in pe.relocations().map_err(|e| e.to_string())? {
                if r.kind == IMAGE_REL_BASED_DIR64 {
                    let at = base + r.rva as u64;
                    let v = self.mem.read_u64(at).map_err(|e| format!("{name}: relocation: {e:?}"))?;
                    self.mem.write_u64(at, v.wrapping_add(delta)).map_err(|e| format!("{name}: relocation: {e:?}"))?;
                }
            }
        }
        let mut all = pe.imports().map_err(|e| e.to_string())?;
        all.extend(pe.delay_imports().unwrap_or_default());
        for dll in all.iter().filter(|d| !d.delay) {
            for e in &dll.entries {
                let function = match &e.what {
                    ImportName::Name { name, .. } => name.clone(),
                    ImportName::Ordinal(o) => format!("#{o}"),
                };
                let trap = self.import_address(&dll.dll, &function);
                self.mem.write_u64(base + e.iat_rva as u64, trap).map_err(|e| format!("{name}: import table: {e:?}"))?;
            }
        }
        if let Some(tls) = pe.tls().map_err(|e| e.to_string())? {
            let r = |va: u64| va.wrapping_add(delta);
            self.win.tls_templates.push(crate::win::TlsTemplate {
                start: r(tls.start_raw),
                len: tls.template_len(),
                zero_fill: tls.size_of_zero_fill as u64,
                index_address: r(tls.address_of_index),
                callbacks: if tls.address_of_callbacks != 0 { r(tls.address_of_callbacks) } else { 0 },
                module_base: base,
            });
        }
        self.modules.push(Module { name: name.to_string(), base, size, entry: if pe.entry_rva != 0 { base + pe.entry_rva as u64 } else { 0 } });
        Ok(self.modules.len() - 1)
    }

    /// Give the first thread its stack, TEB and static TLS, and point it at the entry of module
    /// `index` (the executable). `Guest::run` then starts the title.
    pub fn start(&mut self, index: usize, stack_size: u64) -> Result<(), String> {
        let f = |e| format!("thread setup: {e:?}");
        let stack = self.mem.alloc(page_up(stack_size), PROT_R | PROT_W).map_err(f)?;
        let top = stack + page_up(stack_size);
        self.win.setup_thread(&mut self.cpu, &mut self.mem, stack, top).map_err(|e| format!("thread setup: {e:?}"))?;
        self.cpu.r[RSP] = top - 0x40;
        let entry = self.modules[index].entry;
        let ret = self.cpu.trap_address(SLOT_ENTRY_RETURN);
        abi::call(&mut self.cpu, &mut self.mem, entry, &[], ret).map_err(|e| format!("thread setup: {e:?}"))?;
        // TLS callbacks run before the entry point (DLL_PROCESS_ATTACH)
        let entry_state = (self.cpu.r[RSP], self.cpu.rip);
        for t in self.win.tls_templates.clone() {
            let mut at = t.callbacks;
            while at != 0 {
                let callback = self.mem.read_u64(at).map_err(|e| format!("TLS callbacks: {e:?}"))?;
                if callback == 0 {
                    break;
                }
                self.call(callback, &[t.module_base, 1, 0]).map_err(|e| format!("TLS callback {callback:#x}: {e:?}"))?;
                at += 8;
            }
        }
        (self.cpu.r[RSP], self.cpu.rip) = entry_state;
        Ok(())
    }
}
