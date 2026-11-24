//! Maps a PE image into guest memory: sections, relocations, imports as trap slots, static TLS
//! and the thread's stack and TEB.

use xcl::pe::{ImportName, Layout, Pe};
use xcl_emu::cpu::RSP;
use xcl_emu::mem::{page_up, PROT_R, PROT_W, PROT_X};

use crate::{Guest, Module, SLOT_ENTRY_RETURN};

const IMAGE_REL_BASED_DIR64: u16 = 10;

/// The name of a function a title imports by its number: the host finds its functions by name.
fn ordinal_name(dll: &str, ordinal: u16) -> Option<&'static str> {
    match (dll.to_ascii_lowercase().as_str(), ordinal) {
        ("gameinput.dll", 1) => Some("GameInputCreate"),
        _ => None,
    }
}

impl Guest {
    /// Map the image `data` (a PE file) and bind its imports. Returns its index in `modules`.
    pub fn map_image(&mut self, name: &str, data: &[u8]) -> Result<usize, String> {
        let pe = Pe::parse(data, Layout::File).map_err(|e| e.to_string())?;
        // the executable is a file of the game folder, as on the console: a title opens its own
        // image (to read resources appended to it, or just as a file that is surely there)
        if !pe.is_dll() && self.vfs.game_file(name).is_none() {
            self.vfs.add_game_file(name, data.to_vec());
        }
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
        let all = pe.imports().map_err(|e| e.to_string())?;
        for dll in &all {
            // a DLL shipped in the game folder is mapped and its exports bound; everything else
            // is the host's (a trap slot per function)
            let game_dll = self.game_dll(&dll.dll)?;
            for e in &dll.entries {
                let function = match &e.what {
                    ImportName::Name { name, .. } => name.clone(),
                    ImportName::Ordinal(o) => ordinal_name(&dll.dll, *o).map_or_else(|| format!("#{o}"), str::to_string),
                };
                let target = match game_dll.and_then(|m| self.modules[m].exports.get(&function).copied()) {
                    Some(address) => address,
                    None => self.import_address(&dll.dll, &function),
                };
                self.mem.write_u64(base + e.iat_rva as u64, target).map_err(|e| format!("{name}: import table: {e:?}"))?;
            }
        }
        let mut exports = std::collections::HashMap::new();
        if let Some(table) = pe.exports().map_err(|e| e.to_string())? {
            for (export, index) in &table.names {
                let rva = table.functions.get(*index as usize).copied().unwrap_or(0);
                if rva != 0 && !table.is_forwarder_rva(rva) {
                    exports.insert(export.clone(), base + rva as u64);
                }
            }
        }
        if let Some(tls) = pe.tls().map_err(|e| e.to_string())? {
            let r = |va: u64| va.wrapping_add(delta);
            let template = crate::win::TlsTemplate {
                start: r(tls.start_raw),
                len: tls.template_len(),
                zero_fill: tls.size_of_zero_fill as u64,
                index_address: r(tls.address_of_index),
                callbacks: if tls.address_of_callbacks != 0 { r(tls.address_of_callbacks) } else { 0 },
                module_base: base,
            };
            self.win.add_tls_template(template, pe.is_dll());
        }
        let pdata = pe.dir(xcl::pe::DIR_EXCEPTION);
        self.modules.push(Module {
            name: name.to_string(),
            base,
            size,
            entry: if pe.entry_rva != 0 { base + pe.entry_rva as u64 } else { 0 },
            is_dll: pe.is_dll(),
            exports,
            pdata: base + pdata.rva as u64,
            pdata_count: pdata.size / 12,
        });
        Ok(self.modules.len() - 1)
    }

    /// The module of the game folder's DLL `dll`, mapped now if it was not: its index, or
    /// `None` when the game folder has no such file.
    pub fn game_dll(&mut self, dll: &str) -> Result<Option<usize>, String> {
        let file = dll.rsplit(['\\', '/']).next().unwrap_or(dll);
        if let Some(i) = self.modules.iter().position(|m| m.name.eq_ignore_ascii_case(file)) {
            return Ok(Some(i));
        }
        let Some(data) = crate::vfs::resolve(file).and_then(|p| self.vfs.read(&p).map(<[u8]>::to_vec)) else { return Ok(None) };
        if self.loading.iter().any(|l| l.eq_ignore_ascii_case(file)) {
            return Err(format!("{file}: DLLs that import each other are not supported"));
        }
        self.loading.push(file.to_string());
        let mapped = self.map_image(file, &data);
        self.loading.pop();
        mapped.map(Some)
    }

    /// Give the first thread its stack, TEB and static TLS, and point it at the entry of module
    /// `index` (the executable). `Guest::run` then starts the title.
    pub fn start(&mut self, index: usize, stack_size: u64) -> Result<(), String> {
        let f = |e| format!("thread setup: {e:?}");
        let stack = self.mem.alloc(page_up(stack_size), PROT_R | PROT_W).map_err(f)?;
        let top = stack + page_up(stack_size);
        self.exe = index;
        let entry = self.modules[index].entry;
        let main = self.add_thread(stack, top, entry, 0, SLOT_ENTRY_RETURN).map_err(|e| format!("thread setup: {e:?}"))?;
        self.current = main;
        self.cpu = self.threads[main].cpu.clone();
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
        // … and the game DLLs' entry points (DllMain, DLL_PROCESS_ATTACH), in the order mapped
        for m in self.modules.clone().into_iter().filter(|m| m.is_dll && m.entry != 0) {
            self.call(m.entry, &[m.base, 1, 0]).map_err(|e| format!("{}: entry point: {e:?}", m.name))?;
        }
        (self.cpu.r[RSP], self.cpu.rip) = entry_state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_function_imported_by_number_has_its_name() {
        assert_eq!(ordinal_name("GameInput.dll", 1), Some("GameInputCreate"));
        assert_eq!(ordinal_name("GameInput.dll", 2), None);
        // the number alone says nothing: another DLL's #1 is another function
        assert_eq!(ordinal_name("xaudio2_9.dll", 1), None);
        assert!(Guest::new().has_handler("GameInputCreate"));
    }
}
