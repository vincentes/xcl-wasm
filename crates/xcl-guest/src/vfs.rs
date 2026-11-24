//! The title's file system: the game folder (read-only) under an overlay that takes every
//! change, as in XCL's Windows runtime – a write copies, a delete hides, the game's files are
//! never touched. The overlay is in memory; the host puts the game folder in, a file whole
//! (`Vfs::add_game_file`) or as something it reads ranges of when the title does
//! (`Vfs::add_game_source`: a title's archives are far larger than memory), and can take the
//! overlay out (`Vfs::overlay`).
//!
//! Console paths: `G:\`, `GAME:\` and `D:\` are the game folder (so is a relative path), `T:\`
//! the title's scratch storage, `S:\` its persistent local storage, `U:\` the user's data
//! (game saves).

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Node {
    File(Vec<u8>),
    Dir,
    /// hides a file of the game folder
    Deleted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    File(u64),
    Dir,
}

/// A file of the game folder the host reads on demand: `read(offset, buffer)` fills the buffer
/// from that offset of the file (the range is inside the file) and says whether it could.
pub struct Source {
    pub size: u64,
    pub read: Box<dyn Fn(u64, &mut [u8]) -> bool>,
}

#[derive(Default)]
pub struct Vfs {
    /// the game folder: lower-case path → (path as given, contents)
    game: BTreeMap<String, (String, Vec<u8>)>,
    /// … and its files that are read on demand
    sources: BTreeMap<String, (String, Source)>,
    /// what the title changed: lower-case path → (path as given, node)
    over: BTreeMap<String, (String, Node)>,
}

const ROOTS: [&str; 4] = ["g:\\", "t:\\", "s:\\", "u:\\"];

/// The full path a title's path names: `G:\data\hello.txt` for `GAME:/data/./hello.txt`,
/// `\\?\G:\data\hello.txt` or `data\hello.txt`. `None`: not a path of the title's drives.
pub fn resolve(path: &str) -> Option<String> {
    let p = path.replace('/', "\\");
    let p = p.strip_prefix("\\\\?\\").or_else(|| p.strip_prefix("\\??\\")).unwrap_or(&p);
    let (drive, rest) = match p.find(':') {
        Some(i) => (p[..i].to_ascii_uppercase(), &p[i + 1..]),
        None => ("G".to_string(), p),
    };
    let drive = match drive.as_str() {
        "G" | "GAME" | "D" => 'G',
        "T" => 'T',
        "S" => 'S',
        "U" => 'U',
        _ => return None,
    };
    let mut parts: Vec<&str> = Vec::new();
    for part in rest.split('\\') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            name => parts.push(name),
        }
    }
    Some(format!("{drive}:\\{}", parts.join("\\")))
}

fn parent(path: &str) -> &str {
    match path.rfind('\\') {
        Some(i) if i > 2 => &path[..i],
        _ => &path[..3],
    }
}

impl Vfs {
    /// A file of the game folder, by its path in it (`data\hello.txt`).
    pub fn add_game_file(&mut self, relative: &str, data: Vec<u8>) {
        if let Some(path) = resolve(relative) {
            self.game.insert(path.to_ascii_lowercase(), (path, data));
        }
    }

    /// A file of the game folder that is read on demand, by its path in it.
    pub fn add_game_source(&mut self, relative: &str, source: Source) {
        if let Some(path) = resolve(relative) {
            self.sources.insert(path.to_ascii_lowercase(), (path, source));
        }
    }

    /// The size of the file at `path` when it is one that is read on demand (and the title has
    /// not replaced or deleted it).
    pub fn source_size(&self, path: &str) -> Option<u64> {
        let key = path.to_ascii_lowercase();
        if self.over.contains_key(&key) {
            return None;
        }
        self.sources.get(&key).map(|(_, s)| s.size)
    }

    /// Read the file at `path` from `offset` into `buffer`: the number of bytes there were
    /// (fewer than the buffer at the file's end). Works for every file, however it is held.
    pub fn read_at(&self, path: &str, offset: u64, buffer: &mut [u8]) -> Option<usize> {
        let key = path.to_ascii_lowercase();
        let whole = match self.over.get(&key) {
            Some((_, Node::File(d))) => Some(d.as_slice()),
            Some(_) => return None,
            None => self.game.get(&key).map(|(_, d)| d.as_slice()),
        };
        if let Some(data) = whole {
            let from = (offset as usize).min(data.len());
            let n = buffer.len().min(data.len() - from);
            buffer[..n].copy_from_slice(&data[from..from + n]);
            return Some(n);
        }
        let (_, source) = self.sources.get(&key)?;
        let from = offset.min(source.size);
        let n = (buffer.len() as u64).min(source.size - from) as usize;
        (n == 0 || (source.read)(from, &mut buffer[..n])).then_some(n)
    }

    /// A file of the game folder as the host put it in (whatever the title did since).
    pub fn game_file(&self, relative: &str) -> Option<&[u8]> {
        self.game.get(&resolve(relative)?.to_ascii_lowercase()).map(|(_, d)| d.as_slice())
    }

    /// What the title changed: full path → node.
    pub fn overlay(&self) -> impl Iterator<Item = (&str, &Node)> {
        self.over.values().map(|(p, n)| (p.as_str(), n))
    }

    /// Whether anything visible lies below the directory `key` (lower case, no trailing `\`).
    fn has_children(&self, key: &str) -> bool {
        let prefix = if key.ends_with('\\') { key.to_string() } else { format!("{key}\\") };
        let below = |k: &String| k.starts_with(&prefix);
        self.over.range(prefix.clone()..).take_while(|(k, _)| below(k)).any(|(_, (_, n))| *n != Node::Deleted)
            || self.game.range(prefix.clone()..).take_while(|(k, _)| below(k)).any(|(k, _)| !matches!(self.over.get(k), Some((_, Node::Deleted))))
            || self.sources.range(prefix.clone()..).take_while(|(k, _)| below(k)).any(|(k, _)| !matches!(self.over.get(k), Some((_, Node::Deleted))))
    }

    /// What is at `path` (a resolved path).
    pub fn kind(&self, path: &str) -> Option<Kind> {
        let key = path.to_ascii_lowercase();
        if ROOTS.contains(&key.as_str()) {
            return Some(Kind::Dir);
        }
        match self.over.get(&key) {
            Some((_, Node::File(d))) => return Some(Kind::File(d.len() as u64)),
            Some((_, Node::Dir)) => return Some(Kind::Dir),
            Some((_, Node::Deleted)) => return None,
            None => {}
        }
        if let Some((_, d)) = self.game.get(&key) {
            return Some(Kind::File(d.len() as u64));
        }
        if let Some((_, s)) = self.sources.get(&key) {
            return Some(Kind::File(s.size));
        }
        self.has_children(&key).then_some(Kind::Dir)
    }

    pub fn read(&self, path: &str) -> Option<&[u8]> {
        let key = path.to_ascii_lowercase();
        match self.over.get(&key) {
            Some((_, Node::File(d))) => Some(d),
            Some(_) => None,
            None => self.game.get(&key).map(|(_, d)| d.as_slice()),
        }
    }

    /// Write a file (into the overlay). `false`: its folder does not exist, or it is a folder.
    pub fn write(&mut self, path: &str, data: Vec<u8>) -> bool {
        if self.kind(parent(path)) != Some(Kind::Dir) || self.kind(path) == Some(Kind::Dir) {
            return false;
        }
        self.over.insert(path.to_ascii_lowercase(), (path.to_string(), Node::File(data)));
        true
    }

    pub fn mkdir(&mut self, path: &str) -> bool {
        if self.kind(path).is_some() || self.kind(parent(path)) != Some(Kind::Dir) {
            return false;
        }
        self.over.insert(path.to_ascii_lowercase(), (path.to_string(), Node::Dir));
        true
    }

    /// Remove a file, or an empty folder. A file of the game folder is hidden, not removed.
    pub fn remove(&mut self, path: &str) -> bool {
        let key = path.to_ascii_lowercase();
        match self.kind(path) {
            None => return false,
            Some(Kind::Dir) if self.has_children(&key) || ROOTS.contains(&key.as_str()) => return false,
            _ => {}
        }
        if self.game.contains_key(&key) || self.sources.contains_key(&key) {
            self.over.insert(key, (path.to_string(), Node::Deleted));
        } else {
            self.over.remove(&key);
        }
        true
    }

    /// Move a file.
    pub fn rename(&mut self, from: &str, to: &str) -> bool {
        let Some(data) = self.read(from).map(<[u8]>::to_vec) else { return false };
        self.write(to, data) && self.remove(from)
    }

    /// The entries of the folder `path`: (name, kind), by name.
    pub fn list(&self, path: &str) -> Vec<(String, Kind)> {
        let key = path.to_ascii_lowercase();
        let prefix = if key.ends_with('\\') { key } else { format!("{key}\\") };
        let mut names: BTreeMap<String, String> = BTreeMap::new();
        let all = self.game.iter().map(|(k, (p, _))| (k, p)).chain(self.sources.iter().map(|(k, (p, _))| (k, p))).chain(self.over.iter().map(|(k, (p, _))| (k, p)));
        for (k, display) in all.filter(|(k, _)| k.starts_with(&prefix)) {
            let name = display[prefix.len()..].split('\\').next().unwrap_or("");
            if !name.is_empty() {
                names.entry(k[prefix.len()..].split('\\').next().unwrap_or("").to_string()).or_insert_with(|| name.to_string());
            }
        }
        let base = &path[..prefix.len().min(path.len())];
        let base = base.trim_end_matches('\\');
        names.into_values().filter_map(|name| self.kind(&format!("{base}\\{name}")).map(|k| (name, k))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game() -> Vfs {
        let mut v = Vfs::default();
        v.add_game_file("data\\hello.txt", b"hello".to_vec());
        v.add_game_file("data/sub/nested.bin", vec![1, 2]);
        v.add_game_file("Title.exe", vec![0]);
        v
    }

    #[test]
    fn console_paths_name_the_game_folder() {
        assert_eq!(resolve("GAME:/data/./x/../hello.txt").as_deref(), Some("G:\\data\\hello.txt"));
        assert_eq!(resolve("\\\\?\\g:\\data\\hello.txt").as_deref(), Some("G:\\data\\hello.txt"));
        assert_eq!(resolve("data\\hello.txt").as_deref(), Some("G:\\data\\hello.txt"));
        assert_eq!(resolve("D:\\").as_deref(), Some("G:\\"));
        assert_eq!(resolve("t:\\a.tmp").as_deref(), Some("T:\\a.tmp"));
        assert_eq!(resolve("C:\\Windows\\win.ini"), None);
        let v = game();
        assert_eq!(v.kind("G:\\DATA\\Hello.TXT"), Some(Kind::File(5)));
        assert_eq!(v.kind("G:\\data"), Some(Kind::Dir));
        assert_eq!(v.kind("G:\\data\\sub"), Some(Kind::Dir));
        assert_eq!(v.kind("G:\\"), Some(Kind::Dir));
        assert_eq!(v.kind("G:\\nothing"), None);
        assert_eq!(v.read("G:\\data\\sub\\nested.bin"), Some(&[1u8, 2][..]));
    }

    #[test]
    fn a_file_read_on_demand_is_a_file_like_the_others() {
        use std::cell::RefCell;
        use std::rc::Rc;
        let mut v = game();
        // 1 TB that nobody holds: byte i is i modulo 251
        let asked = Rc::new(RefCell::new(Vec::new()));
        let log = asked.clone();
        let read = move |offset: u64, buffer: &mut [u8]| {
            log.borrow_mut().push((offset, buffer.len()));
            for (i, b) in buffer.iter_mut().enumerate() {
                *b = ((offset + i as u64) % 251) as u8;
            }
            true
        };
        v.add_game_source("pack/levels.rpf", Source { size: 1 << 40, read: Box::new(read) });
        assert_eq!(v.kind("G:\\Pack\\LEVELS.rpf"), Some(Kind::File(1 << 40)));
        assert_eq!(v.kind("G:\\pack"), Some(Kind::Dir));
        assert_eq!(v.list("G:\\pack"), [("levels.rpf".to_string(), Kind::File(1 << 40))]);
        assert_eq!((v.source_size("G:\\pack\\levels.rpf"), v.source_size("G:\\data\\hello.txt")), (Some(1 << 40), None));
        assert_eq!(v.read("G:\\pack\\levels.rpf"), None, "not held whole");
        let mut buffer = [0u8; 4];
        let far = (1u64 << 40) - 2;
        assert_eq!(v.read_at("G:\\pack\\levels.rpf", 500, &mut buffer), Some(4));
        assert_eq!(buffer, [(500 % 251) as u8, (501 % 251) as u8, (502 % 251) as u8, (503 % 251) as u8]);
        // at the end there are fewer bytes, past it none: only the range inside the file is asked for
        assert_eq!(v.read_at("G:\\pack\\levels.rpf", far, &mut buffer), Some(2));
        assert_eq!(v.read_at("G:\\pack\\levels.rpf", far + 9, &mut buffer), Some(0));
        assert_eq!(*asked.borrow(), [(500, 4), (far, 2)]);
        // the files held whole are read the same way
        assert_eq!(v.read_at("G:\\data\\hello.txt", 3, &mut buffer), Some(2));
        assert_eq!(&buffer[..2], b"lo");
        assert_eq!(v.read_at("G:\\nothing", 0, &mut buffer), None);
        // deleted, it is hidden like any file of the game folder
        assert!(v.remove("G:\\pack\\levels.rpf"));
        assert_eq!((v.kind("G:\\pack\\levels.rpf"), v.source_size("G:\\pack\\levels.rpf")), (None, None));
        assert_eq!(v.read_at("G:\\pack\\levels.rpf", 0, &mut buffer), None);
    }

    #[test]
    fn changes_go_to_the_overlay_and_the_game_folder_stays() {
        let mut v = game();
        assert!(v.write("G:\\data\\hello.txt", b"changed".to_vec()));
        assert!(v.write("G:\\data\\new.bin", vec![9]));
        assert!(!v.write("G:\\nowhere\\new.bin", vec![9]), "no such folder");
        assert!(v.mkdir("G:\\data\\newdir") && v.mkdir("G:\\data\\newdir\\deeper") && !v.mkdir("G:\\data"));
        assert!(v.rename("G:\\data\\new.bin", "G:\\data\\renamed.bin"));
        assert!(v.remove("G:\\Title.exe") && !v.remove("G:\\Title.exe"));
        assert!(!v.remove("G:\\data"), "not empty");
        assert_eq!(v.read("G:\\data\\hello.txt"), Some(&b"changed"[..]));
        assert_eq!(v.kind("G:\\Title.exe"), None);
        assert_eq!(v.kind("G:\\data\\new.bin"), None);
        let names: Vec<String> = v.list("G:\\data").into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["hello.txt", "newdir", "renamed.bin", "sub"]);
        assert_eq!(v.list("G:\\"), [("data".to_string(), Kind::Dir)]);
        // the game folder itself is as it was
        assert_eq!(v.game.get("g:\\data\\hello.txt").map(|(_, d)| d.as_slice()), Some(&b"hello"[..]));
        assert!(v.game.contains_key("g:\\title.exe"));
        assert!(v.overlay().any(|(p, n)| p == "G:\\Title.exe" && *n == Node::Deleted));
        // scratch storage starts empty and takes files
        assert!(v.list("T:\\").is_empty());
        assert!(v.write("T:\\scratch.tmp", vec![1]));
        assert_eq!(v.kind("T:\\scratch.tmp"), Some(Kind::File(1)));
    }
}
