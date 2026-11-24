//! A title that keeps running: the page starts it, then gives it the time and a budget of
//! instructions again and again (`xcl_session_step`), showing its frames and playing its sound
//! in between. The guest runs in real time: the page's clock is its clock, and when every
//! thread waits the step says until when.

use std::cell::RefCell;

use xcl_guest::{Guest, Stop};

use super::{answer, bytes, FRAME, GAME};

struct Session {
    guest: Guest,
    /// bytes of the title's output and lines of the host's log that were handed out
    printed: usize,
    logged: usize,
    ended: bool,
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
    /// sound of the session that was not taken yet, then what `xcl_session_audio` handed out
    static AUDIO: RefCell<(Vec<i16>, Vec<i16>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// Start a title (a PE file's bytes) with the game folder of `xcl_game_file`; a session that
/// was running is dropped.
///
/// # Safety
/// `name` (UTF-8), `data` and `command_line` (UTF-8; empty: the name): blocks from `xcl_alloc`.
#[no_mangle]
pub unsafe extern "C" fn xcl_session_start(name: *const u8, name_len: usize, data: *const u8, data_len: usize, command_line: *const u8, command_line_len: usize) -> u32 {
    let name = String::from_utf8_lossy(bytes(name, name_len)).into_owned();
    let command_line = String::from_utf8_lossy(bytes(command_line, command_line_len)).into_owned();
    let mut guest = Guest::new();
    guest.real_time = true;
    guest.command_line = if command_line.is_empty() { name.clone() } else { command_line };
    for (path, data) in GAME.lock().unwrap().iter() {
        guest.vfs.add_game_file(path, data.clone());
    }
    let started = guest.map_image(&name, bytes(data, data_len)).and_then(|exe| guest.start(exe, 1 << 20));
    *FRAME.lock().unwrap() = None;
    AUDIO.with_borrow_mut(|a| *a = (Vec::new(), Vec::new()));
    match started {
        Ok(()) => {
            SESSION.set(Some(Session { guest, printed: 0, logged: 0, ended: false }));
            answer(Ok("started".into()))
        }
        Err(e) => {
            SESSION.set(None);
            answer(Err(e))
        }
    }
}

/// Let the session run: the page's clock is `now_ns` (nanoseconds since it started the
/// session), and at most `max_instructions` are executed. The answer's first line says how it
/// stands – `running` (the budget is used up), `idle <ns>` (every thread waits; the first wait
/// ends at that time) or `ended <why>` – and the rest is what the title wrote since the last
/// step (and the host's notes). The frame (`xcl_frame`) and the sound (`xcl_session_audio`)
/// are the session's.
#[no_mangle]
pub extern "C" fn xcl_session_step(now_ns: u64, max_instructions: u64) -> u32 {
    SESSION.with_borrow_mut(|session| {
        let Some(s) = session.as_mut() else { return answer(Err("no session".into())) };
        if s.ended {
            return answer(Ok("ended (before)\n".into()));
        }
        s.guest.set_time(now_ns);
        let stop = s.guest.run(max_instructions.max(1));
        let state = match &stop {
            Stop::Limit => "running".to_string(),
            Stop::Idle { until } => format!("idle {until}"),
            other => {
                s.ended = true;
                format!("ended {other:x?}")
            }
        };
        let mut text = state + "\n";
        text.push_str(&String::from_utf8_lossy(&s.guest.output[s.printed..]));
        s.printed = s.guest.output.len();
        for line in &s.guest.log[s.logged..] {
            text.push_str(line);
            text.push('\n');
        }
        s.logged = s.guest.log.len();
        // the frame presented last, and what the voices played
        let d3d = &mut s.guest.win.d3d;
        if let Some(f) = d3d.frame.take() {
            *FRAME.lock().unwrap() = Some((f.width, f.height, f.format, d3d.presents, f.pixels));
        }
        let played = s.guest.win.audio.take_output();
        AUDIO.with_borrow_mut(|a| a.0.extend(played));
        answer(Ok(text))
    })
}

/// The sound the session's voices played since the last call: the number of samples
/// (interleaved 48 kHz stereo, 16 bits) at `xcl_session_audio_ptr()`.
#[no_mangle]
pub extern "C" fn xcl_session_audio() -> usize {
    AUDIO.with_borrow_mut(|a| {
        a.1 = std::mem::take(&mut a.0);
        a.1.len()
    })
}

#[no_mangle]
pub extern "C" fn xcl_session_audio_ptr() -> *const i16 {
    AUDIO.with_borrow(|a| a.1.as_ptr())
}
