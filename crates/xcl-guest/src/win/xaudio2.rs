//! XAudio2 on the guest: the engine, a mastering voice and source voices whose buffers play on
//! guest time. A buffer's callbacks (OnBufferStart, OnBufferEnd, OnStreamEnd) come when its
//! samples have been played, on the host-driven thread; what the voices play is mixed to
//! 48 kHz stereo for the host to put out (`State::take_output`).
//!
//! Formats: 16-bit and 32-bit float PCM, and XMA1 / XMA2, decoded with XCL's decoder
//! (`xcl::hle::xma2`). **Opus (0x704F) plays as silence** of the right length: there is no Opus
//! decoder here (XCL's Windows runtime uses the system's). Not there: submix voices, effects,
//! filters, output matrices, loops inside a buffer, the per-pass callbacks.

#![allow(non_snake_case)]

use std::collections::{HashMap, VecDeque};

use xcl::hle::xma2;

use super::register;
use crate::{Guest, Handler, Stop};

type H = Result<u64, Stop>;

const E_INVALIDARG: u64 = 0x8007_0057;
/// XAUDIO2_E_INVALID_CALL
const E_INVALID_CALL: u64 = 0x8896_0001;
const OUTPUT_RATE: u64 = 48_000;
const END_OF_STREAM: u32 = 0x40;
const TAG_PCM: u16 = 1;
const TAG_FLOAT: u16 = 3;
const TAG_XMA1: u16 = 0x165;
const TAG_XMA2: u16 = 0x166;
const TAG_OPUS: u16 = 0x704F;

/// A submitted buffer, decoded: interleaved samples of the voice's channels.
struct Buffer {
    samples: Vec<f32>,
    context: u64,
    end_of_stream: bool,
    started: bool,
}

struct Voice {
    source: bool,
    channels: u32,
    rate: u32,
    tag: u16,
    bits: u16,
    /// the title's IXAudio2VoiceCallback (0: none)
    callback: u64,
    volume: f32,
    ratio: f32,
    started: bool,
    queue: VecDeque<Buffer>,
    /// frames of the current buffer that have been played (fractional: the voice's rate is not
    /// the output's)
    position: f64,
    played: u64,
    decoder: Option<xma2::Decoder>,
}

#[derive(Default)]
pub struct State {
    voices: HashMap<u64, Voice>,
    engine: u64,
    /// guest time up to which the voices have played
    clock_ns: u64,
    /// 48 kHz stereo, for the host; the oldest second is dropped when nobody takes it
    output: VecDeque<i16>,
}

impl State {
    /// What the voices played since the last call: interleaved 48 kHz stereo.
    pub fn take_output(&mut self) -> Vec<i16> {
        self.output.drain(..).collect()
    }
}

const VOICE_METHODS: [&str; 29] = [
    "XA2Voice_GetVoiceDetails",
    "XA2Voice::#1",
    "XA2Voice::#2",
    "XA2Voice::#3",
    "XA2Voice::#4",
    "XA2Voice::#5",
    "XA2Voice::#6",
    "XA2Voice::#7",
    "XA2Voice::#8",
    "XA2Voice::#9",
    "XA2Voice::#10",
    "XA2Voice::#11",
    "XA2Voice_SetVolume",
    "XA2Voice_GetVolume",
    "XA2Voice::#14",
    "XA2Voice::#15",
    "XA2Voice::#16",
    "XA2Voice::#17",
    "XA2Voice_DestroyVoice",
    "XA2Voice_Start",
    "XA2Voice_Stop",
    "XA2Voice_SubmitSourceBuffer",
    "XA2Voice_FlushSourceBuffers",
    "XA2Voice_Discontinuity",
    "XA2Voice_ExitLoop",
    "XA2Voice_GetState",
    "XA2Voice_SetFrequencyRatio",
    "XA2Voice_GetFrequencyRatio",
    "XA2Voice_SetSourceSampleRate",
];

/// XAudio2Create(engine out, flags, processor): the console's flags and core masks are taken
fn XAudio2Create(g: &mut Guest) -> H {
    let out = g.arg(0)?;
    if out == 0 {
        return Ok(E_INVALIDARG);
    }
    if g.win.audio.engine == 0 {
        let methods = [
            "XA2_QueryInterface",
            "XA2_AddRef",
            "XA2_Release",
            "XA2::#3",
            "XA2::#4",
            "XA2_CreateSourceVoice",
            "XA2::#6",
            "XA2_CreateMasteringVoice",
            "XA2_StartEngine",
            "XA2_StopEngine",
            "XA2_CommitChanges",
            "XA2::#11",
            "XA2_SetDebugConfiguration",
        ];
        g.win.audio.engine = g.com_object("IXAudio2", &methods)?;
        g.win.audio.clock_ns = g.now_ns();
    }
    let engine = g.win.audio.engine;
    g.mem.write_u64(out, engine)?;
    Ok(0)
}

fn XA2_QueryInterface(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(2)?);
    if out == 0 {
        return Ok(E_INVALIDARG);
    }
    g.mem.write_u64(out, this)?;
    Ok(0)
}

fn one(_: &mut Guest) -> H {
    Ok(1)
}

fn ok(_: &mut Guest) -> H {
    Ok(0)
}

/// (tag, channels, rate, bits, the bytes after the WAVEFORMATEX) of a format; a
/// WAVE_FORMAT_EXTENSIBLE names its real format in its sub-format.
fn format(g: &Guest, at: u64) -> Result<(u16, u32, u32, u16, Vec<u8>), Stop> {
    let (mut tag, channels, rate, bits, extra_len) = (g.mem.read_u16(at)?, g.mem.read_u16(at + 2)? as u32, g.mem.read_u32(at + 4)?, g.mem.read_u16(at + 14)?, g.mem.read_u16(at + 16)?);
    let mut extra = vec![0u8; if matches!(tag, TAG_PCM | TAG_FLOAT) { 0 } else { extra_len as usize }];
    g.mem.read(at + 18, &mut extra)?;
    if tag == 0xFFFE && extra.len() >= 22 {
        tag = u16::from_le_bytes([extra[6], extra[7]]);
    }
    Ok((tag, channels, rate, bits, extra))
}

/// CreateSourceVoice(voice out, format, flags, maximum frequency ratio, callback, sends, effects)
fn XA2_CreateSourceVoice(g: &mut Guest) -> H {
    let (out, at, callback) = (g.arg(1)?, g.arg(2)?, g.arg(5)?);
    if out == 0 || at == 0 {
        return Ok(E_INVALIDARG);
    }
    let (tag, channels, rate, bits, extra) = format(g, at)?;
    if channels == 0 || rate == 0 {
        return Ok(E_INVALIDARG);
    }
    let decoder = match tag {
        TAG_XMA2 | TAG_XMA1 => {
            let config = if tag == TAG_XMA2 { xma2::StreamConfig::from_xma2_wfx(channels as u16, rate, &extra) } else { xma2::StreamConfig::from_xma1_wfx(channels as u16, rate, &extra) };
            match config.and_then(xma2::Decoder::new) {
                Ok(d) => Some(d),
                Err(e) => {
                    g.log.push(format!("XAudio2: an XMA voice that cannot be decoded: {e}"));
                    return Ok(E_INVALIDARG);
                }
            }
        }
        _ => None,
    };
    let voice = g.com_object("IXAudio2SourceVoice", &VOICE_METHODS)?;
    g.win.audio.voices.insert(voice, Voice { source: true, channels, rate, tag, bits, callback, volume: 1.0, ratio: 1.0, started: false, queue: VecDeque::new(), position: 0.0, played: 0, decoder });
    // the callbacks come on a thread of the host's
    if callback != 0 {
        super::xgameruntime::ensure_worker(g)?;
    }
    g.mem.write_u64(out, voice)?;
    Ok(0)
}

/// CreateMasteringVoice(voice out, channels, rate, flags, device id, effects, category): any
/// device id is the one output there is
fn XA2_CreateMasteringVoice(g: &mut Guest) -> H {
    let (out, channels, rate) = (g.arg(1)?, g.arg(2)? as u32, g.arg(3)? as u32);
    if out == 0 {
        return Ok(E_INVALIDARG);
    }
    let voice = g.com_object("IXAudio2SourceVoice", &VOICE_METHODS)?;
    let (channels, rate) = (if channels == 0 { 2 } else { channels }, if rate == 0 { OUTPUT_RATE as u32 } else { rate });
    g.win.audio.voices.insert(voice, Voice { source: false, channels, rate, tag: TAG_PCM, bits: 16, callback: 0, volume: 1.0, ratio: 1.0, started: false, queue: VecDeque::new(), position: 0.0, played: 0, decoder: None });
    g.mem.write_u64(out, voice)?;
    Ok(0)
}

/// GetVoiceDetails(details { creation flags, active flags, channels, rate })
fn XA2Voice_GetVoiceDetails(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(1)?);
    if let Some((channels, rate)) = g.win.audio.voices.get(&this).map(|v| (v.channels, v.rate)) {
        g.mem.write_u64(out, 0)?;
        g.mem.write_u32(out + 8, channels)?;
        g.mem.write_u32(out + 12, rate)?;
    }
    Ok(0)
}

/// A float argument of the first four (XMM0..3), as a single.
fn float_arg(g: &Guest, i: usize) -> f32 {
    f32::from_bits(g.cpu.v[i][0] as u32)
}

fn XA2Voice_SetVolume(g: &mut Guest) -> H {
    let (this, volume) = (g.arg(0)?, float_arg(g, 1));
    if let Some(v) = g.win.audio.voices.get_mut(&this) {
        v.volume = volume;
    }
    Ok(0)
}

fn XA2Voice_GetVolume(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(1)?);
    let volume = g.win.audio.voices.get(&this).map_or(1.0, |v| v.volume);
    g.mem.write_u32(out, volume.to_bits())?;
    Ok(0)
}

fn XA2Voice_SetFrequencyRatio(g: &mut Guest) -> H {
    let (this, ratio) = (g.arg(0)?, float_arg(g, 1));
    if let Some(v) = g.win.audio.voices.get_mut(&this).filter(|_| ratio > 0.0) {
        v.ratio = ratio;
    }
    Ok(0)
}

fn XA2Voice_GetFrequencyRatio(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(1)?);
    let ratio = g.win.audio.voices.get(&this).map_or(1.0, |v| v.ratio);
    g.mem.write_u32(out, ratio.to_bits())?;
    Ok(0)
}

fn XA2Voice_DestroyVoice(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    pump(g);
    g.win.audio.voices.remove(&this);
    Ok(0)
}

fn start(g: &mut Guest, on: bool) -> H {
    let this = g.arg(0)?;
    pump(g);
    match g.win.audio.voices.get_mut(&this) {
        Some(v) => {
            v.started = on;
            Ok(0)
        }
        None => Ok(E_INVALID_CALL),
    }
}

fn XA2Voice_Start(g: &mut Guest) -> H {
    start(g, true)
}

fn XA2Voice_Stop(g: &mut Guest) -> H {
    start(g, false)
}

/// Frames of sound in the Opus packets of a block ([u16 length][packet] …, as the console's
/// titles frame them): from each packet's table-of-contents byte.
fn opus_frames(block: &[u8], rate: u32) -> u64 {
    let (mut at, mut micros) = (0usize, 0u64);
    while at + 2 <= block.len() {
        let len = u16::from_le_bytes([block[at], block[at + 1]]) as usize;
        if len == 0 || at + 2 + len > block.len() {
            break;
        }
        let packet = &block[at + 2..at + 2 + len];
        let config = packet[0] >> 3;
        let frame = match config {
            0..=11 => [10_000, 20_000, 40_000, 60_000][(config & 3) as usize],
            12..=15 => [10_000, 20_000][(config & 1) as usize],
            _ => [2_500, 5_000, 10_000, 20_000][(config & 3) as usize],
        };
        let count = match packet[0] & 3 {
            0 => 1,
            1 | 2 => 2,
            _ => packet.get(1).map_or(0, |b| b & 0x3F) as u64,
        };
        micros += frame * count;
        at += 2 + len;
    }
    micros * rate as u64 / 1_000_000
}

/// SubmitSourceBuffer(XAUDIO2_BUFFER { flags, bytes, data @8, play begin @16, play length @20,
/// loop begin / length / count, context @36 }, WMA buffer): decoded now, played in order
fn XA2Voice_SubmitSourceBuffer(g: &mut Guest) -> H {
    let (this, at) = (g.arg(0)?, g.arg(1)?);
    if at == 0 {
        return Ok(E_INVALIDARG);
    }
    let (flags, bytes, data) = (g.mem.read_u32(at)?, g.mem.read_u32(at + 4)? as usize, g.mem.read_u64(at + 8)?);
    let (begin, length, context) = (g.mem.read_u32(at + 16)? as usize, g.mem.read_u32(at + 20)? as usize, g.mem.read_u64(at + 36)?);
    let mut raw = vec![0u8; bytes.min(256 << 20)];
    g.mem.read(data, &mut raw)?;
    let end_of_stream = flags & END_OF_STREAM != 0;
    let Some(v) = g.win.audio.voices.get_mut(&this).filter(|v| v.source) else { return Ok(E_INVALID_CALL) };
    let channels = v.channels as usize;
    let mut samples: Vec<f32> = match (v.tag, v.bits) {
        (TAG_PCM, 16) => raw.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect(),
        (TAG_FLOAT, 32) => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        (TAG_XMA1 | TAG_XMA2, _) => match v.decoder.as_mut().map(|d| d.decode_buffer_i16(&raw, end_of_stream)) {
            Some(Ok(pcm)) => pcm.into_iter().map(|s| s as f32 / 32768.0).collect(),
            Some(Err(e)) => {
                g.log.push(format!("XAudio2: an XMA buffer that cannot be decoded: {e}"));
                return Ok(E_INVALIDARG);
            }
            None => Vec::new(),
        },
        // no decoder: silence for as long as the packets are
        (TAG_OPUS, _) => vec![0.0; opus_frames(&raw, v.rate) as usize * channels],
        (tag, bits) => {
            g.log.push(format!("XAudio2: format {tag:#x} with {bits} bits is not played (silence)"));
            vec![0.0; raw.len() / 2]
        }
    };
    // the part of the buffer to play, in frames
    let frames = samples.len() / channels;
    let begin = begin.min(frames);
    let end = if length == 0 { frames } else { (begin + length).min(frames) };
    samples.truncate(end * channels);
    samples.drain(..begin * channels);
    v.queue.push_back(Buffer { samples, context, end_of_stream, started: false });
    Ok(0)
}

fn XA2Voice_FlushSourceBuffers(g: &mut Guest) -> H {
    let this = g.arg(0)?;
    pump(g);
    // what has not started playing is dropped, with its OnBufferEnd
    let (callback, dropped): (u64, Vec<u64>) = match g.win.audio.voices.get_mut(&this) {
        Some(v) => {
            let keep = v.queue.iter().take_while(|b| b.started).count();
            (v.callback, v.queue.drain(keep..).map(|b| b.context).collect())
        }
        None => return Ok(E_INVALID_CALL),
    };
    for context in dropped {
        notify(g, callback, 4, Some(context));
    }
    Ok(0)
}

/// GetState(state { current buffer context, buffers queued @8, samples played @12 }, flags)
fn XA2Voice_GetState(g: &mut Guest) -> H {
    let (this, out) = (g.arg(0)?, g.arg(1)?);
    pump(g);
    let Some(v) = g.win.audio.voices.get(&this) else { return Ok(0) };
    let (context, queued, played) = (v.queue.front().map_or(0, |b| b.context), v.queue.len() as u32, v.played);
    g.mem.write_u64(out, context)?;
    g.mem.write_u32(out + 8, queued)?;
    g.mem.write_u64(out + 12, played)?;
    Ok(0)
}

/// Queue a call of slot `slot` of the title's voice callback (2 OnStreamEnd, 3 OnBufferStart,
/// 4 OnBufferEnd) for the host-driven thread.
fn notify(g: &mut Guest, callback: u64, slot: u64, context: Option<u64>) {
    if callback == 0 {
        return;
    }
    let function = g.mem.read_u64(callback).and_then(|vtable| g.mem.read_u64(vtable + 8 * slot));
    if let Ok(function) = function {
        let mut args = vec![callback];
        args.extend(context);
        g.win.host_calls.push_back((function, args));
    }
}

/// Let the started voices play up to now: their samples go to the output, and the callbacks
/// of buffers that began or ended are queued.
pub(crate) fn pump(g: &mut Guest) {
    if g.win.audio.engine == 0 {
        return;
    }
    let now = g.now_ns();
    let frames = (now.saturating_sub(g.win.audio.clock_ns)) * OUTPUT_RATE / 1_000_000_000;
    if frames == 0 {
        return;
    }
    g.win.audio.clock_ns += frames * 1_000_000_000 / OUTPUT_RATE;
    let mut mix = vec![0f32; frames as usize * 2];
    let mut events: Vec<(u64, u64, Option<u64>)> = Vec::new();
    for v in g.win.audio.voices.values_mut().filter(|v| v.source && v.started) {
        let step = v.rate as f64 * v.ratio as f64 / OUTPUT_RATE as f64;
        let channels = v.channels as usize;
        for out in mix.chunks_exact_mut(2) {
            let Some(b) = v.queue.front_mut() else { break };
            if !b.started {
                b.started = true;
                events.push((v.callback, 3, Some(b.context)));
            }
            let frame = v.position as usize;
            if frame * channels + channels <= b.samples.len() {
                let (left, right) = (b.samples[frame * channels], b.samples[frame * channels + channels.min(2) - 1]);
                out[0] += left * v.volume;
                out[1] += right * v.volume;
                v.position += step;
            }
            // the buffer is played out
            if v.position as usize * channels >= b.samples.len() {
                v.played += (b.samples.len() / channels) as u64;
                v.position = 0.0;
                events.push((v.callback, 4, Some(b.context)));
                if b.end_of_stream {
                    events.push((v.callback, 2, None));
                }
                v.queue.pop_front();
            }
        }
    }
    let output = &mut g.win.audio.output;
    output.extend(mix.iter().map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16));
    // nobody listening: keep the last second
    let excess = output.len().saturating_sub(OUTPUT_RATE as usize * 2);
    output.drain(..excess);
    for (callback, slot, context) in events {
        notify(g, callback, slot, context);
    }
}

pub(super) fn register(map: &mut HashMap<&'static str, Handler>) {
    register!(map; XAudio2Create, XA2_QueryInterface, XA2_CreateSourceVoice, XA2_CreateMasteringVoice);
    register!(map; XA2Voice_GetVoiceDetails, XA2Voice_SetVolume, XA2Voice_GetVolume, XA2Voice_SetFrequencyRatio, XA2Voice_GetFrequencyRatio, XA2Voice_DestroyVoice);
    register!(map; XA2Voice_Start, XA2Voice_Stop, XA2Voice_SubmitSourceBuffer, XA2Voice_FlushSourceBuffers, XA2Voice_GetState);
    map.insert("XA2_AddRef", one as Handler);
    map.insert("XA2_Release", ok as Handler);
    for name in ["XA2_StartEngine", "XA2_StopEngine", "XA2_CommitChanges", "XA2_SetDebugConfiguration", "XA2Voice_Discontinuity", "XA2Voice_ExitLoop", "XA2Voice_SetSourceSampleRate"] {
        map.insert(name, ok as Handler);
    }
}

#[cfg(test)]
mod tests {
    use xcl_emu::cpu::{RCX, RDX, RSP};
    use xcl_emu::mem::{PROT_R, PROT_W};

    use super::*;

    fn guest() -> (Guest, u64) {
        let mut g = Guest::new();
        let stack = g.mem.alloc(0x1_0000, PROT_R | PROT_W).unwrap();
        let t = g.add_thread(stack, stack + 0x1_0000, 0, 0, crate::SLOT_ENTRY_RETURN).unwrap();
        g.current = t;
        g.cpu = g.threads[t].cpu.clone();
        let scratch = g.mem.alloc(0x1_0000, PROT_R | PROT_W).unwrap();
        (g, scratch)
    }

    fn call(g: &mut Guest, name: &str, args: &[u64]) -> u64 {
        for (i, &a) in args.iter().enumerate() {
            match i {
                0 => g.cpu.r[RCX] = a,
                1 => g.cpu.r[RDX] = a,
                2 => g.cpu.r[8] = a,
                3 => g.cpu.r[9] = a,
                _ => g.mem.write_u64(g.cpu.r[RSP] + 8 + 8 * i as u64, a).unwrap(),
            }
        }
        let handler = *g.handlers.get(name).unwrap_or_else(|| panic!("no host function {name}"));
        handler(g).unwrap()
    }

    #[test]
    fn a_pcm_buffer_plays_on_guest_time_with_its_callbacks() {
        let (mut g, m) = guest();
        assert_eq!(call(&mut g, "XAudio2Create", &[m, 0x41, 0x3C]), 0);
        let engine = g.mem.read_u64(m).unwrap();
        assert_eq!(call(&mut g, "XA2_CreateMasteringVoice", &[engine, m, 2, 48_000, 0, 0, 0, 6]), 0);
        // the title's callback object: a vtable of seven functions
        let (callback, vtable) = (m + 0x100, m + 0x200);
        g.mem.write_u64(callback, vtable).unwrap();
        for slot in 0..7 {
            g.mem.write_u64(vtable + 8 * slot, 0x7000 + slot).unwrap();
        }
        // WAVEFORMATEX: PCM, mono, 48 kHz, 16 bits
        let format = m + 0x300;
        for (at, v) in [(0u64, 1u16), (2, 1), (12, 2), (14, 16)] {
            g.mem.write_u16(format + at, v).unwrap();
        }
        g.mem.write_u32(format + 4, 48_000).unwrap();
        assert_eq!(call(&mut g, "XA2_CreateSourceVoice", &[engine, m, format, 0, 0, callback, 0, 0]), 0);
        let voice = g.mem.read_u64(m).unwrap();
        call(&mut g, "XA2Voice_GetVoiceDetails", &[voice, m + 0x40]);
        assert_eq!((g.mem.read_u32(m + 0x48).unwrap(), g.mem.read_u32(m + 0x4C).unwrap()), (1, 48_000));

        // 480 frames (10 ms) of half scale, the last buffer of its stream, context 0x55
        let (data, buffer) = (m + 0x1000, m + 0x400);
        for i in 0..480 {
            g.mem.write_u16(data + 2 * i, 16384).unwrap();
        }
        g.mem.write_u32(buffer, END_OF_STREAM).unwrap();
        g.mem.write_u32(buffer + 4, 960).unwrap();
        g.mem.write_u64(buffer + 8, data).unwrap();
        g.mem.write_u64(buffer + 36, 0x55).unwrap();
        assert_eq!(call(&mut g, "XA2Voice_SubmitSourceBuffer", &[voice, buffer, 0]), 0);
        call(&mut g, "XA2Voice_GetState", &[voice, m + 0x40, 0]);
        assert_eq!((g.mem.read_u64(m + 0x40).unwrap(), g.mem.read_u32(m + 0x48).unwrap()), (0x55, 1));

        // nothing plays before Start; then half of the buffer in 5 ms
        g.win.time_ns += 5_000_000;
        pump(&mut g);
        assert!(g.win.audio.take_output().iter().all(|&s| s == 0));
        assert_eq!(call(&mut g, "XA2Voice_Start", &[voice, 0, 0]), 0);
        g.win.time_ns += 5_000_000;
        pump(&mut g);
        let out = g.win.audio.take_output();
        assert_eq!(out.len(), 240 * 2);
        assert!(out.iter().all(|&s| (16383..=16384).contains(&s)), "{:?}", &out[..4]);
        assert_eq!(g.win.host_calls.drain(..).collect::<Vec<_>>(), [(0x7003, vec![callback, 0x55])], "OnBufferStart");
        // … and the rest: OnBufferEnd with the context, OnStreamEnd, and silence after it
        g.win.time_ns += 10_000_000;
        pump(&mut g);
        let out = g.win.audio.take_output();
        assert_eq!((out.len(), out[0], out[2 * 239], out[2 * 240]), (480 * 2, 16383, 16383, 0));
        assert_eq!(g.win.host_calls.drain(..).collect::<Vec<_>>(), [(0x7004, vec![callback, 0x55]), (0x7002, vec![callback])]);
        call(&mut g, "XA2Voice_GetState", &[voice, m + 0x40, 0]);
        assert_eq!((g.mem.read_u32(m + 0x48).unwrap(), g.mem.read_u64(m + 0x4C).unwrap()), (0, 480));
    }

    #[test]
    fn opus_blocks_are_as_long_as_their_packets_say() {
        // two packets: hybrid fullband 20 ms, one frame (0x78); CELT 10 ms, two frames (0xF1)
        let mut block = vec![3, 0, 0x78, 1, 2, 2, 0, 0xF1, 9];
        block.extend([0; 8]);
        assert_eq!(opus_frames(&block, 48_000), 960 + 2 * 480);
        assert_eq!(opus_frames(&[], 48_000), 0);
    }
}
