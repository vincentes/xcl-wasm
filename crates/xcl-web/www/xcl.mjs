// Loader for xcl_web.wasm (wasm64: pointers and sizes are BigInt on this side).

export async function load(source) {
  const bytes = source instanceof Uint8Array || source instanceof ArrayBuffer ? source : await (await fetch(source)).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const x = instance.exports;
  const enc = new TextEncoder();
  const dec = new TextDecoder();

  // copy `data` (Uint8Array or string) into the module; returns [ptr, len] and remembers it
  function pass(held, data) {
    const b = typeof data === "string" ? enc.encode(data) : data;
    const len = BigInt(b.length);
    const ptr = x.xcl_alloc(len);
    new Uint8Array(x.memory.buffer, Number(ptr), b.length).set(b);
    held.push([ptr, len]);
    return [ptr, len];
  }

  function call(fn, ...inputs) {
    const held = [];
    try {
      const args = [];
      for (const i of inputs) {
        if (typeof i === "number" || typeof i === "bigint") args.push(i);
        else args.push(...pass(held, i));
      }
      const code = fn(...args);
      const text = dec.decode(new Uint8Array(x.memory.buffer, Number(x.xcl_out_ptr()), Number(x.xcl_out_len())).slice());
      if (code !== 0) throw new Error(text);
      return text;
    } finally {
      for (const [ptr, len] of held) x.xcl_free(ptr, len);
    }
  }

  return {
    version: () => call(x.xcl_version),
    /** `xcl inspect`: report of a PE file (text, or JSON text with `json`). */
    inspect: (name, data, json = false) => call(x.xcl_inspect, name, data, json ? 1 : 0),
    /** `xcl disasm`: RDNA2 listing of a shader blob; `args` as on the command line. */
    disasm: (name, data, args = []) => call(x.xcl_disasm, name, data, args.join("\n")),
    /** Put a file into the game folder the next `run` sees (`path` like "data\\hello.txt"). */
    gameFile: (path, data) => call(x.xcl_game_file, path, data),
    /** Empty the game folder. */
    clearGame: () => call(x.xcl_game_file, "", new Uint8Array(0)),
    /** Run a title on the x86-64 guest: its output, then a `--- stopped: …` line. */
    run: (name, data, commandLine = "", maxInstructions = 0) => call(x.xcl_run, name, data, commandLine, BigInt(maxInstructions)),
    /** Assemble RDNA2 instructions (lines of `NAME ENCODING FIELD=value …`) to code. */
    assemble: (lines) => Uint8Array.from(call(x.xcl_rdna2_assemble, lines.join("\n")).match(/../g) ?? [], (h) => parseInt(h, 16)),
    /** An RDNA2 program as a WGSL compute shader in kernel mode (throws what is missing). */
    wgslKernel: (code) => call(x.xcl_wgsl_kernel, code),
    /** An RDNA2 vertex program as a WGSL vertex stage: `attributes` at `@location(k)`, four VGPRs each from `attrBase`. */
    wgslVertex: (code, attrBase, attributes) => call(x.xcl_wgsl_stage, code, 1, attrBase, attributes),
    /** An RDNA2 pixel program as a WGSL fragment stage (`inputEna`: SPI_PS_INPUT_ENA). */
    wgslPixel: (code, inputEna) => call(x.xcl_wgsl_stage, code, 2, inputEna, 0),
    /** Start a title that keeps running (`step`); the game folder is `gameFile`'s. */
    start: (name, data, commandLine = "") => call(x.xcl_session_start, name, data, commandLine),
    /**
     * Let the started title run: `nowMs` is the page's clock (milliseconds since `start`), at
     * most `maxInstructions` are executed. Returns { state: "running" | "idle" | "ended",
     * until (ms, when idle: the first wait ends then), why (when ended), output (what the
     * title wrote since the last step) }. `frame()` and `audio()` are the session's.
     */
    step: (nowMs, maxInstructions = 3_000_000) => {
      const text = call(x.xcl_session_step, BigInt(Math.max(0, Math.round(nowMs * 1e6))), BigInt(maxInstructions));
      const end = text.indexOf("\n");
      const [state, ...rest] = text.slice(0, end).split(" ");
      return { state, until: state === "idle" ? Number(rest[0]) / 1e6 : undefined, why: state === "ended" ? rest.join(" ") : undefined, output: text.slice(end + 1) };
    },
    /** What the session's voices played since the last call: Int16Array, 48 kHz stereo. */
    audio: () => new Int16Array(x.memory.buffer, Number(x.xcl_session_audio_ptr()), Number(x.xcl_session_audio())).slice(),
    /**
     * The frame the last run presented last, or null: { width, height, format (a DXGI_FORMAT:
     * 28 R8G8B8A8, 87 B8G8R8A8), presented (frames so far), pixels (4 bytes each, top down) }.
     */
    frame: () => {
      let info;
      try {
        info = call(x.xcl_frame);
      } catch {
        return null;
      }
      const [width, height, format, presented] = info.split(" ").map(Number);
      const pixels = new Uint8Array(x.memory.buffer, Number(x.xcl_frame_ptr()), width * height * 4).slice();
      return { width, height, format, presented, pixels };
    },
  };
}
