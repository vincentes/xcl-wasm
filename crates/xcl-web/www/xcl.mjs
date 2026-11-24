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
  };
}
