// Smoke test of the built module under Node (22+ for 64-bit memories):
//   node crates/xcl-web/www/smoke.mjs target/wasm64-unknown-unknown/release/xcl_web.wasm [xcl_testtitle.exe]
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { load } from "./xcl.mjs";

// a minimal x64 PE with one import table (as `inspect`'s unit test builds it)
function peWithImports(imports) {
  const RVA = 0x1000, RAW = 0x200;
  const sec = new Array((imports.length + 1) * 20).fill(0);
  const put32 = (b, at, v) => { for (let i = 0; i < 4; i++) b[at + i] = (v >>> (8 * i)) & 0xff; };
  const str = (s) => [...Buffer.from(s, "latin1"), 0];
  imports.forEach(([dll, functions], i) => {
    const thunks = sec.length;
    for (let k = 0; k < (functions.length + 1) * 8; k++) sec.push(0);
    functions.forEach((f, k) => {
      put32(sec, thunks + k * 8, sec.length + RVA);
      sec.push(0, 0, ...str(f));
    });
    const name = sec.length + RVA;
    sec.push(...str(dll));
    put32(sec, i * 20, thunks + RVA);
    put32(sec, i * 20 + 12, name);
    put32(sec, i * 20 + 16, thunks + RVA);
  });
  const pe = new Array(RAW).fill(0);
  pe[0] = 0x4d; pe[1] = 0x5a;
  put32(pe, 0x3c, 0x40);
  put32(pe, 0x40, 0x4550);
  const coff = 0x44, opt = coff + 20, s = opt + 240;
  put32(pe, coff, 0x8664 | (1 << 16));
  put32(pe, coff + 16, 240 | (0x22 << 16));
  put32(pe, opt, 0x20b);
  put32(pe, opt + 16, 0x1000);
  put32(pe, opt + 24, 0x40000000); put32(pe, opt + 28, 1);
  put32(pe, opt + 32, 0x1000);
  put32(pe, opt + 36, 0x200);
  put32(pe, opt + 56, 0x2000);
  put32(pe, opt + 60, RAW);
  put32(pe, opt + 68, 2);
  put32(pe, opt + 72, 0x100000);
  put32(pe, opt + 108, 16);
  put32(pe, opt + 120, RVA);
  put32(pe, opt + 124, (imports.length + 1) * 20);
  [..."\.rdata"].forEach((c, i) => (pe[s + i] = c.charCodeAt(0)));
  put32(pe, s + 8, sec.length);
  put32(pe, s + 12, RVA);
  put32(pe, s + 16, sec.length);
  put32(pe, s + 20, RAW);
  put32(pe, s + 36, 0x40000040);
  return Uint8Array.from([...pe, ...sec]);
}

const path = process.argv[2];
assert.ok(path, "usage: node smoke.mjs <xcl_web.wasm>");
const wasm = readFileSync(path);
const memory = WebAssembly.Module.exports(new WebAssembly.Module(wasm)).find((e) => e.name === "memory");
assert.ok(memory, "the module exports its memory");
const xcl = await load(wasm);

assert.match(xcl.version(), /^xcl \d+\.\d+\.\d+$/);

const pe = peWithImports([["kernelx.dll", ["XMemAlloc", "CreateFileW"]], ["d3d12_xs.dll", ["D3D12XboxCreateDevice"]], ["ucrtbase.dll", ["malloc"]]]);
const text = xcl.inspect("game.exe", pe);
assert.match(text, /Console game:  yes \(imports kernelx\.dll, d3d12_xs\.dll\)/);
assert.match(text, /Imports: 4 functions from 3 DLLs/);
const report = JSON.parse(xcl.inspect("game.exe", pe, true));
assert.equal(report.image_base, "0x140000000");
assert.deepEqual(report.imports.map((e) => e.function), ["XMemAlloc", "CreateFileW", "D3D12XboxCreateDevice", "malloc"]);
assert.throws(() => xcl.inspect("x.bin", new TextEncoder().encode("not a PE")), /MZ/);

// s_endpgm
const code = Uint8Array.from([0x00, 0x00, 0x81, 0xbf]);
assert.match(xcl.disasm("blob.bin", code, ["--offset", "0"]), /s_endpgm/);
assert.throws(() => xcl.disasm("blob.bin", code, ["--offset", "8"]), /past the end/);
assert.match(xcl.disasm("blob.bin", new Uint8Array(0)), /no RDNA2 programs found/);

// many calls: the page's blocks are freed again
for (let i = 0; i < 2000; i++) xcl.inspect("game.exe", pe);

// a title on the x64 guest (the test title, when it is built: scripts/cross-testtitle.sh)
const titlePath = process.argv[3];
let ran = "no title given: run not checked";
if (titlePath) {
  const t0 = performance.now();
  // its game folder: the repository's test data and its DLL (built next to it)
  const data = new URL("../../../tests/data/testgame/data/", import.meta.url);
  for (const f of ["hello.txt", "delete_me.txt", "sub/nested.bin"]) xcl.gameFile("data\\" + f.replace("/", "\\"), readFileSync(new URL(f, data)));
  xcl.gameFile("testgame_mw.dll", readFileSync(titlePath.replace(/[^\\/]+$/, "testgame_mw.dll")));
  const out = xcl.run("xcl_testtitle.exe", readFileSync(titlePath), "xcl_testtitle.exe --hello");
  xcl.clearGame();
  const lines = out.split("\n");
  const passes = lines.filter((l) => l.startsWith("[PASS] ")).length;
  const fails = lines.filter((l) => l.startsWith("[FAIL] ")).length;
  assert.match(out, /^xcl_testtitle: start\n/);
  // it runs to its end (functions the host lacks answer 0, so later checks fail), and the
  // checks up to XGameRuntime pass
  assert.match(out, /\nxcl_testtitle: done, failures: \d+\n/);
  assert.match(out, /--- stopped: Exit\(/);
  assert.ok(lines.slice(1, 26).every((l) => l.startsWith("[PASS] ")), `the first 25 checks:\n${lines.slice(0, 28).join("\n")}`);
  assert.ok(passes >= 25, `only ${passes} checks passed`);
  ran = `test title ran to its end: ${passes} checks passed, ${fails} failed, in ${Math.round(performance.now() - t0)} ms`;
}

console.log(`xcl-web smoke: ok (${xcl.version()}, ${wasm.length} B; ${ran})`);
