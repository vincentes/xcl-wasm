// The WGSL back end of the shader recompiler, on a real WebGPU: RDNA2 programs are assembled
// and translated by the module, run as compute shaders in headless Chromium, and what they
// leave in the register file is compared with what the instructions are specified to do
// (computed here in JavaScript). A vertex and a pixel program then draw a triangle, and the
// pixels are read back.
//   node crates/xcl-web/www/wgsl.test.mjs target/wasm64-unknown-unknown/release/xcl_web.wasm
// Needs Playwright's Chromium; WebGPU there is SwiftShader (software) unless a GPU is exposed.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { chromium } from "playwright";
import { load } from "./xcl.mjs";

const xcl = await load(readFileSync(process.argv[2]));
const [SGPRS, VGPRS, LANES] = [128, 256, 64];
const DWORDS = SGPRS + LANES * VGPRS;
const bits = (f) => new Uint32Array(new Float32Array([f]).buffer)[0];
const float = (u) => new Float32Array(new Uint32Array([u]).buffer)[0];
const f32 = Math.fround;

// a register file: every lane on, registers filled by the callbacks
function registers({ s = {}, v = {}, exec = [0xffffffff, 0xffffffff], vcc = [0, 0], scc = 0 } = {}) {
  const io = new Uint32Array(DWORDS);
  for (const [n, value] of Object.entries(s)) io[n] = value >>> 0;
  [io[126], io[127]] = exec;
  [io[106], io[107]] = vcc;
  io[125] = scc;
  for (const [n, lane] of Object.entries(v)) for (let l = 0; l < LANES; l++) io[SGPRS + l * VGPRS + Number(n)] = lane(l) >>> 0;
  return io;
}
const vreg = (file, n) => Array.from({ length: LANES }, (_, l) => file[SGPRS + l * VGPRS + n]);
// SGPR and VGPR operand codes
const [S, V] = [(n) => n, (n) => 256 + n];
const mix = (l, k) => (Math.imul(l + 1, 0x9e3779b1) ^ Math.imul(k + 7, 0x85ebca6b)) >>> 0;

const cases = [];
const test = (name, program, io, check) => cases.push({ name, program, io, check });

// --- 32-bit float and integer vector instructions, one each: v5 = op(v1, v2)
const A = (l) => bits(l * 0.5 - 7.25), B = (l) => bits(1.25 + (l % 5));
for (const [op, expect] of [
  ["V_ADD_F32", (a, b) => bits(f32(float(a) + float(b)))],
  ["V_SUB_F32", (a, b) => bits(f32(float(a) - float(b)))],
  ["V_SUBREV_F32", (a, b) => bits(f32(float(b) - float(a)))],
  ["V_MUL_F32", (a, b) => bits(f32(float(a) * float(b)))],
  ["V_MIN_F32", (a, b) => bits(Math.min(float(a), float(b)))],
  ["V_MAX_F32", (a, b) => bits(Math.max(float(a), float(b)))],
]) {
  test(op.toLowerCase(), [`${op} ENC_VOP2 VDST=5 SRC0=${V(1)} VSRC1=2`, "S_ENDPGM ENC_SOPP"], registers({ v: { 1: A, 2: B } }), (res) =>
    assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => expect(A(l), B(l)))));
}
const X = (l) => mix(l, 1), Y = (l) => mix(l, 2);
for (const [op, enc, expect] of [
  ["V_ADD_NC_U32", "ENC_VOP2", (a, b) => (a + b) >>> 0],
  ["V_SUB_NC_U32", "ENC_VOP2", (a, b) => (a - b) >>> 0],
  ["V_SUBREV_NC_U32", "ENC_VOP2", (a, b) => (b - a) >>> 0],
  ["V_AND_B32", "ENC_VOP2", (a, b) => (a & b) >>> 0],
  ["V_OR_B32", "ENC_VOP2", (a, b) => (a | b) >>> 0],
  ["V_XOR_B32", "ENC_VOP2", (a, b) => (a ^ b) >>> 0],
  ["V_LSHLREV_B32", "ENC_VOP2", (a, b) => (b << (a & 31)) >>> 0],
  ["V_LSHRREV_B32", "ENC_VOP2", (a, b) => b >>> (a & 31)],
  ["V_ASHRREV_I32", "ENC_VOP2", (a, b) => ((b | 0) >> (a & 31)) >>> 0],
  ["V_MIN_U32", "ENC_VOP2", (a, b) => Math.min(a, b)],
  ["V_MAX_I32", "ENC_VOP2", (a, b) => Math.max(a | 0, b | 0) >>> 0],
  ["V_MUL_LO_U32", "ENC_VOP3", (a, b) => Math.imul(a, b) >>> 0],
  ["V_MUL_HI_U32", "ENC_VOP3", (a, b) => Number((BigInt(a) * BigInt(b)) >> 32n)],
]) {
  const operands = enc === "ENC_VOP3" ? `SRC0=${V(1)} SRC1=${V(2)}` : `SRC0=${V(1)} VSRC1=2`;
  test(op.toLowerCase(), [`${op} ${enc} VDST=5 ${operands}`, "S_ENDPGM ENC_SOPP"], registers({ v: { 1: X, 2: Y } }), (res) =>
    assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => expect(X(l), Y(l)))));
}
test("v_fma_f32 and the conversions", [
  `V_FMA_F32 ENC_VOP3 VDST=5 SRC0=${V(1)} SRC1=${V(2)} SRC2=${V(3)}`,
  `V_CVT_F32_I32 ENC_VOP1 VDST=6 SRC0=${V(4)}`,
  `V_CVT_I32_F32 ENC_VOP1 VDST=7 SRC0=${V(1)}`,
  `V_FLOOR_F32 ENC_VOP1 VDST=8 SRC0=${V(1)}`,
  "S_ENDPGM ENC_SOPP",
], registers({ v: { 1: A, 2: B, 3: (l) => bits(l), 4: (l) => l - 30 } }), (res) => {
  // products and sums of these values are exact in single precision
  assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => bits(f32(float(A(l)) * float(B(l)) + l))));
  assert.deepEqual(vreg(res, 6), Array.from({ length: LANES }, (_, l) => bits(l - 30)));
  assert.deepEqual(vreg(res, 7), Array.from({ length: LANES }, (_, l) => Math.trunc(float(A(l))) >>> 0));
  assert.deepEqual(vreg(res, 8), Array.from({ length: LANES }, (_, l) => bits(Math.floor(float(A(l))))));
});

// --- operands: scalar registers, constants and a literal are the same for every lane
test("scalar, constant and literal operands of a vector instruction", [
  `V_ADD_NC_U32 ENC_VOP2 VDST=5 SRC0=${S(3)} VSRC1=1`, // v5 = s3 + v1
  `V_MOV_B32 ENC_VOP1 VDST=6 SRC0=242`, // 1.0
  `V_MOV_B32 ENC_VOP1 VDST=7 SRC0=193`, // -1
  `V_MOV_B32 ENC_VOP1 VDST=8 SRC0=255 LIT=0x12345678`,
  "S_ENDPGM ENC_SOPP",
], registers({ s: { 3: 1000 }, v: { 1: (l) => l } }), (res) => {
  assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => 1000 + l));
  assert.deepEqual([...new Set(vreg(res, 6))], [bits(1)]);
  assert.deepEqual([...new Set(vreg(res, 7))], [0xffffffff]);
  assert.deepEqual([...new Set(vreg(res, 8))], [0x12345678]);
});

// --- EXEC: lanes that are off keep what they had; a compare gives them a 0 bit
const HALF = [0x0000ffff, 0xff000000];
const lit = (l) => (l < 32 ? (HALF[0] >>> l) & 1 : (HALF[1] >>> (l - 32)) & 1);
test("EXEC masks a write and a compare", [
  `V_MOV_B32 ENC_VOP1 VDST=5 SRC0=${V(1)}`,
  `V_CMP_LT_U32 ENC_VOPC SRC0=${V(1)} VSRC1=2`, // vcc = v1 < v2
  "S_ENDPGM ENC_SOPP",
], registers({ exec: HALF, v: { 1: (l) => l, 2: () => 40, 5: (l) => 0xdead0000 + l } }), (res) => {
  assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => (lit(l) ? l : 0xdead0000 + l)));
  // lanes 0..15 are on and below 40; lanes 56..63 are on and not
  assert.deepEqual([res[106], res[107]], [0x0000ffff, 0]);
});

// --- compares and selects
test("v_cmp_*_f32 into VCC, v_cndmask_b32", [
  `V_CMP_LT_F32 ENC_VOPC SRC0=${V(1)} VSRC1=2`,
  `V_CNDMASK_B32 ENC_VOP2 VDST=6 SRC0=${V(1)} VSRC1=2`, // vcc ? v2 : v1  → the larger
  "S_ENDPGM ENC_SOPP",
], registers({ v: { 1: (l) => bits(l), 2: () => bits(31.5) } }), (res) => {
  assert.deepEqual([res[106], res[107]], [0xffffffff, 0]);
  assert.deepEqual(vreg(res, 6), Array.from({ length: LANES }, (_, l) => bits(Math.max(l, 31.5))));
});
test("a compare with a NaN: ordered is false, unordered true", [
  `V_CMP_GE_F32 ENC_VOP3 VDST=10 SRC0=${V(1)} SRC1=${V(2)}`, // into s10:s11
  `V_CMP_NGE_F32 ENC_VOP3 VDST=12 SRC0=${V(1)} SRC1=${V(2)}`,
  `V_CMP_GT_I32 ENC_VOP3 VDST=14 SRC0=${V(3)} SRC1=128`, // v3 > 0, signed
  "S_ENDPGM ENC_SOPP",
], registers({ v: { 1: (l) => (l === 3 ? 0x7fc00000 : bits(l)), 2: () => bits(2), 3: (l) => (l % 2 ? -5 : 5) } }), (res) => {
  // lanes 2.. are ≥ 2 but for the NaN in lane 3
  assert.deepEqual([res[10], res[11]], [0xfffffff4, 0xffffffff]);
  assert.deepEqual([res[12], res[13]], [0x0000000b, 0]);
  assert.deepEqual([res[14], res[15]], [0x55555555, 0x55555555]);
});

// --- the scalar unit
test("scalar arithmetic and its carry", [
  "S_MOVK_I32 ENC_SOPK SDST=3 SIMM16=0xfffb", // -5
  `S_ADD_U32 ENC_SOP2 SDST=4 SSRC0=${S(0)} SSRC1=${S(1)}`, // carries
  `S_CSELECT_B32 ENC_SOP2 SDST=5 SSRC0=129 SSRC1=130`, // scc ? 1 : 2
  `S_SUB_I32 ENC_SOP2 SDST=6 SSRC0=${S(3)} SSRC1=132`, // -5 - 4, no overflow
  `S_CSELECT_B32 ENC_SOP2 SDST=7 SSRC0=129 SSRC1=130`,
  `S_LSHL_B32 ENC_SOP2 SDST=8 SSRC0=${S(2)} SSRC1=132`,
  `S_ASHR_I32 ENC_SOP2 SDST=9 SSRC0=${S(3)} SSRC1=129`,
  `S_MUL_I32 ENC_SOP2 SDST=10 SSRC0=${S(3)} SSRC1=${S(2)}`,
  `S_AND_B64 ENC_SOP2 SDST=12 SSRC0=${S(20)} SSRC1=${S(22)}`,
  `S_ANDN2_B64 ENC_SOP2 SDST=14 SSRC0=${S(20)} SSRC1=${S(22)}`,
  "S_ENDPGM ENC_SOPP",
], registers({ s: { 0: 0xfffffff0, 1: 0x20, 2: 0x01020304, 20: 0xff00ff00, 21: 0x0f0f0f0f, 22: 0x0ff00ff0, 23: 0xffff0000 } }), (res) => {
  assert.deepEqual([...res.slice(3, 11)], [0xfffffffb, 0x10, 1, 0xfffffff7, 2, 0x10203040, 0xfffffffd, Math.imul(-5, 0x01020304) >>> 0]);
  assert.deepEqual([...res.slice(12, 16)], [0x0f000f00, 0x0f0f0000, 0xf000f000, 0x00000f0f]);
});
test("a loop: the sum of 1 … 10", [
  "S_MOVK_I32 ENC_SOPK SDST=0 SIMM16=0",
  "S_MOVK_I32 ENC_SOPK SDST=1 SIMM16=10",
  `S_ADD_U32 ENC_SOP2 SDST=0 SSRC0=${S(0)} SSRC1=${S(1)}`, // +8
  `S_SUB_U32 ENC_SOP2 SDST=1 SSRC0=${S(1)} SSRC1=129`,
  `S_CMP_LG_U32 ENC_SOPC SSRC0=${S(1)} SSRC1=128`,
  "S_CBRANCH_SCC1 ENC_SOPP SIMM16=0xfffc", // back to +8
  "S_ENDPGM ENC_SOPP",
], registers(), (res) => assert.deepEqual([res[0], res[1], res[125]], [55, 0, 0]));
test("if / else on lanes: s_and_saveexec_b64, a masked write, EXEC restored", [
  `V_CMP_GT_U32 ENC_VOPC SRC0=${V(1)} VSRC1=2`, // vcc = v1 > v2
  `S_AND_SAVEEXEC_B64 ENC_SOP1 SDST=10 SSRC0=106`, // s10:s11 = exec; exec &= vcc
  "S_CBRANCH_EXECZ ENC_SOPP SIMM16=1", // nothing to do: skip
  `V_MOV_B32 ENC_VOP1 VDST=5 SRC0=242`, // 1.0 where v1 > v2
  `S_MOV_B64 ENC_SOP1 SDST=126 SSRC0=${S(10)}`,
  "S_ENDPGM ENC_SOPP",
], registers({ v: { 1: (l) => l, 2: () => 50, 5: () => 7 } }), (res) => {
  assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => (l > 50 ? bits(1) : 7)));
  assert.deepEqual([res[126], res[127], res[10], res[11]], [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
});

// --- a draw: a vertex and a pixel program (the HLSL back end's gpu_graphics_vs_ps_draw)
const draw = {
  // position (attribute 0, v4 v5) to pos0 with z 0 and w 1; the colour (attribute 1) to param 0
  vertex: xcl.wgslVertex(xcl.assemble([
    `V_MOV_B32 ENC_VOP1 VDST=0 SRC0=${V(4)}`,
    `V_MOV_B32 ENC_VOP1 VDST=1 SRC0=${V(5)}`,
    "V_MOV_B32 ENC_VOP1 VDST=2 SRC0=128",
    "V_MOV_B32 ENC_VOP1 VDST=3 SRC0=242",
    "EXP ENC_EXP TGT=12 EN=15 VSRC0=0 VSRC1=1 VSRC2=2 VSRC3=3 DONE=1",
    "EXP ENC_EXP TGT=32 EN=15 VSRC0=8 VSRC1=9 VSRC2=10 VSRC3=11",
  ]), 4, 2),
  // pixels left of x = 8 are killed; the others get the colour with red × s4 (0.5, a uniform)
  pixel: xcl.wgslPixel(xcl.assemble([
    `V_CMPX_LT_F32 ENC_VOPC SRC0=255 VSRC1=2 LIT=${bits(8)}`,
    "V_INTERP_P1_F32 ENC_VINTRP VDST=4 VSRC=0 ATTR=0 ATTRCHAN=0",
    "V_INTERP_P2_F32 ENC_VINTRP VDST=4 VSRC=1 ATTR=0 ATTRCHAN=0",
    "V_INTERP_P2_F32 ENC_VINTRP VDST=5 VSRC=1 ATTR=0 ATTRCHAN=1",
    "V_INTERP_P2_F32 ENC_VINTRP VDST=6 VSRC=1 ATTR=0 ATTRCHAN=2",
    "V_MOV_B32 ENC_VOP1 VDST=7 SRC0=242",
    `V_MUL_F32 ENC_VOP2 VDST=4 SRC0=${S(4)} VSRC1=4`,
    "EXP ENC_EXP TGT=0 EN=15 VSRC0=4 VSRC1=5 VSRC2=6 VSRC3=7 DONE=1 VM=1",
  ]), 0x302),
  sgprs: Object.assign(new Array(128).fill(0), { 4: bits(0.5) }),
  // a triangle over the whole target: x y, r g b a
  vertices: [[-1, -1], [-1, 3], [3, -1]].flatMap((p) => [...p, 0.8, 0.6, 0.4, 1]),
  clear: [0, 0, 1, 1],
  size: 16,
};

// --- run them
const programs = cases.map((c) => ({ name: c.name, wgsl: xcl.wgslKernel(xcl.assemble(c.program)), io: [...c.io] }));
assert.throws(() => xcl.wgslKernel(xcl.assemble([`V_SIN_F32 ENC_VOP1 VDST=0 SRC0=${V(1)}`])), /v_sin_f32/, "what is missing is named");

const server = createServer((req, res) => res.writeHead(200, { "content-type": "text/html" }).end("<!doctype html><title>wgsl</title>"));
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU"] });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const { adapter, results, drawn } = await page.evaluate(async ({ programs, draw }) => {
    const gpuAdapter = await navigator.gpu?.requestAdapter();
    if (!gpuAdapter) return { adapter: null, results: [] };
    const device = await gpuAdapter.requestDevice();
    const results = [];
    for (const { wgsl, io } of programs) {
      device.pushErrorScope("validation");
      const module = device.createShaderModule({ code: wgsl });
      const messages = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
      if (messages.length) {
        await device.popErrorScope();
        results.push({ error: messages.join("; ") });
        continue;
      }
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      const size = io.length * 4;
      const input = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(input, 0, new Uint32Array(io));
      const output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const read = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(output, 0, read, 0, size);
      device.queue.submit([encoder.finish()]);
      const error = await device.popErrorScope();
      if (error) {
        results.push({ error: error.message });
        continue;
      }
      await read.mapAsync(GPUMapMode.READ);
      results.push({ res: [...new Uint32Array(read.getMappedRange())] });
    }
    // the draw: both stages in one pipeline, the SGPRs as a uniform buffer, a float target read back
    let drawn;
    {
      device.pushErrorScope("validation");
      const errors = [];
      const stage = async (code) => {
        const module = device.createShaderModule({ code });
        errors.push(...(await module.getCompilationInfo()).messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`));
        return module;
      };
      const [vs, ps] = [await stage(draw.vertex), await stage(draw.pixel)];
      if (errors.length) {
        await device.popErrorScope();
        drawn = { error: errors.join("; ") };
      } else {
        const pipeline = device.createRenderPipeline({
          layout: "auto",
          vertex: { module: vs, entryPoint: "main", buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }, { shaderLocation: 1, offset: 8, format: "float32x4" }] }] },
          fragment: { module: ps, entryPoint: "main", targets: [{ format: "rgba32float" }] },
        });
        const vertices = device.createBuffer({ size: draw.vertices.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(vertices, 0, new Float32Array(draw.vertices));
        const sgprs = device.createBuffer({ size: 512, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(sgprs, 0, new Uint32Array(draw.sgprs));
        const n = draw.size;
        const target = device.createTexture({ size: [n, n], format: "rgba32float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
        const read = device.createBuffer({ size: n * n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), clearValue: draw.clear, loadOp: "clear", storeOp: "store" }] });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: sgprs } }] }));
        pass.setVertexBuffer(0, vertices);
        pass.draw(3);
        pass.end();
        encoder.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: n * 16 }, [n, n]);
        device.queue.submit([encoder.finish()]);
        const error = await device.popErrorScope();
        if (error) drawn = { error: error.message };
        else {
          await read.mapAsync(GPUMapMode.READ);
          drawn = { pixels: [...new Float32Array(read.getMappedRange())] };
        }
      }
    }
    const info = gpuAdapter.info ?? {};
    return { adapter: `${info.vendor ?? "?"} ${info.architecture ?? ""} ${info.description ?? ""}`.trim(), results, drawn };
  }, { programs, draw });
  assert.ok(adapter, "no WebGPU adapter in this Chromium");
  let failed = 0;
  cases.forEach((c, i) => {
    try {
      assert.ok(!results[i].error, results[i].error);
      c.check(new Uint32Array(results[i].res));
    } catch (e) {
      failed += 1;
      console.log(`FAIL ${c.name}: ${String(e.message).slice(0, 600)}`);
    }
  });
  try {
    assert.ok(!drawn.error, drawn.error);
    for (let y = 0; y < draw.size; y++) {
      for (let x = 0; x < draw.size; x++) {
        const got = drawn.pixels.slice((y * draw.size + x) * 4, (y * draw.size + x) * 4 + 4);
        const want = x >= 8 ? [0.4, 0.6, 0.4, 1] : draw.clear;
        assert.ok(got.every((g, i) => Math.abs(g - want[i]) < 1e-5), `pixel (${x}, ${y}) = ${got}, expected ${want}`);
      }
    }
  } catch (e) {
    failed += 1;
    console.log(`FAIL a draw with a vertex and a pixel program: ${String(e.message).slice(0, 600)}`);
  }
  const total = cases.length + 1;
  console.log(`wgsl tests on WebGPU (${adapter}): ${total - failed} passed, ${failed} failed (${cases.length} kernels, 1 draw)`);
  process.exitCode = failed ? 1 : 0;
} finally {
  await browser.close();
  server.close();
}
