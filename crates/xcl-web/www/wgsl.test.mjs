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
const EXPORTS = SGPRS + LANES * VGPRS;
// the registers, then the exports of 64 targets
const DWORDS = EXPORTS + 64 * LANES * 4;
const exported = (file, target, lane, c) => file[EXPORTS + (target * LANES + lane) * 4 + c];
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
const test = (name, program, io, check, mem) => cases.push({ name, program, io, check, mem });

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

// --- more of the vector unit: one instruction each, against the same operation done here
const Z = (l) => mix(l, 3);
const u = (x) => x >>> 0;
const i24 = (x) => (x << 8) >> 8;
const big = (x) => BigInt(x >>> 0), sbig = (x) => BigInt(x | 0);
const lowBits = (w) => (w === 0 ? 0 : 0xffffffff >>> (32 - w));
const med3 = (a, b, c) => Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));
const clz = (x) => (x === 0 ? 0xffffffff : Math.clz32(x));
const ctz = (x) => (x === 0 ? 0xffffffff : 31 - Math.clz32(x & -x));
const bfeI = (v, o, w) => (w === 0 ? 0 : u((((v | 0) >> o) << (32 - w)) >> (32 - w)));
for (const [op, expect] of [
  ["V_LSHL_ADD_U32", (a, b, c) => u((a << (b & 31)) + c)],
  ["V_ADD_LSHL_U32", (a, b, c) => u((a + b) << (c & 31))],
  ["V_LSHL_OR_B32", (a, b, c) => u((a << (b & 31)) | c)],
  ["V_AND_OR_B32", (a, b, c) => u((a & b) | c)],
  ["V_OR3_B32", (a, b, c) => u(a | b | c)],
  ["V_XOR3_B32", (a, b, c) => u(a ^ b ^ c)],
  ["V_ADD3_U32", (a, b, c) => u(a + b + c)],
  ["V_XAD_U32", (a, b, c) => u((a ^ b) + c)],
  ["V_BFE_U32", (a, b, c) => u((a >>> (b & 31)) & lowBits(c & 31))],
  ["V_BFE_I32", (a, b, c) => bfeI(a, b & 31, c & 31)],
  ["V_BFI_B32", (a, b, c) => u((a & b) | (~a & c))],
  ["V_ALIGNBIT_B32", (a, b, c) => (c & 31 ? u((b >>> (c & 31)) | (a << (32 - (c & 31)))) : b)],
  ["V_MIN3_U32", (a, b, c) => Math.min(a, b, c)],
  ["V_MAX3_U32", (a, b, c) => Math.max(a, b, c)],
  ["V_MED3_U32", (a, b, c) => med3(a, b, c)],
  ["V_MIN3_I32", (a, b, c) => u(Math.min(a | 0, b | 0, c | 0))],
  ["V_MED3_I32", (a, b, c) => u(med3(a | 0, b | 0, c | 0))],
  ["V_MAD_U32_U24", (a, b, c) => Number((big(a & 0xffffff) * big(b & 0xffffff) + big(c)) & 0xffffffffn)],
  ["V_MAD_I32_I24", (a, b, c) => Number(BigInt.asUintN(32, BigInt(i24(a)) * BigInt(i24(b)) + sbig(c)))],
  ["V_SAD_U32", (a, b, c) => u(Math.max(a, b) - Math.min(a, b) + c)],
  ["V_MUL_HI_I32", (a, b) => Number(BigInt.asUintN(32, (sbig(a) * sbig(b)) >> 32n))],
  ["V_BFM_B32", (a, b) => u(lowBits(a & 31) << (b & 31))],
  ["V_BCNT_U32_B32", (a, b) => u(a.toString(2).replaceAll("0", "").length + b)],
  ["V_MUL_U32_U24", (a, b) => Number((big(a & 0xffffff) * big(b & 0xffffff)) & 0xffffffffn)],
  ["V_MUL_HI_U32_U24", (a, b) => Number((big(a & 0xffffff) * big(b & 0xffffff)) >> 32n)],
  ["V_MUL_I32_I24", (a, b) => Number(BigInt.asUintN(32, BigInt(i24(a)) * BigInt(i24(b))))],
]) {
  const three = expect.length === 3;
  test(op.toLowerCase(), [`${op} ENC_VOP3 VDST=5 SRC0=${V(1)} SRC1=${V(2)}${three ? ` SRC2=${V(3)}` : ""}`, "S_ENDPGM ENC_SOPP"], registers({ v: { 1: X, 2: Y, 3: Z } }), (res) =>
    assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => expect(X(l), Y(l), Z(l)))));
}
// one source; lanes 0 and 1 are 0 and all ones
const W = (l) => (l === 0 ? 0 : l === 1 ? 0xffffffff : mix(l, 4) >>> (l % 29));
for (const [op, expect] of [
  ["V_BFREV_B32", (a) => parseInt(a.toString(2).padStart(32, "0").split("").reverse().join(""), 2)],
  ["V_FFBH_U32", (a) => clz(a)],
  ["V_FFBL_B32", (a) => ctz(a)],
  ["V_FFBH_I32", (a) => clz(u((a | 0) < 0 ? ~a : a))],
  ["V_NOT_B32", (a) => u(~a)],
  ["V_CVT_F32_UBYTE1", (a) => bits((a >>> 8) & 0xff)],
  ["V_CVT_F32_UBYTE3", (a) => bits(a >>> 24)],
  ["V_CVT_F32_U32", (a) => bits(a)],
]) {
  test(op.toLowerCase(), [`${op} ENC_VOP1 VDST=5 SRC0=${V(1)}`, "S_ENDPGM ENC_SOPP"], registers({ v: { 1: W } }), (res) =>
    assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => expect(W(l)))));
}
// float to integer: out of range saturates, a NaN is 0
const FI = [0, -0, 0.5, -0.5, 1.5, -1.5, 2.5, 1e10, -1e10, NaN, Infinity, -Infinity, 2147483648, -2147483649, 4294967296, 7.99];
const F = (l) => bits(FI[l % FI.length] * (l < FI.length ? 1 : l));
const sat = (x, lo, hi) => (Number.isNaN(x) ? 0 : u(Math.min(hi, Math.max(lo, Math.trunc(x)))));
for (const [op, expect] of [
  ["V_CVT_I32_F32", (x) => sat(x, -2147483648, 2147483647)],
  ["V_CVT_U32_F32", (x) => sat(x, 0, 4294967295)],
  ["V_CVT_FLR_I32_F32", (x) => sat(Math.floor(x), -2147483648, 2147483647)],
  ["V_CVT_RPI_I32_F32", (x) => sat(Math.floor(f32(x + 0.5)), -2147483648, 2147483647)],
]) {
  test(op.toLowerCase(), [`${op} ENC_VOP1 VDST=5 SRC0=${V(1)}`, "S_ENDPGM ENC_SOPP"], registers({ v: { 1: F } }), (res) =>
    vreg(res, 5).forEach((got, l) => assert.equal(got, expect(float(F(l))), `${op} lane ${l}: ${float(F(l))}`)));
}
test("legacy multiply, multiply-adds with a constant and into the destination", [
  `V_MUL_LEGACY_F32 ENC_VOP2 VDST=5 SRC0=${V(1)} VSRC1=2`,
  `V_MADMK_F32 ENC_VOP2 VDST=6 SRC0=${V(3)} VSRC1=4 LIT=${bits(3)}`, // v3 · 3 + v4
  `V_MADAK_F32 ENC_VOP2 VDST=7 SRC0=${V(3)} VSRC1=4 LIT=${bits(3)}`, // v3 · v4 + 3
  `V_MOV_B32 ENC_VOP1 VDST=8 SRC0=${V(4)}`,
  `V_FMAC_F32 ENC_VOP2 VDST=8 SRC0=${V(3)} VSRC1=3`, // v8 += v3 · v3
  `V_MED3_F32 ENC_VOP3 VDST=9 SRC0=${V(3)} SRC1=${V(4)} SRC2=242`,
  `V_FRACT_F32 ENC_VOP1 VDST=10 SRC0=${V(3)}`,
  "S_ENDPGM ENC_SOPP",
  // 0 · ∞ and 0 · NaN are 0 in the legacy multiply
], registers({ v: { 1: (l) => (l < 2 ? 0 : bits(l)), 2: (l) => (l === 0 ? 0x7f800000 : l === 1 ? 0x7fc00000 : bits(0.5)), 3: (l) => bits(l * 0.25 - 4), 4: (l) => bits(l % 7) } }), (res) => {
  const [a, b] = [(l) => l * 0.25 - 4, (l) => l % 7];
  assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => (l < 2 ? 0 : bits(l * 0.5))));
  assert.deepEqual(vreg(res, 6), Array.from({ length: LANES }, (_, l) => bits(a(l) * 3 + b(l))));
  assert.deepEqual(vreg(res, 7), Array.from({ length: LANES }, (_, l) => bits(a(l) * b(l) + 3)));
  assert.deepEqual(vreg(res, 8), Array.from({ length: LANES }, (_, l) => bits(b(l) + a(l) * a(l))));
  assert.deepEqual(vreg(res, 9), Array.from({ length: LANES }, (_, l) => bits(med3(a(l), b(l), 1))));
  assert.deepEqual(vreg(res, 10), Array.from({ length: LANES }, (_, l) => bits(a(l) - Math.floor(a(l)))));
});
test("operand modifiers: |x|, −x, a scale, a clamp", [
  `V_ADD_F32 ENC_VOP3 VDST=5 SRC0=${V(1)} SRC1=${V(2)} ABS=1 NEG=2`, // |v1| − v2
  `V_MUL_F32 ENC_VOP3 VDST=6 SRC0=${V(1)} SRC1=${V(2)} OMOD=1`, // · 2
  `V_ADD_F32 ENC_VOP3 VDST=7 SRC0=${V(1)} SRC1=${V(2)} OMOD=3 CLAMP=1`, // clamp((v1 + v2) / 2)
  `V_CMP_LT_F32 ENC_VOP3 VDST=10 SRC0=${V(1)} SRC1=${V(2)} ABS=1 NEG=1`, // −|v1| < v2
  "S_ENDPGM ENC_SOPP",
], registers({ v: { 1: (l) => bits(l - 31.5), 2: (l) => bits((l % 4) - 2) } }), (res) => {
  const [a, b] = [(l) => l - 31.5, (l) => (l % 4) - 2];
  assert.deepEqual(vreg(res, 5), Array.from({ length: LANES }, (_, l) => bits(Math.abs(a(l)) - b(l))));
  assert.deepEqual(vreg(res, 6), Array.from({ length: LANES }, (_, l) => bits(a(l) * b(l) * 2)));
  assert.deepEqual(vreg(res, 7), Array.from({ length: LANES }, (_, l) => bits(Math.min(1, Math.max(0, (a(l) + b(l)) / 2)))));
  let m = 0n;
  for (let l = 0; l < LANES; l++) if (-Math.abs(a(l)) < b(l)) m |= 1n << BigInt(l);
  assert.deepEqual([res[10], res[11]], [Number(m & 0xffffffffn), Number(m >> 32n)]);
});
test("exp, log, sin and cos (to a few units in the last places)", [
  `V_EXP_F32 ENC_VOP1 VDST=5 SRC0=${V(1)}`,
  `V_LOG_F32 ENC_VOP1 VDST=6 SRC0=${V(2)}`,
  `V_SIN_F32 ENC_VOP1 VDST=7 SRC0=${V(3)}`,
  `V_COS_F32 ENC_VOP1 VDST=8 SRC0=${V(3)}`,
  "S_ENDPGM ENC_SOPP",
], registers({ v: { 1: (l) => bits(l * 0.25 - 8), 2: (l) => bits(0.01 + l * 3.7), 3: (l) => bits(l / 64) } }), (res) => {
  const near = (r, want, what, by = 2e-5) => vreg(res, r).forEach((got, l) => assert.ok(Math.abs(float(got) - want(l)) <= by * Math.max(1, Math.abs(want(l))), `${what} lane ${l}: ${float(got)} for ${want(l)}`));
  near(5, (l) => 2 ** (l * 0.25 - 8), "exp");
  near(6, (l) => Math.log2(f32(0.01 + l * 3.7)), "log");
  // the angle is in revolutions; a GPU's sine is an approximation (SwiftShader's is off by
  // up to 1.5e-4 here)
  near(7, (l) => Math.sin((l / 64) * 2 * Math.PI), "sin", 1e-3);
  near(8, (l) => Math.cos((l / 64) * 2 * Math.PI), "cos", 1e-3);
});
test("lanes by number: the first one on, a read, a write", [
  `V_READFIRSTLANE_B32 ENC_VOP1 VDST=4 SRC0=${V(1)}`,
  `V_READLANE_B32 ENC_VOP3 VDST=5 SRC0=${V(1)} SRC1=${S(0)}`,
  `V_WRITELANE_B32 ENC_VOP3 VDST=2 SRC0=${S(1)} SRC1=${S(0)}`, // off in EXEC, written all the same
  "S_ENDPGM ENC_SOPP",
], registers({ exec: [0xffffff00, 0], s: { 0: 40, 1: 0xabcdef }, v: { 1: (l) => 1000 + l, 2: (l) => l } }), (res) => {
  assert.deepEqual([res[4], res[5]], [1008, 1040]);
  assert.deepEqual(vreg(res, 2), Array.from({ length: LANES }, (_, l) => (l === 40 ? 0xabcdef : l)));
});

// --- more of the scalar unit: s10 = op(s0, s1) with SCC set before, then SCC into s12
const [SA, SB] = [0xf0001234, 0x00050009];
const scalar = (name, line, expect, a = SA, b = SB) =>
  test(name, [line, "S_CSELECT_B32 ENC_SOP2 SDST=12 SSRC0=129 SSRC1=128", "S_ENDPGM ENC_SOPP"], registers({ scc: 1, s: { 0: a, 1: b, 2: 0x80000001, 3: 0x3, 10: 0x00ff00ff, 11: 0x5 } }), (res) => {
    const want = expect(a, b);
    assert.deepEqual([res[10], res[11], res[12]].map((x, k) => (want[k] === null ? null : x)), want.map((x) => (x === null ? null : u(x))), name);
  });
// [result, s11 (5: untouched), scc (null: not this instruction's)]
for (const [op, expect] of [
  ["S_AND_B32", (a, b) => [a & b, 5, (a & b) !== 0]],
  ["S_NAND_B32", (a, b) => [~(a & b), 5, 1]],
  ["S_NOR_B32", (a, b) => [~(a | b), 5, 1]],
  ["S_XNOR_B32", (a, b) => [~(a ^ b), 5, 1]],
  ["S_ORN2_B32", (a, b) => [a | ~b, 5, 1]],
  ["S_ADDC_U32", (a, b) => [a + b + 1, 5, 0]],
  ["S_SUBB_U32", (a, b) => [a - b - 1, 5, 0]],
  ["S_MUL_HI_U32", (a, b) => [Number((big(a) * big(b)) >> 32n), 5, 1]],
  ["S_MUL_HI_I32", (a, b) => [Number(BigInt.asUintN(32, (sbig(a) * sbig(b)) >> 32n)), 5, 1]],
  ["S_LSHR_B32", (a, b) => [a >>> (b & 31), 5, 1]],
  ["S_LSHL2_ADD_U32", (a, b) => [(a << 2) + b, 5, 1]], // the shift carries out
  ["S_BFM_B32", (a, b) => [lowBits(a & 31) << (b & 31), 5, 1]],
  ["S_BFE_U32", (a, b) => [(a >>> 9) & 31, 5, 1]], // offset 9, 5 bits: 0b01001
  ["S_BFE_I32", (a, b) => [bfeI(a, 9, 5), 5, 1]],
  ["S_MAX_I32", (a, b) => [b, 5, 0]],
  ["S_MIN_U32", (a, b) => [b, 5, 0]],
  ["S_ABSDIFF_I32", (a, b) => [Math.abs((a | 0) - (b | 0)), 5, 1]],
  ["S_PACK_LH_B32_B16", (a, b) => [(a & 0xffff) | (b & 0xffff0000), 5, 1]],
]) scalar(op.toLowerCase(), `${op} ENC_SOP2 SDST=10 SSRC0=${S(0)} SSRC1=${S(1)}`, (a, b) => expect(a, b).map((x) => Number(x)));
for (const [op, expect] of [
  ["S_BREV_B32", (a) => [parseInt(a.toString(2).padStart(32, "0").split("").reverse().join(""), 2), 5, 1]],
  ["S_BCNT1_I32_B32", (a) => [10, 5, 1]], // f, 1, 2, 3, c
  ["S_BCNT0_I32_B32", (a) => [22, 5, 1]],
  ["S_FF1_I32_B32", (a) => [ctz(a), 5, 1]],
  ["S_FF0_I32_B32", (a) => [ctz(u(~a)), 5, 1]],
  ["S_FLBIT_I32_B32", (a) => [clz(a), 5, 1]],
  ["S_FLBIT_I32", (a) => [clz(u(~a)), 5, 1]],
  ["S_ABS_I32", (a) => [Math.abs(a | 0), 5, 1]],
  ["S_SEXT_I32_I8", (a) => [(a << 24) >> 24, 5, 1]],
  ["S_SEXT_I32_I16", (a) => [(a << 16) >> 16, 5, 1]],
  ["S_CMOV_B32", (a) => [a, 5, 1]],
  ["S_BITSET1_B32", (a) => [0x00ff00ff | (1 << (a & 31)), 5, 1]],
  ["S_BITSET0_B32", (a) => [0x00ff00ff & ~(1 << (a & 31)), 5, 1]],
]) scalar(op.toLowerCase(), `${op} ENC_SOP1 SDST=10 SSRC0=${S(0)}`, expect, 0xf000123c);
scalar("s_cmpk_gt_i32", "S_CMPK_GT_I32 ENC_SOPK SDST=0 SIMM16=0xffff", () => [0x00ff00ff, 5, 0]); // negative > −1: no
scalar("s_cmpk_gt_u32", "S_CMPK_GT_U32 ENC_SOPK SDST=0 SIMM16=0xffff", () => [0x00ff00ff, 5, 1]);
scalar("s_addk_i32", "S_ADDK_I32 ENC_SOPK SDST=10 SIMM16=0xfff0", () => [0x00ff00ef, 5, 0]);
scalar("s_mulk_i32", "S_MULK_I32 ENC_SOPK SDST=10 SIMM16=0xfffe", () => [Math.imul(0x00ff00ff, -2), 5, 1]);
scalar("s_cmovk_i32", "S_CMOVK_I32 ENC_SOPK SDST=10 SIMM16=0x8000", () => [0xffff8000, 5, 1]);
scalar("s_bitcmp1_b32", `S_BITCMP1_B32 ENC_SOPC SSRC0=${S(0)} SSRC1=132`, () => [0x00ff00ff, 5, 1]); // bit 4 of …34
// 64 bits: s0:s1 and s2:s3 (0x00000003_80000001)
const wide = (a, b) => (big(b) << 32n) | big(a);
const halves = (x) => [Number(x & 0xffffffffn), Number((x >> 32n) & 0xffffffffn)];
scalar("s_lshl_b64", `S_LSHL_B64 ENC_SOP2 SDST=10 SSRC0=${S(0)} SSRC1=${S(3)}`, (a, b) => [...halves(wide(a, b) << 3n), 1]);
scalar("s_lshl_b64 by 36", `S_LSHL_B64 ENC_SOP2 SDST=10 SSRC0=${S(0)} SSRC1=164`, (a, b) => [...halves(wide(a, b) << 36n), 1]);
scalar("s_lshr_b64", `S_LSHR_B64 ENC_SOP2 SDST=10 SSRC0=${S(0)} SSRC1=${S(3)}`, (a, b) => [...halves(wide(a, b) >> 3n), 1]);
scalar("s_lshr_b64 by 36", `S_LSHR_B64 ENC_SOP2 SDST=10 SSRC0=${S(0)} SSRC1=164`, (a, b) => [...halves(wide(a, b) >> 36n), 1]);
scalar("s_nand_b64", `S_NAND_B64 ENC_SOP2 SDST=10 SSRC0=${S(0)} SSRC1=${S(2)}`, (a, b) => [~(a & 0x80000001), ~(b & 3), 1]);
scalar("s_cselect_b64", `S_CSELECT_B64 ENC_SOP2 SDST=10 SSRC0=${S(0)} SSRC1=${S(2)}`, (a, b) => [a, b, 1]);
scalar("s_bcnt1_i32_b64", `S_BCNT1_I32_B64 ENC_SOP1 SDST=10 SSRC0=${S(2)}`, () => [4, 5, 1]);
scalar("s_ff1_i32_b64", `S_FF1_I32_B64 ENC_SOP1 SDST=10 SSRC0=${S(0)}`, () => [36, 5, 1], 0, 0x10);
scalar("s_cmp_eq_u64", `S_CMP_EQ_U64 ENC_SOPC SSRC0=${S(0)} SSRC1=${S(2)}`, () => [0x00ff00ff, 5, 0]);
scalar("s_cmp_lg_u64", `S_CMP_LG_U64 ENC_SOPC SSRC0=${S(0)} SSRC1=${S(2)}`, () => [0x00ff00ff, 5, 1]);
test("s_andn1_saveexec_b64: EXEC saved, then the lanes the source does not have", [
  `S_ANDN1_SAVEEXEC_B64 ENC_SOP1 SDST=10 SSRC0=${S(0)}`,
  "S_ENDPGM ENC_SOPP",
], registers({ exec: [0xffff0000, 0xff], s: { 0: 0x0ff00000, 1: 0xf } }), (res) => assert.deepEqual([res[10], res[11], res[126], res[127], res[125]], [0xffff0000, 0xff, 0xf00f0000, 0xf0, 1]));

// --- memory and exports (the HLSL back end's gpu_memory_and_exports)
const MEM = Array.from({ length: 256 }, (_, i) => 0x1000 + i);
test("loads through a pointer and a descriptor, exports", [
  "S_LOAD_DWORDX2 ENC_SMEM SDATA=8 SBASE=0 OFFSET=0x10 SOFFSET=125",
  "S_BUFFER_LOAD_DWORD ENC_SMEM SDATA=10 SBASE=6 OFFSET=4 SOFFSET=125",
  "S_BUFFER_LOAD_DWORD ENC_SMEM SDATA=11 SBASE=6 OFFSET=0x400 SOFFSET=125", // out of range
  "BUFFER_LOAD_DWORD ENC_MUBUF VDATA=3 VADDR=0 SRSRC=3 SOFFSET=128 OFFSET=8 OFFEN=1",
  `V_MOV_B32 ENC_VOP1 VDST=4 SRC0=${S(8)}`,
  "EXP ENC_EXP TGT=12 EN=15 VSRC0=3 VSRC1=4 VSRC2=0 VSRC3=0 DONE=1",
  "V_CVT_PKRTZ_F16_F32 ENC_VOP2 VDST=5 SRC0=242 VSRC1=6",
  "EXP ENC_EXP TGT=0 EN=3 VSRC0=5 VSRC1=5 COMPR=1 DONE=1 VM=1",
  "S_ENDPGM ENC_SOPP",
  // s0: a pointer to 0; s12..14: a descriptor of 0x100 bytes at 0x40, no stride
], registers({ s: { 0: 0, 12: 0x40, 13: 0, 14: 0x100 }, v: { 0: (l) => 4 * l, 6: () => bits(0.5) } }), (res) => {
  assert.deepEqual([res[8], res[9], res[10], res[11]], [0x1004, 0x1005, 0x1000 + 0x44 / 4, 0]);
  for (let l = 0; l < LANES; l++) {
    const want = 8 + 4 * l < 0x100 ? 0x1000 + (0x40 + 8 + 4 * l) / 4 : 0;
    assert.equal(res[SGPRS + l * VGPRS + 3], want, `lane ${l} buffer_load_dword`);
    assert.deepEqual([0, 1, 2].map((c) => exported(res, 12, l, c)), [want, 0x1004, 4 * l], `lane ${l} pos0`);
    assert.deepEqual([exported(res, 0, l, 0), exported(res, 0, l, 1)], [0x3c00, 0x3800], `lane ${l} mrt0 (1.0, 0.5) as halves`);
  }
}, MEM);
test("a descriptor with a stride: records by index, x4, past the end 0", [
  "BUFFER_LOAD_DWORDX4 ENC_MUBUF VDATA=4 VADDR=0 SRSRC=3 SOFFSET=128 OFFSET=4 IDXEN=1",
  "S_ENDPGM ENC_SOPP",
  // 20 records of 16 bytes at 0x20
], registers({ s: { 12: 0x20, 13: 16 << 16, 14: 20 }, v: { 0: (l) => l } }), (res) => {
  for (let l = 0; l < LANES; l++) {
    const first = 0x1000 + (0x20 + 16 * l + 4) / 4;
    assert.deepEqual([4, 5, 6, 7].map((r) => res[SGPRS + l * VGPRS + r]), l < 20 ? [first, first + 1, first + 2, first + 3] : [0, 0, 0, 0], `lane ${l}`);
  }
}, MEM);
// halves rounded toward zero, against a conversion done here
const halfRtz = (f) => {
  const b = bits(f), sign = (b >>> 16) & 0x8000, u = b & 0x7fffffff, e = (u >>> 23) - 127;
  if (u > 0x7f800000) return sign | 0x7e00;
  if (u === 0x7f800000) return sign | 0x7c00;
  if (e > 15) return sign | 0x7bff;
  // the largest half not above |f|: search (halves are ordered as their bits)
  let h = 0;
  const value = (x) => ((x >> 10) === 0 ? (x & 0x3ff) * 2 ** -24 : (0x400 | (x & 0x3ff)) * 2 ** ((x >> 10) - 25));
  for (let bit = 0x4000; bit; bit >>= 1) if ((h | bit) < 0x7c00 && value(h | bit) <= Math.abs(f)) h |= bit;
  return sign | h;
};
const HALVES = [0, -0, 1, -1, 0.1, -0.1, 1 / 3, 65504, 65519.9, 65520, 1e9, -1e9, Infinity, -Infinity, NaN, 2 ** -14, 2 ** -14 * 0.999, 2 ** -24, 2 ** -24 * 1.5, 2 ** -25, 2 ** -24 * 0.99, 6.1e-5, 1.0009765625, 1.00048828125, 1.0014, 3.14159, 1000.7, -2.5e-6];
const H = (l) => bits(HALVES[l % HALVES.length] * (l >= HALVES.length ? 1.37 + l : 1));
test("v_cvt_pkrtz_f16_f32 rounds toward zero", [
  `V_CVT_PKRTZ_F16_F32 ENC_VOP2 VDST=5 SRC0=${V(1)} VSRC1=2`,
  "S_ENDPGM ENC_SOPP",
], registers({ v: { 1: H, 2: (l) => H(63 - l) } }), (res) => {
  for (let l = 0; l < LANES; l++) {
    const want = (halfRtz(float(H(l))) | (halfRtz(float(H(63 - l))) << 16)) >>> 0;
    assert.equal(res[SGPRS + l * VGPRS + 5].toString(16), want.toString(16), `lane ${l}: ${float(H(l))}, ${float(H(63 - l))}`);
  }
});

// --- draws: a vertex and a pixel program (the HLSL back end's gpu_graphics_vs_ps_draw)
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
  expect: (x) => (x >= 8 ? [0.4, 0.6, 0.4, 1] : [0, 0, 1, 1]),
};
// the same triangle; the pixel program takes its colour from memory through a descriptor
// (s8..), adds the attribute's red and exports halves
const draw2 = {
  ...draw,
  pixel: xcl.wgslPixel(xcl.assemble([
    "S_BUFFER_LOAD_DWORDX4 ENC_SMEM SDATA=16 SBASE=4 OFFSET=8 SOFFSET=125", // s16..19: the colour
    "V_INTERP_P2_F32 ENC_VINTRP VDST=4 VSRC=1 ATTR=0 ATTRCHAN=0",
    `V_ADD_F32 ENC_VOP2 VDST=4 SRC0=${S(16)} VSRC1=4`,
    `V_MOV_B32 ENC_VOP1 VDST=6 SRC0=${S(17)}`,
    `V_MOV_B32 ENC_VOP1 VDST=8 SRC0=${S(18)}`,
    `V_MOV_B32 ENC_VOP1 VDST=9 SRC0=${S(19)}`,
    `V_CVT_PKRTZ_F16_F32 ENC_VOP2 VDST=5 SRC0=${V(4)} VSRC1=6`,
    `V_CVT_PKRTZ_F16_F32 ENC_VOP2 VDST=7 SRC0=${V(8)} VSRC1=9`,
    "EXP ENC_EXP TGT=0 EN=15 VSRC0=5 VSRC1=7 COMPR=1 DONE=1 VM=1",
  ]), 0x302),
  sgprs: Object.assign(new Array(128).fill(0), { 8: 16, 9: 0, 10: 64 }),
  // the descriptor: 64 bytes at 16; the colour is its dwords 2..5 (0.125 0.25 0.5 1: exact halves)
  mem: [...new Array(6).fill(0), bits(0.125), bits(0.25), bits(0.5), bits(1), ...new Array(22).fill(0)],
  // 0.8 as a float plus 0.125, cut to a half
  expect: () => [halfValue(halfRtz(f32(f32(0.8) + 0.125))), 0.25, 0.5, 1],
};
function halfValue(h) {
  return (h >> 10) === 0 ? (h & 0x3ff) * 2 ** -24 : (0x400 | (h & 0x3ff)) * 2 ** ((h >> 10) - 25);
}
// a texture: the pixel program samples it at the attribute (a coordinate over the target, so
// each pixel gets one texel with a nearest sampler) and loads the texel at the pixel's place
const texel = (x, y) => [x * 16 + y, 255 - x * 8, y * 16, 255];
const draw3 = {
  ...draw,
  pixel: xcl.wgslPixel(xcl.assemble([
    "V_INTERP_P2_F32 ENC_VINTRP VDST=4 VSRC=1 ATTR=0 ATTRCHAN=0",
    "V_INTERP_P2_F32 ENC_VINTRP VDST=5 VSRC=1 ATTR=0 ATTRCHAN=1",
    "IMAGE_SAMPLE ENC_MIMG VDATA=8 VADDR=4 SRSRC=2 SSAMP=4 DMASK=7 DIM=1", // v8..10 = rgb
    `V_CVT_I32_F32 ENC_VOP1 VDST=12 SRC0=${V(2)}`, // the pixel's x and y
    `V_CVT_I32_F32 ENC_VOP1 VDST=13 SRC0=${V(3)}`,
    "IMAGE_LOAD ENC_MIMG VDATA=11 VADDR=12 SRSRC=2 DMASK=1 DIM=1", // v11 = red
    "EXP ENC_EXP TGT=0 EN=15 VSRC0=8 VSRC1=9 VSRC2=10 VSRC3=11 DONE=1 VM=1",
  ]), 0x302),
  vertices: [[-1, -1, 0, 0], [-1, 3, 0, 2], [3, -1, 2, 0]].flatMap((v) => [...v, 0, 0]),
  textures: [{ size: 16, data: Array.from({ length: 256 }, (_, k) => texel(k % 16, k >> 4)).flat() }],
  // slot 0: sampled; slot 1: the same texture loaded
  images: [{ texture: 0, sampler: true }, { texture: 0 }],
  // the target's rows go down, the coordinate up
  expect: (x, y) => [...texel(x, 15 - y).slice(0, 3), texel(x, y)[0]].map((c) => c / 255),
};
const draws = [draw, draw2, draw3].map(({ expect, ...d }) => d);

// --- run them
const programs = cases.map((c) => ({ name: c.name, wgsl: xcl.wgslKernel(xcl.assemble(c.program)), io: [...c.io], mem: c.mem }));
assert.throws(() => xcl.wgslKernel(xcl.assemble([`V_CVT_F16_F32 ENC_VOP1 VDST=0 SRC0=${V(1)}`])), /v_cvt_f16_f32/, "what is missing is named");

const server = createServer((req, res) => res.writeHead(200, { "content-type": "text/html" }).end("<!doctype html><title>wgsl</title>"));
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU"] });
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const { adapter, results, drawn } = await page.evaluate(async ({ programs, draws }) => {
    const gpuAdapter = await navigator.gpu?.requestAdapter();
    if (!gpuAdapter) return { adapter: null, results: [] };
    const device = await gpuAdapter.requestDevice();
    const results = [];
    for (const { wgsl, io, mem } of programs) {
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
      const entries = [{ binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } }];
      if (mem) {
        const memory = device.createBuffer({ size: mem.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(memory, 0, new Uint32Array(mem));
        entries.push({ binding: 2, resource: { buffer: memory } });
      }
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
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
    const drawn = [];
    for (const draw of draws) {
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
        drawn.push({ error: errors.join("; ") });
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
        const entries = [{ binding: 0, resource: { buffer: sgprs } }];
        if (draw.mem) {
          const memory = device.createBuffer({ size: draw.mem.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(memory, 0, new Uint32Array(draw.mem));
          entries.push({ binding: 1, resource: { buffer: memory } });
        }
        pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
        if (draw.images) {
          const textures = draw.textures.map((t) => {
            const texture = device.createTexture({ size: [t.size, t.size], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
            device.queue.writeTexture({ texture }, new Uint8Array(t.data), { bytesPerRow: t.size * 4 }, [t.size, t.size]);
            return texture;
          });
          const sampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
          const images = draw.images.flatMap((image, k) => [{ binding: 2 * k, resource: textures[image.texture].createView() }, ...(image.sampler ? [{ binding: 2 * k + 1, resource: sampler }] : [])]);
          pass.setBindGroup(1, device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: images }));
        }
        pass.setVertexBuffer(0, vertices);
        pass.draw(3);
        pass.end();
        encoder.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: n * 16 }, [n, n]);
        device.queue.submit([encoder.finish()]);
        const error = await device.popErrorScope();
        if (error) drawn.push({ error: error.message });
        else {
          await read.mapAsync(GPUMapMode.READ);
          drawn.push({ pixels: [...new Float32Array(read.getMappedRange())] });
        }
      }
    }
    const info = gpuAdapter.info ?? {};
    return { adapter: `${info.vendor ?? "?"} ${info.architecture ?? ""} ${info.description ?? ""}`.trim(), results, drawn };
  }, { programs, draws });
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
  [draw, draw2, draw3].forEach((d, n) => {
    try {
      assert.ok(!drawn[n].error, drawn[n].error);
      for (let y = 0; y < d.size; y++) {
        for (let x = 0; x < d.size; x++) {
          const got = drawn[n].pixels.slice((y * d.size + x) * 4, (y * d.size + x) * 4 + 4);
          const want = d.expect(x, y);
          assert.ok(got.every((g, i) => Math.abs(g - want[i]) < 1e-5), `pixel (${x}, ${y}) = ${got}, expected ${want}`);
        }
      }
    } catch (e) {
      failed += 1;
      console.log(`FAIL draw ${n + 1} with a vertex and a pixel program: ${String(e.message).slice(0, 600)}`);
    }
  });
  const total = cases.length + 3;
  console.log(`wgsl tests on WebGPU (${adapter}): ${total - failed} passed, ${failed} failed (${cases.length} kernels, 3 draws)`);
  process.exitCode = failed ? 1 : 0;
} finally {
  await browser.close();
  server.close();
}
