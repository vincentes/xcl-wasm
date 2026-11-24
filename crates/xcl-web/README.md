# xcl-web

The part of XCL that needs no Windows, built as a WebAssembly module for the browser.

**What it does today:** `xcl inspect` (a game exe / DLL: headers, imports and the XCL module each
imported DLL routes to) and `xcl disasm` (console shader blobs: containers and RDNA2 listing), on a
file picked in the page. The file stays in the tab. It does **not** run games: the loader, the
console libraries and D3D12.X are Windows code and are not in this build.

`inspect` here does not say whether XCL implements each imported function (the Windows build
does): the function tables are part of the Windows-only code.

## Build

The target is `wasm64-unknown-unknown` (64-bit memory, for title-sized address spaces later). It
is a tier 3 Rust target: it needs a nightly toolchain and builds its own standard library.

```sh
rustup toolchain install nightly --component rust-src
cargo +nightly build -p xcl-web --release --target wasm64-unknown-unknown -Zbuild-std=std,panic_abort
cp target/wasm64-unknown-unknown/release/xcl_web.wasm crates/xcl-web/www/
```

## Test

```sh
cargo test -p xcl --lib          # the logic (also runs off Windows: Windows-only tests are left out)
node crates/xcl-web/www/smoke.mjs target/wasm64-unknown-unknown/release/xcl_web.wasm   # Node 22+
```

## Run

Serve `crates/xcl-web/www/` with any static file server and open it, e.g.
`python3 -m http.server -d crates/xcl-web/www`. The browser needs 64-bit WebAssembly memory
(memory64).

## Layout

- `src/lib.rs` – the module's exports (no bindings generator; pointers and sizes are 64-bit)
- `www/xcl.mjs` – loader: `load(url | bytes)` → `{ version, inspect, disasm }`
- `www/index.html` – the page
- `www/smoke.mjs` – smoke test of the built module under Node
