// Builds the page's demo into www/ (demo.js, demo.css), next to the page:
//   npm ci --prefix crates/xcl-web/demo && npm run build --prefix crates/xcl-web/demo
import { build } from "esbuild";
import { fileURLToPath } from "node:url";

const here = (path) => fileURLToPath(new URL(path, import.meta.url));
await build({
  entryPoints: { demo: here("src/main.tsx") },
  outdir: here("../www"),
  bundle: true,
  format: "esm",
  target: "es2022",
  minify: true,
  jsx: "automatic",
  define: { "process.env.NODE_ENV": '"production"' },
  logLevel: "info",
});
