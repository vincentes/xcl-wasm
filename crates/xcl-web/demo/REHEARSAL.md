# Rehearsal: XclDemo (the browser page's hero)

Written with Claude Code (Claude Fable 5.2) using the Stageshow skill, 1.1.1.

Gold: `hero-demo` (kept: 720 wide, a browser window pinned one 32px gutter in on a quiet well and
running off the bottom, a compact design at 358; changed: the screen is XCL's page, no satellite,
400 tall) and `dark-hero` (kept: one status pill whose mark and word change in place).
Beats: 1 "run" is picked, the pill arrives as Running · 2 the title's log replaces the output and
the pill turns to Finished · payoff holds 2.6s · 3 "inspect" is picked again and the first output
returns. Loop about 8.5s. Poster: the finished run.
Captured: on the real page (`node www/server.mjs`), headless Chrome on macOS, at 1360 and 390 wide:
the loop, `--states`, and the reduced-motion poster at 1360. Light theme only (the page has no dark one).
Gate: `rehearsal passed (necessary, not sufficient: references/verify.md and the fresh eye in review.md still stand)`
on all four runs.
Film:
- `empty pane` (the log arriving one line at a time left the panel empty) → the log now replaces
  the output in one step, which is also what the page does: `run` returns all of its output at once.
- `dark slab` (a dark output panel in a light window) → the panel is a light well, in the demo and on the page.
- `framing` at 390 (window narrower than tall) → the compact stage is 320 tall.
- `spills its slot` at 390 → the compact controls are a column.
- `kit override` (unused phone styles in the bundle) → the scene imports each kit part from its own file.
- `lands on a transform` → the pill, the views and the log lines keep a layer.
- `lopsided: span.ss-badge "Running" 5px left, 9px right` → stands: the spinner's arc is narrower than
  its 8px cell, which the pill pads as `dark-hero` does; the dot that replaces it is not flagged.
- `near aligned: "XCL in the browser" and "Choose file" start 8px apart` → stands: the button's
  outline is on the title's edge and its label is inset by the button's own padding.
Review: no fresh-eye review (`references/review.md`) was run.
Assumptions: the lines shown are ones the page prints for `xcl_testtitle.exe` given the exe alone
(checked against two builds of the module on 2026-11-23). No counts of passed checks are shown: they
depend on the build and on what the page is given.
Unresolved: film reports `inconclusive` for sampling gaps over 27ms in headless capture.
Not checked: Safari, Firefox, a real phone, live frame pacing, pause and resume when scrolled
offscreen, a resize across the 600px switch between the two designs.
