// Each part from its own file, not kit/index.ts: the page then carries only these parts' styles
import type { Act } from '../kit/act'
import { Badge } from '../kit/controls/Badge'
import { Button } from '../kit/controls/Button'
import { Segmented } from '../kit/controls/Segmented'
import type { Layout } from '../kit/looks'
import { Roll } from '../kit/motion/Roll'
import { Spinner } from '../kit/parts/Spinner'
import { Stage } from '../kit/stage'
import { Bar, Window } from '../kit/window'
import './xcl-demo.css'

// The hero of XCL's browser page: the page itself, in a browser window. A
// title's exe is already picked and inspected; the cursor picks "run", the
// pill says Running while the title's x64 code runs on the guest (the page
// keeps the old output meanwhile, as the real one does), then the title's own
// log replaces it and the same pill says Finished. Picking "inspect" again is
// the reset, so the loop ends where it began.
//
// Every line of output is one the page prints for XCL's test title
// (tests/testtitle): keep them true when the output changes. No counts of
// passed checks are shown, because they depend on what the page is given.
//
// Real strings: the file, the three commands, the status, and three or four
// lines of each output. The rest of each output is placeholder at the lines'
// own pitch, running off the bottom of the stage.

type S = { cmd: number; state: 'idle' | 'running' | 'done'; log: boolean }

const initial: S = { cmd: 0, state: 'idle', log: false }

const FILE = 'xcl_testtitle.exe'
const COMMANDS = ['inspect', 'disasm', 'run']

const INSPECT: Record<Layout, [string, string][]> = {
  wide: [
    ['File:', FILE],
    ['Type:', 'EXE x64, subsystem 3'],
    ['Console game:', 'yes (imports kernelx.dll, xgameruntime.dll)'],
    ['Imports:', '258 functions from 30 DLLs'],
  ],
  compact: [
    ['File:', FILE],
    ['Type:', 'EXE x64, subsystem 3'],
    ['Imports:', '258 functions from 30 DLLs'],
  ],
}

const PASSED: Record<Layout, string[]> = {
  wide: [
    'static TLS: template on main thread',
    'kernelx: XMemAlloc alignment + zeroing',
    'kernelx: TitleMemoryStatus',
    'modules: GetModuleHandle(NULL) is the game image',
  ],
  compact: ['static TLS: template on main thread', 'kernelx: TitleMemoryStatus'],
}

// Placeholder lines under each output, as [indent, width] in px
const IMPORTS = [[14, 150], [14, 118], [14, 172], [14, 96], [14, 140], [14, 128], [14, 164], [14, 110], [14, 146], [14, 132]]
const LOG = [[0, 232], [0, 176], [0, 258], [0, 150], [0, 204], [0, 240], [0, 168], [0, 222], [0, 190]]

async function script(a: Act<S>) {
  await a.wait(900)

  // 1. Pick "run": the thumb slides and the pill arrives, already working
  a.mark('run')
  await a.move({ at: 'cmd-run', x: 0.78, y: 0.7 })
  await a.wait(200)
  await a.click()
  await a.set({ cmd: 2, state: 'running' })
  await a.wait(500)
  await a.move('rest')
  await a.wait(1300)

  // 2. It ran to its end: the title's log takes the output's place, and the
  //    same pill says so a moment after
  a.mark('done')
  await a.set({ log: true })
  await a.wait(350)
  await a.set({ state: 'done' })
  await a.wait(2600)

  // 3. Back to "inspect": the log and the pill leave together and the first
  //    output returns
  a.mark('reset')
  await a.move({ at: 'cmd-inspect', x: 0.78, y: 0.7 })
  await a.wait(200)
  await a.click()
  await a.set({ cmd: 0, state: 'idle', log: false })
  await a.wait(500)
  await a.move('rest')
  await a.wait(300)
}

export default function XclDemo() {
  return (
    <Stage<S>
      width={720}
      height={400}
      compact={{ below: 600, width: 358, height: 320 }}
      initial={initial}
      poster={{ cmd: 2, state: 'done', log: true }}
      script={script}
      cursor={{ at: 'rest' }}
      className="xd"
    >
      {(s, _rect, layout) => {
        const nth = (i: number) => ({ ['--i' as string]: i })
        return (
          <Window url="localhost:8000" className="xd-win">
            <div className="xd-page">
              <div className="xd-title">XCL in the browser</div>
              <div className="xd-controls">
                <span className="xd-file">
                  <Button size="small">Choose file</Button>
                  <span className="xd-name">{FILE}</span>
                </span>
                <span className="xd-cmd">
                  <Segmented className="xd-seg" items={COMMANDS} value={s.cmd} targets={['cmd-inspect', 'cmd-disasm', 'cmd-run']} />
                  <span className="xd-state" data-on={s.state !== 'idle' || undefined}>
                    <Badge tone={s.state === 'done' ? 'success' : 'neutral'} icon={<Mark done={s.state === 'done'} />}>
                      <Roll variant="fade">{s.state === 'done' ? 'Finished' : 'Running'}</Roll>
                    </Badge>
                  </span>
                </span>
              </div>
              <div className="xd-out" data-crop>
                <div className="xd-view" data-on={!s.log || undefined}>
                  {INSPECT[layout].map(([k, v]) => (
                    <div key={k} className="xd-line">
                      <span className="xd-key">{k}</span>
                      {v}
                    </div>
                  ))}
                  {IMPORTS.map(([x, w], i) => (
                    <div key={i} className="xd-line">
                      <Bar w={w} h={7} r={3.5} tone={3} style={{ marginLeft: x }} />
                    </div>
                  ))}
                </div>
                <div className="xd-view xd-log" data-on={s.log || undefined}>
                  <div className="xd-line xd-in" style={nth(0)}>
                    <span className="xd-dim">xcl_testtitle: start</span>
                  </div>
                  {PASSED[layout].map((t, i) => (
                    <div key={t} className="xd-line xd-in" style={nth(i + 1)}>
                      <span className="xd-ok">[PASS]</span>
                      {t}
                    </div>
                  ))}
                  {LOG.map(([x, w], i) => (
                    <div key={i} className="xd-line xd-in" style={nth(PASSED[layout].length + 1 + i)}>
                      <Bar w={w} h={7} r={3.5} tone={3} style={{ marginLeft: x }} />
                    </div>
                  ))}
                </div>
                <span className="xd-rest" data-target="rest" />
              </div>
            </div>
          </Window>
        )
      }}
    </Stage>
  )
}

// The pill's mark: a spinner while the title runs and a dot once it has
// finished, crossfading in one cell so the word beside it never moves
function Mark({ done }: { done: boolean }) {
  return (
    <span className="xd-mark" data-done={done || undefined}>
      <Spinner size={8} />
      <i className="xd-dot" />
    </span>
  )
}
