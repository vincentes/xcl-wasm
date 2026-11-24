// The page's demo: mounted into #demo by www/index.html when www/demo.js is there.
import { createRoot } from 'react-dom/client'
import XclDemo from './XclDemo'

const slot = document.getElementById('demo')
if (slot) createRoot(slot).render(<XclDemo />)
