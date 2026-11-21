// Produce the static GitHub Pages site. The deployed application has no
// functions, API rewrites, credentials, cookies, or external compute origin.
import { execFileSync } from 'node:child_process'
import { copyFileSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

// Project pages are served from /<repo>/; override with PAGES_BASE for a custom domain.
const base = process.env.PAGES_BASE ?? '/hypercast/'
const output = resolve('build/pages')
rmSync(output, { recursive: true, force: true })
execFileSync('npm', ['run', 'build', '--', '--outDir', output, '--emptyOutDir', '--base', base], {
  cwd: resolve('web'),
  stdio: 'inherit',
  env: { ...process.env, VITE_OFFLINE_DEMO: 'true', VITE_PUBLIC_DEMO: 'false' },
})
// GitHub Pages serves 404.html for unknown paths, which gives direct-route reloads the SPA.
copyFileSync(resolve(output, 'index.html'), resolve(output, '404.html'))
writeFileSync(resolve(output, '.nojekyll'), '')
