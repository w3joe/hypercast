// Vercel Build Output API: static React files plus a same-origin API rewrite.
// No Python, GPU work, cookies, or credentials are bundled into Vercel Functions.
import { execFileSync } from 'node:child_process'
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const raw = process.env.DEMO_API_ORIGIN
if (!raw) throw new Error('Set DEMO_API_ORIGIN to the deployed Modal API HTTPS origin.')
const origin = new URL(raw)
if (origin.protocol !== 'https:' || !origin.hostname.endsWith('.modal.run') ||
    origin.username || origin.password || origin.search || origin.hash || origin.pathname !== '/') {
  throw new Error('DEMO_API_ORIGIN must be an HTTPS *.modal.run origin, without a path or credentials.')
}
const output = resolve('.vercel/output')
rmSync(output, { recursive: true, force: true })
mkdirSync(output, { recursive: true })
execFileSync('npm', ['run', 'build', '--', '--outDir', resolve(output, 'static')], {
  cwd: resolve('web'), stdio: 'inherit', env: { ...process.env, VITE_PUBLIC_DEMO: 'true' },
})
writeFileSync(resolve(output, 'config.json'), JSON.stringify({
  version: 3,
  routes: [
    { src: '/api/(.*)', dest: `${origin.origin}/api/$1`, headers: { 'Cache-Control': 'no-store' } },
    { src: '/(.*)', headers: { 'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'strict-origin-when-cross-origin',
        'X-Frame-Options': 'DENY' }, continue: true },
    { handle: 'filesystem' },
    { src: '/(.*)', dest: '/index.html' },
  ],
}, null, 2))
