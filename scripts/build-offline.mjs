// Produce a static Vercel Build Output directory. The deployed application has
// no functions, API rewrites, credentials, cookies, or external compute origin.
import { execFileSync } from 'node:child_process'
import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const output = resolve('.vercel/output')
rmSync(output, { recursive: true, force: true })
mkdirSync(output, { recursive: true })
execFileSync('npm', ['run', 'build', '--', '--outDir', resolve(output, 'static')], {
  cwd: resolve('web'),
  stdio: 'inherit',
  env: { ...process.env, VITE_OFFLINE_DEMO: 'true', VITE_PUBLIC_DEMO: 'false' },
})
writeFileSync(resolve(output, 'config.json'), JSON.stringify({
  version: 3,
  routes: [
    {
      src: '/(.*)',
      headers: {
        'X-Content-Type-Options': 'nosniff',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'X-Frame-Options': 'DENY',
        'Permissions-Policy': 'camera=(), microphone=(), geolocation=(), payment=()',
      },
      continue: true,
    },
    { handle: 'filesystem' },
    { src: '/(.*)', dest: '/index.html' },
  ],
}, null, 2))
