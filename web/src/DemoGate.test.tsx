import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import DemoGate from './DemoGate'
import { EvaluationPanel, RunControls } from './BuilderControls'
import type { Catalog, EvaluationSpec, GraphSpec } from './types'

const limits = { run_seconds: 120, epochs: 5, runs_per_day: 3, window: 60, horizon: 20, parameters: 15000000, rows: 512 }
const evaluation = { preset: 'quick', target_column: 'Signal', cells: [{ window: 10, horizon: 1 }], seeds: [7], epochs: 3 } as EvaluationSpec
let authenticated = false
let remaining = 3
let submitted: unknown

function wrap(child: React.ReactNode) {
  const cache = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={cache}>{child}</QueryClientProvider>)
}

beforeEach(() => {
  authenticated = false; remaining = 3; submitted = undefined
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
    const path = String(input)
    let data: unknown = {}
    if (path.endsWith('/session')) data = { authenticated, remaining_runs: remaining, demo_available: true, limits }
    if (path.endsWith('/request-code')) data = { challenge_id: 'challenge' }
    if (path.endsWith('/verify')) {
      const body = JSON.parse(String(options?.body))
      if (body.code !== '123456') return new Response(JSON.stringify({ detail: 'That code is invalid or expired.' }), { status: 422 })
      authenticated = true; data = { authenticated }
    }
    if (path.endsWith('/logout')) authenticated = false
    if (path.endsWith('/jobs') && options?.method === 'POST') {
      submitted = JSON.parse(String(options.body)); remaining -= 1
      data = { id: 'demo-job', status: { state: 'queued' } }
    }
    return new Response(JSON.stringify(data), { headers: { 'Content-Type': 'application/json' } })
  }))
})

it('verifies email without a password and opens the workspace only after a correct code', async () => {
  wrap(<DemoGate><p>Private workspace</p></DemoGate>)
  fireEvent.change(await screen.findByLabelText('Email address'), { target: { value: 'visitor@example.com' } })
  expect(screen.queryByText('Private workspace')).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Email me a code' }))
  fireEvent.change(await screen.findByLabelText('Verification code'), { target: { value: '000000' } })
  fireEvent.click(screen.getByRole('button', { name: 'Open the demo' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('invalid or expired')
  expect(screen.queryByText('Private workspace')).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Verification code'), { target: { value: '123456' } })
  fireEvent.click(screen.getByRole('button', { name: 'Open the demo' }))
  expect(await screen.findByText('Private workspace')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Sign out' }))
  expect(await screen.findByLabelText('Email address')).toBeInTheDocument()
  expect(screen.queryByText('Private workspace')).not.toBeInTheDocument()
})

it('exposes only the limited demo settings', () => {
  wrap(<EvaluationPanel catalog={{ demo: limits } as Catalog} evaluation={evaluation} onChange={vi.fn()} />)
  expect(screen.getByLabelText('Epochs')).toHaveAttribute('max', '5')
  expect(screen.getByLabelText('Window')).toHaveAttribute('max', '60')
  expect(screen.queryByLabelText('Preset')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('Device')).not.toBeInTheDocument()
})

it('fixes submission to L4 and refreshes the remaining allowance', async () => {
  authenticated = true
  const onQueued = vi.fn()
  wrap(<RunControls demo={limits} architecture={{} as GraphSpec} evaluation={evaluation} disabled={false} onQueued={onQueued} />)
  const run = screen.getByRole('button', { name: 'Run demo experiment' })
  await waitFor(() => expect(run).toBeEnabled())
  expect(screen.queryByLabelText('Run method')).not.toBeInTheDocument()
  fireEvent.click(run)
  await waitFor(() => expect(onQueued).toHaveBeenCalled())
  expect(submitted).toMatchObject({ execution: { target: 'modal', gpu: 'L4' } })
  expect(await screen.findByText(/2 runs left today/)).toBeInTheDocument()
})

it('disables runs when the daily allowance is exhausted', async () => {
  authenticated = true; remaining = 0
  wrap(<RunControls demo={limits} architecture={{} as GraphSpec} evaluation={evaluation} disabled={false} onQueued={vi.fn()} />)
  await screen.findByText(/0 runs left today/)
  expect(screen.getByRole('button', { name: 'Run demo experiment' })).toBeDisabled()
})
