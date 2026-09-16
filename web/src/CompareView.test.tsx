import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ReactNode } from 'react'
import type { Job, ArchivedComparisonJob } from './types'
import CompareView from './CompareView'

vi.mock('recharts', () => {
  const Box = ({ children }: { children?: ReactNode }) => <div>{children}</div>
  return { ResponsiveContainer: Box, LineChart: Box, CartesianGrid: () => null, Line: () => null, ReferenceLine: () => null, Tooltip: () => null, XAxis: () => null, YAxis: () => null }
})
vi.mock('./ForecastReplay', () => ({ default: ({ compact }: { compact: boolean }) => <div data-testid="forecast-overlay">{compact ? 'Single forecast chart' : 'Two forecast charts'}</div> }))
function job(id: string, phase = 'validation', preset = 'standard') {
  return { id, request: { evaluation: { protocol: 'chronological-v1', data_path: 'copper.csv', target_column: 'Copper', preset, seeds: [1], epochs: 50, learning_rate: .001, loss: 'mae' } },
    status: { state: 'complete', phase, preset, architecture_name: id }, summary: [{ window: 32, horizon: 5 }], per_lead: [],
    runs: [{ window: 32, horizon: 5, seed: 1, fold: 1, mae: 1, mse: 3, persistence_mae: 2, persistence_mse: 2, parameters: 100, directional_accuracy: .6, bias: -.1 }] } as unknown as Job
}
const scores = () => within(screen.getByRole('tabpanel', { name: 'Scores' }))
describe('simplified comparison workspace', () => {
  it('separates validation, final evaluations and explicitly selected quick checks', () => {
    render(<CompareView jobs={[job('development-model'), job('final-model', 'final_test'), job('smoke-model', 'validation', 'quick')]} finalAction={() => null} />)
    expect(scores().getByText('development-model')).toBeVisible()
    expect(screen.queryByText('smoke-model')).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Comparison stage'), { target: { value: 'final_test' } })
    expect(scores().getByText('final-model')).toBeVisible()
    expect(screen.queryByText('development-model')).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Comparison stage'), { target: { value: 'quick' } })
    expect(scores().getByText('smoke-model')).toBeVisible()
    expect(screen.getByText(/scores are preliminary/)).toBeVisible()
  })
  it('updates metric values and allows an explicitly empty run selection', () => {
    render(<CompareView jobs={[job('run-a')]} finalAction={() => null} />)
    expect(scores().getByText('0.5000')).toBeVisible()
    expect(scores().getByText('One repeat')).toBeVisible()
    fireEvent.change(screen.getByLabelText('Comparison metric'), { target: { value: 'mse_ratio' } })
    expect(scores().getByText('1.5000')).toBeVisible()
    fireEvent.click(screen.getByText('Compare runs', { selector: 'summary' }))
    fireEvent.click(screen.getByRole('checkbox', { name: /run-a/ }))
    expect(screen.getByRole('heading', { name: 'Select runs in the top bar' })).toBeVisible()
    expect(screen.queryByRole('tabpanel')).not.toBeInTheDocument()
  })
  it('opens on the group with multiple comparable models instead of the newest singleton', () => {
    render(<CompareView jobs={[job('only-robust', 'validation', 'robust'), job('standard-a'), job('standard-b')]} finalAction={() => null} />)
    expect(scores().getByText('standard-a')).toBeVisible()
    expect(scores().getByText('standard-b')).toBeVisible()
    expect(screen.queryByText('only-robust')).not.toBeInTheDocument()
    const group = screen.getByLabelText('Comparison evaluation group') as HTMLSelectElement
    fireEvent.change(group, { target: { value: [...group.options].find(o => o.text.includes('robust'))!.value } })
    expect(scores().getByText('only-robust')).toBeVisible()
    expect(screen.getByText(/Only one run uses this split plan/)).toBeVisible()
  })
  it('keeps chosen runs when changing window or horizon, with no silent reselection', () => {
    const jobs = ['a', 'b', 'c', 'd'].map(id => {
      const j = job(id)
      j.runs.push({ ...j.runs[0], window: 64, horizon: 10, mae: 2 })
      return j
    })
    render(<CompareView jobs={jobs} finalAction={() => null} />)
    fireEvent.click(screen.getByText('Compare runs', { selector: 'summary' }))
    fireEvent.click(screen.getByRole('checkbox', { name: /^b / }))
    fireEvent.click(screen.getByRole('checkbox', { name: /^d / }))
    fireEvent.change(screen.getByLabelText('Comparison window'), { target: { value: '64' } })
    expect(screen.getByLabelText('Comparison horizon')).toHaveValue('10')
    const names = scores().getAllByRole('rowheader').map(e => e.textContent)
    expect(names).toEqual(['aa', 'cc', 'dd'])
    fireEvent.change(screen.getByLabelText('Comparison window'), { target: { value: '32' } })
    expect(screen.getByLabelText('Comparison horizon')).toHaveValue('5')
    expect(scores().getAllByRole('rowheader').map(e => e.textContent)).toEqual(names)
  })
  it('has three primary chart panels and only loads the single-chart forecast when opened', () => {
    render(<CompareView jobs={[job('run-a'), job('run-b')]} finalAction={() => null} />)
    expect(screen.getByRole('article', { name: 'Score comparison chart' })).toBeVisible()
    expect(screen.getByRole('article', { name: 'Evaluation settings chart' })).toBeVisible()
    expect(screen.getByRole('article', { name: 'Does a larger model earn its size?' })).toBeVisible()
    expect(screen.queryByTestId('forecast-overlay')).not.toBeInTheDocument()
    // Native details toggles expose their open state to React.
    const disclosure = screen.getByText('Forecast vs actual', { exact: true }).closest('details')!
    disclosure.open = true
    fireEvent(disclosure, new Event('toggle'))
    expect(screen.getByTestId('forecast-overlay')).toHaveTextContent('Single forecast chart')
  })
  it('shows diagnostics as numbers from the same matched repeats rather than extra graphs', () => {
    const a = job('a'), b = job('b')
    a.runs.push({ ...a.runs[0], seed: 2, directional_accuracy: 1, bias: 999 })
    render(<CompareView jobs={[a, b]} finalAction={() => null} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Diagnostics' }))
    const panel = within(screen.getByRole('tabpanel', { name: 'Diagnostics' }))
    expect(panel.getAllByText('60.0%')).toHaveLength(2)
    expect(panel.queryByText('100.0%')).not.toBeInTheDocument()
    expect(panel.getAllByText('-0.1000')).toHaveLength(2)
    expect(panel.getAllByRole('table')).toHaveLength(1)
    fireEvent.click(screen.getByRole('tab', { name: 'Settings' }))
    expect(screen.getByRole('columnheader', { name: 'Learning rate' })).toBeVisible()
    expect(screen.queryByRole('columnheader', { name: 'Direction accuracy' })).not.toBeInTheDocument()
  })
  it('opens registered studies under final tests, lists every variant, and keeps them read-only', () => {
    const archived = (id: string, variant: string) => ({ ...job(id, 'final_test', 'archived-study'), archive: {
      collection: 'TSLib 15 · ETTh1', backbone: 'dlinear', variant, source: 'study', target_scale: 'Original OT units',
      task: 'Fixed short-horizon task', forecasts: true, split_description: 'test rows 100–199', note: 'Three paired seeds',
    } }) as ArchivedComparisonJob
    const action = vi.fn(() => <button>Start final test</button>)
    render(<CompareView jobs={[job('validation'), archived('DLinear-native', 'native'), archived('DLinear-quaternion', 'quaternion')]} finalAction={action} />)
    expect(screen.getByLabelText('Comparison stage')).toHaveValue('final_test')
    expect(scores().getByText('DLinear-native')).toBeVisible()
    expect(scores().getByText('DLinear-quaternion')).toBeVisible()
    expect(action).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('Compare runs', { selector: 'summary' }))
    fireEvent.change(screen.getByLabelText('Comparison variant filter'), { target: { value: 'native' } })
    expect(screen.getByRole('checkbox', { name: /DLinear-native/ })).toBeVisible()
    expect(screen.queryByRole('checkbox', { name: /DLinear-quaternion/ })).not.toBeInTheDocument()
    expect(scores().getByText('DLinear-quaternion')).toBeVisible() // A search filter never silently changes selection.
    fireEvent.click(screen.getByRole('tab', { name: 'Settings' }))
    expect(screen.getAllByText('test rows 100–199').length).toBeGreaterThan(0)
  })

})
