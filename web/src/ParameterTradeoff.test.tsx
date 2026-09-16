import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ParameterTradeoff from './ParameterTradeoff'
import { comparisonColor, comparisonData } from './comparison'
import type { ComparisonJob } from './types'

const job = (id: string, parameters: number | null, errors: number[], variant = 'native', backbone = id) => ({
  id, status: { architecture_name: id }, archive: { variant, backbone },
  runs: errors.map((mae, i) => ({ window: 32, horizon: 5, seed: i + 1, fold: 1, parameters, mae, persistence_mae: 2 })),
}) as unknown as ComparisonJob
const scores = (jobs: ComparisonJob[]) => comparisonData(jobs, 32, 5, 'mae', true)

describe('model size comparison', () => {
  it('shows exact matched scores and observed variability on hover and keyboard focus', () => {
    const data = scores([job('small', 1250, [1, 3]), job('accurate', 12000, [.5, 1.5]), job('dominated', 15000, [2, 4])])
    render(<ParameterTradeoff data={data} metric="mae" />)
    expect(screen.getByText('Total model parameters · log scale')).toBeVisible()
    fireEvent.mouseEnter(screen.getByRole('button', { name: /^small\./ }))
    const readout = within(screen.getByRole('status'))
    expect(readout.getByText('1,250')).toBeVisible()
    expect(readout.getByText('Range 1–3 · 2 repeats')).toBeVisible()
    expect(readout.getByText('On selected-run frontier')).toBeVisible()
    fireEvent.focus(screen.getByRole('button', { name: /^dominated\./ }))
    expect(readout.getByText('Outside selected-run frontier')).toBeVisible()
    fireEvent.keyDown(screen.getByRole('button', { name: /^dominated\./ }), { key: 'Escape' })
    expect(readout.getByText(/Hover, tap/)).toBeVisible()
  })
  it('omits missing, varying, negative and fractional parameter counts without fabricating points', () => {
    const varying = job('varies', 100, [1, 2]); varying.runs[1].parameters = 200
    const jobs = [job('missing', null, [1, 2]), varying, job('negative', -1, [1, 2]), job('fractional', 1.5, [1, 2])]
    render(<ParameterTradeoff data={scores(jobs)} metric="mae" />)
    expect(screen.getByText('No comparable parameter counts')).toBeVisible()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
    expect(screen.getByText(/4 selected run\(s\) omitted/)).toBeVisible()
  })
  it('includes zero-parameter models on a linear scale and keeps tied points accessible', () => {
    render(<ParameterTradeoff data={scores([job('a', 0, [0]), job('b', 0, [0])])} metric="mae" />)
    expect(screen.getByText('Total model parameters · linear scale (includes zero)')).toBeVisible()
    expect(screen.getAllByRole('button')).toHaveLength(2)
    fireEvent.focus(screen.getByRole('button', { name: /^b\./ }))
    expect(within(screen.getByRole('status')).getByText('One repeat · variability unavailable')).toBeVisible()
  })
  it('uses one stable backbone colour for Real and HyperDense, with different markers', () => {
    const real = job('Real model', 100, [1], 'real', 'tsmixer'), hyper = job('Quaternion model', 50, [2], 'quaternion', 'tsmixer')
    expect(comparisonColor(real)).toBe(comparisonColor(hyper))
    expect(scores([hyper, real])[1].color).toBe(scores([real])[0].color)
    render(<ParameterTradeoff data={scores([real, hyper])} metric="mae" />)
    expect(screen.getByText('Real', { exact: true })).toBeVisible()
    expect(screen.getByText('HyperDense', { exact: true })).toBeVisible()
    expect(screen.getByRole('button', { name: /^Real model\./ }).querySelector('rect')).not.toBeNull()
    expect(screen.getByRole('button', { name: /^Quaternion model\./ }).querySelector('path')).not.toBeNull()
  })
  it('updates the selected metric and clears a readout whose run is no longer selected', () => {
    const jobs = [job('a', 100, [1, 3])]
    const { rerender } = render(<ParameterTradeoff data={scores(jobs)} metric="mae" />)
    fireEvent.focus(screen.getByRole('button'))
    rerender(<ParameterTradeoff data={comparisonData(jobs, 32, 5, 'mae_ratio', true)} metric="mae_ratio" />)
    expect(screen.getByRole('button')).toHaveAccessibleName(/Mean MAE \/ persistence: 1\./)
    expect(screen.getByText('Persistence = 1')).toBeVisible()
    rerender(<ParameterTradeoff data={scores([job('b', 1000, [2])])} metric="mae" />)
    expect(within(screen.getByRole('status')).getByText(/Hover, tap/)).toBeVisible()
  })
})
