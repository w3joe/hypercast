import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import ForecastReplay, { alignForecasts } from './ForecastReplay'
import { api } from './api'
import { heatmapData } from './PerformanceHeatmap'
import { comparisonProtocol } from './comparison'
import type { ForecastResponse, ForecastRow } from './ForecastReplay'
import type { Job } from './types'

vi.mock('recharts', () => {
  const Box = ({ children }: { children?: ReactNode }) => <div>{children}</div>
  return { ResponsiveContainer: Box, LineChart: Box, CartesianGrid: () => null, Line: () => null, ReferenceLine: () => null, Tooltip: () => null, XAxis: () => null, YAxis: () => null }
})

const row: ForecastRow = { origin_row: 1, forecast_origin: '2026-01-01', target_date: '2026-01-02', actual: 5, prediction: 7, persistence: 4 }
const response = (rows: ForecastRow[]): ForecastResponse => ({ rows, total: rows.length, invalid: 0, limit: 2000 })
describe('forecast alignment', () => {
  it('aligns exact origins and computes signed error without averaging', () => {
    const points = alignForecasts([response([row]), response([{ ...row, prediction: 3 }])])
    expect(points[0]).toMatchObject({ actual: 5, prediction0: 7, prediction1: 3, error0: 2, error1: -2 })
  })
  it('excludes duplicates, mismatched observed values and nonshared origins', () => {
    expect(alignForecasts([response([row, row]), response([row])])).toEqual([])
    expect(alignForecasts([response([row]), response([{ ...row, actual: 6 }])])).toEqual([])
    expect(alignForecasts([response([row]), response([{ ...row, target_date: '2026-02-01' }])])).toEqual([])
  })
  it.each([false, true])('scrubs saved observations and fetches a separate forecast lead (compact: %s)', async compact => {
    const fetch = vi.spyOn(api, 'forecasts').mockResolvedValue(response([row, { ...row, origin_row: 2, target_date: '2026-01-03', actual: 8 }]))
    const job = { id: 'run-a', status: { architecture_name: 'Model A' }, runs: [{ window: 10, horizon: 2, seed: 7, fold: 0 }] } as unknown as Job
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={client}><ForecastReplay jobs={[job]} window={10} horizon={2} compact={compact} /></QueryClientProvider>)
    await screen.findByText('5.0000')
    expect(screen.queryAllByRole('heading', { name: 'Signed error' })).toHaveLength(compact ? 0 : 1)
    fireEvent.change(screen.getByLabelText('Forecast cursor'), { target: { value: '1' } })
    expect(screen.getByText('8.0000')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('Replay lead'), { target: { value: '2' } })
    await waitFor(() => expect(fetch).toHaveBeenLastCalledWith('run-a', 10, 2, 7, 0, 2, ''))
    view.unmount(); client.clear(); fetch.mockRestore()
  })
})
describe('performance heatmap cohorts', () => {
  it('keeps cells missing and separates datasets and evaluation stages', () => {
    const job = { id: 'a', request: { evaluation: { protocol: 'chronological-v1', data_path: 'a.csv', target_column: 'value', preset: 'standard' } }, status: { architecture_name: 'A', phase: 'validation' }, summary: [{ window: 10, horizon: 1 }, { window: 10, horizon: 5 }], runs: [{ window: 10, horizon: 1, mae: 2, persistence_mae: 4, seed: 1, fold: 0 }] } as unknown as Job
    const other = { ...job, id: 'b', status: { ...job.status, phase: 'final_test' } } as Job
    const { horizons, rows } = heatmapData([job, other], comparisonProtocol(job), 10, 'mae_ratio')
    expect(horizons).toEqual([1, 5])
    expect(rows).toHaveLength(1)
    expect(rows[0].cells.map(c => c.mean)).toEqual([.5, null])
  })
})
