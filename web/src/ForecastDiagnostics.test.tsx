import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { ReactNode } from 'react'
import ForecastDiagnostics, { diagnosticCells } from './ForecastDiagnostics'
import type { Job } from './types'

vi.mock('recharts', () => {
  const Box = ({ children }: { children?: ReactNode }) => <div>{children}</div>
  return { ResponsiveContainer: Box, LineChart: ({ data, children }: { data: unknown; children: ReactNode }) => <div data-testid="chart" data-points={JSON.stringify(data)}>{children}</div>,
    CartesianGrid: () => null, Legend: () => null, Line: () => null, ReferenceLine: () => null, Tooltip: () => null, XAxis: () => null, YAxis: () => null }
})
function job(id: string, rows: Job['per_lead'], preset = 'standard'): Job {
  return { id, request: { evaluation: { preset, protocol: 'chronological-v1', data_path: 'data/raw/paper_data.xlsx', target_column: 'Copper' } },
    status: { architecture_name: id, phase: 'validation' }, per_lead: rows } as Job
}
const row = { window: 10, horizon: 1, lead: 1, mae: 2, persistence_mae: 3, directional_accuracy: .75, direction_baseline_accuracy: .5, return_correlation: null, bias: -1, p95_abs_error: 4, large_move_mae: null, large_move_mae_ratio: null, large_move_count: 0, sample_count: 12, large_move_threshold: .02 }

describe('forecast diagnostics', () => {
  it('limits crowded comparisons and lets users choose other runs', () => {
    render(<ForecastDiagnostics jobs={Array.from({ length: 20 }, (_, i) => job(`run-${i}`, [row]))} />)
    expect(screen.getByText('Compare runs · 4 of 20 selected')).toBeInTheDocument()
    const points = () => JSON.parse(screen.getAllByTestId('chart')[0].getAttribute('data-points')!)
    expect(points()).toHaveLength(4)
    expect(points()[3].model3).toBe(2)
    expect(points()[0].model3).toBeUndefined()
    fireEvent.click(screen.getByText('Compare runs · 4 of 20 selected'))
    const choices = screen.getAllByRole('checkbox')
    fireEvent.click(choices[4]); fireEvent.click(choices[5])
    expect(choices[6]).toBeDisabled()
    fireEvent.click(choices[0])
    expect(choices[6]).not.toBeDisabled()
    fireEvent.click(choices[6])
    expect(screen.getByText('Compare runs · 6 of 20 selected')).toBeInTheDocument()
    expect(screen.getAllByRole('link', { name: 'Download CSV' })).toHaveLength(6)
  })
  it('shows all diagnostics, missing values and a forecast download', () => {
    render(<ForecastDiagnostics jobs={[job('test-model',[row])]} />)
    expect(screen.getByRole('heading',{name:'Direction accuracy'})).toBeInTheDocument()
    expect(screen.getAllByTestId('chart')).toHaveLength(6)
    expect(screen.getByRole('link',{name:'Download CSV'})).toHaveAttribute('href','/api/v1/jobs/test-model/predictions.csv')
    expect(screen.getAllByText('N/A')).toHaveLength(3)
    const correlation = JSON.parse(screen.getAllByTestId('chart')[2].getAttribute('data-points')!)
    expect(correlation[0].model0).toBeNull()
    expect(screen.getByText('75.0%')).toBeInTheDocument()
  })
  it('does not mix windows, horizons, or validation protocols', () => {
    const jobs = [job('a',[row,{...row,window:20,horizon:5,lead:1,bias:99}]),job('robust',[row],'robust')]
    expect(diagnosticCells(jobs)).toHaveLength(3)
    render(<ForecastDiagnostics jobs={jobs} />)
    expect(screen.queryByText('99.000')).not.toBeInTheDocument()
    const select = screen.getByRole('combobox',{name:'Diagnostic evaluation cell'})
    const cell = diagnosticCells(jobs).find(item=>item.window===20)!
    fireEvent.change(select,{target:{value:cell.key}})
    expect(screen.getByText('99.000')).toBeInTheDocument()
    expect(screen.queryByText('robust')).not.toBeInTheDocument()
  })
  it('explains why old runs have no diagnostics without inventing scores', () => {
    render(<ForecastDiagnostics jobs={[job('old',[{window:10,horizon:1,lead:1,mae:2,persistence_mae:3}])]} />)
    expect(screen.getByRole('status')).toHaveTextContent('predate forecast diagnostics')
    expect(screen.getAllByTestId('chart')).toHaveLength(1)
    expect(screen.queryByRole('link',{name:'Download CSV'})).not.toBeInTheDocument()
  })
})
