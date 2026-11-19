import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { EvaluationPanel } from './BuilderControls'
import type { Catalog, EvaluationSpec } from './types'

describe('batch size dropdown', () => {
  const evaluation = {
    preset: 'quick', target_column: 'Copper', cells: [{ window: 10, horizon: 1 }],
    seeds: [7], epochs: 3, batch_size: 32, learning_rate: 0.001, loss: 'mse',
    early_stopping_patience: null, device: 'cpu', shuffle: true, restore_best_weights: false,
  } as EvaluationSpec

  it('offers standard sizes and updates only the numeric batch size', () => {
    const onChange = vi.fn()
    render(<EvaluationPanel evaluation={evaluation} catalog={{} as Catalog} onChange={onChange} />)
    const select = screen.getByLabelText('Batch size') as HTMLSelectElement
    expect(select.tagName).toBe('SELECT')
    expect(select.value).toBe('32')
    expect(Array.from(select.options, option => option.value)).toEqual(['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])
    fireEvent.change(select, { target: { value: '128' } })
    expect(onChange).toHaveBeenCalledWith({ ...evaluation, batch_size: 128 })
  })

  it('preserves a nonstandard size from an existing configuration', () => {
    render(<EvaluationPanel evaluation={{ ...evaluation, batch_size: 48 }} catalog={{} as Catalog} onChange={vi.fn()} />)
    const select = screen.getByLabelText('Batch size') as HTMLSelectElement
    expect(select.value).toBe('48')
    expect(Array.from(select.options, option => option.value).filter(value => value === '48')).toHaveLength(1)
  })

  it('switches dataset and updates target column options automatically', () => {
    const onChange = vi.fn()
    const catalog = {
      datasets: [
        { id: 'copper', label: 'Copper (4D paper)', group: 'Paper', data_path: 'data/raw/paper_data.xlsx', target_column: 'Copper', features: 4, rows: 2008, frequency: 'daily', columns: ['Copper', 'FCX', 'CLP', 'SCCO'], available: true, download_command: '', description: '' },
        { id: 'weather', label: 'Weather', group: 'TSLib', data_path: 'data/external/tslib/weather.csv', target_column: 'OT', features: 21, rows: 52696, frequency: '10-min', columns: ['p (mbar)', 'T (degC)', 'OT'], available: true, download_command: '', description: '' },
        { id: 'nyc_taxi', label: 'NYC Taxi Trips', group: 'NYC TLC', data_path: 'data/external/nyc_taxi/nyc_taxi.csv', target_column: 'trip_count', features: 8, rows: 4368, frequency: 'hourly', columns: ['manhattan_pickups', 'trip_count'], available: true, download_command: '', description: '' },
      ],
    } as unknown as Catalog
    render(<EvaluationPanel evaluation={{ ...evaluation, data_path: 'data/raw/paper_data.xlsx' }} catalog={catalog} onChange={onChange} />)
    const datasetSelect = screen.getByLabelText('Dataset') as HTMLSelectElement
    expect(datasetSelect.options).toHaveLength(3)
    fireEvent.change(datasetSelect, { target: { value: 'data/external/nyc_taxi/nyc_taxi.csv' } })
    expect(onChange).toHaveBeenCalledWith({
      ...evaluation,
      data_path: 'data/external/nyc_taxi/nyc_taxi.csv',
      target_column: 'trip_count',
    })
  })
})
