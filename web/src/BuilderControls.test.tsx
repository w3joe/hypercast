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
})
