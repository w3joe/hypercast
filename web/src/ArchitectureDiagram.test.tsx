import { render, screen, fireEvent } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ArchitectureDiagram from './ArchitectureDiagram'
import type { SceneBlock } from './architectureScene'

// jsdom has no layout engine. Keep the real React Flow controls and card DOM;
// only its ResizeObserver measurements are unavailable in this test.
afterEach(() => vi.unstubAllGlobals())
describe('node diagram accessibility', () => {
  it('renders an editable tree counterpart without requiring WebGL', async () => {
    vi.stubGlobal('ResizeObserver', class { observe() {}; disconnect() {}; unobserve() {} })
    vi.stubGlobal('WebGL2RenderingContext', undefined)
    const select = vi.fn(), toggle = vi.fn()
    const block: SceneBlock = { id: 'group:core', nodeIds: ['a'], stage: 'core', label: 'Core', shape: '2 × 64', color: '#5793d0', kind: 'dense', disconnected: false, missingInputs: [], output: false, x: 0, y: 0, width: 224, height: 122, inputs: [], outputs: [] }
    const { container } = render(<ArchitectureDiagram scene={{ blocks: [block], routes: [], width: 400, height: 300 }} selected={['group:core']} selectedEdge={null} active stale={false} inserting={false} expandedGroups={[]} onCamera={vi.fn()} onSelect={select} onEdge={vi.fn()} onInsert={vi.fn()} onToggle={toggle} />)
    expect(container.querySelector('canvas')).toBeNull()
    expect(container.querySelector('.react-flow')).not.toBeNull()
    const node = await screen.findByRole('button', { name: 'Select Core' })
    expect(node).toHaveAttribute('aria-pressed', 'true')
    fireEvent.keyDown(node, { key: 'Enter', shiftKey: true })
    expect(select).toHaveBeenCalledWith('group:core', true)
    fireEvent.click(screen.getByRole('button', { name: 'Expand Core' }))
    expect(toggle).toHaveBeenCalledWith('core')
  })
})
