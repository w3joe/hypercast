import { describe, it, expect } from 'vitest'
import { prepareDenseSwap } from './layerSwap'
import type { GraphSpec, GraphNodeInfo } from './types'
const graph: GraphSpec = { schema_version: 2, revision: 'test', name: 'Test', sources: {}, groups: [], output: 'dense', nodes: [
  { id: 'input', kind: 'source', params: {} }, { id: 'dense', kind: 'source', source_ref: { source: 's0', node: 'linear' }, module_ref: 'shared', group: 'model', params: {} },
], edges: [{ source: 'input', target: 'dense', port: 'args/0' }] }
const info: Record<string, GraphNodeInfo> = {
  input: { label: 'Input', category: 'placeholder', settings: {}, ports: [], shape: [2, 10, 16] },
  dense: { label: 'Linear', category: 'call_module', settings: { out_features: 32, bias: false }, ports: ['args/0'], shape: [2, 10, 32] },
}
describe('Dense type conversion', () => {
  it('preserves identity, connections, output width and bias without sharing old weights', () => {
    const next = prepareDenseSwap(graph, 'dense', 'hyper_dense', info)
    expect(next.nodes[1]).toEqual({ id: 'dense', kind: 'hyper_dense', label: 'HyperDense', group: 'model', params: { units: 8, bias: false, algebra: 'quaternion', shape_mode: 'preserve' } })
    expect(next.edges).toEqual([{ source: 'input', target: 'dense', port: 'x' }])
    expect(next.output).toBe(graph.output)
    expect(graph.nodes[1].kind).toBe('source')
    const restored = prepareDenseSwap(next, 'dense', 'dense', { ...info, dense: { ...info.dense, category: 'custom', label: 'hyper_dense', settings: next.nodes[1].params } })
    expect(restored.nodes[1].params).toEqual({ units: 32, bias: false })
  })
  it.each([2, 3, 4])('supports %i-dimensional feature tensors', rank => {
    const metadata = { ...info, input: { ...info.input, shape: [...Array(rank - 1).fill(2), 16] }, dense: { ...info.dense, shape: [...Array(rank - 1).fill(2), 32] } }
    expect(prepareDenseSwap(graph, 'dense', 'hyper_dense', metadata).nodes[1].params.units).toBe(8)
  })
  it.each([
    ['complex', 2], ['split_complex', 2], ['tricomplex', 3],
    ['quaternion', 4], ['coquaternion', 4], ['cl11', 4], ['octonion', 8],
  ])('preserves width for %s HyperDense with %i components', (algebra, dimension) => {
    const metadata = {
      ...info,
      input: { ...info.input, shape: [2, 10, 24] },
      dense: { ...info.dense, shape: [2, 10, 24] },
    }
    const next = prepareDenseSwap(graph, 'dense', 'hyper_dense', metadata, algebra)
    expect(next.nodes[1].params).toEqual({ units: 24 / Number(dimension), bias: false, algebra, shape_mode: 'preserve' })
  })
  it('uses the selected algebra dimension when auto-fitting nondivisible widths', () => {
    const next = prepareDenseSwap(graph, 'dense', 'hyper_dense', info, 'tricomplex')
    expect(next.nodes[1].params).toEqual({ units: 11, bias: false, algebra: 'tricomplex', shape_mode: 'preserve' })
  })
  it('changes algebra while retaining the expanded real output width', () => {
    const denseMetadata = {
      ...info,
      input: { ...info.input, shape: [2, 10, 24] },
      dense: { ...info.dense, shape: [2, 10, 24] },
    }
    const quaternion = prepareDenseSwap(graph, 'dense', 'hyper_dense', denseMetadata, 'quaternion')
    const hyperMetadata = {
      ...denseMetadata,
      dense: { ...denseMetadata.dense, category: 'custom', label: 'HyperDense' },
    }
    const tricomplex = prepareDenseSwap(quaternion, 'dense', 'hyper_dense', hyperMetadata, 'tricomplex')
    expect(quaternion.nodes[1].params.units).toBe(6)
    expect(tricomplex.nodes[1].params).toEqual({ units: 8, bias: false, algebra: 'tricomplex', shape_mode: 'preserve' })
  })
  it('pads and crops incompatible widths without mutating the source graph', () => {
    const smallInput = prepareDenseSwap(graph, 'dense', 'hyper_dense', { ...info, input: { ...info.input, shape: [2, 10, 3] } })
    const unevenOutput = prepareDenseSwap(graph, 'dense', 'hyper_dense', { ...info, dense: { ...info.dense, shape: [2, 10, 7] } })
    expect(smallInput.nodes[1].params).toMatchObject({ units: 8, shape_mode: 'preserve' })
    expect(unevenOutput.nodes[1].params).toMatchObject({ units: 2, shape_mode: 'preserve' })
    expect(graph.nodes[1].kind).toBe('source')
  })
  it('requires known validated dimensions', () => {
    expect(() => prepareDenseSwap(graph, 'dense', 'hyper_dense', {})).toThrow()
  })
})
