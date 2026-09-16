import { describe, expect, it } from 'vitest'
import { buildArchitectureScene, formatShape } from './architectureScene'
import { edgeId, emptyView, safeView } from './graphCanvas'
import type { GraphSpec, GraphNodeInfo } from './types'

const graph: GraphSpec = { schema_version: 2, revision: 'test', name: 'Branching model', sources: {}, groups: [{ id: 'core', label: 'Core' }],
  nodes: [{ id: 'input', kind: 'source', params: {} }, { id: 'a', kind: 'dense', params: { units: 4 }, group: 'core' }, { id: 'b', kind: 'dense', params: { units: 4 }, group: 'core' }, { id: 'join', kind: 'add', params: {} }, { id: 'unused', kind: 'dropout', params: {} }],
  edges: [{ source: 'input', target: 'a', port: 'x' }, { source: 'a', target: 'b', port: 'x' }, { source: 'b', target: 'join', port: 'a' }, { source: 'input', target: 'join', port: 'b' }], output: 'join' }

function metadata(shape: number[]): Record<string, GraphNodeInfo> {
  return Object.fromEntries(graph.nodes.map(n => [n.id, { label: n.id, ports: [], category: 'custom', settings: {}, shape }]))
}

describe('node diagram projection', () => {
  it('uses non-overlapping cards independent of tensor size and retains exact shape labels', async () => {
    const small = await buildArchitectureScene(graph, metadata([2, 4]), emptyView())
    const large = await buildArchitectureScene(graph, metadata([2, 16, 4096, 128]), emptyView())
    expect(small.blocks.map(b => [b.x, b.y, b.width, b.height])).toEqual(large.blocks.map(b => [b.x, b.y, b.width, b.height]))
    expect(large.blocks.find(b => b.output)?.shape).toBe('2 × 16 × 4096 × 128')
    for (const a of large.blocks) for (const b of large.blocks) {
      if (a.id === b.id) continue
      expect(a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y).toBe(true)
    }
  })
  it('preserves boundary ports, skip connections and immutable executable data when collapsing stages', async () => {
    const before = JSON.stringify(graph)
    const scene = await buildArchitectureScene(graph, {}, { ...emptyView(), collapsed: ['core'] })
    expect(scene.blocks.map(b => b.id)).toEqual(['input', 'group:core', 'join', 'unused'])
    expect(scene.routes.map(r => [r.source, r.target, r.port])).toEqual(expect.arrayContaining([['input', 'a', 'x'], ['b', 'join', 'a'], ['input', 'join', 'b']]))
    expect(scene.routes.some(r => r.id === edgeId(graph.edges[1]))).toBe(false)
    const stage = scene.blocks.find(b => b.stage === 'core')!
    expect(stage.inputs).toEqual([expect.objectContaining({ nodeId: 'a', port: 'x' })])
    expect(stage.outputs).toEqual([expect.objectContaining({ nodeId: 'b' })])
    expect(JSON.stringify(graph)).toBe(before)
  })
  it('lays out deterministically left to right and puts unused operations in a separate area', async () => {
    const first = await buildArchitectureScene(graph, {}, emptyView()), second = await buildArchitectureScene(graph, {}, emptyView())
    expect(first).toEqual(second)
    const block = (id: string) => first.blocks.find(b => b.id === id)!
    expect(block('input').x).toBeLessThan(block('a').x)
    expect(block('a').x).toBeLessThan(block('b').x)
    expect(block('b').x).toBeLessThan(block('join').x)
    expect(block('unused').disconnected).toBe(true)
    expect(block('unused').y).toBeGreaterThan(first.disconnectedY!)
    expect(first.routes.every(r => r.points.length >= 2 && Number.isFinite(r.insert.x))).toBe(true)
  })
  it('gives parallel input ports separate handles even when validation metadata is stale', async () => {
    const parallel = { ...graph, edges: [...graph.edges, { source: 'a', target: 'b', port: 'other' }] }
    const scene = await buildArchitectureScene(parallel, metadata([2, 4]), emptyView())
    const routes = scene.routes.filter(r => r.source === 'a' && r.target === 'b')
    expect(routes).toHaveLength(2)
    expect(new Set(routes.map(r => r.targetHandle)).size).toBe(2)
    for (const route of scene.routes) {
      const source = scene.blocks.find(b => b.id === route.sourceBlock)!, target = scene.blocks.find(b => b.id === route.targetBlock)!
      expect(source.outputs.find(p => p.id === route.sourceHandle)?.nodeId).toBe(route.source)
      expect(target.inputs.find(p => p.id === route.targetHandle)).toMatchObject({ nodeId: route.target, port: route.port })
      expect(route.points[0].x).toBeCloseTo(source.x + source.width)
      expect(route.points.at(-1)!.x).toBeCloseTo(target.x)
      expect(route.points.at(-1)!.y).toBeCloseTo(target.y + target.height * (target.inputs.findIndex(p => p.id === route.targetHandle) + 1) / (target.inputs.length + 1))
    }
  })
  it('exposes disconnected members of a collapsed stage and never drops a connection', async () => {
    const mixed = { ...graph, nodes: graph.nodes.map(n => n.id === 'unused' ? { ...n, group: 'core' } : n), edges: [...graph.edges, { source: 'a', target: 'unused', port: 'x' }] }
    const scene = await buildArchitectureScene(mixed, {}, { ...emptyView(), collapsed: ['core'] })
    expect(scene.blocks.some(b => b.id === 'group:core')).toBe(true)
    expect(scene.blocks.find(b => b.id === 'unused')?.disconnected).toBe(true)
    expect(scene.routes).toHaveLength(mixed.edges.length - 1)
  })
  it('ignores old manual positions and keeps automatic layout without changing the executable graph', async () => {
    const before = JSON.stringify(graph)
    const scene = await buildArchitectureScene(graph, {}, { ...emptyView(), positions: { a: { x: -150, y: 800 } } })
    expect(scene).toEqual(await buildArchitectureScene(graph, {}, emptyView()))
    expect(JSON.stringify(graph)).toBe(before)
  })
  it('loads legacy views without treating Three.js camera coordinates as node positions', () => {
    expect(safeView({ positions: { a: { x: 9999, y: -200 } }, direction: 'LR', collapsed: ['core', 'missing'] }, graph)).toEqual({ ...emptyView(), collapsed: ['core'] })
    const legacy = safeView({ renderer: 'three', version: 1, camera: { panX: 300, panY: 120, zoom: .8 }, expandedStages: ['core'] }, graph)
    expect(legacy.camera).toBeUndefined()
    expect(legacy.expandedStages).toEqual(['core'])
    expect(legacy.renderer).toBe('flow')
    expect(safeView({ renderer: 'flow', version: 1, camera: { panX: 9999, panY: 120, zoom: .8 }, positions: { a: { x: 9999, y: 120 } } }, graph)).toEqual({ ...emptyView(), collapsed: ['core'] })
  })
  it('reports missing inputs inside collapsed stages and exposes their connectable ports', async () => {
    const draft = { ...graph, edges: graph.edges.filter(e => e.target !== 'a') }
    const scene = await buildArchitectureScene(draft, {}, { ...emptyView(), collapsed: ['core'] })
    const stage = scene.blocks.find(b => b.id === 'group:core')!
    expect(stage.missingInputs).toEqual(['a · x'])
    expect(stage.inputs).toEqual([expect.objectContaining({ nodeId: 'a', port: 'x' })])
    expect(stage.shape).toBe('Shape unavailable')
    expect(stage.disconnected).toBe(false)
  })
  it('reports every distinct boundary shape instead of choosing an arbitrary one', async () => {
    const branching = { ...graph, edges: [...graph.edges, { source: 'a', target: 'join', port: 'c' }] }
    const info = metadata([2, 4]); info.a.shape = [2, 16, 512]; info.b.shape = [2, 1]
    const scene = await buildArchitectureScene(branching, info, { ...emptyView(), collapsed: ['core'] })
    const stage = scene.blocks.find(b => b.stage === 'core')!
    expect(stage.shape).toBe('2 output tensors')
    expect(stage.outputs.map(p => p.nodeId).sort()).toEqual(['a', 'b'])
    expect(formatShape(undefined)).toBe('Shape unavailable')
  })
})
