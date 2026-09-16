import { describe, expect, it } from 'vitest'
import { architectureOverview, overviewView } from './architectureOverview'
import { buildArchitectureScene } from './architectureScene'
import { emptyView, safeView } from './graphCanvas'
import type { GraphNodeInfo, GraphNodeSpec, GraphSpec } from './types'

const source = (id: string, group?: string): GraphNodeSpec => ({ id, kind: 'source', params: {}, source_ref: { source: 's0', node: id }, group })
const graph: GraphSpec = { schema_version: 2, revision: 'test', name: 'CNN', sources: {},
  groups: [{ id: 'conv', label: 'causal_conv' }, { id: 'relu', label: 'activation' }, { id: 'flatten', label: 'flatten' }, { id: 'dense', label: 'dense' }, { id: 'dropout', label: 'dropout' }],
  nodes: [source('inputs'), source('getitem'), source('getitem_1'), source('transpose', 'conv'), source('convolution', 'conv'), source('activation', 'relu'), source('flatten', 'flatten'), source('linear', 'dense'), source('dropout', 'dropout'), source('output'), source('output_1'), { id: 'draft', kind: 'dense', params: { units: 8 } }],
  edges: [ ['inputs', 'getitem'], ['inputs', 'getitem_1'], ['getitem_1', 'transpose'], ['transpose', 'convolution'], ['convolution', 'activation'], ['activation', 'flatten'], ['flatten', 'linear'], ['linear', 'dropout'], ['dropout', 'output'], ['output', 'output_1'] ].map(([source, target]) => ({ source, target, port: 'x' })), output: 'output_1' }
const info: Record<string, GraphNodeInfo> = Object.fromEntries(graph.nodes.map(n => [n.id, { label: ({ inputs: 'Input', activation: 'ReLU', convolution: 'Conv1d', linear: 'Linear', output: 'Linear', output_1: 'Forecast output' } as Record<string, string>)[n.id] ?? n.id, category: n.id === 'inputs' ? 'placeholder' : n.id === 'output_1' ? 'output' : ['convolution', 'activation', 'linear', 'dropout', 'output'].includes(n.id) ? 'call_module' : 'call_function', ports: n.id === 'inputs' ? [] : ['x'], settings: {}, shape: ['output', 'output_1'].includes(n.id) ? [2, 1] : [2, 10, 16] }]))

describe('semantic architecture overview', () => {
  it('summarizes source scaffolding and linear combinations without changing the executable graph', async () => {
    const before = JSON.stringify(graph), overview = architectureOverview(graph, info)
    expect(overview.stages.map(s => s.label)).toEqual(['Input', 'Conv1D + ReLU', 'Dense', 'Forecast'])
    const scene = await buildArchitectureScene(overview.graph, info, overviewView(overview, emptyView()), overview)
    expect(scene.blocks.map(b => b.label)).toEqual(['Input', 'Conv1D + ReLU', 'Dense', 'Forecast', 'draft'])
    expect(scene.blocks.find(b => b.label === 'Input')?.nodeIds).toEqual(['inputs', 'getitem', 'getitem_1'])
    expect(scene.blocks.find(b => b.label === 'Dense')?.nodeIds).toEqual(['flatten', 'linear', 'dropout'])
    expect(scene.blocks.find(b => b.label === 'Forecast')?.shape).toBe('2 × 1')
    expect(scene.blocks.find(b => b.label === 'Conv1D + ReLU')?.nodeIds).toEqual(['transpose', 'convolution', 'activation'])
    expect(scene.blocks.find(b => b.id === 'draft')?.disconnected).toBe(true)
    expect(JSON.stringify(graph)).toBe(before)
    expect(overview.graph.edges).toBe(graph.edges)
    expect(overview.graph.sources).toBe(graph.sources)
  })
  it('retains all real operations, ports, and edges when expanded', async () => {
    const overview = architectureOverview(graph, info)
    const expanded = overviewView(overview, { ...emptyView(), expandedStages: overview.stages.map(s => s.id) })
    const scene = await buildArchitectureScene(overview.graph, info, expanded, overview)
    expect(scene.blocks.map(b => b.id)).toEqual(graph.nodes.map(n => n.id))
    expect(scene.routes.map(({ source, target, port }) => ({ source, target, port }))).toEqual(expect.arrayContaining(graph.edges))
    expect(scene.routes).toHaveLength(graph.edges.length)
  })
  it('tucks unused source shape lookups into their stage without inventing an extra tensor output', async () => {
    const traced: GraphSpec = { ...graph, nodes: [...graph.nodes, source('getattr_9', 'conv'), source('getitem_9', 'conv')], edges: [...graph.edges, { source: 'convolution', target: 'getattr_9', port: 'x' }, { source: 'getattr_9', target: 'getitem_9', port: 'x' }] }
    const overview = architectureOverview(traced, info)
    const scene = await buildArchitectureScene(overview.graph, info, overviewView(overview, emptyView()), overview)
    expect(scene.blocks.find(b => b.id === 'group:conv')?.shape).toBe('2 × 10 × 16')
    expect(scene.blocks.some(b => b.id === 'getattr_9')).toBe(false)
    expect(overview.stages.find(s => s.id === 'conv')?.nodeIds).toContain('getattr_9')
  })
  it('does not merge an activation across a residual branch', async () => {
    const branch: GraphSpec = { ...graph, nodes: [...graph.nodes, { id: 'join', kind: 'add', params: {} }], edges: [...graph.edges.filter(e => e.target !== 'flatten'), { source: 'convolution', target: 'join', port: 'a' }, { source: 'activation', target: 'join', port: 'b' }, { source: 'join', target: 'flatten', port: 'x' }] }
    const overview = architectureOverview(branch, info)
    expect(overview.membership.get('activation')).not.toBe(overview.membership.get('convolution'))
    const scene = await buildArchitectureScene(overview.graph, info, overviewView(overview, emptyView()), overview)
    expect(scene.routes).toEqual(expect.arrayContaining([expect.objectContaining({ source: 'convolution', target: 'join', port: 'a' }), expect.objectContaining({ source: 'activation', target: 'join', port: 'b' })]))
  })
  it('keeps disconnected edits and independent duplicates visible even inside a live stage', async () => {
    const edited: GraphSpec = { ...graph, nodes: [...graph.nodes, { ...source('getitem'), id: 'copy-getitem', group: 'conv' }, { id: 'custom-add', kind: 'add', params: {}, group: 'conv' }], edges: [...graph.edges, { source: 'convolution', target: 'custom-add', port: 'a' }] }
    const overview = architectureOverview(edited, info)
    expect(overview.tuckedNodeIds.has('copy-getitem')).toBe(false)
    const scene = await buildArchitectureScene(overview.graph, info, overviewView(overview, emptyView()), overview)
    expect(scene.blocks.find(b => b.id === 'copy-getitem')?.disconnected).toBe(true)
    expect(scene.blocks.find(b => b.id === 'custom-add')?.missingInputs).toContain('custom-add · b')
    expect(scene.routes.some(r => r.target === 'custom-add')).toBe(true)
  })
  it('loads summary expansion metadata and ignores obsolete stage IDs', () => {
    const overview = architectureOverview(graph, info)
    const restored = safeView({ renderer: 'flow', version: 1, expandedStages: ['conv', 'missing', 4], camera: { panX: 1, panY: 2, zoom: .7 } }, graph)
    expect(restored.expandedStages).toEqual(['conv', 'missing'])
    expect(overviewView(overview, restored).collapsed).not.toContain('conv')
    expect(overviewView(overview, restored).collapsed).not.toContain('missing')
    expect(restored.camera?.zoom).toBe(.7)
    expect(overviewView(overview, safeView({ positions: { conv: { x: 100, y: -500 } } }, graph)).collapsed).toEqual(overview.stages.map(s => s.id))
  })
})
