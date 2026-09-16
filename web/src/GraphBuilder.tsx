import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import YAML from 'yaml'
import { Undo2, Redo2, ZoomIn, ZoomOut, Maximize, RotateCcw, Search, Plus, ArrowLeft, ArrowRight, Crosshair, X } from 'lucide-react'
import { api } from './api'
import { EvaluationPanel, RunControls, evaluationFromPreset } from './BuilderControls'
import MethodCollection from './MethodCollection'
import { SidePanel, useWorkspacePanels } from './WorkspacePanels'
import WeightInspector from './WeightInspector'
import type { ArchitectureSpec, Catalog, GraphSpec, GraphNodeSpec, GraphNodeInfo, GraphRecord, GraphViewState } from './types'
import { appendGraph, connectGraph, duplicateGraph, edgeId, emptyView, insertOnEdge, portsFor, safeView, uid } from './graphCanvas'
import ArchitectureDiagram from './ArchitectureDiagram'
import type { DiagramHandle } from './ArchitectureDiagram'
import { buildArchitectureScene, formatShape } from './architectureScene'
import { architectureOverview, overviewView } from './architectureOverview'
import { isLayerOperation, replaceLayer, replacementPorts } from './architectureEditing'
import type { ArchitectureScene } from './architectureScene'
import type { DiagramCamera } from './types'
import { DEFAULT_ALGEBRA_DIMENSIONS, denseKind } from './layerSwap'

type Snapshot = { graph: GraphSpec; view: GraphViewState }
const extras = [
  { type: 'add', label: 'Residual / add', defaults: {} }, { type: 'multiply', label: 'Multiply', defaults: {} },
  { type: 'concat', label: 'Concatenate', defaults: { dim: -1 } },
  { type: 'reshape', label: 'Reshape (0 copies axis)', defaults: { shape: [0, -1] } },
  { type: 'permute', label: 'Permute axes', defaults: { dims: [0, 2, 1] } },
  { type: 'softmax', label: 'Softmax', defaults: { dim: -1 } },
]
const allowed = new Set(['dense', 'activation', 'dropout', 'flatten', 'mean_pool', 'last_state', 'layer_norm', 'causal_conv', 'tcn', 'gru', 'lstm', 'hyper_dense'])

export default function GraphBuilder({ catalog, active = true }: { catalog: Catalog; active?: boolean }) {
  const diagram = useRef<DiagramHandle>(null)
  const workspace = useWorkspacePanels()
  const revealPanel = () => {
    workspace?.openSidebar?.()
    const sidebar = workspace?.sidebar?.closest<HTMLElement>('.workspace-sidebar')
    if (sidebar) sidebar.scrollTop = 0
  }
  const camera = useRef<DiagramCamera | undefined>(undefined)
  const [loadKey, setLoadKey] = useState(0)
  const [scene, setScene] = useState<ArchitectureScene>({ blocks: [], routes: [], width: 400, height: 300 })
  const canvasRef = useRef<HTMLDivElement>(null)
  const queryClient = useQueryClient()
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const current = useRef(snapshot); current.current = snapshot
  const past = useRef<Snapshot[]>([]), future = useRef<Snapshot[]>([])
  const [selected, setSelected] = useState<string[]>([]), [selectedEdge, setSelectedEdge] = useState<string | null>(null)
  const [metadata, setMetadata] = useState<Record<string, GraphNodeInfo>>({})
  const [error, setError] = useState(''), [validationError, setValidationError] = useState('')
  const [parameters, setParameters] = useState<number | null>(null)
  const [validatedGraph, setValidatedGraph] = useState<GraphSpec | null>(null)
  const [validating, setValidating] = useState(false), [loading, setLoading] = useState(false)
  const [warnings, setWarnings] = useState<string[]>([]), [message, setMessage] = useState('')
  const [savedId, setSavedId] = useState<string>(), [library, setLibrary] = useState(false)
  const [advanced, setAdvanced] = useState(false)
  const [panel, setPanel] = useState<'architecture' | 'inspect' | 'add' | 'run'>('architecture')
  useEffect(() => {
    if (!active) return
    const sidebar = workspace?.sidebar?.closest<HTMLElement>('.workspace-sidebar')
    if (sidebar) sidebar.scrollTop = 0
  }, [panel, workspace?.sidebar])
  const [swapping, setSwapping] = useState(false), [swapError, setSwapError] = useState('')
  const [replacementAlgebra, setReplacementAlgebra] = useState('quaternion')
  const [weightsOpen, setWeightsOpen] = useState(false)
  const [layerSearch, setLayerSearch] = useState('')
  const [paletteSearch, setPaletteSearch] = useState('')
  const [moveId, setMoveId] = useState<string | null>(null)
  const pendingFocus = useRef<string | null>(null)
  const searchInput = useRef<HTMLInputElement>(null)
  const [showOperations, setShowOperations] = useState(false)
  const [allowDisconnected, setAllowDisconnected] = useState(false)
  const findLayer = () => {
    setLibrary(false); setPanel('architecture'); revealPanel()
    requestAnimationFrame(() => { searchInput.current?.focus(); searchInput.current?.scrollIntoView?.({ block: 'nearest' }) })
  }
  useEffect(() => {
    if (!active) return
    const keydown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); findLayer() }
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  }, [active, workspace])
  useEffect(() => { setSwapError('') }, [selected[0]])
  const [evaluation, setEvaluation] = useState(() => evaluationFromPreset(catalog, 'quick'))
  const evaluationRef = useRef(evaluation); evaluationRef.current = evaluation
  const records = useQuery({ queryKey: ['architectures'], queryFn: api.graphRecords })
  const graph = snapshot?.graph
  const overview = useMemo(() => graph ? architectureOverview(graph, metadata) : undefined, [graph, metadata])
  const overviewRef = useRef(overview); overviewRef.current = overview
  const stageMembers = (id: string) => overview?.stages.find(s => s.id === id)?.nodeIds ?? []
  const selectionIds = (ids: string[]) => [...new Set(ids.flatMap(id => id.startsWith('group:') ? stageMembers(id.slice(6)) : [id]))]
  const presetGroups = [
    { label: 'Research models', items: catalog.presets.filter(p => p.preset_id?.startsWith('tslib-')) },
    { label: 'Paper baselines', items: catalog.presets.filter(p => p.preset_id?.startsWith('paper-')) },
    { label: 'Other starting points', items: catalog.presets.filter(p => !p.preset_id?.startsWith('tslib-') && !p.preset_id?.startsWith('paper-')) },
  ]
  const presetOptions = presetGroups.filter(g => g.items.length).map(g => <optgroup label={g.label} key={g.label}>{g.items.map(p => <option key={p.preset_id} value={p.preset_id}>{p.name.replace(/ \(TSLib core, editable\)$/, '')}</option>)}</optgroup>)
  const palette = [...catalog.categories.flatMap(c => c.layers).filter(l => allowed.has(l.type)), ...extras]
  const loadSequence = useRef(0)
  const load = async (spec: ArchitectureSpec | GraphSpec, record?: GraphRecord) => {
    const sequence = ++loadSequence.current
    setLoading(true); setError('')
    try {
      const cell = evaluation.cells[0] ?? { window: 10, horizon: 1 }
      const converted = await api.convert(spec, cell.window, cell.horizon)
      const description = await api.describeGraph(converted, cell.window, cell.horizon)
      if (sequence !== loadSequence.current) return
      const next = { ...converted, locked: Boolean(spec.locked), preset_id: spec.preset_id }
      const view = safeView(record?.view, converted)
      camera.current = view.camera
      setScene({ blocks: [], routes: [], width: 400, height: 300 })
      setSnapshot({ graph: next, view }); setLoadKey(key => key + 1)
      setSavedId(spec.schema_version === 2 ? record?.id : undefined)
      setMetadata(description.graph_nodes ?? {}); setValidatedGraph(null); setSelected([]); setSelectedEdge(null); setPanel('architecture'); setWeightsOpen(false); setMoveId(null)
      past.current = []; future.current = []; setLibrary(false)
      setMessage(''); setLayerSearch(''); setAllowDisconnected(false)
    } catch (e) { if (sequence === loadSequence.current) setError(String(e)) }
    finally { if (sequence === loadSequence.current) setLoading(false) }
  }
  useEffect(() => { void load(catalog.presets.find(p => p.preset_id === 'tslib-tsmixer') ?? catalog.presets[0]) }, []) // Initial preset only.
  const commit = useCallback((next: Snapshot) => {
    if (current.current) past.current = [...past.current.slice(-49), current.current]
    future.current = []; current.current = next; setSnapshot(next); setError('')
  }, [])
  const edit = (next: GraphSpec) => { if (snapshot && !graph?.locked) commit({ ...snapshot, graph: next }) }
  const toggle = useCallback((id: string) => {
    const s = current.current
    if (s) {
      const expanded = s.view.expandedStages ?? [], opening = !expanded.includes(id)
      pendingFocus.current = opening ? overviewRef.current?.stages.find(stage => stage.id === id)?.primaryId ?? '__all__' : `group:${id}`
      commit({ ...s, view: { ...s.view, expandedStages: opening ? [...expanded, id] : expanded.filter(g => g !== id) } })
    }
  }, [commit])
  useEffect(() => {
    if (!snapshot || !overview) return
    let cancelled = false
    void buildArchitectureScene(overview.graph, metadata, overviewView(overview, snapshot.view), overview).then(result => {
      if (cancelled) return
      setScene(result)
      if (pendingFocus.current) {
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (cancelled) return
          const id = pendingFocus.current; pendingFocus.current = null
          if (id) diagram.current?.fit(id === '__all__' ? undefined : id)
        }))
      }
    }).catch(e => { if (!cancelled) setError(`Layout: ${e}`) })
    return () => { cancelled = true }
  }, [snapshot, metadata, overview])
  const select = (id: string, multiple = false) => {
    setAllowDisconnected(false)
    diagram.current?.hold()
    setSelected(previous => multiple ? previous.includes(id) ? previous.filter(n => n !== id) : [...previous, id] : [id])
    setSelectedEdge(null); setPanel('inspect'); if (!multiple) revealPanel(); else workspace?.openSidebar?.()
  }
  const selectFromTree = (id: string, multiple = false) => {
    select(id, multiple)
    const s = current.current, group = overview?.membership.get(id)
    if (s && group && !(s.view.expandedStages ?? []).includes(group)) {
      pendingFocus.current = id; commit({ ...s, view: { ...s.view, expandedStages: [...s.view.expandedStages ?? [], group] } })
    } else { if (pendingFocus.current) pendingFocus.current = id; diagram.current?.fit(id) }
  }
  const selectEdge = (id: string) => { setSelectedEdge(id); setSelected([]); setPanel('inspect'); revealPanel() }
  const chooseInsertion = (id: string) => {
    setAllowDisconnected(false); setPaletteSearch('')
    if (moveId && current.current) {
      try {
        const node = current.current.graph.nodes.find(n => n.id === moveId)
        if (!node) throw new Error('The selected layer no longer exists.')
        finishInsertion(insertOnEdge(current.current.graph, node, id, metadata), node.id)
        setMoveId(null)
      } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    } else { setSelectedEdge(id); setPanel('add'); revealPanel() }
  }
  const undo = (redo = false) => {
    setAllowDisconnected(false); setMessage(''); setError(''); setSwapError('')
    const history = redo ? future : past, destination = redo ? past : future
    const next = history.current.pop()
    if (next && current.current) { destination.current.push(current.current); current.current = next; setSnapshot(next); setMoveId(null); setSelectedEdge(null); setSelected(ids => ids.filter(id => next.graph.nodes.some(n => n.id === id) || architectureOverview(next.graph, metadata).stages.some(g => `group:${g.id}` === id))) }
  }
  useEffect(() => {
    if (!graph) return
    let cancelled = false
    setValidating(true); setValidatedGraph(null)
    const timer = setTimeout(async () => {
      try {
        if (!evaluation.cells.length) throw new Error('Select at least one evaluation cell.')
        let first: Record<string, GraphNodeInfo> | undefined
        const notices: string[] = []
        let parameterCount: number | null = null
        for (const cell of evaluation.cells) {
          try {
            const result = await api.validateGraph(graph, cell.window, cell.horizon)
            first ??= result.graph_nodes; parameterCount ??= result.parameters; notices.push(...result.warnings)
          } catch (e) { throw new Error(`${cell.window}/${cell.horizon}: ${e instanceof Error ? e.message : e}`) }
          if (cancelled) return
        }
        if (!cancelled) { setMetadata(previous => ({ ...previous, ...first })); setValidatedGraph(graph); setParameters(parameterCount); setValidationError(''); setWarnings([...new Set(notices)]) }
      } catch (e) { if (!cancelled) setValidationError(String(e)) }
      finally { if (!cancelled) setValidating(false) }
    }, 300)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [graph, evaluation.cells])
  const addModel = async (id: string) => {
    const preset = catalog.presets.find(p => p.preset_id === id)
    if (!preset || !graph || graph.locked) return
    setLoading(true)
    try {
      const cell = evaluation.cells[0] ?? { window: 10, horizon: 1 }
      const template = await api.convert(preset, cell.window, cell.horizon)
      const description = await api.describeGraph(template, cell.window, cell.horizon)
      const s = current.current
      if (!s || s.graph !== graph) throw new Error('The graph changed while loading. Please add the model again.')
      const combined = appendGraph(graph, template, description.graph_nodes)
      const contracts = await api.describeGraph(combined, cell.window, cell.horizon)
      if (current.current?.graph !== graph) throw new Error('The graph changed while loading. Please add the model again.')
      setMetadata(previous => ({ ...previous, ...contracts.graph_nodes }))
      commit({ ...s, graph: combined })
      setMessage('Model added as a disconnected subgraph. Connect its Model input, then join its output or select it as the forecast output.')
    } catch (e) { setError(String(e)) }
    finally { setLoading(false) }
  }
  const remove = () => {
    if (!graph || graph.locked) return
    const ids = new Set(selectionIds(selected))
    if (ids.has(graph.output)) { setError('Choose another forecast output before deleting this node.'); return }
    edit({ ...graph, nodes: graph.nodes.filter(n => !ids.has(n.id)), edges: graph.edges.filter(e => !ids.has(e.source) && !ids.has(e.target) && edgeId(e) !== selectedEdge) })
    setSelected([]); setSelectedEdge(null)
  }
  const finishInsertion = (next: GraphSpec, nodeId: string) => {
    if (!snapshot) return
    // Reflow to make room for the inserted layer; one undo restores wiring and layout.
    const stage = architectureOverview(next, metadata).membership.get(nodeId)
    commit({ graph: next, view: { ...snapshot.view, direction: 'LR', positions: {}, expandedStages: [...new Set([...snapshot.view.expandedStages ?? [], ...(stage ? [stage] : [])])] } })
    setSelected([nodeId]); setSelectedEdge(null); setPanel('inspect'); pendingFocus.current = nodeId
    setMessage('Layer inserted. Connections updated automatically. Undo restores the previous wiring and layout.')
  }
  const add = (mode: 'add' | 'insert' | 'replace', kind: string, targetEdge = selectedEdge) => {
    if (!graph || graph.locked || !snapshot) return
    const choice = palette.find(p => p.type === kind)
    if (!choice || (mode === 'add' && !allowDisconnected)) return
    const node: GraphNodeSpec = { id: uid(kind), kind, params: { ...choice.defaults, ...(kind === 'hyper_dense' ? { shape_mode: 'preserve' } : {}) }, label: choice.label }
    let next = { ...graph, nodes: [...graph.nodes, node], edges: [...graph.edges] }
    if (mode === 'insert') {
      try { finishInsertion(insertOnEdge(graph, node, targetEdge ?? '', metadata), node.id) }
      catch (e) { setError(e instanceof Error ? e.message : String(e)) }
      return
    } else if (mode === 'replace') {
      try { next = replaceLayer(graph, selected[0], node, metadata) }
      catch (e) { setError(e instanceof Error ? e.message : String(e)); return }
    }
    pendingFocus.current = node.id
    const stage = architectureOverview(next, metadata).membership.get(node.id)
    commit({ graph: next, view: { ...snapshot.view, positions: {}, expandedStages: [...new Set([...snapshot.view.expandedStages ?? [], ...(stage ? [stage] : [])])] } })
    setSelected([node.id]); setSelectedEdge(null); setPanel('inspect')
    setMessage(mode === 'replace' ? 'Layer replaced. Connections kept. Undo restores the previous layer.' : 'Unconnected layer created. Choose its input sources in Inspect.')
  }
  const inspect = graph?.nodes.find(n => n.id === selected[0])
  const info = inspect ? metadata[inspect.id] : undefined
  const settings = inspect ? { ...info?.settings, ...inspect.params } : {}
  const layerKind = inspect ? denseKind(inspect, info) : null
  const autoFit = inspect?.kind === 'hyper_dense' && inspect.params.shape_mode === 'preserve'
  const shapeFit = autoFit && validatedGraph === graph ? info?.shape_fit : undefined
  const algebraDimensions = catalog.algebra_dimensions ?? DEFAULT_ALGEBRA_DIMENSIONS
  const switchLayer = async (target: 'dense' | 'hyper_dense', algebra?: string) => {
    if (!graph || !inspect || graph.locked || validatedGraph !== graph || swapping) return
    setSwapError(''); setSwapping(true)
    const selectedEvaluation = evaluation
    try {
      const selectedAlgebra = algebra ?? replacementAlgebra
      const { spec: next } = await api.editGraph(graph, { action: 'replace', id: inspect.id, kind: target, params: { algebra: selectedAlgebra } }, selectedEvaluation.cells)
      let preview: Record<string, GraphNodeInfo> | undefined
      for (const cell of selectedEvaluation.cells) {
        const before = await api.validateGraph(graph, cell.window, cell.horizon)
        const after = await api.validateGraph(next, cell.window, cell.horizon)
        if (JSON.stringify(before.graph_nodes[inspect.id]?.shape) !== JSON.stringify(after.graph_nodes[inspect.id]?.shape)) throw new Error(`The replacement changes this layer’s shape for window ${cell.window} / horizon ${cell.horizon}. Original layer kept.`)
        preview ??= after.graph_nodes
      }
      if (current.current?.graph !== graph || evaluationRef.current !== selectedEvaluation) throw new Error('The experiment changed while checking. Please try the switch again.')
      commit({ ...current.current, graph: next })
      setMetadata(previous => ({ ...previous, ...preview }))
      if (target === 'hyper_dense') setReplacementAlgebra(selectedAlgebra)
      setMessage(`${target === 'hyper_dense' ? 'HyperDense' : 'Dense'} applied. Connections and output shape preserved; replacement weights are freshly initialized.`)
    } catch (e) { setSwapError(e instanceof Error ? e.message : String(e)) }
    finally { setSwapping(false) }
  }
  const editableLayers = graph?.nodes ?? []
  const updateParams = (key: string, value: unknown) => {
    if (!graph || !inspect) return
    edit({ ...graph, nodes: graph.nodes.map(n => n.id === inspect.id || (inspect.module_ref && n.module_ref === inspect.module_ref) ? { ...n, params: { ...n.params, [key]: value } } : n) })
  }
  const selectedNodeIds = selectionIds(selected).filter(id => graph?.nodes.some(n => n.id === id))
  const save = async () => {
    if (!graph || !snapshot) return
    try { const record = await api.saveGraph(graph, { ...snapshot.view, renderer: 'flow', version: 1, camera: camera.current, direction: 'LR' }, savedId); setSavedId(record.id); setMessage('Graph and canvas layout saved.'); void queryClient.invalidateQueries({ queryKey: ['architectures'] }) }
    catch (e) { setError(String(e)) }
  }
  const exportGraph = () => {
    if (!snapshot) return
    const url = URL.createObjectURL(new Blob([YAML.stringify({ ...snapshot.graph, view: { ...snapshot.view, renderer: 'flow', version: 1, camera: camera.current, direction: 'LR' } })], { type: 'text/yaml' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'architecture-graph.yaml'; anchor.click(); URL.revokeObjectURL(url)
  }
  const openAdd = () => { setAllowDisconnected(false); setMoveId(null); setPaletteSearch(''); setPanel('add'); revealPanel() }
  const inspectStage = overview?.stages.find(stage => stage.nodeIds.includes(inspect?.id ?? ''))
  const nearbyLayers = graph?.nodes.filter(n => (!inspectStage || inspectStage.nodeIds.includes(n.id)) && (isLayerOperation(n, metadata[n.id]) || n.id === inspect?.id)) ?? []
  const nearbyIndex = nearbyLayers.findIndex(n => n.id === inspect?.id)
  const selectedConnection = graph?.edges.find(e => edgeId(e) === selectedEdge)
  const name = (id: string) => graph?.nodes.find(n => n.id === id)?.label ?? metadata[id]?.label ?? id
  const treeNodes = editableLayers.filter(n => layerSearch.trim()
    ? `${n.id} ${n.label ?? ''} ${metadata[n.id]?.label ?? n.kind} ${metadata[n.id]?.source_path ?? ''} ${overview?.stages.find(g => g.id === overview.membership.get(n.id))?.label ?? ''}`.toLowerCase().includes(layerSearch.trim().toLowerCase())
    : showOperations || isLayerOperation(n, metadata[n.id]) || ['placeholder', 'output'].includes(metadata[n.id]?.category ?? ''))
  const treeRow = (id: string, ordinal?: number) => <div key={id} className="architecture-tree-row"><input type="checkbox" aria-label={`Include ${name(id)} in selection`} checked={selected.includes(id)} onChange={() => { select(id, true); setPanel('architecture') }} /><button aria-label={`Select ${id}`} aria-pressed={selected.includes(id)} onClick={e => selectFromTree(id, e.shiftKey)}><strong>{ordinal ? `${ordinal}. ` : ''}{name(id)}</strong><small>{metadata[id]?.category === 'call_module' ? 'Layer' : graph?.nodes.find(n => n.id === id)?.kind} · {formatShape(metadata[id]?.shape)}</small></button></div>
  return <section className={`graph-workspace node-workspace ${library ? 'library-open' : 'canvas-mode'}`} aria-label="Architecture workspace" onKeyDown={e => {
      if (e.key === 'Escape' && (moveId || selectedEdge || panel === 'add')) { setMoveId(null); setSelectedEdge(null); setAllowDisconnected(false); setPanel('inspect'); return }
      if ((e.target as HTMLElement).closest('input, select, textarea')) return
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(e.shiftKey) }
      if ((e.key === 'Delete' || e.key === 'Backspace') && !(e.target as HTMLElement).closest('button')) { e.preventDefault(); remove() }
    }}>
    <SidePanel active={active}><div className="builder-panel-header"><span className="eyebrow">Architecture workspace</span><h2>{graph?.name ?? 'Build an experiment'}</h2><button onClick={() => setLibrary(!library)}>{library ? 'Architecture diagram' : 'Method collection'}</button>{graph?.locked && snapshot && <button onClick={() => { commit({ ...snapshot, graph: { ...graph, name: `${graph.name} experiment`.slice(0, 80), locked: false, preset_id: undefined } }); setSavedId(undefined) }}>Clone to edit</button>}{!library && <div className="builder-quick-actions"><button onClick={findLayer} title="Find a layer · ⌘/Ctrl K"><Search size={15} />Find layer<kbd>⌘ K</kbd></button><button className="primary-action" disabled={!graph || graph.locked} onClick={openAdd}><Plus size={15} />Add layer</button></div>}</div>
      {!library && <><div className="builder-section-tabs" role="tablist" aria-label="Architecture tools">{(['architecture', 'inspect', 'add', 'run'] as const).map(key => <button key={key} role="tab" id={`builder-tab-${key}`} aria-controls={`builder-panel-${key}`} aria-selected={panel === key} onClick={() => { setPanel(key); setAllowDisconnected(false); if (key !== 'add') setMoveId(null) }}>{key === 'architecture' ? 'Architecture' : key === 'inspect' ? 'Inspect' : key === 'add' ? 'Add' : 'Run'}</button>)}</div>
      <div role="tabpanel" id={`builder-panel-${panel}`} aria-labelledby={`builder-tab-${panel}`} className="builder-tab-content">
      {panel === 'architecture' && <>
        <label>Starting model<select aria-label="Load graph preset" disabled={loading} value={graph?.preset_id ?? ''} onChange={e => { const p = catalog.presets.find(p => p.preset_id === e.target.value); if (p) void load(p) }}><option value="" disabled>{graph?.name ?? 'Choose a starting point…'}</option>{presetOptions}</select></label>
        <h3>Stages & layers</h3><div className="layer-search-field"><Search size={16} /><input ref={searchInput} aria-label="Search layers" placeholder="Find a layer…" value={layerSearch} onChange={e => setLayerSearch(e.target.value)} />{layerSearch && <button aria-label="Clear layer search" onClick={() => { setLayerSearch(''); searchInput.current?.focus() }}><X size={14} /></button>}</div><label className="show-operations"><input type="checkbox" checked={showOperations} onChange={e => setShowOperations(e.target.checked)} />Show tensor operations</label>{layerSearch && <p className="search-result-count" role="status">{treeNodes.length ? `${treeNodes.length} matching operations` : 'No matching layers. Try a name such as Dense or LSTM.'}</p>}
        <div className="architecture-tree" aria-label="Architecture tree" onKeyDown={e => {
          const buttons = [...e.currentTarget.querySelectorAll<HTMLButtonElement>('button')], index = buttons.indexOf(e.target as HTMLButtonElement)
          if (index < 0) return
          const next = e.key === 'ArrowDown' ? Math.min(index + 1, buttons.length - 1) : e.key === 'ArrowUp' ? Math.max(0, index - 1) : e.key === 'Home' ? 0 : e.key === 'End' ? buttons.length - 1 : -1
          if (next >= 0) { e.preventDefault(); buttons[next].focus() }
        }}>{overview?.stages.map(g => { const members = treeNodes.filter(n => g.nodeIds.includes(n.id)); if (!members.length && layerSearch.trim()) return null; const expanded = (snapshot?.view.expandedStages ?? []).includes(g.id); return <div className="architecture-tree-stage" key={g.id}><div className="architecture-stage-row"><input type="checkbox" aria-label={`Include stage ${g.label} in selection`} checked={selected.includes(`group:${g.id}`)} onChange={() => { select(`group:${g.id}`, true); setPanel('architecture') }} /><button onClick={() => select(`group:${g.id}`)}>{g.label}<small>{g.nodeIds.length} operations</small></button><button aria-label={`${expanded ? 'Collapse' : 'Expand'} ${g.label}`} aria-expanded={expanded} onClick={() => toggle(g.id)}>{expanded ? '−' : '+'}</button></div>{(expanded || layerSearch) && members.map((n, index) => treeRow(n.id, index + 1))}</div> })}{treeNodes.filter(n => !overview?.membership.has(n.id)).map(n => treeRow(n.id))}</div>
        <div className="architecture-model-files"><h3>Model & files</h3>
        <details><summary>Open saved</summary><label>Saved experiments<select aria-label="Load saved graph" disabled={loading || !records.data?.length} value="" onChange={e => { const r = records.data?.find(r => r.id === e.target.value); if (r) void load(r.spec, r) }}><option value="">{records.data?.length ? 'Open an experiment…' : 'No saved experiments yet'}</option>{records.data?.map(r => <option key={r.id} value={r.id}>{r.spec.name}</option>)}</select></label></details>
        {graph && <label>Model name<input aria-label="Graph name" key={`${loadKey}:${graph.name}`} defaultValue={graph.name} disabled={graph.locked} onBlur={e => { if (e.target.value !== graph.name) edit({ ...graph, name: e.target.value }) }} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }} /></label>}
        <div className="builder-button-row"><button onClick={save} disabled={!graph || graph.locked || loading}>Save graph</button><button onClick={exportGraph} disabled={!graph}>Export YAML</button></div>
        <label className="graph-import">Import architecture<input type="file" accept=".json,.yaml,.yml" onChange={async e => { const file = e.target.files?.[0]; if (!file) return; try { const raw = YAML.parse(await file.text()); await load(raw, raw.schema_version === 2 ? { id: '', spec: raw, view: raw.view } : undefined) } catch (err) { setError(String(err)) } e.target.value = '' }} /></label>
        </div>
        <details><summary>Add another model</summary><select aria-label="Add another model" value="" disabled={!graph || graph.locked || loading} onChange={e => void addModel(e.target.value)}><option value="" disabled>Choose a model…</option>{presetOptions}</select></details>
      </>}
      {panel === 'add' && <><h3>{moveId ? `Move ${name(moveId)}` : 'Add a layer'}</h3><p>{moveId ? 'Choose a connection marker in the diagram or a destination below.' : selectedConnection ? `Insert between ${name(selectedConnection.source)} and ${name(selectedConnection.target)}.` : allowDisconnected ? 'Create a draft and choose its input sources in Inspect.' : 'Click a + between nodes, or choose a connection below. Then choose the layer to insert.'}</p>
        {selectedConnection && <div className="insertion-context"><span>Insert between</span><strong>{name(selectedConnection.source)}</strong><ArrowRight size={15} /><strong>{name(selectedConnection.target)}</strong><small>Both connections will be wired automatically.</small></div>}
        <label>Insertion destination<select aria-label="Insertion destination" value={selectedEdge ?? ''} disabled={graph?.locked} onChange={e => { if (moveId && e.target.value) chooseInsertion(e.target.value); else { setSelectedEdge(e.target.value || null); setAllowDisconnected(false) } }}><option value="">{allowDisconnected ? 'Unconnected layer' : 'Choose a connection…'}</option>{graph?.edges.map(e => <option key={edgeId(e)} value={edgeId(e)}>{name(e.source)} → {name(e.target)} · {e.port}</option>)}</select></label>
        {moveId ? <button onClick={() => { setMoveId(null); setSelectedEdge(null) }}>Cancel move</button> : <><input aria-label="Search layer palette" placeholder="Find a layer…" value={paletteSearch} onChange={e => setPaletteSearch(e.target.value)} /><div className="architecture-palette">{palette.filter(p => `${p.label} ${p.type}`.toLowerCase().includes(paletteSearch.toLowerCase())).map(p => <button key={p.type} disabled={!graph || graph.locked || (!selectedEdge && !allowDisconnected) || (Boolean(selectedEdge) && ['add', 'multiply', 'concat'].includes(p.type))} title={`Add ${p.label}`} onClick={() => add(selectedEdge ? 'insert' : 'add', p.type)}><strong>{p.label}</strong><small>{['add', 'multiply', 'concat'].includes(p.type) ? selectedEdge ? 'Add separately: this layer needs multiple inputs' : 'Multiple inputs · connect in Inspect' : selectedEdge ? 'Insert at selected connection' : allowDisconnected ? 'Add unconnected draft' : 'Choose a connection first'}</small></button>)}</div></>}
        {!moveId && <details className="advanced-add"><summary>Advanced</summary><button disabled={!graph || graph.locked} onClick={() => { setSelectedEdge(null); setAllowDisconnected(true) }}>Create unconnected layer</button></details>}
      </>}
      {panel === 'inspect' && graph && snapshot && <>
        {selectedConnection && <section><h3>Connection</h3><p>{name(selectedConnection.source)} → {name(selectedConnection.target)}</p><label>Source<select aria-label="Connection source" disabled={graph.locked} value={selectedConnection.source} onChange={e => { try { edit(connectGraph(graph, { ...selectedConnection, source: e.target.value }, selectedEdge!)); setSelectedEdge(edgeId({ ...selectedConnection, source: e.target.value })) } catch (err) { setError(String(err)) } }}>{graph.nodes.filter(n => n.id !== selectedConnection.target).map(n => <option key={n.id} value={n.id}>{name(n.id)}</option>)}</select></label><label>Destination port<select aria-label="Connection destination port" disabled={graph.locked} value={JSON.stringify([selectedConnection.target, selectedConnection.port])} onChange={e => { try { const [target, port] = JSON.parse(e.target.value); const next = { ...selectedConnection, target, port }; edit(connectGraph(graph, next, selectedEdge!)); setSelectedEdge(edgeId(next)) } catch (err) { setError(String(err)) } }}>{graph.nodes.filter(n => n.id !== selectedConnection.source).flatMap(n => portsFor(n, graph, metadata).map(port => <option key={JSON.stringify([n.id, port])} value={JSON.stringify([n.id, port])}>{name(n.id)} · {port}</option>))}</select></label><div className="builder-button-row"><button disabled={graph.locked} onClick={() => setPanel('add')}>Insert a layer</button><button disabled={graph.locked} onClick={remove}>Remove connection</button></div></section>}
        {overview?.stages.filter(g => selected.includes(`group:${g.id}`)).map(g => {
          const layers = g.nodeIds.filter(id => { const n = graph.nodes.find(n => n.id === id); return n && isLayerOperation(n, metadata[id]) })
          const other = g.nodeIds.filter(id => !layers.includes(id))
          return <section key={g.id} className="stage-inspector"><span className="eyebrow">Stage</span><h3>{g.label}</h3><p className="inspector-shape">{scene.blocks.find(b => b.id === `group:${g.id}`)?.shape ?? formatShape(metadata[g.primaryId]?.shape)}</p><div className="builder-button-row"><button className="primary-action" onClick={() => selectFromTree(g.primaryId)}>Inspect main layer</button><button onClick={() => toggle(g.id)}>{(snapshot.view.expandedStages ?? []).includes(g.id) ? 'Collapse stage' : 'Expand stage'}</button></div>
          {layers.length > 0 && <><h3>Layers in this stage <small>{layers.length}</small></h3><div className="architecture-tree stage-layer-list">{layers.map((id, index) => treeRow(id, index + 1))}</div></>}
          {other.length > 0 && <details className="stage-tensor-operations" open={!layers.length}><summary>Tensor operations ({other.length})</summary><div className="architecture-tree">{other.map(id => treeRow(id))}</div></details>}
          {advanced && g.originalGroup && <label>Stage name<input key={`${g.id}:${graph.groups.find(item => item.id === g.originalGroup)?.label}`} defaultValue={graph.groups.find(item => item.id === g.originalGroup)?.label} disabled={graph.locked} onBlur={e => { if (e.target.value !== graph.groups.find(item => item.id === g.originalGroup)?.label) edit({ ...graph, groups: graph.groups.map(item => item.id === g.originalGroup ? { ...item, label: e.target.value } : item) }) }} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }} /></label>}</section>
        })}
            {inspect ? <><div className="inspector-navigation"><button className="stage-back" onClick={() => { if (inspectStage) select(`group:${inspectStage.id}`); else { setPanel('architecture'); revealPanel() } }}><ArrowLeft size={20} aria-hidden="true" /><span>Back to {inspectStage?.label ?? 'architecture'}</span></button><div className="layer-stepper"><button aria-label="Previous layer" title="Previous layer" disabled={nearbyIndex <= 0} onClick={() => selectFromTree(nearbyLayers[nearbyIndex - 1].id)}><ArrowLeft size={15} /></button><select aria-label="Layer in stage" value={inspect.id} onChange={e => selectFromTree(e.target.value)}>{nearbyLayers.map((n, i) => <option key={n.id} value={n.id}>{i + 1}. {name(n.id)}</option>)}</select><button aria-label="Next layer" title="Next layer" disabled={nearbyIndex < 0 || nearbyIndex >= nearbyLayers.length - 1} onClick={() => selectFromTree(nearbyLayers[nearbyIndex + 1].id)}><ArrowRight size={15} /></button></div><button className="focus-layer" onClick={() => { if (pendingFocus.current) pendingFocus.current = inspect.id; diagram.current?.fit(inspect.id) }}><Crosshair size={14} />Locate on canvas</button></div><h3>{name(inspect.id)}</h3><p className="inspector-shape">{formatShape(info?.shape)}{validatedGraph !== graph ? ' · last valid shape' : ''}</p>
              <label>Layer type<select aria-label="Layer type" value={layerKind ?? inspect.kind} disabled={graph.locked || swapping} onChange={e => {
                const kind = e.target.value
                if (layerKind && (kind === 'dense' || kind === 'hyper_dense')) void switchLayer(kind)
                else add('replace', kind)
              }}>{!palette.some(p => p.type === (layerKind ?? inspect.kind)) && <option value={inspect.kind} disabled>{name(inspect.id)} (current)</option>}{palette.map(p => {
                const arity = portsFor({ id: '__replacement__', kind: p.type, params: p.defaults }, graph, {}).length
                const compatible = arity === replacementPorts(inspect, graph, metadata).length
                const needsValidation = layerKind && p.type !== layerKind && ['dense', 'hyper_dense'].includes(p.type) && (validating || validatedGraph !== graph)
                return <option key={p.type} value={p.type} disabled={!compatible || Boolean(needsValidation)}>{p.label}{compatible ? '' : ` · needs ${arity} inputs`}</option>
              })}</select></label><p className="replacement-hint">Choose another type to replace this layer. Its connections stay in place.</p>
              {inspect.kind === 'hyper_dense' && <label><input type="checkbox" aria-label="Auto-fit connection" disabled={graph.locked || swapping} checked={autoFit} onChange={e => updateParams('shape_mode', e.target.checked ? 'preserve' : 'manual')} />Auto-fit connection</label>}
              {autoFit && <p>Feature axis: last (−1). {shapeFit ? `${shapeFit.input_width} features → ${shapeFit.padded_width} padded → ${shapeFit.output_width} output; ${shapeFit.units} hypercomplex units, ${shapeFit.padding} zeros added, ${shapeFit.crop} outputs cropped.` : 'Dimensions will be calculated when this connection validates.'}</p>}
              {Object.entries(settings).filter(([key]) => key !== 'algebra' && key !== 'shape_mode' && !(autoFit && ['units', 'out_features'].includes(key))).map(([key, value]) => <label key={`${inspect.id}:${key}`}>{layerKind === 'hyper_dense' && ['units', 'out_features'].includes(key) ? `Hypercomplex units (${algebraDimensions[String(settings.algebra)]} features each)` : ({ out_features: 'Neurons (output width)', units: 'Neurons', p: 'Dropout rate', bias: 'Use bias', out_channels: 'Output channels', kind: 'Activation', num_layers: 'Number of layers' } as Record<string, string>)[key] ?? key.replaceAll('_', ' ')}
                {typeof value === 'boolean' ? <input type="checkbox" disabled={graph.locked || swapping} checked={value} onChange={e => updateParams(key, e.target.checked)} /> : <input aria-label={key} key={`${inspect.id}:${key}:${JSON.stringify(value)}`} disabled={graph.locked || swapping} defaultValue={typeof value === 'object' ? JSON.stringify(value) : String(value)} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }} onBlur={e => { try { const raw = e.target.value; const parsed = typeof value === 'number' ? Number(raw) : typeof value === 'object' ? JSON.parse(raw) : raw; if (JSON.stringify(parsed) !== JSON.stringify(value)) updateParams(key, parsed) } catch (err) { setError(String(err)) } }} />}
              </label>)}
              {!Object.keys(settings).length && <p>This is a structural operation. Reconnect its inputs or replace it with a palette operation.</p>}
              {layerKind && <div className="layer-type-card">
                <label>{layerKind === 'dense' ? 'Target algebra' : 'Algebra'}<select aria-label="Algebra" disabled={graph.locked || swapping || (!autoFit && (validating || validatedGraph !== graph))} value={layerKind === 'hyper_dense' ? String(settings.algebra) : replacementAlgebra} onChange={e => autoFit ? updateParams('algebra', e.target.value) : layerKind === 'hyper_dense' ? void switchLayer('hyper_dense', e.target.value) : setReplacementAlgebra(e.target.value)}>{layerKind === 'hyper_dense' && settings.algebra === 'tricomplex' && <option value="tricomplex" disabled>Cyclic tricomplex (existing layer)</option>}{catalog.algebras.filter(algebra => algebra !== 'tricomplex').map(algebra => <option key={algebra} value={algebra}>{({ complex: 'Complex (2D)', split_complex: 'Split-complex (2D)', quaternion: 'Quaternion (4D)', coquaternion: 'Coquaternion (4D)', cl11: 'Cl(1,1) (4D)', octonion: 'Octonion (8D)' } as Record<string, string>)[algebra] ?? `${algebra} (${algebraDimensions[algebra]}D)`}</option>)}</select></label>
                <p>{autoFit ? 'Output shape matches the input connection.' : layerKind === 'hyper_dense' ? `${Number(settings.units ?? settings.out_features) * algebraDimensions[String(settings.algebra)]} output features = ${settings.units ?? settings.out_features} hypercomplex units × ${algebraDimensions[String(settings.algebra)]} components.` : `Switch to ${replacementAlgebra} HyperDense without reconnecting this layer.`}</p>
                <small>{autoFit ? 'Checks every evaluation size. Zero-padding and cropping use no extra trainable projections.' : 'Checks every evaluation size. New weights; no hidden padding.'}</small>
                {layerKind === 'hyper_dense' && <button type="button" onClick={() => { pendingFocus.current = null; diagram.current?.hold(); setWeightsOpen(true) }}>Inspect weight structure</button>}
                {inspect.module_ref && graph.nodes.filter(n => n.module_ref === inspect.module_ref).length > 1 && <p>Switching affects only this call and gives it independent weights.</p>}
                {swapping && <p role="status">Checking replacement compatibility…</p>}
                {swapError && <p className="swap-error" role="alert">{swapError}</p>}
              </div>}
              {advanced && <label>Label<input key={`${inspect.id}:${inspect.label}`} defaultValue={inspect.label ?? info?.label ?? inspect.kind} disabled={graph.locked} onBlur={e => { if (e.target.value !== (inspect.label ?? info?.label ?? inspect.kind)) edit({ ...graph, nodes: graph.nodes.map(n => n.id === inspect.id ? { ...n, label: e.target.value } : n) }) }} onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur() }} /></label>}
              {advanced && <><code>{info?.source_path ?? inspect.kind}</code><p>Output shape: {JSON.stringify(info?.shape ?? 'Not validated')}</p></>}
              {validatedGraph !== graph && <small>Shape information is from the last valid graph.</small>}
              {portsFor(inspect, graph, metadata).length > 0 && <details key={`inputs:${inspect.id}`} className="canvas-input-picker" open={portsFor(inspect, graph, metadata).some(port => !graph.edges.some(e => e.target === inspect.id && e.port === port)) || undefined}><summary>Input connections</summary><p>Choose a source without dragging a wire. Changing a source replaces this input only.</p>
                {portsFor(inspect, graph, metadata).map(port => {
                  const existing = graph.edges.find(e => e.target === inspect.id && e.port === port)
                  return <label key={port}>{port}<select aria-label={`Source for ${port}`} disabled={graph.locked} value={existing?.source ?? ''} onChange={e => {
                    try { edit(e.target.value ? connectGraph(graph, { source: e.target.value, target: inspect.id, port }, existing ? edgeId(existing) : undefined) : { ...graph, edges: graph.edges.filter(edge => edge !== existing) }) }
                    catch (err) { setError(String(err)) }
                  }}><option value="">Not connected</option>{graph.nodes.filter(n => n.id !== inspect.id).map(n => <option key={n.id} value={n.id}>{n.label ?? metadata[n.id]?.label ?? n.kind} · {n.id}</option>)}</select></label>
                })}</details>}
              {advanced && info?.ports.map(port => <small key={port} className="graph-port-row">{port} ← {graph.edges.find(e => e.target === inspect.id && e.port === port)?.source ?? 'unconnected'}</small>)}
              {advanced && inspect.module_ref && <p>Shared weights: <code>{inspect.module_ref}</code> · {graph.nodes.filter(n => n.module_ref === inspect.module_ref).length} calls. Settings apply to every shared call.</p>}
            {advanced && <>
              <button disabled={graph.locked} onClick={() => edit({ ...graph, output: inspect.id })}>Use as forecast output</button>
              {inspect.module_ref && <button disabled={graph.locked} onClick={() => edit({ ...graph, nodes: graph.nodes.map(n => n.id === inspect.id ? { ...n, module_ref: uid('weights') } : n) })}>Make weights independent</button>}
              </>}
            </> : !selectedConnection && !selected.length ? <div className="builder-panel-empty"><Search size={24} /><h3>Choose a layer to get started</h3><p>Select a node or search by name. Its settings and layer type will appear here. Use + between nodes to add a layer.</p><button onClick={findLayer}>Find a layer</button></div> : null}
            {inspect && advanced && <button disabled={graph.locked} onClick={() => { setMoveId(inspect.id); setSelectedEdge(null); setPanel('add') }}>Move selected layer</button>}
            {(selected.length > 0) && <>
            <button disabled={graph.locked || !selectedNodeIds.length} onClick={() => edit(duplicateGraph(graph, selectedNodeIds))}>Duplicate independently</button>
            <button disabled={graph.locked || (!selected.length && !selectedEdge)} onClick={remove}>Delete selected</button>
            </>}
            <button aria-pressed={advanced} onClick={() => setAdvanced(!advanced)}>Advanced tools</button>
            {advanced && <>
            <button disabled={graph.locked || selectedNodeIds.length < 2} onClick={() => {
              const calls = graph.nodes.filter(n => selectedNodeIds.includes(n.id))
              if (calls.some(n => n.kind === 'source' && metadata[n.id]?.category !== 'call_module') || calls.some(n => n.kind !== calls[0].kind)) { setError('Select matching layer operations to share weights.'); return }
              const ref = uid('shared')
              edit({ ...graph, nodes: graph.nodes.map(n => selectedNodeIds.includes(n.id) ? { ...n, module_ref: ref, params: { ...calls[0].params } } : n) })
            }}>Share selected weights</button>
            <button disabled={graph.locked || !selectedNodeIds.length} onClick={() => { const id = uid('group'); edit({ ...graph, groups: [...graph.groups, { id, label: 'Custom group' }], nodes: graph.nodes.map(n => selectedNodeIds.includes(n.id) ? { ...n, group: id } : n) }) }}>Group selected</button>
            <button disabled={graph.locked || !selectedNodeIds.length} onClick={() => edit({ ...graph, nodes: graph.nodes.map(n => selectedNodeIds.includes(n.id) ? { ...n, group: undefined } : n) })}>Ungroup selected</button>
            </>}

      </>}
      {panel === 'run' && graph && <><h3>Run experiment</h3><div className="graph-validation" role="status">{validating ? 'Checking your architecture…' : validatedGraph === graph ? `Ready to run · ${parameters?.toLocaleString() ?? '—'} parameters` : 'Not ready yet — fix the connection or shape issue below.'}</div><EvaluationPanel catalog={catalog} evaluation={evaluation} onChange={setEvaluation} /><RunControls architecture={graph} evaluation={evaluation} disabled={loading || swapping || validating || validatedGraph !== graph} onQueued={job => setMessage(`Queued run ${job.id}`)} /></>}
      </div>
      {validationError && <div className="error-notice" role="alert">{validationError}</div>}{!!warnings.length && <details className="builder-validation-notes"><summary>{warnings.length} disconnected operations</summary>{warnings.map(w => <p key={w}>{w}</p>)}</details>}
      {loading && <p role="status">Preparing architecture…</p>}{error && <div role="alert" className="error-notice">{error}</div>}{message && <p className="builder-feedback" role="status">{message}</p>}</>}
    </SidePanel>
    {library ? <MethodCollection catalog={catalog} active={active} onLoad={id => { const p = catalog.presets.find(p => p.preset_id === id); if (p) void load(p) }} /> : <div className="graph-canvas architecture-canvas" ref={canvasRef} aria-label="Architecture canvas" tabIndex={0}>
      <ArchitectureDiagram key={loadKey} ref={diagram} scene={scene} selected={selected} selectedEdge={selectedEdge} active={active} stale={validatedGraph !== graph} inserting={!graph?.locked && (panel === 'add' || Boolean(moveId))} camera={camera.current} onCamera={next => { camera.current = next }} onSelect={select} onClearSelection={() => { setSelected([]); setSelectedEdge(null) }} onEdge={selectEdge} onInsert={chooseInsertion} onToggle={toggle} expandedGroups={overview?.stages.filter(g => (snapshot?.view.expandedStages ?? []).includes(g.id)) ?? []} locked={Boolean(graph?.locked)} />
      {snapshot && <div className="canvas-action-toolbar" role="group" aria-label="Canvas actions"><button type="button" aria-label="Undo" title="Undo" disabled={!past.current.length} onClick={() => undo()}><Undo2 size={18} /></button><button type="button" aria-label="Redo" title="Redo" disabled={!future.current.length} onClick={() => undo(true)}><Redo2 size={18} /></button><span className="canvas-action-divider" /><button type="button" aria-label="Zoom out canvas" title="Zoom out" onClick={() => diagram.current?.zoom(1 / 1.2)}><ZoomOut size={18} /></button><button type="button" aria-label="Zoom in canvas" title="Zoom in" onClick={() => diagram.current?.zoom(1.2)}><ZoomIn size={18} /></button><button type="button" aria-label="Fit diagram" title="Fit diagram" onClick={() => { pendingFocus.current = '__all__'; diagram.current?.fit() }}><Maximize size={18} /></button><span className="canvas-action-divider" /><button type="button" aria-label="Reset layout" title="Reset layout · stage overview" onClick={() => { pendingFocus.current = '__all__'; commit({ ...snapshot, view: { ...emptyView(), collapsed: snapshot.graph.groups.map(g => g.id) } }) }}><RotateCcw size={18} /></button></div>}
      {weightsOpen && graph && <div className="canvas-weight-overlay"><WeightInspector graph={graph} nodeId={inspect?.id} window={evaluation.cells[0]?.window ?? 10} horizon={evaluation.cells[0]?.horizon ?? 1} onClose={() => setWeightsOpen(false)} active={active} /></div>}
    </div>}
  </section>
}
