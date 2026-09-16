import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import type { CSSProperties, Ref } from 'react'
import { ReactFlow, ReactFlowProvider, Handle, Position, Background, BackgroundVariant, MiniMap, BaseEdge, EdgeLabelRenderer, MarkerType, ViewportPortal, applyNodeChanges, getSmoothStepPath } from '@xyflow/react'
import type { Node, Edge, NodeProps, EdgeProps, Viewport } from '@xyflow/react'
import { ChevronDown, ChevronUp, Layers3, Plus, AlertCircle } from 'lucide-react'
import type { ArchitectureScene, SceneBlock, SceneRoute, Point } from './architectureScene'
import type { DiagramCamera } from './types'
import '@xyflow/react/dist/style.css'

export type DiagramHandle = { fit: (id?: string) => void; zoom: (factor: number) => void; hold: () => void }
type Props = { scene: ArchitectureScene; selected: string[]; selectedEdge: string | null; active: boolean; stale: boolean; inserting: boolean; locked?: boolean; camera?: DiagramCamera; onCamera: (camera: DiagramCamera) => void; onSelect: (id: string, multiple: boolean) => void; onClearSelection?: () => void; onEdge: (id: string) => void; onInsert: (id: string) => void; onToggle: (id: string) => void; expandedGroups: { id: string; label: string }[] }
type CardData = { block: SceneBlock; stale: boolean; muted: boolean; onToggle: Props['onToggle']; onSelect: Props['onSelect'] }
type Card = Node<CardData, 'architecture'>
type WireData = { route: SceneRoute; showInsert: boolean; related: boolean; muted: boolean; onInsert: Props['onInsert'] }
type Wire = Edge<WireData, 'architecture'>

function ArchitectureNode({ data: { block, stale, muted, onToggle, onSelect }, selected }: NodeProps<Card>) {
  return <div className={`architecture-node nopan ${selected ? 'is-selected' : ''} ${muted ? 'is-muted' : ''} ${block.disconnected ? 'is-disconnected' : ''} ${block.missingInputs.length ? 'is-invalid' : ''}`} style={{ '--node-accent': block.color } as CSSProperties} data-block-id={block.id} role="button" aria-label={`Select ${block.label}`} aria-pressed={selected} tabIndex={0} onKeyDown={e => { if (e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); e.stopPropagation(); onSelect(block.id, e.shiftKey) } }}>
    <div className="architecture-node-category"><span>{block.output ? 'Forecast output' : block.stage ? `${block.nodeIds.length} operations` : block.disconnected ? 'Disconnected draft' : block.kind === 'structural' ? 'Operation' : block.kind}</span>{block.stage && <button className="nodrag nopan node-expand" aria-label={`Expand ${block.label}`} title="Open the layers in this stage" onClick={e => { e.stopPropagation(); onToggle(block.stage!) }}><Layers3 size={12} />Open<ChevronDown size={12} /></button>}</div>
    <div className="architecture-node-title"><strong title={block.label}>{block.label}</strong></div>
    <div className={`architecture-node-shape ${stale ? 'is-stale' : ''}`} title={`${stale ? 'Last validated: ' : ''}${block.shape}`}><code>{block.shape}</code>{stale && <span>stale</span>}</div>
    {block.missingInputs.length > 0 && <div className="architecture-node-warning" title={block.missingInputs.join(', ')}><AlertCircle size={12} />{block.missingInputs.length} missing input{block.missingInputs.length > 1 ? 's' : ''}</div>}
    {block.inputs.map((p, i) => <Handle key={p.id} id={p.id} type="target" position={Position.Left} isConnectable={false} isConnectableStart={false} isConnectableEnd={false} style={{ top: `${100 * (i + 1) / (block.inputs.length + 1)}%` }} title={`Input · ${p.label}`} aria-label={`Input ${p.nodeId} ${p.port}`} />)}
    {block.outputs.map((p, i) => <Handle key={p.id} id={p.id} type="source" position={Position.Right} isConnectable={false} isConnectableStart={false} isConnectableEnd={false} style={{ top: `${100 * (i + 1) / (block.outputs.length + 1)}%` }} title={`Output · ${p.label}`} aria-label={`Output ${p.nodeId}`} />)}
  </div>
}

function roundedPath(points: Point[]) {
  const unique = points.filter((p, i) => !i || p.x !== points[i - 1].x || p.y !== points[i - 1].y)
  if (unique.length < 2) return ''
  let path = `M ${unique[0].x} ${unique[0].y}`
  for (let i = 1; i < unique.length - 1; i++) {
    const a = unique[i - 1], b = unique[i], c = unique[i + 1], before = Math.hypot(b.x - a.x, b.y - a.y), after = Math.hypot(c.x - b.x, c.y - b.y), radius = Math.min(9, before / 2, after / 2)
    path += ` L ${b.x - (b.x - a.x) / before * radius} ${b.y - (b.y - a.y) / before * radius} Q ${b.x} ${b.y} ${b.x + (c.x - b.x) / after * radius} ${b.y + (c.y - b.y) / after * radius}`
  }
  return `${path} L ${unique.at(-1)!.x} ${unique.at(-1)!.y}`
}
function ArchitectureEdge(props: EdgeProps<Wire>) {
  const { data, sourceX, sourceY, targetX, targetY, selected } = props
  if (!data) return null
  const { route } = data, start = route.points[0], end = route.points.at(-1)!
  const moved = Math.hypot(sourceX - start.x, sourceY - start.y, targetX - end.x, targetY - end.y) > 2
  const smooth = getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition: Position.Right, targetPosition: Position.Left, borderRadius: 9 })
  const path = moved ? smooth[0] : roundedPath(route.points), x = moved ? smooth[1] : route.insert.x, y = moved ? smooth[2] : route.insert.y
  return <><BaseEdge id={props.id} path={path} markerEnd={props.markerEnd} interactionWidth={32} style={{ stroke: selected ? '#6b3eda' : data.related ? '#9180c6' : '#93a5b4', strokeWidth: selected ? 2.8 : data.related ? 2 : 1.5, opacity: data.muted ? .32 : 1 }} />
    {data.showInsert && <EdgeLabelRenderer><button className="node-insert nodrag nopan" data-edge-id={route.id} style={{ transform: `translate(-50%, -50%) translate(${x}px, ${y}px)` }} aria-label={`Insert on ${route.label}`} title={`Insert on ${route.label}`} onClick={e => { e.stopPropagation(); data.onInsert(route.id) }}><Plus size={15} /></button></EdgeLabelRenderer>}</>
}
const nodeTypes = { architecture: ArchitectureNode }, edgeTypes = { architecture: ArchitectureEdge }

function Diagram(props: Props & { diagramRef: Ref<DiagramHandle> }) {
  const host = useRef<HTMLDivElement>(null), latest = useRef(props); latest.current = props
  const [size, setSize] = useState({ width: 1000, height: 650 })
  const [camera, setCamera] = useState<DiagramCamera>(props.camera ?? { panX: props.scene.width / 2, panY: props.scene.height / 2, zoom: 1 })
  const cameraRef = useRef(camera); cameraRef.current = camera
  const autoFit = useRef(!props.camera), measured = useRef(false), moving = useRef(false)
  const setView = (view: DiagramCamera, automatic = false) => { if (!automatic) autoFit.current = false; cameraRef.current = view; setCamera(view); latest.current.onCamera(view) }
  const fit = (id?: string, automatic = false) => {
    const rect = host.current?.getBoundingClientRect(); if (!rect?.width || !rect.height) return
    const blocks = latest.current.scene.blocks.filter(b => !id || b.id === id || b.nodeIds.includes(id))
    if (!blocks.length) return
    const left = Math.min(...blocks.map(b => b.x)), right = Math.max(...blocks.map(b => b.x + b.width)), top = Math.min(...blocks.map(b => b.y)), bottom = Math.max(...blocks.map(b => b.y + b.height))
    setView({ panX: (left + right) / 2, panY: (top + bottom) / 2, zoom: Math.max(.025, Math.min(id ? 1.3 : 1.1, (rect.width - 100) / (right - left + 70), (rect.height - 180) / (bottom - top + 80))) }, automatic)
  }
  useImperativeHandle(props.diagramRef, () => ({ fit, hold: () => { autoFit.current = false }, zoom: factor => setView({ ...cameraRef.current, zoom: Math.max(.025, Math.min(3, cameraRef.current.zoom * factor)) }) }))
  useEffect(() => {
    if (!props.active) return
    const observer = new ResizeObserver(() => {
      const rect = host.current?.getBoundingClientRect(); if (!rect?.width || !rect.height) return
      measured.current = true; setSize({ width: rect.width, height: rect.height })
      if (autoFit.current) fit(undefined, true)
    })
    observer.observe(host.current!); return () => observer.disconnect()
  }, [props.active])
  useEffect(() => { if (props.active && measured.current && autoFit.current) fit(undefined, true) }, [props.scene, props.active])
  const selectedBlocks = useMemo(() => new Set(props.scene.blocks.filter(b => props.selected.includes(b.id) || b.nodeIds.some(id => props.selected.includes(id))).map(b => b.id)), [props.scene, props.selected])
  const relatedRoutes = useMemo(() => new Set(props.scene.routes.filter(r => r.id === props.selectedEdge || selectedBlocks.has(r.sourceBlock) || selectedBlocks.has(r.targetBlock)).map(r => r.id)), [props.scene, props.selectedEdge, selectedBlocks])
  const relatedBlocks = useMemo(() => new Set([...selectedBlocks, ...props.scene.routes.filter(r => relatedRoutes.has(r.id)).flatMap(r => [r.sourceBlock, r.targetBlock])]), [props.scene, selectedBlocks, relatedRoutes])
  const hasSelection = selectedBlocks.size > 0 || Boolean(props.selectedEdge)
  const [nodes, setNodes] = useState<Card[]>([])
  useEffect(() => {
    setNodes(props.scene.blocks.map(b => ({ id: b.id, type: 'architecture', position: { x: b.x, y: b.y }, width: b.width, height: b.height, style: { width: b.width, height: b.height }, selected: selectedBlocks.has(b.id), data: { block: b, stale: props.stale, muted: hasSelection && !relatedBlocks.has(b.id), onToggle: id => latest.current.onToggle(id), onSelect: (id, multiple) => latest.current.onSelect(id, multiple) } })))
  }, [props.scene, props.stale, selectedBlocks, relatedBlocks, hasSelection])
  const edges: Wire[] = useMemo(() => props.scene.routes.map(r => ({ id: r.id, type: 'architecture', source: r.sourceBlock, target: r.targetBlock, sourceHandle: r.sourceHandle, targetHandle: r.targetHandle, selected: props.selectedEdge === r.id, ariaLabel: r.label, markerEnd: { type: MarkerType.ArrowClosed, color: props.selectedEdge === r.id ? '#6b3eda' : relatedRoutes.has(r.id) ? '#9180c6' : '#93a5b4', width: 16, height: 16 }, data: { route: r, showInsert: !props.locked && (props.inserting || props.selectedEdge === r.id || (camera.zoom >= .65 && (!hasSelection || relatedRoutes.has(r.id)))), related: relatedRoutes.has(r.id), muted: hasSelection && !relatedRoutes.has(r.id), onInsert: id => latest.current.onInsert(id) } })), [props.scene, props.selectedEdge, props.inserting, props.locked, camera.zoom, relatedRoutes, hasSelection])
  const viewport = { x: size.width / 2 - camera.panX * camera.zoom, y: size.height / 2 - camera.panY * camera.zoom, zoom: camera.zoom }
  const onViewport = (view: Viewport) => {
    // React Flow also emits this while synchronizing controlled props. Only
    // user gestures may feed a transform back into our saved camera state.
    if (!moving.current || !measured.current || !latest.current.scene.blocks.length) return
    const next = { panX: (size.width / 2 - view.x) / view.zoom, panY: (size.height / 2 - view.y) / view.zoom, zoom: view.zoom }
    if (Math.abs(next.panX - cameraRef.current.panX) + Math.abs(next.panY - cameraRef.current.panY) + Math.abs(next.zoom - cameraRef.current.zoom) < .00001) return
    cameraRef.current = next; setCamera(next); props.onCamera(next)
  }
  return <div ref={host} className="architecture-diagram node-diagram" aria-label="Node-based architecture editor" data-zoom={camera.zoom.toFixed(4)}>
    <ReactFlow<Card, Wire> nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} viewport={viewport} onViewportChange={onViewport} minZoom={.025} maxZoom={3}
      onNodesChange={changes => setNodes(current => applyNodeChanges(changes.filter(change => change.type === 'dimensions'), current))}
      onNodeDoubleClick={(_, node) => { if (node.data.block.stage) props.onToggle(node.data.block.stage) }}
      onNodeClick={(event, node) => { autoFit.current = false; props.onSelect(node.id, event.shiftKey) }} onEdgeClick={(_, edge) => { autoFit.current = false; props.onEdge(edge.id) }}
      onPaneClick={() => { autoFit.current = false; props.onClearSelection?.() }} onMoveStart={event => { if (event?.type) { autoFit.current = false; moving.current = true } }} onMoveEnd={event => { if (event?.type) moving.current = false }}
      nodesConnectable={false} edgesReconnectable={false} connectOnClick={false} nodesFocusable={false} nodesDraggable={false} edgesFocusable deleteKeyCode={null} selectionKeyCode={null} multiSelectionKeyCode={null} panOnScroll zoomOnScroll={false} zoomOnPinch zoomOnDoubleClick={false} onlyRenderVisibleElements>
      {(props.scene.blocks.length > 4 || camera.zoom < .8) && <MiniMap<Card> className="architecture-minimap" style={{ width: 172, height: 100 }} position="bottom-left" ariaLabel="Model overview map. Click to navigate." nodeColor={node => node.data.block.color} nodeStrokeColor={node => node.selected ? '#6b3eda' : '#ffffff'} nodeStrokeWidth={3} maskColor="#64748b24" onClick={(_, point) => setView({ ...cameraRef.current, panX: point.x, panY: point.y })} onNodeClick={(_, node) => fit(node.id)} />}
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#d6dfe7" />
      {props.scene.disconnectedY !== undefined && <ViewportPortal><div className="disconnected-label" style={{ transform: `translate(38px, ${props.scene.disconnectedY}px)` }}>Disconnected drafts · not executed</div></ViewportPortal>}
    </ReactFlow>
    <div className="diagram-heading"><span className="eyebrow">Architecture / {props.expandedGroups.length ? 'layers & connections' : 'stage overview'}</span><span>Use + to add · select a layer to change its type · drag background to pan</span></div>
    <div className="diagram-scale" aria-label="Canvas zoom">{Math.round(camera.zoom * 100)}%{(props.scene.blocks.length > 4 || camera.zoom < .8) && <span> · Click the map to navigate</span>}</div>
    <div className="diagram-stage-breadcrumbs">{props.expandedGroups.map(g => <button key={g.id} aria-label={`Collapse ${g.label}`} onClick={() => props.onToggle(g.id)}><ChevronUp size={13} />{g.label}</button>)}</div>
  </div>
}
export default forwardRef<DiagramHandle, Props>(function ArchitectureDiagram(props, ref) { return <ReactFlowProvider><Diagram {...props} diagramRef={ref} /></ReactFlowProvider> })
