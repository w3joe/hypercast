import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Download, X } from 'lucide-react'
import { api } from './api'
import { SidePanel } from './WorkspacePanels'
import type { GraphSpec, Job, ShapeFit } from './types'

type WeightLayer = { name: string; node_ids: string[]; error?: string; shape_fit?: ShapeFit; algebra: string; dimension: number; in_features: number; out_features: number; input_indices: number[]; output_indices: number[]; components: number[][][]; effective: number[][]; weight_parameters: number; dense_parameters: number }
export type WeightSnapshot = { source: 'initialized' | 'trained'; trial_id?: string; seed: number; fold?: number; window?: number; horizon?: number; layers: WeightLayer[]; omitted_layers: number }

function Matrix({ values, bound, title, rowIndices, columnIndices, blockRows, blockColumns }: { values: number[][]; bound: number; title: string; rowIndices: number[]; columnIndices: number[]; blockRows?: number; blockColumns?: number }) {
  const rows = values.length, columns = values[0]?.length ?? 0
  const step = 10
  return <svg className="weight-matrix" viewBox={`0 0 ${columns * step} ${rows * step}`} role="img" aria-label={`${title}. Rows are input features; columns are output features. Hover a cell for its exact value, or export the matrix.`}>
    <title>{title}</title>{values.flatMap((row, i) => row.map((v, j) => <rect key={`${i}:${j}`} x={j * step} y={i * step} width={step} height={step} fill={`hsl(${v < 0 ? 191 : 257} 72% ${97 - Math.min(1, Math.abs(v) / bound) * 55}%)`}><title>Input {rowIndices[i]}, output {columnIndices[j]}: {v.toPrecision(7)}</title></rect>))}
    {blockRows && Array.from({ length: Math.ceil(rows / blockRows) - 1 }, (_, i) => <line key={`r${i}`} x1={0} x2={columns * step} y1={(i + 1) * blockRows * step} y2={(i + 1) * blockRows * step} stroke="#fff" strokeWidth={.8} />)}
    {blockColumns && Array.from({ length: Math.ceil(columns / blockColumns) - 1 }, (_, i) => <line key={`c${i}`} y1={0} y2={rows * step} x1={(i + 1) * blockColumns * step} x2={(i + 1) * blockColumns * step} stroke="#fff" strokeWidth={.8} />)}
  </svg>
}

export default function WeightInspector({ graph, nodeId, jobs = [], window, horizon, onClose, active = true }: { graph?: GraphSpec; nodeId?: string; jobs?: Job[]; window: number; horizon: number; onClose?: () => void; active?: boolean }) {
  const [jobId, setJobId] = useState(''), [trialKey, setTrialKey] = useState(''), [layerName, setLayerName] = useState('')
  const job = jobs.find(j => j.id === jobId) ?? jobs[0]
  const preview = useQuery({ queryKey: ['weight-preview', graph, window, horizon], queryFn: () => api.previewWeights(graph!, window, horizon), enabled: Boolean(graph) && active, retry: false, staleTime: Infinity })
  const trained = useQuery({ queryKey: ['trained-weights', job?.id], queryFn: () => api.trainedWeights(job!.id), enabled: !graph && Boolean(job) && active, retry: false, staleTime: Infinity })
  const trials = Object.entries(trained.data ?? {}).filter(([, s]) => s.window === window && s.horizon === horizon)
  const trial = trials.find(([key]) => key === trialKey) ?? trials[0]
  const snapshot = graph ? preview.data : trial?.[1]
  const layers = snapshot?.layers ?? []
  const layer = layers.find(l => l.name === layerName) ?? layers.find(l => l.node_ids?.includes(nodeId ?? '')) ?? layers[0]
  const query = graph ? preview : trained
  const bound = layer && !layer.error ? Math.max(1e-12, ...layer.components.flat(2).map(Math.abs), ...layer.effective.flat().map(Math.abs)) : 1
  const exportMatrix = () => {
    if (!layer || layer.error) return
    const header = ['matrix', 'input_feature', 'output_feature', 'weight']
    const rows: (string | number)[][] = []
    layer.components.forEach((matrix, c) => matrix.forEach((row, i) => row.forEach((value, j) => rows.push([`component_${c}`, layer.input_indices[i], layer.output_indices[j], value]))))
    layer.effective.forEach((row, i) => row.forEach((value, j) => rows.push(['effective', Math.floor(i / layer.input_indices.length) * layer.in_features + layer.input_indices[i % layer.input_indices.length], Math.floor(j / layer.output_indices.length) * layer.out_features + layer.output_indices[j % layer.output_indices.length], value])))
    const url = URL.createObjectURL(new Blob([[header, ...rows].map(row => row.join(',')).join('\n')], { type: 'text/csv' }))
    const a = document.createElement('a'); a.href = url; a.download = `weights-${snapshot?.source}-${layer.name}.csv`; a.click(); URL.revokeObjectURL(url)
  }
  return <article className="chart-panel panel-surface weight-inspector" aria-label="Hypercomplex weight inspector">
    <SidePanel active={active}><section className="sidebar-section"><h3>Weight inspector</h3>{!graph && <label>Weight run<select aria-label="Weight run" value={job?.id ?? ''} onChange={e => { setJobId(e.target.value); setLayerName('') }}>{jobs.map(j => <option key={j.id} value={j.id}>{j.status.architecture_name} · {j.id.slice(-8)}</option>)}</select></label>}{!graph && <label>Weight replicate<select aria-label="Weight replicate" value={trial?.[0] ?? ''} onChange={e => setTrialKey(e.target.value)}>{!trials.length && <option value="">No saved weights</option>}{trials.map(([key, s]) => <option key={key} value={key}>Seed {s.seed} · fold {s.fold}{s.trial_id ? ` · ${s.trial_id}` : ""}</option>)}</select></label>}<label>HyperDense layer<select aria-label="Weight layer" value={layer?.name ?? ''} onChange={e => setLayerName(e.target.value)}>{!layers.length && <option value="">No HyperDense layers</option>}{layers.map(l => <option key={l.name} value={l.name}>{l.node_ids?.join(', ') || l.name} · {l.algebra}</option>)}</select></label><button className="secondary-button" disabled={!layer || Boolean(layer.error)} onClick={exportMatrix}><Download size={15} />Export displayed weights</button></section></SidePanel>
    <div className="weight-heading"><div className="chart-heading"><span className="eyebrow">Inside HyperDense</span><h3>From components to a real matrix</h3><p>{graph ? 'Initialized architecture preview · seed 0 · not trained weights.' : 'Saved trained weights · one evaluation replicate.'}</p></div>{onClose && <button className="weight-close" aria-label="Close weight inspector" title="Close weight inspector" onClick={onClose}><X size={19} /></button>}</div>
    {query.isPending ? <p role="status">Loading weight matrices…</p> : query.error ? <p role="alert">{query.error.message}</p> : !layer ? <p role="status">{snapshot ? 'This model has no HyperDense layers.' : 'No weight snapshot is available for this evaluation cell.'}</p> : layer.error ? <p role="alert">{layer.error}</p> : <>
      {layer.shape_fit && <p>Auto-fit on last axis (−1): {layer.shape_fit.input_width} features → {layer.shape_fit.padded_width} padded → {layer.shape_fit.output_width} output. {layer.shape_fit.units} hypercomplex units; {layer.shape_fit.padding} zeros added and {layer.shape_fit.crop} outputs cropped. The matrices below show the full padded map before cropping.</p>}
      <div className="weight-stats"><span><small>Algebra</small>{layer.algebra} · {layer.dimension} components</span><span><small>Real map dimensions</small>{layer.dimension * layer.in_features} × {layer.dimension * layer.out_features}</span><span><small>Weight parameters</small>{layer.weight_parameters.toLocaleString()} / {layer.dense_parameters.toLocaleString()} dense</span></div>
      <div className="weight-layout"><section><h4>{layer.dimension} component parameter matrices</h4><div className="weight-components">{layer.components.map((m, i) => <figure key={i}><Matrix values={m} bound={bound} title={`Component W${i}`} rowIndices={layer.input_indices} columnIndices={layer.output_indices} /><figcaption>W<sub>{i}</sub> · {i === 0 ? 'real' : `basis e${i}`}</figcaption></figure>)}</div></section><section><h4>Structured effective matrix · x @ W</h4><Matrix values={layer.effective} bound={bound} title="Effective real matrix" rowIndices={Array.from({ length: layer.dimension }, (_, c) => layer.input_indices.map(i => c * layer.in_features + i)).flat()} columnIndices={Array.from({ length: layer.dimension }, (_, c) => layer.output_indices.map(i => c * layer.out_features + i)).flat()} blockRows={layer.input_indices.length} blockColumns={layer.output_indices.length} /><p className="viz-caption">Rows: input components · columns: output components. White lines separate algebra blocks. Bias is excluded.</p></section></div>
      <div className="weight-color-scale"><span>−{bound.toPrecision(3)}</span><i /><span>+{bound.toPrecision(3)}</span></div><p className="viz-caption">Shared symmetric color scale; white is zero. Exact sampled weights, with no interpolation or averaging. Input unit indices: {layer.input_indices.join(', ')}. Output unit indices: {layer.output_indices.join(', ')}. At most 8 units per axis are displayed; dimensions above describe the full layer. {snapshot?.omitted_layers ? `${snapshot.omitted_layers} additional layers are omitted from this snapshot.` : ''}</p>
    </>}
  </article>
}
