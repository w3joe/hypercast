import { useId, useState } from 'react'
import { comparisonMetrics, paretoIds } from './comparison'
import type { comparisonData, ComparisonMetric } from './comparison'

type Scores = ReturnType<typeof comparisonData>
type MarkerKind = 'native' | 'real' | 'hyperdense' | 'lowrank' | 'unknown'
const markerKind = (variant: string | null): MarkerKind => variant === 'native' || variant === 'real' ? variant
  : ['complex', 'quaternion', 'octonion'].includes(variant ?? '') ? 'hyperdense'
    : /^lowrank\d+$/.test(variant ?? '') ? 'lowrank' : 'unknown'
const markerNames: Record<MarkerKind, string> = { native: 'Native', real: 'Real', hyperdense: 'HyperDense', lowrank: 'Low rank', unknown: 'Other / unspecified' }
const compact = (v: number) => Intl.NumberFormat('en', { notation: 'compact', maximumSignificantDigits: 3 }).format(v)
const exact = (v: number) => v.toLocaleString('en', { maximumSignificantDigits: 16 })

function Marker({ kind, color }: { kind: MarkerKind; color: string }) {
  const style = { fill: color, stroke: 'white', strokeWidth: 2 }
  if (kind === 'real') return <rect x={-6} y={-6} width={12} height={12} rx={1} {...style} />
  if (kind === 'hyperdense') return <path d="M 0,-8 L 8,0 L 0,8 L -8,0 Z" {...style} />
  if (kind === 'lowrank') return <path d="M 0,-8 L 8,6 L -8,6 Z" {...style} />
  return <circle r={6.5} {...style} />
}

export default function ParameterTradeoff({ data, metric }: { data: Scores; metric: ComparisonMetric }) {
  const titleId = useId(), [activeId, setActiveId] = useState<string | null>(null)
  const points = data.filter((r): r is Scores[number] & { mean: number; parameters: number; min: number; max: number } =>
    r.mean !== null && r.parameters !== null && r.parameters >= 0 && r.min !== null && r.max !== null)
  const frontierIds = paretoIds(points)
  const frontier = [...points].filter(r => frontierIds.has(r.id)).sort((a, b) => a.parameters - b.parameters || a.mean - b.mean)
    .filter((r, i, all) => i === 0 || r.parameters !== all[i - 1].parameters || r.mean !== all[i - 1].mean)
  const log = points.length > 0 && points.every(r => r.parameters > 0)
  const minimum = Math.min(...points.map(r => r.parameters)), maximum = Math.max(...points.map(r => r.parameters))
  const low = log ? Math.log10(minimum) - .18 : 0, high = log ? Math.log10(maximum) + .18 : Math.max(1, maximum * 1.12)
  const yMax = Math.max(comparisonMetrics[metric].ratio ? 1 : 0, ...points.map(r => r.max)) * 1.12 || 1
  const left = 74, right = 824, top = 24, bottom = 278
  const x = (value: number) => left + ((log ? Math.log10(value) : value) - low) / (high - low) * (right - left)
  const y = (value: number) => bottom - value / yMax * (bottom - top)
  const active = points.find(r => r.id === activeId)
  const kinds = [...new Set(points.map(r => markerKind(r.variant)))]
  const describe = (r: typeof points[number]) => `${r.label}. ${r.parameters.toLocaleString('en')} total parameters. Mean ${comparisonMetrics[metric].label}: ${exact(r.mean)}. ${r.n > 1 ? `Observed range ${exact(r.min)} to ${exact(r.max)} across ${r.n} repeats` : 'One repeat; variability unavailable'}. ${frontierIds.has(r.id) ? 'On the selected-run Pareto frontier.' : 'Outside the selected-run Pareto frontier.'}`
  return <article className="chart-panel panel-surface parameter-tradeoff" aria-labelledby={titleId}>
    <div className="chart-heading"><span className="eyebrow">03 / Error & model size</span><h3 id={titleId}>Does a larger model earn its size?</h3><p>{comparisonMetrics[metric].label} vs total model parameters · bottom-left is better.</p></div>
    {points.length ? <>
      <div className="parameter-plot-scroll"><svg className="parameter-plot" viewBox="0 0 860 340" role="group" aria-label="Error versus parameter count. Focus or select a point for exact values.">
        {Array.from({ length: 5 }, (_, i) => yMax * i / 4).map(t => <g key={`y${t}`}><line x1={left} x2={right} y1={y(t)} y2={y(t)} stroke="#e7eceb" /><text x={left - 12} y={y(t) + 4} textAnchor="end" className="plot-tick">{Number(t.toPrecision(3))}</text></g>)}
        {Array.from({ length: 5 }, (_, i) => low + (high - low) * (i + .25) / 4.5).map(t => { const value = log ? 10 ** t : t; return <g key={`x${t}`}><line x1={x(value)} x2={x(value)} y1={top} y2={bottom} stroke="#edf1f0" strokeDasharray="2 4" /><text x={x(value)} y={bottom + 24} textAnchor="middle" className="plot-tick">{compact(value)}</text></g> })}
        <text x={(left + right) / 2} y={332} textAnchor="middle" className="parameter-axis-label">Total model parameters · {log ? 'log scale' : 'linear scale (includes zero)'}</text>
        <text transform="translate(17 150) rotate(-90)" textAnchor="middle" className="parameter-axis-label">{comparisonMetrics[metric].label}</text>
        {comparisonMetrics[metric].ratio && <g><line x1={left} x2={right} y1={y(1)} y2={y(1)} stroke="#869795" strokeDasharray="5 4" /><text x={right} y={y(1) - 7} textAnchor="end" className="plot-tick">Persistence = 1</text></g>}
        {frontier.length > 1 && <polyline className="parameter-frontier" points={frontier.map(r => `${x(r.parameters)},${y(r.mean)}`).join(' ')} fill="none" stroke="#7c9391" strokeWidth={1.5} strokeDasharray="5 5"><title>Pareto frontier among selected runs, based on observed means</title></polyline>}
        {active && active.n > 1 && <g stroke={active.color} strokeWidth={1.5} opacity={.7} pointerEvents="none"><line x1={x(active.parameters)} x2={x(active.parameters)} y1={y(active.min)} y2={y(active.max)} />{[active.min, active.max].map((v, i) => <line key={i} x1={x(active.parameters) - 5} x2={x(active.parameters) + 5} y1={y(v)} y2={y(v)} />)}</g>}
        {points.map(r => <g key={r.id} transform={`translate(${x(r.parameters)} ${y(r.mean)})`} className="parameter-point" role="button" tabIndex={0} aria-label={describe(r)} aria-pressed={active?.id === r.id}
          onMouseEnter={() => setActiveId(r.id)} onFocus={() => setActiveId(r.id)} onClick={() => setActiveId(r.id)} onKeyDown={e => { if (e.key === 'Escape') setActiveId(null); if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setActiveId(r.id) } }}>
          <title>{describe(r)}</title><circle className="parameter-point-halo" r={12} fill={r.color} fillOpacity={active?.id === r.id ? .16 : 0} stroke={active?.id === r.id ? r.color : 'transparent'} strokeOpacity={.35} />
          <Marker kind={markerKind(r.variant)} color={r.color} />
        </g>)}
      </svg></div>
      <div className="parameter-legend" aria-label="Parameter chart legend">{kinds.map(kind => <span key={kind}><svg width="20" height="20" viewBox="-10 -10 20 20" aria-hidden="true"><Marker kind={kind} color="#526e70" /></svg>{markerNames[kind]}</span>)}<span><i />Selected-run Pareto frontier</span><span>Colour = backbone / model</span></div>
      <div className="parameter-readout" role="status" aria-live="polite">{active ? <><strong style={{ color: active.color }}>{active.label}</strong><span><b>{active.parameters.toLocaleString('en')}</b> parameters</span><span>Mean <b>{exact(active.mean)}</b></span><span>{active.n > 1 ? `Range ${exact(active.min)}–${exact(active.max)} · ${active.n} repeats` : 'One repeat · variability unavailable'}</span><span>{frontierIds.has(active.id) ? 'On selected-run frontier' : 'Outside selected-run frontier'}</span></> : <span>Hover, tap, or keyboard-focus a point for exact parameters, mean error, and observed repeat range.</span>}</div>
    </> : <div className="comparison-chart-empty"><strong>No comparable parameter counts</strong><p>This chart needs a recorded, consistent total parameter count and a valid score for the same included repeats.</p></div>}
    <p className="comparison-footnote">Same selected runs, evaluation cell, and included repeats as Scores. {data.length - points.length > 0 && `${data.length - points.length} selected run(s) omitted: missing score or missing / varying parameter count. `}The frontier joins models for which no selected model has both no more parameters and no higher mean error, with at least one strictly lower. It is descriptive, not evidence of statistical significance. Counts measure model size, not runtime.{points.length > 0 && ' Coincident points remain individually keyboard-accessible; exact values are also in Scores.'}</p>
  </article>
}
