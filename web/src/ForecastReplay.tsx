import { useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Pause, Play, SkipBack } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { SidePanel } from './WorkspacePanels'
import { api } from './api'
import { comparisonColor, comparisonRunLabel } from './comparison'
import type { ComparisonJob } from './types'

export type ForecastRow = { origin_row: number; forecast_origin: string; target_date: string; actual: number; prediction: number; persistence: number }
export type ForecastResponse = { rows: ForecastRow[]; total: number; invalid: number; limit: number }

export function alignForecasts(responses: ForecastResponse[]) {
  const key = (r: ForecastRow) => JSON.stringify([r.origin_row, r.forecast_origin, r.target_date])
  const maps = responses.map(response => {
    const map = new Map<string, ForecastRow>(), duplicates = new Set<string>()
    response.rows.forEach(row => { const k = key(row); if (map.has(k)) duplicates.add(k); map.set(k, row) })
    duplicates.forEach(k => map.delete(k))
    return map
  })
  if (!maps.length) return []
  return [...maps[0].entries()].filter(([k, row]) => maps.every(m => {
    const other = m.get(k)
    return other && other.actual === row.actual && other.persistence === row.persistence
  })).sort(([, a], [, b]) => a.origin_row - b.origin_row).map(([k, row], index) => {
    const point: Record<string, string | number> = { index, date: row.target_date, origin: row.forecast_origin, actual: row.actual, persistence: row.persistence }
    maps.forEach((m, i) => { point[`prediction${i}`] = m.get(k)!.prediction; point[`error${i}`] = m.get(k)!.prediction - row.actual })
    return point
  })
}

export default function ForecastReplay({ jobs, window, horizon, compact = false }: { jobs: ComparisonJob[]; window: number; horizon: number; compact?: boolean }) {
  const keys = jobs.map(j => [...new Set(j.runs.filter(r => r.window === window && r.horizon === horizon && r.seed != null && r.fold != null).map(r => JSON.stringify([Number(r.seed), Number(r.fold), r.trial_id ?? ''])))])
  const shared = (keys[0] ?? []).filter(k => keys.every(list => list.includes(k)))
  const [chosen, setChosen] = useState(''), [chosenLead, setLead] = useState(1), [cursor, setCursor] = useState(0), [playing, setPlaying] = useState(false)
  const replicate = shared.includes(chosen) ? chosen : shared[0]
  const [seed, fold, trial] = replicate ? JSON.parse(replicate) as [number, number, string] : [0, 0, '']
  const lead = Math.min(horizon, chosenLead)
  const query = useQuery({ queryKey: ['forecast-replay', jobs.map(j => j.id), window, horizon, seed, fold, lead, trial], enabled: Boolean(replicate), retry: false,
    queryFn: () => Promise.all(jobs.map(j => ('archive' in j ? api.archivedForecasts(j.id, window, horizon, seed, fold, lead) : api.forecasts(j.id, window, horizon, seed, fold, lead, trial)))), staleTime: Infinity })
  const points = useMemo(() => alignForecasts(query.data ?? []), [query.data])
  const position = Math.min(cursor, Math.max(0, points.length - 1))
  useEffect(() => { setCursor(0); setPlaying(false) }, [query.data])
  useEffect(() => {
    if (!playing || points.length < 2) return
    const timer = globalThis.setInterval(() => setCursor(previous => Math.min(previous + 1, points.length - 1)), 180)
    return () => globalThis.clearInterval(timer)
  }, [playing, points.length])
  useEffect(() => { if (points.length && cursor >= points.length - 1) setPlaying(false) }, [cursor, points.length])
  const start = Math.max(0, Math.min(position - 40, points.length - 120))
  const displayed = points.slice(start, start + 120)
  const selected = points[position]
  return <article className="chart-panel panel-surface forecast-replay" aria-label="Forecast replay">
    <SidePanel><section className="sidebar-section"><h3>Forecast replay</h3><label>Replicate<select aria-label="Replay replicate" value={replicate ?? ''} onChange={e => setChosen(e.target.value)}>{!shared.length && <option value="">No shared replicates</option>}{shared.map(k => { const [s, f, t] = JSON.parse(k); return <option key={k} value={k}>Seed {s} · fold {f}{t ? ` · ${t}` : ""}</option> })}</select></label><label>Forecast lead<select aria-label="Replay lead" value={lead} onChange={e => setLead(Number(e.target.value))}>{Array.from({ length: horizon }, (_, i) => <option key={i} value={i + 1}>{i + 1} observations ahead</option>)}</select></label><div className="replay-controls"><button aria-label={playing ? 'Pause replay' : 'Play replay'} title={playing ? 'Pause' : 'Play'} disabled={points.length < 2} onClick={() => { if (position === points.length - 1) setCursor(0); setPlaying(!playing) }}>{playing ? <Pause size={18} /> : <Play size={18} />}</button><button aria-label="Restart replay" title="Restart replay" disabled={!points.length} onClick={() => { setCursor(0); setPlaying(false) }}><SkipBack size={18} /></button></div><label>Forecast cursor<input aria-label="Forecast cursor" type="range" min={0} max={Math.max(0, points.length - 1)} value={position} disabled={!points.length} onChange={e => { setPlaying(false); setCursor(Number(e.target.value)) }} /></label><p className="viz-caption">{points.length ? `${position + 1} / ${points.length} aligned forecasts` : 'No aligned forecasts'}</p></section></SidePanel>
    <div className="chart-heading"><span className="eyebrow">Forecast replay / lead {lead}</span><h3>{compact ? 'Forecast vs actual' : 'Follow the forecast through time'}</h3><p>{compact ? 'One seed, fold and forecast lead. Observed values, persistence and the selected models in target units.' : 'One seed and fold at a time. Both charts follow the same cursor; error is prediction − observed, in target units.'}</p></div>
    {!replicate ? <p role="status">Select runs with a shared seed and fold to replay forecasts.</p> : query.isPending ? <p role="status">Loading saved forecasts…</p> : query.error ? <p role="alert">{query.error.message}</p> : !points.length ? <p role="status">No forecasts share the same origin, target date, observed value and persistence value. Duplicate origins are excluded.</p> : <>
      <div className="replay-readout"><span><small>Forecast origin</small>{selected.origin}</span><span><small>Target date</small>{selected.date}</span><span><small>Observed</small>{Number(selected.actual).toFixed(4)}</span>{jobs.map((j, i) => <span key={j.id} style={{ color: comparisonColor(j) }}><small>{comparisonRunLabel(j)}</small>{Number(selected[`prediction${i}`]).toFixed(4)}</span>)}</div>
      {(compact ? [false] : [false, true]).map(error => <div key={String(error)} className="replay-chart"><h4>{error ? 'Signed error' : 'Observed and predicted'}</h4><ResponsiveContainer width="100%" height={error ? 175 : 290}><LineChart data={displayed} syncId="forecast-replay" margin={{ top: 10, right: 20, left: 5, bottom: 8 }}>
        <CartesianGrid vertical={false} stroke="#e7eceb" /><XAxis dataKey="index" type="number" domain={['dataMin', 'dataMax']} tickFormatter={v => String(points[Number(v)]?.date ?? '').slice(0, 10)} allowDecimals={false} minTickGap={55} /><YAxis domain={['auto', 'auto']} width={70} tickFormatter={v => Number(v).toLocaleString('en', { maximumFractionDigits: 2 })} /><Tooltip labelFormatter={v => String(points[Number(v)]?.date ?? '')} formatter={v => typeof v === 'number' ? v.toFixed(5) : String(v)} />
        {error ? <ReferenceLine y={0} stroke="#88949d" strokeDasharray="4 4" /> : <><Line dataKey="actual" name="Observed" stroke="#182b40" strokeWidth={2.5} dot={false} isAnimationActive={false} /><Line dataKey="persistence" name="Persistence" stroke="#99a5af" strokeDasharray="5 4" dot={false} isAnimationActive={false} /></>}
        {jobs.map((j, i) => <Line key={j.id} dataKey={`${error ? 'error' : 'prediction'}${i}`} name={comparisonRunLabel(j)} stroke={comparisonColor(j)} strokeWidth={1.8} dot={displayed.length === 1} connectNulls={false} isAnimationActive={false} />)}<ReferenceLine x={position} stroke="#526db5" strokeDasharray="3 3" />
      </LineChart></ResponsiveContainer></div>)}
      <p className="viz-caption">Seed {seed} · fold {fold} · {points.length} aligned origins. Displaying up to 120 consecutive aligned points; each run loads at most its last 2,000 saved forecasts for this lead. {query.data?.map((r, i) => `${comparisonRunLabel(jobs[i])}: ${r.total} valid, ${r.invalid} invalid records`).join(' · ')}. No smoothing or averaging; these overlapping forecasts are not independent samples.</p>
    </>}
  </article>
}
