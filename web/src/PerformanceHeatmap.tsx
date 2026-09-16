import type { Job } from './types'
import { comparisonData, comparisonMetrics, comparisonProtocol } from './comparison'
import type { ComparisonMetric } from './comparison'

export function heatmapData(jobs: Job[], protocol: string, window: number, metric: ComparisonMetric) {
  const cohort = jobs.filter(j => comparisonProtocol(j) === protocol && j.summary.some(r => r.window === window))
  const horizons = [...new Set(cohort.flatMap(j => j.summary.filter(r => r.window === window).map(r => r.horizon)))].sort((a, b) => a - b)
  return { horizons, rows: cohort.map(job => ({ job, cells: horizons.map(h => {
    const mixedTrials = new Set(job.runs.filter(r => r.window === window && r.horizon === h).map(r => r.trial_id ?? '')).size > 1
    const cell = comparisonData([job], window, h, metric, false)[0]
    return { ...cell, mean: mixedTrials ? null : cell.mean, mixedTrials }
  }) })) }
}

export default function PerformanceHeatmap({ jobs, protocol, window, metric, onSelect }: {
  jobs: Job[]; protocol: string; window: number; metric: ComparisonMetric; onSelect: (job: Job, horizon: number) => void
}) {
  const { horizons, rows } = heatmapData(jobs, protocol, window, metric)
  const values = rows.flatMap(r => r.cells.flatMap(c => c.mean === null ? [] : [c.mean]))
  const min = Math.min(...values), max = Math.max(...values)
  const ratio = comparisonMetrics[metric].ratio
  const extent = Math.max(.01, ...values.map(v => Math.abs(v - 1)))
  const color = (value: number) => {
    if (ratio) return `hsl(${value <= 1 ? 191 : 25} 53% ${96 - Math.abs(value - 1) / extent * 58}%)`
    return `hsl(222 62% ${96 - (max === min ? .5 : (value - min) / (max - min)) * 59}%)`
  }
  return <article className="chart-panel panel-surface performance-heatmap" aria-label="Performance heatmap">
    <div className="chart-heading"><span className="eyebrow">Across forecast horizons</span><h3>Where each model performs best</h3><p>{comparisonMetrics[metric].label} · context {window}. Descriptive means of available replicates; counts can differ. Click a cell to inspect its runs.</p></div>
    <div className="heatmap-scroll"><table><thead><tr><th>Architecture / run</th>{horizons.map(h => <th key={h}>Horizon {h}</th>)}</tr></thead><tbody>{rows.map(({ job, cells }) => <tr key={job.id}><th scope="row">{job.status.architecture_name}<small>{job.id.slice(-8)}</small></th>{cells.map((c, i) => <td key={horizons[i]}>{c.mean === null ? <div className="heatmap-missing" title={c.mixedTrials ? "Different trial recipes are not averaged together" : "No valid replicate scores"}>—<small>{c.mixedTrials ? "Mixed trials" : "Missing"}</small></div> : <button style={{ background: color(c.mean), color: (ratio ? Math.abs(c.mean - 1) / extent : max === min ? .5 : (c.mean - min) / (max - min)) > .65 ? '#fff' : '#17313e' }} aria-label={`${job.status.architecture_name}, horizon ${horizons[i]}: ${c.mean.toFixed(4)}, ${c.n} replicates`} title={`Mean ${c.mean.toFixed(5)} · observed range ${c.min?.toFixed(5)}–${c.max?.toFixed(5)} · ${c.n} replicates`} onClick={() => onSelect(job, horizons[i])}>{c.mean.toFixed(3)}<small>n = {c.n}</small></button>}</td>)}</tr>)}</tbody></table></div>
    <p className="viz-caption">{ratio ? 'Cyan: better than persistence · white: ratio 1 · amber: worse. Color is shared across this heatmap.' : `Shared color scale: ${Number.isFinite(min) ? min.toFixed(3) : '—'} (light) to ${Number.isFinite(max) ? max.toFixed(3) : '—'} (dark). Lower is better.`} Rows retain individual runs and training recipes. Missing cells are never scored as zero.</p>
  </article>
}
