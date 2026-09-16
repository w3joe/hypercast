import { useState } from 'react'
import type { ReactNode } from 'react'
import { BarChart3, Download, SlidersHorizontal, Search } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import ForecastReplay from './ForecastReplay'
import ParameterTradeoff from './ParameterTradeoff'
import { SidePanel, TopbarTools } from './WorkspacePanels'
import { comparisonCells, comparisonData, comparisonGroups, comparisonMetrics, comparisonProfile, comparisonLeadProfile, defaultComparisonCell, meanRecorded, trainingRecipe, comparisonRunTag } from './comparison'
import type { ComparisonMetric } from './comparison'
import type { Job, ComparisonJob } from './types'

const format = (n: number | null, digits = 4) => n === null ? 'Unavailable' : n.toFixed(digits)
function RangePlot({ data, metric }: { data: ReturnType<typeof comparisonData>; metric: ComparisonMetric }) {
  const valid = data.filter(r => r.mean !== null).sort((a, b) => a.mean! - b.mean!), ratio = comparisonMetrics[metric].ratio
  const max = Math.max(ratio ? 1 : 0, ...valid.map(r => r.max!)) * 1.12 || 1
  const left = 220, width = 400, height = Math.max(170, valid.length * 54 + 58)
  const x = (v: number) => left + v / max * width
  return <svg className="replicate-plot" viewBox={`0 0 680 ${height}`} role="img" aria-label={`${comparisonMetrics[metric].label}: mean and observed replicate range. Exact values in the comparison table.`}>
    <title>Mean and observed range, not confidence intervals</title>
    {Array.from({ length: 5 }, (_, i) => max * i / 4).map(t => <g key={t}><line x1={x(t)} x2={x(t)} y1={12} y2={height - 34} stroke="#e7eceb" /><text x={x(t)} y={height - 12} textAnchor="middle" className="plot-tick">{t.toFixed(2)}</text></g>)}
    {ratio && <g><line x1={x(1)} x2={x(1)} y1={10} y2={height - 34} stroke="#6a767b" strokeDasharray="4 4" /><text x={x(1)} y={9} textAnchor="middle" className="plot-tick">Persistence</text></g>}
    {valid.map((r, i) => { const y = 36 + i * 54, label = shortName(r.name).replace(' · input ', ' · '); return <g key={r.id}>
      <title>{r.label}: mean {format(r.mean)}, range {format(r.min)} to {format(r.max)}, n={r.n}</title>
      <text x={0} y={y - 4} className="plot-name">{label.length > 28 ? label.slice(0, 16) + '…' + label.slice(-10) : label}</text>
      <text x={0} y={y + 13} className="plot-tick">{r.tag} · {r.n} replicate{r.n === 1 ? '' : 's'}</text>
      <line x1={x(r.min!)} x2={x(r.max!)} y1={y} y2={y} stroke={r.color} strokeWidth={3} />
      {[r.min!, r.max!].map((v, j) => <line key={j} x1={x(v)} x2={x(v)} y1={y - 5} y2={y + 5} stroke={r.color} />)}
      <circle cx={x(r.mean!)} cy={y} r={5.5} fill={r.color} stroke="white" strokeWidth={2} />
      <text x={674} y={y + 4} textAnchor="end" className="plot-value">{format(r.mean, 3)}</text>
    </g> })}
  </svg>
}

const shortName = (name: string) => name.replace(/ \(TSLib core, editable\)$/, '')
const counts = (values: number[]) => !values.length ? '0' : Math.min(...values) === Math.max(...values) ? String(values[0]) : `${Math.min(...values)}–${Math.max(...values)}`
const duration = (seconds: number | null) => seconds === null ? '—' : seconds >= 60 ? `${(seconds / 60).toFixed(1)} min` : `${seconds.toFixed(2)} s`
const settingsValue = (value: unknown) => value == null ? 'Not recorded' : String(value)
const diagnostics = [
  ['directional_accuracy', 'Direction accuracy', true], ['direction_baseline_accuracy', 'Direction baseline', true],
  ['return_correlation', 'Return correlation', false], ['bias', 'Bias', false],
  ['p95_abs_error', 'Mean replicate P95 error', false], ['large_move_mae_ratio', 'Large-move MAE / persistence', false],
] as const

function EvaluationProfile({ jobs, metric, paired, window, horizon, onSelect }: {
  jobs: ComparisonJob[]; metric: ComparisonMetric; paired: boolean; window: number; horizon: number; onSelect: (window: number, horizon: number) => void
}) {
  const settings = comparisonProfile(jobs, metric, paired), byLead = settings.length === 1 && horizon > 1
  const profile = byLead ? comparisonLeadProfile(jobs, window, horizon, metric, paired) : settings
  const points = profile.map(cell => ({ label: cell.label, ...Object.fromEntries(cell.scores.map((r, i) => [`run${i}`, r.mean])) }))
  const series = profile[0]?.scores ?? []
  return <article className="chart-panel panel-surface" aria-label="Evaluation settings chart">
    <div className="chart-heading"><span className="eyebrow">02 / {byLead ? 'Across forecast leads' : 'Across evaluation settings'}</span><h3>{byLead ? 'How error changes further ahead' : 'How scores change with the task'}</h3><p>{comparisonMetrics[metric].label} · {byLead ? `window ${window}, horizon ${horizon}` : 'each point is a window / horizon pair'}. Lower is better.</p></div>
    {profile.length > 1 ? <ResponsiveContainer width="100%" height={300}><LineChart data={points} margin={{ top: 20, right: 22, bottom: 26, left: 0 }}>
      <CartesianGrid vertical={false} stroke="#e7eceb" /><XAxis dataKey="label" tick={{ fontSize: 11 }} label={{ value: byLead ? 'Forecast lead (observations ahead)' : 'Context window / forecast horizon', position: 'insideBottom', offset: -16, fontSize: 12 }} />
      <YAxis tick={{ fontSize: 11 }} domain={[0, 'auto']} width={55} tickFormatter={v => Number(v).toLocaleString('en', { maximumFractionDigits: 2 })} />
      <Tooltip labelFormatter={v => byLead ? String(v) : `Window / horizon: ${v}`} formatter={v => typeof v === 'number' ? v.toFixed(4) : 'Unavailable'} />
      {comparisonMetrics[metric].ratio && <ReferenceLine y={1} ifOverflow="extendDomain" stroke="#7b8992" strokeDasharray="5 4" />}
      {!byLead && <ReferenceLine x={`${window} / ${horizon}`} stroke="#cad2e1" strokeDasharray="3 3" />}
      {series.map((r, i) => <Line key={r.id} dataKey={`run${i}`} name={r.label} stroke={r.color} strokeWidth={2.5} dot={{ r: 4, strokeWidth: 2, fill: 'white' }} activeDot={{ r: 6 }} connectNulls={false} isAnimationActive={false} />)}
    </LineChart></ResponsiveContainer> : <div className="comparison-chart-empty"><strong>One evaluation setting available</strong><p>These runs only include {window} / {horizon}. Additional window or horizon results will appear here as separate points.</p></div>}
    <p className="comparison-footnote">{byLead ? 'Means of included repeats at each lead; missing or nonshared scores stay absent. Exact coverage is below.' : 'Each point is a separate evaluation. Matched repeats can differ by cell; gaps mean missing or unmatched results. Lines connect discrete settings, without smoothing.'}</p>
    <details className="profile-values"><summary>Exact values by {byLead ? 'lead' : 'setting'}</summary><div className="comparison-table"><table><caption>Mean score and included replicate count.</caption><thead><tr><th>{byLead ? 'Forecast lead' : 'Window / horizon'}</th>{series.map(r => <th key={r.id}>{shortName(r.name)}<small>{r.tag}</small></th>)}</tr></thead><tbody>{profile.map(c => <tr key={c.label}><th>{byLead ? c.label : <button onClick={() => onSelect(c.window, c.horizon)}>{c.label}</button>}</th>{c.scores.map(r => <td key={r.id}>{format(r.mean)}<small>n = {r.n}</small></td>)}</tr>)}</tbody></table></div></details>
  </article>
}

export default function CompareView({ jobs, finalAction }: { jobs: ComparisonJob[]; finalAction: (job: Job) => ReactNode }) {
  const [chosenStage, setStage] = useState<string | null>(null), [chosenGroup, setChosenGroup] = useState(''), [chosenCell, setChosenCell] = useState('')
  const [metric, setMetric] = useState<ComparisonMetric>('mae_ratio'), [paired, setPaired] = useState(true)
  const [selection, setSelection] = useState<Record<string, string[]>>({}), [runSearch, setRunSearch] = useState('')
  const stage = chosenStage ?? (jobs.some(j => 'archive' in j) ? 'final_test' : 'validation')
  const [variant, setVariant] = useState('all')
  const [table, setTable] = useState<'scores' | 'diagnostics' | 'settings'>('scores'), [replayOpen, setReplayOpen] = useState(false)
  const complete = jobs.filter(j => j.status.state === 'complete' && (j.summary.length || j.runs.length))
  const forStage = (value: string) => complete.filter(j => value === 'quick' ? j.status.phase === 'validation' && j.status.preset === 'quick' : j.status.phase === value && j.status.preset !== 'quick')
  const candidates = forStage(stage), groups = comparisonGroups(candidates), group = groups.find(g => g.key === chosenGroup) ?? groups[0]
  const eligible = group?.jobs ?? [], cells = comparisonCells(eligible), cell = cells.find(c => c.key === chosenCell) ?? defaultComparisonCell(eligible)
  const archive = eligible.find(j => 'archive' in j)
  const archiveInfo = archive && 'archive' in archive ? archive.archive : null
  const variants = [...new Set(eligible.flatMap(j => 'archive' in j ? [j.archive.variant] : []))]
  const selected = group ? selection[group.key] ?? eligible.slice(0, 3).map(j => j.id) : []
  const visible = eligible.filter(j => selected.includes(j.id))
  const data = cell ? comparisonData(visible, cell.window, cell.horizon, metric, paired) : [], valid = data.filter(r => r.mean !== null)
  const best = [...valid].sort((a, b) => a.mean! - b.mean!)[0]
  const mixedRecipes = new Set(visible.map(trainingRecipe)).size > 1
  const excluded = data.some(r => r.n < r.available)
  const windows = [...new Set(cells.map(c => c.window))], horizons = cells.filter(c => c.window === cell?.window).map(c => c.horizon)
  const evaluation = eligible[0]?.request.evaluation
  const chooseCell = (window: number, horizon: number) => {
    const next = cells.find(c => c.window === window && c.horizon === horizon) ?? cells.find(c => c.window === window)
    if (next) { setChosenCell(next.key); setReplayOpen(false) }
  }
  const toggleRun = (id: string, checked: boolean) => group && setSelection(previous => ({ ...previous, [group.key]: checked ? [...new Set([...selected, id])].slice(0, 6) : selected.filter(x => x !== id) }))
  const rowName = (r: typeof data[number]) => <><strong><i className="comparison-swatch" style={{ background: r.color }} />{shortName(r.name)}</strong><small>{r.tag}</small></>
  const value = (r: typeof data[number], key: string, percent = false) => {
    const stat = meanRecorded(r.included, key)
    return <span title={`${stat.n} of ${r.n} included replicates have this measure`}>{stat.value === null ? '—' : percent ? `${(stat.value * 100).toFixed(1)}%` : format(stat.value)}</span>
  }
  const exportScores = () => {
    const csv = [['run_id', 'architecture', 'metric', 'mean', 'observed_min', 'observed_max', 'included', 'available', 'seeds', 'folds', 'parameters', 'mean_train_seconds', 'pairing', 'window', 'horizon', 'stage', 'dataset', ...diagnostics.map(([key]) => key)],
      ...data.map(r => [r.id, r.name, metric, r.mean, r.min, r.max, r.n, r.available, r.seeds, r.folds, r.parameters, meanRecorded(r.included, 'train_seconds').value, paired ? 'matched' : 'descriptive', cell?.window, cell?.horizon, stage, evaluation?.data_path, ...diagnostics.map(([key]) => meanRecorded(r.included, key).value)])]
      .map(row => row.map(v => `"${String(v ?? '').replaceAll('"', '""')}"`).join(',')).join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' })), a = document.createElement('a')
    a.href = url; a.download = 'comparison.csv'; a.click(); URL.revokeObjectURL(url)
  }
  return <section className="content-view research-comparison simplified-comparison">
    <SidePanel><section className="sidebar-section"><span className="eyebrow">Evaluation</span><h2>Compare runs</h2>
      <label>Evaluation stage<select aria-label="Comparison stage" value={stage} onChange={e => { setStage(e.target.value); setReplayOpen(false) }}><option value="validation">Validation ({forStage('validation').length})</option><option value="final_test">Final test ({forStage('final_test').length})</option><option value="quick">Quick checks ({forStage('quick').length})</option></select></label>
      <label>Dataset & split plan<select aria-label="Comparison evaluation group" value={group?.key ?? ''} onChange={e => { setChosenGroup(e.target.value); setChosenCell(''); setReplayOpen(false); setRunSearch(''); setVariant('all') }}>{!groups.length && <option value="">No completed evaluations</option>}{groups.map(g => { const e = g.jobs[0].request.evaluation; return <option key={g.key} value={g.key}>{'archive' in g.jobs[0] ? g.jobs[0].archive.collection : `${e.target_column} · ${e.preset}`} · {g.jobs.length} runs · {e.data_path.split(/[\\/]/).at(-1)}</option> })}</select></label>
      {evaluation && <p className="comparison-footnote">{evaluation.data_path}<br />{eligible.length} compatible run{eligible.length === 1 ? '' : 's'} of {candidates.length} in this stage. Different split plans stay separate.</p>}
      <div className="comparison-cell-controls"><label>Context window<select aria-label="Comparison window" disabled={!cell} value={cell?.window ?? ''} onChange={e => chooseCell(Number(e.target.value), cell?.horizon ?? 1)}>{windows.map(w => <option key={w} value={w}>{w}</option>)}</select></label><label>Forecast horizon<select aria-label="Comparison horizon" disabled={!cell} value={cell?.horizon ?? ''} onChange={e => chooseCell(cell!.window, Number(e.target.value))}>{horizons.map(h => <option key={h} value={h}>{h}</option>)}</select></label></div>
      <label>Score<select aria-label="Comparison metric" value={metric} onChange={e => setMetric(e.target.value as ComparisonMetric)}>{Object.entries(comparisonMetrics).map(([k, v]) => <option value={k} key={k}>{v.label}</option>)}</select></label>
      <label className="checkbox-row"><input type="checkbox" checked={paired} onChange={e => setPaired(e.target.checked)} /><span>Match seeds and folds</span></label>
      <p className="comparison-footnote">{paired ? 'Same seed, fold and recorded split sizes within each evaluation cell.' : 'All available repeats. Descriptive scores can use different seeds and folds.'}</p>
      <button className="secondary-button" disabled={!valid.length} onClick={exportScores}><Download size={15} />Export scores & diagnostics</button>
    </section><details className="sidebar-section comparison-method"><summary>How comparisons are checked</summary><p>Dots are arithmetic means; whiskers are observed minimum–maximum ranges, not confidence intervals. One repeat has no variability estimate.</p><p>Matching excludes duplicate or nonshared replicate keys. Missing scores stay unavailable. Diagnostic numbers use the same included repeats as the selected score; unavailable measures are omitted from each mean.</p><p>Dataset paths and recorded split metadata define groups. Matching paths alone cannot verify identical file contents. Repeated seeds and overlapping windows do not establish independent evidence or statistical significance.</p>{stage === 'final_test' && <p>A final-test label does not establish prospective confirmation.</p>}{archiveInfo && <><p>{archiveInfo.note}</p><p>{archiveInfo.split_description}</p></>}</details></SidePanel>
    <TopbarTools><details className="topbar-run-picker"><summary><SlidersHorizontal size={15} />Compare runs <span>{visible.length} / {eligible.length}</span></summary><div className="topbar-run-menu"><strong>Select up to 6 runs</strong><p>Selection stays the same when you change window or horizon.</p><label className="compare-run-search"><Search size={14} /><input aria-label="Find comparison runs" placeholder="Find a model or run…" value={runSearch} onChange={e => setRunSearch(e.target.value)} /></label>{!eligible.length && <p>No completed runs in this group.</p>}{variants.length > 0 && <label className="compare-variant-filter">Filter variant<select aria-label="Comparison variant filter" value={variants.includes(variant) ? variant : 'all'} onChange={e => setVariant(e.target.value)}><option value="all">All variants</option>{variants.map(v => <option key={v} value={v}>{v}</option>)}</select></label>}{eligible.filter(j => (!variants.includes(variant) || ('archive' in j && j.archive.variant === variant)) && `${j.status.architecture_name} ${j.id}`.toLowerCase().includes(runSearch.toLowerCase())).map(j => <label className="checkbox-row" key={j.id}><input type="checkbox" checked={selected.includes(j.id)} disabled={!selected.includes(j.id) && visible.length >= 6} onChange={e => toggleRun(j.id, e.target.checked)} /><span>{shortName(j.status.architecture_name)}<small>{comparisonRunTag(j)} · {j.request.evaluation.epochs} epochs{cell && !j.runs.some(r => r.window === cell.window && r.horizon === cell.horizon) ? ' · Missing this setting' : ''}</small></span></label>)}</div></details></TopbarTools>
    <div className="view-heading"><div><span className="eyebrow">{stage === 'final_test' ? 'Final evaluations' : stage === 'quick' ? 'Technical checks' : 'Development evidence'}</span><h2>Compare evaluations</h2><p className="muted-copy">{cell ? `${archiveInfo?.collection ?? `${evaluation?.target_column} · ${evaluation?.preset}`} · window ${cell.window} / horizon ${cell.horizon}` : 'Choose completed evaluations to begin.'}</p></div><span className="analysis-mode">{paired ? 'Matched repeats' : 'Descriptive comparison'}</span></div>
    {!candidates.length ? <div className="large-empty panel-surface"><BarChart3 size={32} /><h3>No completed {stage === 'final_test' ? 'final evaluations' : stage === 'quick' ? 'quick checks' : 'validation runs'}</h3><p>{stage === 'final_test' ? 'Current validation scores are available under Validation. Final-test results are kept separate.' : 'Select another evaluation stage to see its completed runs.'}</p></div> : !visible.length ? <div className="large-empty panel-surface"><h3>Select runs in the top bar</h3><p>Choose up to six runs from this dataset and split plan.</p></div> : <>
      {archiveInfo && <p className="comparison-archive-note"><strong>{new Set(eligible.flatMap(j => 'archive' in j ? [j.archive.backbone] : [])).size} backbones · {variants.length} variants · {evaluation?.seeds.length} seeds</strong> · {archiveInfo.target_scale}. {archiveInfo.task}. Archived results are read-only.</p>}
      <div className="comparison-stats" aria-label="Comparison summary"><article><span>Runs selected</span><strong>{visible.length}<small> / {eligible.length}</small></strong><p>Same dataset and split plan</p></article><article><span>{paired ? 'Matched' : 'Included'} repeats per run</span><strong>{counts(data.map(r => r.n))}</strong><p>For window {cell?.window} / horizon {cell?.horizon}</p></article><article><span>Lowest {comparisonMetrics[metric].label}</span><strong>{best ? format(best.mean, 3) : '—'}</strong><p title={best?.name}>{best ? shortName(best.name) : 'No comparable scores'}</p></article></div>
      <ul className="comparison-legend" aria-label="Selected runs">{data.map(r => <li key={r.id}><i style={{ background: r.color }} /><span>{shortName(r.name)}<small>{r.tag}</small></span><button aria-label={`Remove ${r.name} from comparison`} onClick={() => toggleRun(r.id, false)}>×</button></li>)}</ul>
      {(visible.length < 2 || stage === 'quick' || mixedRecipes || excluded) && <div className="comparison-checks" role="status">{visible.length < 2 && <p>{eligible.length < 2 ? 'Only one run uses this split plan. Choose another dataset & split plan to compare models.' : 'Select another run in the top bar for a model-to-model comparison.'}</p>}{stage === 'quick' && <p>Quick checks verify execution; their scores are preliminary.</p>}{mixedRecipes && <p>Training settings differ. See the Settings table before attributing differences to architecture.</p>}{excluded && <p>Some repeats were excluded by matching. Included / available counts are in the Scores table.</p>}</div>}
      {!valid.length && <div className="large-empty panel-surface"><h3>No comparable replicate scores</h3><p>These selected runs have no shared valid seed/fold scores for this setting. Check their coverage in Scores, choose another setting, or turn off matching for a descriptive comparison.</p></div>}
      {valid.length > 0 && cell && <div className="charts-grid comparison-primary-charts">
        <article className="chart-panel panel-surface" aria-label="Score comparison chart"><div className="chart-heading"><span className="eyebrow">01 / Score & repeatability</span><h3>Which models have lower error?</h3><p>{comparisonMetrics[metric].label} · {comparisonMetrics[metric].ratio ? 'below 1 beats persistence' : 'lower is better'}</p></div><div className="replicate-plot-scroll"><RangePlot data={data} metric={metric} /></div><p className="comparison-footnote">Dot = mean · whisker = observed range across repeats, not a confidence interval.</p></article>
        <EvaluationProfile jobs={visible} metric={metric} paired={paired} window={cell.window} horizon={cell.horizon} onSelect={chooseCell} />
        <ParameterTradeoff data={data} metric={metric} />
      </div>}
      <section className="panel-surface comparison-numbers" aria-label="Comparison numbers"><div className="comparison-numbers-heading"><h3>Results at a glance</h3><div role="tablist" aria-label="Comparison table">{(['scores', 'diagnostics', 'settings'] as const).map(key => <button role="tab" id={`compare-${key}`} aria-controls={`compare-table-${key}`} aria-selected={table === key} key={key} onClick={() => setTable(key)}>{key[0].toUpperCase() + key.slice(1)}</button>)}</div></div>
        <div role="tabpanel" id={`compare-table-${table}`} aria-labelledby={`compare-${table}`} className="comparison-table">
          {table === 'scores' && <table><caption>{comparisonMetrics[metric].label} · window {cell?.window} / horizon {cell?.horizon}. Training time is a mean per included repeat, not whole-job time.</caption><thead><tr><th>Model / run</th><th>Mean score</th><th>Observed range</th><th>Included / available</th><th>Seeds / folds</th><th>Parameters</th><th>Train / repeat</th>{stage === 'validation' && <th>Final evaluation</th>}</tr></thead><tbody>{data.map(r => <tr key={r.id}><th scope="row">{rowName(r)}</th><td>{format(r.mean)}</td><td>{r.n > 1 ? `${format(r.min)}–${format(r.max)}` : r.n ? 'One repeat' : '—'}</td><td>{r.n} / {r.available}</td><td>{r.seeds} / {r.folds}</td><td>{r.parameters?.toLocaleString() ?? 'Unavailable / varies'}</td><td>{duration(meanRecorded(r.included, 'train_seconds').value)}</td>{stage === 'validation' && <td>{(() => { const job = visible.find(j => j.id === r.id)!; return 'archive' in job ? 'Archived result' : finalAction(job) })()}</td>}</tr>)}</tbody></table>}
          {table === 'diagnostics' && <table><caption>Means over included repeats for this window and horizon. P95 is a mean of replicate percentiles, not a pooled percentile. Missing values are —; hover a value for coverage.</caption><thead><tr><th>Model / run</th><th>MAE</th><th>MSE</th>{diagnostics.map(([key, label]) => <th key={key}>{label}</th>)}<th>Mean samples</th><th>Mean large moves</th></tr></thead><tbody>{data.map(r => <tr key={r.id}><th scope="row">{rowName(r)}</th><td>{value(r, 'mae')}</td><td>{value(r, 'mse')}</td>{diagnostics.map(([key, , percent]) => <td key={key}>{value(r, key, percent)}</td>)}<td>{value(r, 'sample_count')}</td><td>{value(r, 'large_move_count')}</td></tr>)}</tbody></table>}
          {table === 'settings' && <table><caption>Recorded training and evaluation settings. Different windows and horizons are shown as separate evaluations in chart 02.</caption><thead><tr><th>Model / run</th><th>Epoch limit</th><th>Batch size</th><th>Learning rate</th><th>Loss</th><th>Optimizer (Adam)</th><th>Shuffle</th><th>Seeds</th><th>Train / validation splits</th><th>Early stopping</th><th>Restore best</th><th>Compute</th></tr></thead><tbody>{data.map(r => { const job = visible.find(j => j.id === r.id)!, e = job.request.evaluation; return <tr key={r.id}><th scope="row">{rowName(r)}</th><td>{settingsValue(e.epochs)}</td><td>{settingsValue(e.batch_size)}</td><td>{settingsValue(e.learning_rate)}</td><td>{settingsValue(e.loss)}</td><td>β₁ {settingsValue(e.adam_beta1)} · β₂ {settingsValue(e.adam_beta2)}<small>ε {settingsValue(e.adam_epsilon)} · AMSGrad {e.adam_amsgrad == null ? 'Not recorded' : e.adam_amsgrad ? 'Yes' : 'No'}</small></td><td>{e.shuffle == null ? 'Not recorded' : e.shuffle ? 'Yes' : 'No'}</td><td>{e.seeds?.join(', ') ?? 'Not recorded'}</td><td>{'archive' in job ? job.archive.split_description : e.folds?.length ? e.folds.map(f => `${(f.train_fraction * 100).toFixed(0)}% / ${(f.validation_fraction * 100).toFixed(0)}%`).join('; ') : 'Not recorded'}</td><td>{e.early_stopping_patience === undefined ? 'Not recorded' : e.early_stopping_patience === null ? 'Off' : `${e.early_stopping_patience} epochs · min Δ ${settingsValue(e.early_stopping_min_delta)}`}</td><td>{e.restore_best_weights == null ? 'Not recorded' : e.restore_best_weights ? 'Yes' : 'No'}</td><td>{job.request.execution?.target ?? e.device ?? 'Not recorded'}</td></tr> })}</tbody></table>}
        </div>
      </section>
      {cell && valid.length > 0 && <details className="comparison-forecast-disclosure" open={replayOpen} onToggle={e => { if (e.currentTarget.open !== replayOpen) setReplayOpen(e.currentTarget.open) }}><summary><span>Forecast vs actual</span><small>Optional · inspect one shared repeat</small></summary>{replayOpen && <ForecastReplay compact key={cell.key + visible.map(j => j.id).join(',')} jobs={visible.map(j => ({ ...j, runs: data.find(r => r.id === j.id)?.included ?? [] }))} window={cell.window} horizon={cell.horizon} />}</details>}
    </>}
  </section>
}
