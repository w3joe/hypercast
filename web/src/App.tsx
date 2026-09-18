import { useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Activity, BarChart3, Blocks, CircleStop, Clock3, ExternalLink, FlaskConical, PanelLeftClose, PanelLeftOpen, X } from 'lucide-react'
import CompareView from './CompareView'
import { WorkspacePanels, SidePanel } from './WorkspacePanels'
import GraphBuilder from './GraphBuilder'
import BrandLogo from './BrandLogo'
import { ErrorNotice } from './BuilderControls'
import { api, isOfflineDemo } from './api'
import type { Job } from './types'

type View = 'builder' | 'runs' | 'compare'

function formatMetric(value: number | null | undefined, digits = 4) {
  return value == null || Number.isNaN(value) ? '—' : value.toFixed(digits)
}

function statusClass(state: Job['status']['state']) {
  return `status-pill status-${state.replace('_', '-')}`
}

function RunsView({ jobs, legacyRuns, archive = false }: { jobs: Job[]; legacyRuns: Record<string, string | number | null>[]; archive?: boolean }) {
  const queryClient = useQueryClient()
  const [logJob, setLogJob] = useState<string | null>(null)
  const cancelMutation = useMutation({
    mutationFn: api.cancel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
  })
  const log = useQuery({ queryKey: ['job-log', logJob], queryFn: () => api.log(logJob!), enabled: Boolean(logJob), refetchInterval: 2000 })
  const reproduction = legacyRuns.some((run) => run.candidate != null)
  return (
    <section className="content-view">
      <div className="view-heading"><div><span className="eyebrow">Experiment lifecycle</span><h2>Runs</h2></div><span>{jobs.length} {archive ? 'archived results' : 'persisted jobs'}</span></div>
      {archive && <div className="offline-builder-note" role="note"><strong>Completed 120-result study</strong><span>Read-only results from 360 audited final-test fits across 15 TSLib models, eight variants, and three paired seeds on ETTh1.</span></div>}
      {!jobs.length && <div className="large-empty panel-surface"><Clock3 size={34} /><h3>No experiments yet</h3><p>Build an architecture and submit a validation run.</p></div>}
      <div className="jobs-list">
        {jobs.map((job) => {
          const percent = job.status.total ? Math.round(job.status.completed / job.status.total * 100) : 0
          const active = ['queued', 'starting', 'running'].includes(job.status.state)
          return <article className="job-card panel-surface" key={job.id}>
            <div className="job-top"><div><span className={statusClass(job.status.state)}>{job.status.state}</span><h3>{job.status.architecture_name}</h3><p>{job.status.phase.replace('_', ' ')} · {job.status.protocol ?? 'chronological-v1'} · {job.status.preset} · {job.status.execution_target ?? 'local'}{job.status.gpu ? ` · ${job.status.gpu}` : ''} · {job.id}</p></div><div className="job-actions">{job.status.modal_dashboard_url && <a className="text-button" href={job.status.modal_dashboard_url} target="_blank" rel="noreferrer">Modal <ExternalLink size={13} /></a>}{!archive && <button className="text-button" onClick={() => setLogJob(logJob === job.id ? null : job.id)}>View log</button>}{active && <button className="danger-button" disabled={cancelMutation.isPending} onClick={() => cancelMutation.mutate(job.id)}><CircleStop size={14} />Cancel</button>}</div></div>
            <div className="progress-track"><div style={{ width: `${percent}%` }} /></div>
            <div className="job-meta"><span>{job.status.completed} / {job.status.total || '—'} runs</span><span>{percent}%</span>{job.status.current && <span>w{job.status.current.window}/h{job.status.current.horizon} · seed {job.status.current.seed} · epoch {job.status.current.epoch}/{job.status.current.epochs}</span>}<span>Updated {new Date(job.status.updated_at).toLocaleString()}</span></div>
            {job.status.error && <div className="inline-error">{job.status.error}</div>}
            {job.status.execution_target === 'gcp' && <p className="muted-copy">{job.status.gpu_count ?? 1} × GPU · {active ? 'VM startup / training; logs and results arrive when finished.' : 'GCP VM run'}{job.status.gcp_cleanup && job.status.gcp_cleanup !== 'complete' ? ` · Cleanup needs attention: ${job.status.gcp_cleanup}` : ''}</p>}
            {job.summary.length > 0 && <div className="result-strip"><span>Mean MAE ratio <strong>{formatMetric(job.summary.reduce((sum, row) => sum + row.mae_ratio, 0) / job.summary.length, 3)}</strong></span><span>Mean MSE ratio <strong>{formatMetric(job.summary.reduce((sum, row) => sum + row.mse_ratio, 0) / job.summary.length, 3)}</strong></span><span>Parameters <strong>{Math.round(job.summary[0].parameters).toLocaleString()}</strong></span></div>}
            {logJob === job.id && <pre className="training-log">{log.data || 'Waiting for worker output…'}</pre>}
          </article>
        })}
      </div>
      {legacyRuns.length > 0 && <section className="legacy-runs panel-surface"><div className="panel-title"><Activity size={17} /><span>{reproduction ? 'Released-notebook reproduction results' : 'Main evaluation results'}</span></div><p className="muted-copy">Read-only results from the directory supplied with <code>--results</code>. {reproduction ? 'MAE values are normalized cross-validation scores, matching the paper reproduction runner.' : 'MAE and MSE are in original target units, matching the leakage-safe main runner.'}</p><div className="comparison-table"><table><thead><tr><th>Model</th><th>Window</th><th>Horizon</th><th>{reproduction ? 'Candidate' : 'Seed'}</th><th>{reproduction ? 'CV MAE (scaled)' : 'MAE'}</th><th>{reproduction ? 'MAE std' : 'MSE'}</th><th>Parameters</th></tr></thead><tbody>{legacyRuns.slice(-12).reverse().map((run, index) => <tr key={`${run.model}-${run.seed ?? run.candidate}-${index}`}><td><strong>{run.model}</strong></td><td>{run.window}</td><td>{run.horizon}</td><td>{reproduction ? run.candidate : run.seed}</td><td>{formatMetric(Number(run.mae), 5)}</td><td>{formatMetric(reproduction ? Number(run.mae_std) : Number(run.mse), 5)}</td><td>{Number(run.parameters).toLocaleString()}</td></tr>)}</tbody></table></div></section>}
    </section>
  )
}

function FinalTestDialog({ job, onComplete }: { job: Job; onComplete: () => void }) {
  const [acknowledged, setAcknowledged] = useState(false)
  const mutation = useMutation({ mutationFn: () => api.finalTest(job.id), onSuccess: onComplete })
  return <Dialog.Root><Dialog.Trigger asChild><button className="text-button" disabled={job.status.preset === 'quick'}><FlaskConical size={14} />Final test</button></Dialog.Trigger><Dialog.Portal><Dialog.Overlay className="dialog-overlay" /><Dialog.Content className="dialog-content"><Dialog.Title>Unlock the held-out test set?</Dialog.Title><Dialog.Description>This candidate can be tested only once. Use this after architecture selection is complete.</Dialog.Description><label className="checkbox-row acknowledgement"><input type="checkbox" checked={acknowledged} onChange={(event) => setAcknowledged(event.target.checked)} /><span>I understand this consumes the final evaluation for this candidate.</span></label><ErrorNotice error={mutation.error} /><div className="dialog-actions"><Dialog.Close asChild><button className="secondary-button">Cancel</button></Dialog.Close><button className="primary-button" disabled={!acknowledged || mutation.isPending} onClick={() => mutation.mutate()}>Start final test</button></div><Dialog.Close className="dialog-close" aria-label="Close"><X size={17} /></Dialog.Close></Dialog.Content></Dialog.Portal></Dialog.Root>
}

export default function App() {
  const [view, setView] = useState<View>('builder')
  const queryClient = useQueryClient()
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    try { return localStorage.getItem('hypercast.sidebar') !== 'closed' && window.innerWidth > 800 } catch { return window.innerWidth > 800 }
  })
  const [sidebar, setSidebar] = useState<HTMLElement | null>(null)
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    try { return Math.max(260, Math.min(520, Number(localStorage.getItem('hypercast.sidebarWidth')) || 320)) } catch { return 320 }
  })
  const resizeStart = useRef<{ x: number; width: number } | null>(null)
  const resizeSidebar = (width: number) => { const next = Math.max(260, Math.min(520, width)); setSidebarWidth(next); try { localStorage.setItem('hypercast.sidebarWidth', String(next)) } catch { /* Device preference only. */ } }
  const [toolbar, setToolbar] = useState<HTMLElement | null>(null)
  const [runId, setRunId] = useState('')
  const [runState, setRunState] = useState('all')
  const catalog = useQuery({ queryKey: ['catalog'], queryFn: api.catalog, staleTime: Infinity })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs, refetchInterval: isOfflineDemo ? false : 1500, staleTime: isOfflineDemo ? Infinity : 0 })
  const archives = useQuery({ queryKey: ['comparison-archives'], queryFn: api.comparisonArchives, enabled: !isOfflineDemo && view === 'compare', staleTime: 60000 })
  const legacyRuns = useQuery({ queryKey: ['legacy-runs'], queryFn: api.legacyRuns, enabled: !isOfflineDemo, staleTime: 5000 })
  const activeCount = useMemo(() => (jobs.data ?? []).filter((job) => ['queued', 'starting', 'running'].includes(job.status.state)).length, [jobs.data])
  if (catalog.isLoading) return <div className="app-loading"><BrandLogo className="loading-logo" /><p>Loading architecture catalog…</p></div>
  if (catalog.error || !catalog.data) return <div className="app-loading"><ErrorNotice error={catalog.error ?? new Error('Catalog unavailable')} /></div>
  return (
    <WorkspacePanels.Provider value={{ sidebar, toolbar, openSidebar: () => { setSidebarOpen(true); try { localStorage.setItem('hypercast.sidebar', 'open') } catch { /* Device preference only. */ } } }}><div className={`app-shell unified-workspace ${sidebarOpen ? 'sidebar-open' : 'sidebar-closed'} ${view === 'builder' ? 'canvas-shell' : ''}`} style={{ '--rail-width': `${sidebarWidth}px` } as CSSProperties}>
      <header className="app-header">
        <button className="sidebar-toggle" aria-label={sidebarOpen ? 'Hide left panel' : 'Show left panel'} aria-expanded={sidebarOpen} aria-controls="workspace-sidebar" onClick={() => {
          setSidebarOpen(!sidebarOpen)
          try { localStorage.setItem('hypercast.sidebar', sidebarOpen ? 'closed' : 'open') } catch { /* Preference storage is optional. */ }
        }}>{sidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}</button>
        <button className="brand" aria-label="Hypercast Architecture Playground" onClick={() => setView('builder')}><BrandLogo /><small>Architecture Playground</small></button>
        <nav aria-label="Primary navigation">
          <button className={view === 'builder' ? 'active' : ''} onClick={() => setView('builder')}><Blocks size={16} />Builder</button>
          <button className={view === 'runs' ? 'active' : ''} onClick={() => setView('runs')}><Clock3 size={16} />Runs{activeCount > 0 && <span className="nav-count">{activeCount}</span>}</button>
          <button className={view === 'compare' ? 'active' : ''} onClick={() => setView('compare')}><BarChart3 size={16} />Compare</button>
        </nav>
        {isOfflineDemo && <span className="offline-badge">Browser playground</span>}
        <div ref={setToolbar} className="topbar-tools">{view === 'runs' && <label className="topbar-single-picker"><span>Run</span><select aria-label="Select run" value={runId} onChange={e => setRunId(e.target.value)}><option value="">All runs</option>{(jobs.data ?? []).map(j => <option key={j.id} value={j.id}>{j.status.architecture_name} · {j.id.slice(-8)}</option>)}</select></label>}</div>
      </header>
      <div className="workspace-body">
      <aside id="workspace-sidebar" className="workspace-sidebar" aria-label="Workspace controls" hidden={!sidebarOpen}>
        <div ref={setSidebar} className="sidebar-content" />
      </aside>
      <div className="sidebar-resizer" hidden={!sidebarOpen} role="separator" aria-label="Resize left panel" aria-orientation="vertical" aria-valuemin={260} aria-valuemax={520} aria-valuenow={sidebarWidth} tabIndex={0}
        onPointerDown={e => { resizeStart.current = { x: e.clientX, width: sidebarWidth }; e.currentTarget.setPointerCapture(e.pointerId); e.preventDefault() }}
        onPointerMove={e => { if (resizeStart.current) resizeSidebar(resizeStart.current.width + e.clientX - resizeStart.current.x) }}
        onPointerUp={e => { resizeStart.current = null; if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId) }}
        onPointerCancel={() => { resizeStart.current = null }}
        onKeyDown={e => { if (['ArrowLeft', 'ArrowRight', 'Home'].includes(e.key)) { e.preventDefault(); resizeSidebar(e.key === 'Home' ? 320 : sidebarWidth + (e.key === 'ArrowRight' ? 16 : -16)) } }} />
      <main className="app-content">
        <div className="builder-view" hidden={view !== 'builder'}><GraphBuilder catalog={catalog.data} active={view === 'builder'} /></div>
        {view === 'runs' && <><SidePanel><section className="sidebar-section"><span className="eyebrow">Experiment history</span><h2>Runs</h2><label>Status<select aria-label="Filter run status" value={runState} onChange={e => setRunState(e.target.value)}>{['all', 'queued', 'starting', 'running', 'complete', 'failed', 'cancelled', 'interrupted'].map(s => <option key={s} value={s}>{s === 'all' ? 'All statuses' : s}</option>)}</select></label><p className="muted-copy">{isOfflineDemo ? 'Explore the completed read-only TSLib 15 study.' : 'Select a run from the top bar to inspect its progress and saved scores.'}</p></section></SidePanel><ErrorNotice error={jobs.error} /><RunsView archive={isOfflineDemo} jobs={(jobs.data ?? []).filter(j => (!runId || j.id === runId) && (runState === 'all' || j.status.state === runState))} legacyRuns={runId ? [] : legacyRuns.data ?? []} /></>}
        {view === 'compare' && <><ErrorNotice error={jobs.error} /><ErrorNotice error={archives.error} />{isOfflineDemo && <div className="offline-builder-note offline-comparison-note" role="note"><strong>Completed TSLib 15 comparison</strong><span>120 model/variant results from 360 audited final-test fits on ETTh1. Results are read-only and use three paired seeds.</span></div>}<CompareView jobs={[...jobs.data ?? [], ...archives.data ?? []]} offlineExamples={isOfflineDemo} finalAction={job => isOfflineDemo ? <span>Archived result</span> : <FinalTestDialog job={job} onComplete={() => queryClient.invalidateQueries({ queryKey: ['jobs'] })} />} /></>}
      </main>
      </div>
    </div></WorkspacePanels.Provider>
  )
}
