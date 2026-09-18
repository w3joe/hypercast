import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, FlaskConical, Cloud, Play } from 'lucide-react'
import { api } from './api'
import type { ArchitectureSpec, GraphSpec, Catalog, EvaluationSpec, ExecutionSpec, Job, DemoLimits } from './types'

const dataColumns = ['Copper', 'FCX', 'CLP', 'SCCO']
const batchSizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
function clone<T>(value: T): T { return structuredClone(value) }

export function evaluationFromPreset(catalog: Catalog, presetName: 'quick' | 'standard' | 'robust'): EvaluationSpec {
  const preset = catalog.evaluation_presets[presetName]
  return {
    ...clone(catalog.evaluation_defaults),
    preset: presetName,
    cells: clone(preset.cells),
    seeds: [...preset.seeds],
    epochs: preset.epochs,
  }
}

function numberList(value: string): number[] {
  return value
    .split(',')
    .map((part) => Number(part.trim()))
    .filter(Number.isFinite)
}

function cellList(value: string) {
  return value
    .split(',')
    .map((part) => part.trim().split(/[x/]/).map(Number))
    .filter(([window, horizon]) => Number.isFinite(window) && Number.isFinite(horizon))
    .map(([window, horizon]) => ({ window, horizon }))
}

export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null
  return <div className="error-notice" role="alert">{error instanceof Error ? error.message : String(error)}</div>
}

export function EvaluationPanel({
  evaluation,
  catalog,
  onChange,
}: {
  evaluation: EvaluationSpec
  catalog: Catalog
  onChange: (next: EvaluationSpec) => void
}) {
  const loadPreset = (preset: EvaluationSpec['preset']) => onChange(evaluationFromPreset(catalog, preset))
  const batchSizeOptions = [...new Set([...batchSizes, evaluation.batch_size])].sort((a, b) => a - b)
  if (catalog.demo) return <section className="evaluation-panel panel-surface">
    <h4>Demo settings</h4><p className="demo-run-note">512 synthetic observations · one seed and split · early stopping · limited training.</p>
    <div className="evaluation-grid">
      <label>Window<input type="number" min={2} max={catalog.demo.window} value={evaluation.cells[0].window} onChange={e => onChange({ ...evaluation, cells: [{ ...evaluation.cells[0], window: Number(e.target.value) }] })} /></label>
      <label>Horizon<input type="number" min={1} max={catalog.demo.horizon} value={evaluation.cells[0].horizon} onChange={e => onChange({ ...evaluation, cells: [{ ...evaluation.cells[0], horizon: Number(e.target.value) }] })} /></label>
      <label>Epochs<input type="number" min={1} max={catalog.demo.epochs} value={evaluation.epochs} onChange={e => onChange({ ...evaluation, epochs: Number(e.target.value) })} /></label>
    </div><p className="demo-run-note">Up to {catalog.demo.parameters.toLocaleString()} parameters. Export your design for unrestricted training.</p>
  </section>
  return (
    <details className="evaluation-panel panel-surface">
      <summary><FlaskConical size={16} />Evaluation settings<span>{evaluation.preset}</span></summary>
      <div className="protocol-note">
        <Check size={15} />
        <span><strong>Main run protocol</strong> · chronological split · training-only scaling · Adam defaults from <code>configs/evaluation.yaml</code></span>
      </div>
      <div className="evaluation-grid">
        <label><span>Preset</span><select value={evaluation.preset} onChange={(event) => loadPreset(event.target.value as EvaluationSpec['preset'])}><option value="quick">Quick</option><option value="standard">Standard</option><option value="robust">Robust</option></select></label>
        <label><span>Target</span><select value={evaluation.target_column} onChange={(event) => onChange({ ...evaluation, target_column: event.target.value })}>{dataColumns.map((column) => <option key={column}>{column}</option>)}</select></label>
        <label><span>Cells (window/horizon)</span><input value={evaluation.cells.map((cell) => `${cell.window}/${cell.horizon}`).join(', ')} onChange={(event) => onChange({ ...evaluation, cells: cellList(event.target.value) })} /></label>
        <label><span>Seeds</span><input value={evaluation.seeds.join(', ')} onChange={(event) => onChange({ ...evaluation, seeds: numberList(event.target.value) })} /></label>
        <label><span>Epochs</span><input type="number" value={evaluation.epochs} onChange={(event) => onChange({ ...evaluation, epochs: Number(event.target.value) })} /></label>
        <label><span>Batch size</span><select value={evaluation.batch_size} onChange={(event) => onChange({ ...evaluation, batch_size: Number(event.target.value) })}>{batchSizeOptions.map((size) => <option key={size} value={size}>{size}</option>)}</select></label>
        <label><span>Learning rate</span><input type="number" step="0.0001" value={evaluation.learning_rate} onChange={(event) => onChange({ ...evaluation, learning_rate: Number(event.target.value) })} /></label>
        <label><span>Loss</span><select value={evaluation.loss} onChange={(event) => onChange({ ...evaluation, loss: event.target.value as EvaluationSpec['loss'] })}><option value="mse">MSE</option><option value="mae">MAE</option><option value="huber">Huber</option></select></label>
        <label><span>Patience</span><input type="number" value={evaluation.early_stopping_patience ?? ''} onChange={(event) => onChange({ ...evaluation, early_stopping_patience: event.target.value ? Number(event.target.value) : null })} /></label>
        <label><span>Device</span><select value={evaluation.device} onChange={(event) => onChange({ ...evaluation, device: event.target.value as EvaluationSpec['device'] })}><option value="cpu">CPU</option><option value="auto">Auto</option><option value="mps">Apple MPS</option><option value="cuda">CUDA</option></select></label>
        <label className="checkbox-row"><input type="checkbox" checked={evaluation.shuffle} onChange={(event) => onChange({ ...evaluation, shuffle: event.target.checked })} /><span>Shuffle training batches</span></label>
        <label className="checkbox-row"><input type="checkbox" checked={evaluation.restore_best_weights} disabled={evaluation.early_stopping_patience == null} onChange={(event) => onChange({ ...evaluation, restore_best_weights: event.target.checked })} /><span>Restore best weights</span></label>
      </div>
    </details>
  )
}

export function RunControls({
  architecture,
  evaluation,
  disabled,
  onQueued,
  demo,
  offline = false,
}: {
  architecture: ArchitectureSpec | GraphSpec
  evaluation: EvaluationSpec
  disabled: boolean
  onQueued: (job: Job) => void
  demo?: DemoLimits
  offline?: boolean
}) {
  const queryClient = useQueryClient()
  const [target, setTarget] = useState<ExecutionSpec['target']>('local')
  const [gpu, setGpu] = useState('L4')
  const [gcpGpu, setGcpGpu] = useState('L4')
  const [gpuCount, setGpuCount] = useState(1)
  const compute = useQuery({
    queryKey: ['compute-capabilities'],
    queryFn: api.compute,
    enabled: !offline && !demo && target !== 'local',
    staleTime: 10_000,
  })
  const mutation = useMutation({
    mutationFn: (execution: ExecutionSpec) => api.submit(architecture, evaluation, execution),
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['demo-session'] })
      onQueued(job)
    },
  })
  const session = useQuery({ queryKey: ['demo-session'], queryFn: api.demoSession, enabled: Boolean(demo) })
  const modal = compute.data?.modal
  const gcp = compute.data?.gcp
  const gcpGpus = gcp?.gpus ?? [
    { id: 'L4', label: 'L4', counts: [1, 2, 4, 8, 16] },
    { id: 'A100-40GB', label: 'A100 40 GB', counts: [1, 2, 4, 8] },
  ]
  const execution: ExecutionSpec = target === 'modal'
    ? { target: 'modal', gpu }
    : target === 'gcp' ? { target: 'gcp', gpu: gcpGpu, gpu_count: gpuCount }
    : { target: 'local', gpu: null }
  const canSubmit = target === 'local' || Boolean(target === 'gcp' ? gcp?.available : modal?.available)

  if (offline) return <div className="run-control-stack offline-run-note">
    <p>Training is unavailable in this browser playground. Export the YAML and open it in a connected Hypercast installation to validate shapes, inspect weights, and run experiments.</p>
    <button className="primary-button" disabled><Play size={15} />Compute not connected</button>
  </div>

  if (demo) return <div className="run-control-stack">
    <p className="demo-run-note">L4 demo · up to {demo.run_seconds / 60} minutes · {session.data?.remaining_runs ?? '…'} runs left today (UTC). Failed and cancelled experiments count toward your allowance.</p>
    {!session.data?.demo_available && session.data && <p role="status">The shared demo allowance is exhausted for this month.</p>}
    <button className="primary-button" disabled={disabled || mutation.isPending || !session.data?.authenticated || !session.data.demo_available || session.data.remaining_runs < 1}
      onClick={() => mutation.mutate({ target: 'modal', gpu: 'L4' })}><Play size={15} />{mutation.isPending ? 'Queueing…' : 'Run demo experiment'}</button>
    <ErrorNotice error={session.error || mutation.error} />
  </div>

  return (
    <div className="run-control-stack">
      <div className="run-controls" aria-label="Validation execution">
        <label className="run-select">
          <span>Run on</span>
          <select
            aria-label="Run method"
            value={target}
            onChange={(event) => { setTarget(event.target.value as ExecutionSpec['target']); mutation.reset() }}
          >
            <option value="local">Local · {evaluation.device}</option>
            <option value="modal">Modal · GPU</option>
            <option value="gcp">GCP · GPU VM</option>
          </select>
        </label>
        {target === 'modal' && <label className="run-select gpu-select">
          <span>GPU</span>
          <select aria-label="Modal GPU" value={gpu} onChange={(event) => setGpu(event.target.value)}>
            {(modal?.gpus ?? [{ id: 'L4', label: 'L4', description: 'Recommended' }]).map((option) => (
              <option value={option.id} key={option.id}>{option.label} · {option.description}</option>
            ))}
          </select>
        </label>}
        {target === 'gcp' && <>
          <label className="run-select gpu-select"><span>GPU</span>
            <select aria-label="GCP GPU" value={gcpGpu} onChange={(event) => {
              setGcpGpu(event.target.value); setGpuCount(1); mutation.reset()
            }}>
              {gcpGpus.map(option => <option key={option.id} value={option.id}>{option.label}</option>)}
            </select>
          </label>
          <label className="run-select"><span>GPUs</span>
            <select aria-label="GCP GPU count" value={gpuCount} onChange={(event) => setGpuCount(Number(event.target.value))}>
              {gcpGpus.find(option => option.id === gcpGpu)?.counts.map(count => <option key={count} value={count}>{count === 16 ? '16 × GPU · 2 VMs' : `${count} × GPU`}</option>)}
            </select>
          </label>
        </>}
        <button
          className="primary-button"
          disabled={disabled || !canSubmit || compute.isLoading || mutation.isPending}
          onClick={() => mutation.mutate(execution)}
        >
          {target !== 'local' ? <Cloud size={15} /> : <Play size={15} />}
          {mutation.isPending ? 'Queueing…' : 'Run validation'}
        </button>
      </div>
      {target === 'modal' && <div className={`run-status ${modal?.available ? 'ready' : ''}`} role="status">
        {compute.isLoading
          ? 'Checking Modal…'
          : modal?.available
            ? `${gpu} ready on Modal · usage charges may apply`
            : modal
              ? <>{modal.sdk_installed ? 'Authenticate Modal:' : 'Install Modal support:'} <code>{modal.setup_command}</code></>
              : 'Modal availability could not be checked.'}
      </div>}
      {target === 'gcp' && <div className={`run-status ${gcp?.available ? 'ready' : ''}`} role="status">
        {compute.isLoading ? 'Checking GCP…' : gcp?.message ?? 'GCP availability could not be checked.'}
        {gcp?.available && <span> · Billed VM{gpuCount === 16 ? 's: 2 × 8 L4 GPUs' : ''}; dataset and code uploaded to your bucket. {gpuCount > 1 ? 'Trials run in parallel; one model per GPU. Extra GPUs are idle if there are fewer cell/seed trials.' : 'One GPU per trial.'}</span>}
      </div>}
      <ErrorNotice error={(target !== 'local' && compute.error) || mutation.error} />
    </div>
  )
}
