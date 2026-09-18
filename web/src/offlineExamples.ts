import type { ArchitectureSpec, EvaluationSpec, Job, ResultSummary } from './types'

const architecture: ArchitectureSpec = {
  schema_version: 1,
  name: 'Illustrative architecture',
  input: { representation: 'levels', feature_order: [0, 1, 2, 3] },
  layers: [],
  head: { type: 'direct', zero_initialize: false },
}

const evaluation: EvaluationSpec = {
  protocol: 'chronological-v1',
  data_path: 'examples/copper_prices.csv',
  target_column: 'Copper',
  batch_size: 32,
  evaluation_batch_size: 256,
  learning_rate: 0.001,
  adam_beta1: 0.9,
  adam_beta2: 0.999,
  adam_epsilon: 1e-7,
  adam_amsgrad: false,
  loss: 'mse',
  shuffle: true,
  early_stopping_patience: 8,
  early_stopping_min_delta: 0.0001,
  restore_best_weights: true,
  device: 'cuda',
  preset: 'standard',
  cells: [{ window: 10, horizon: 1 }, { window: 24, horizon: 3 }],
  seeds: [7, 19, 42],
  epochs: 40,
  folds: [{ train_fraction: 0.7, validation_fraction: 0.15 }],
}

type Example = { name: string; id: string; parameters: number; mae: [number, number]; mse: [number, number] }

const examples: Example[] = [
  { name: 'DLinear', id: 'example-dlinear', parameters: 2_188, mae: [0.184, 0.247], mse: [0.059, 0.103] },
  { name: 'DLinear + Quaternion HyperDense', id: 'example-dlinear-quaternion', parameters: 3_036, mae: [0.166, 0.221], mse: [0.049, 0.084] },
  { name: 'PatchTST', id: 'example-patchtst', parameters: 82_945, mae: [0.158, 0.214], mse: [0.045, 0.079] },
  { name: 'PatchTST + Octonion HyperDense', id: 'example-patchtst-octonion', parameters: 86_273, mae: [0.149, 0.202], mse: [0.041, 0.071] },
]

const cells = [{ window: 10, horizon: 1 }, { window: 24, horizon: 3 }]
const persistence = [{ mae: 0.211, mse: 0.075 }, { mae: 0.282, mse: 0.128 }]

function makeJob(example: Example): Job {
  const runs = cells.flatMap((cell, cellIndex) => evaluation.seeds.map((seed, seedIndex) => {
    const factor = [0.985, 1, 1.015][seedIndex]
    return {
      ...cell,
      seed,
      fold: 0,
      split: 'illustrative-split-v1',
      mae: example.mae[cellIndex] * factor,
      mse: example.mse[cellIndex] * factor,
      persistence_mae: persistence[cellIndex].mae,
      persistence_mse: persistence[cellIndex].mse,
      parameters: example.parameters,
      train_seconds: (18 + cellIndex * 9 + example.parameters / 10_000) * factor,
      directional_accuracy: 0.55 + (persistence[cellIndex].mae - example.mae[cellIndex]) * 0.5,
      direction_baseline_accuracy: 0.5,
      return_correlation: 0.31 + (persistence[cellIndex].mae - example.mae[cellIndex]),
      bias: (seedIndex - 1) * 0.002,
      p95_abs_error: example.mae[cellIndex] * 2.4,
      large_move_mae_ratio: example.mae[cellIndex] / persistence[cellIndex].mae,
      sample_count: 240,
      large_move_count: 48,
    }
  }))
  const summary: ResultSummary[] = cells.map((cell, index) => ({
    ...cell,
    mae_mean: example.mae[index],
    mae_std: example.mae[index] * 0.015,
    mse_mean: example.mse[index],
    mse_std: example.mse[index] * 0.015,
    persistence_mae: persistence[index].mae,
    persistence_mse: persistence[index].mse,
    parameters: example.parameters,
    train_seconds: 18 + index * 9 + example.parameters / 10_000,
    epochs_median: 34,
    mae_ratio: example.mae[index] / persistence[index].mae,
    mse_ratio: example.mse[index] / persistence[index].mse,
  }))
  return {
    id: example.id,
    request: {
      phase: 'validation',
      candidate_hash: `illustrative-${example.id}`,
      architecture: { ...architecture, name: example.name },
      evaluation,
      execution: { target: 'local', gpu: null },
    },
    status: {
      id: example.id,
      state: 'complete',
      phase: 'validation',
      completed: runs.length,
      total: runs.length,
      created_at: '2026-01-15T09:00:00.000Z',
      updated_at: '2026-01-15T09:42:00.000Z',
      architecture_name: example.name,
      preset: 'standard',
      protocol: 'chronological-v1',
      execution_target: 'local',
      gpu: null,
    },
    summary,
    runs,
    per_lead: [],
  }
}

/** Read-only illustrative data for the static public playground. */
export const offlineExampleJobs: Job[] = examples.map(makeJob)
