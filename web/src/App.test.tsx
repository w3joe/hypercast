import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { prepareDenseSwap } from './layerSwap'

const paperPreset = {
  schema_version: 1,
  name: 'Paper Quaternion',
  input: { representation: 'levels', feature_order: [0, 1, 2, 3] },
  layers: [
    { id: 'hyper', type: 'hyper_dense', params: { units: 8, algebra: 'quaternion' } },
    { id: 'flatten', type: 'flatten', params: {} },
  ],
  head: { type: 'direct', zero_initialize: false },
  locked: true,
  preset_id: 'paper-quaternion',
}

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class { observe() {}; unobserve() {}; disconnect() {} })
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
    const path = String(input)
    if (path.endsWith('/architectures/edit')) {
      const payload = JSON.parse(String(options?.body))
      const response = await fetch('/api/v1/architectures/validate', { method: 'POST', body: JSON.stringify({ architecture: payload.architecture, ...payload.cells[0] }) })
      const validation = await response.json()
      try {
        const spec = prepareDenseSwap(payload.architecture, payload.edit.id, payload.edit.kind, validation.graph_nodes, payload.edit.params.algebra)
        return new Response(JSON.stringify({ spec, warnings: [] }), { status: 200, headers: { 'Content-Type': 'application/json' } })
      } catch (error) {
        return new Response(JSON.stringify({ detail: String(error) }), { status: 422, headers: { 'Content-Type': 'application/json' } })
      }
    }
    const body = path.includes('catalog') ? {
      schema_version: 1,
      categories: [{ name: 'Feature mixing', layers: [{ type: 'dense', label: 'Dense', defaults: { units: 32 } }, { type: 'hyper_dense', label: 'HyperDense', defaults: { units: 8, algebra: 'quaternion' } }, { type: 'dropout', label: 'Dropout', defaults: { p: .1 } }] }],
      input_representations: ['levels', 'centered', 'differences'],
      head_types: ['direct', 'persistence_residual', 'cumulative_residual'],
      algebras: ['complex', 'split_complex', 'tricomplex', 'quaternion', 'coquaternion', 'cl11', 'octonion'],
      algebra_dimensions: { complex: 2, split_complex: 2, tricomplex: 3, quaternion: 4, coquaternion: 4, cl11: 4, octonion: 8 },
      activations: ['relu', 'gelu', 'silu', 'tanh', 'linear'],
      presets: [paperPreset],
      method_collection: {
        sources: { survey: { title: 'Survey', venue: 'FCS', url: 'https://example.com/paper.pdf', pages: '13' } },
        methods: [{ id: 'quaternion', name: 'Quaternion', family: 'Hypercomplex', kind: 'neural', status: 'adaptation', preset_id: 'paper-quaternion', sources: [{ source_id: 'survey', page: 13 }], notes: 'Test preset.' }],
      },
      evaluation_presets: {
        quick: { cells: [{ window: 10, horizon: 1 }], seeds: [7], epochs: 3, folds: [] },
        standard: { cells: [{ window: 10, horizon: 1 }], seeds: [7], epochs: 50, folds: [] },
        robust: { cells: [{ window: 10, horizon: 1 }], seeds: [7], epochs: 50, folds: [] },
      },
      evaluation_defaults: {
        protocol: 'chronological-v1',
        data_path: 'data/raw/paper_data.xlsx',
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
        early_stopping_patience: null,
        early_stopping_min_delta: 0,
        restore_best_weights: false,
        device: 'cpu',
      },
    } : path.includes('compute') ? {
      local: { available: true },
      gcp: { available: true, message: 'Configured · test-zone', gpus: [
        { id: 'L4', label: 'L4', counts: [1, 2, 4, 8, 16] },
        { id: 'A100-40GB', label: 'A100 40 GB', counts: [1, 2, 4, 8] },
      ] },
      modal: {
        available: true,
        sdk_installed: true,
        authenticated: true,
        gpus: [
          { id: 'T4', label: 'T4', description: 'Economy' },
          { id: 'L4', label: 'L4', description: 'Recommended' },
        ],
        setup_command: 'modal setup',
      },
    } : path.includes('convert') ? executableGraph : path.includes('validate') ? {
      graph_nodes: graphInfo, warnings: [],
      valid: true,
      spec: paperPreset,
      input_shape: '[B, 10, 4]',
      output_shape: '[B, 1]',
      parameters: 225,
      receptive_field: 1,
      trace: [],
    } : []
    return new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
  }))
})


vi.mock('./ArchitectureDiagram', async () => {
  const React = await import('react')
  return { default: React.forwardRef(({ scene, onSelect, onToggle, onInsert, inserting }: any, ref: any) => {
    React.useImperativeHandle(ref, () => ({ fit: vi.fn(), zoom: vi.fn(), hold: vi.fn() }))
    return <div data-testid="diagram">{scene.blocks.map((b: any) => <div key={b.id}><button onClick={() => onSelect(b.id, false)}>Select {b.id}</button>{b.stage && <button onClick={() => onToggle(b.stage)}>Expand diagram {b.label}</button>}</div>)}{inserting && scene.routes.map((r: any) => <button key={r.id} onClick={() => onInsert(r.id)}>Insert {r.id}</button>)}</div>
  }) }
})
const executableGraph = { schema_version: 2, revision: 'test', name: 'Paper Quaternion', sources: { s0: paperPreset },
  nodes: [{ id: 'input', kind: 'source', params: {} }, { id: 'linear', kind: 'source', params: {}, group: 'core', source_ref: { source: 's0', node: 'linear' }, module_ref: 'weights' }, { id: 'output', kind: 'source', params: {} }],
  edges: [{ source: 'input', target: 'linear', port: 'args/0' }, { source: 'linear', target: 'output', port: 'args/0' }], groups: [{ id: 'core', label: 'Model core' }], output: 'output' }
const graphInfo = {
  input: { label: 'Input', category: 'placeholder', ports: [], shape: [2, 10, 4], settings: {} },
  linear: { label: 'Linear', category: 'call_module', ports: ['args/0'], shape: [2, 1], settings: { out_features: 1, bias: true }, source_path: 'core.projection' },
  output: { label: 'Output', category: 'output', ports: ['args/0'], shape: [2, 1], settings: {} },
}
function renderApp() { render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><App /></QueryClientProvider>) }

describe('graph-native playground', () => {
  it('docks controls on the left, preserves edits while hidden, and switches panel content with navigation', async () => {
    renderApp()
    fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Graph name' }), { target: { value: 'Preserved canvas' } }); fireEvent.blur(screen.getByRole('textbox', { name: 'Graph name' }))
    const aside = screen.getByRole('complementary', { name: 'Workspace controls' })
    expect(aside).toContainElement(screen.getByRole('combobox', { name: 'Load graph preset' }))
    fireEvent.click(screen.getByRole('button', { name: 'Hide left panel' }))
    expect(aside).not.toBeVisible()
    expect(screen.getByLabelText('Architecture canvas')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Show left panel' }))
    expect(screen.getByRole('textbox', { name: 'Graph name' })).toHaveValue('Preserved canvas')
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }))
    expect(aside).toContainElement(screen.getByLabelText('Comparison metric'))
    expect(screen.queryByRole('combobox', { name: 'Load graph preset' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Builder' }))
    expect(screen.getByRole('textbox', { name: 'Graph name' })).toHaveValue('Preserved canvas')
  })
  it('finds layers with the keyboard even when the sidebar is hidden', async () => {
    renderApp()
    await screen.findByRole('textbox', { name: 'Search layers' })
    fireEvent.click(screen.getByRole('tab', { name: 'Inspect' }))
    fireEvent.click(screen.getByRole('button', { name: 'Hide left panel' }))
    fireEvent.keyDown(window, { key: 'k', ctrlKey: true })
    const search = await screen.findByRole('textbox', { name: 'Search layers' })
    await waitFor(() => expect(search).toHaveFocus())
    expect(screen.getByRole('complementary', { name: 'Workspace controls' })).toBeVisible()
    fireEvent.change(search, { target: { value: 'Linear' } })
    expect(screen.getByRole('button', { name: 'Select linear' })).toBeVisible()
    fireEvent.change(search, { target: { value: 'not-a-layer' } })
    expect(screen.getByText(/No matching layers/)).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Clear layer search' }))
    expect(search).toHaveValue('')
  })
  it('inserts from the arrow plus at a collapsed stage boundary as one undoable action', async () => {
    const originalFetch = fetch
    const saved: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      if (String(input).endsWith('/architectures') && options?.body) { const payload = JSON.parse(String(options.body)); saved.push(payload); return new Response(JSON.stringify({ id: 'saved', spec: payload })) }
      return originalFetch(input, options)
    }))
    renderApp()
    fireEvent.click(await screen.findByRole('button', { name: 'Select group:core' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clone to edit' }))
    expect(screen.queryByRole('button', { name: 'Add after' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'Add' }))
    fireEvent.click(screen.getByRole('button', { name: `Insert ${JSON.stringify(['linear', 'output', 'args/0'])}` }))
    expect(screen.getByLabelText('Insertion destination')).toHaveValue(JSON.stringify(['linear', 'output', 'args/0']))
    fireEvent.click(screen.getByTitle('Add Dense'))
    await screen.findByRole('button', { name: /^Select dense-/ })
    fireEvent.click(screen.getByRole('tab', { name: 'Architecture' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(saved).toHaveLength(1))
    const added = saved[0].nodes.find((n: any) => n.kind === 'dense')
    expect(saved[0].edges).toEqual(expect.arrayContaining([{ source: 'linear', target: added.id, port: 'x' }, { source: added.id, target: 'output', port: 'args/0' }]))
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(saved).toHaveLength(2))
    expect(saved[1].edges).toEqual(executableGraph.edges)
    expect(saved[1].nodes).toEqual(executableGraph.nodes)
  })
  it('auto-fits inserted HyperDense and repairs an invalid manual draft with undo and redo', async () => {
    const originalFetch = fetch, saved: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      const path = String(input), payload = options?.body ? JSON.parse(String(options.body)) : null
      if (path.endsWith('/architectures') && payload) { saved.push(payload); return new Response(JSON.stringify({ id: 'saved', spec: payload })) }
      const node = payload?.architecture?.nodes?.find((n: any) => n.kind === 'hyper_dense')
      if (path.endsWith('/validate') && node) {
        if (node.params.shape_mode !== 'preserve') return new Response(JSON.stringify({ detail: 'Node add_1: width 32 must match 4' }), { status: 422 })
        return new Response(JSON.stringify({ valid: true, parameters: 20, warnings: [], graph_nodes: { ...graphInfo,
          [node.id]: { label: 'hyper_dense', category: 'custom', ports: ['x'], shape: [2, 10, 4], settings: node.params,
            shape_fit: { axis: -1, input_width: 4, padded_width: 4, output_width: 4, units: 1, padding: 0, crop: 0 } },
        } }))
      }
      return originalFetch(input, options)
    }))
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Add' }))
    fireEvent.change(screen.getByLabelText('Insertion destination'), { target: { value: JSON.stringify(['input', 'linear', 'args/0']) } })
    fireEvent.click(screen.getByTitle('Add HyperDense'))
    const autoFit = await screen.findByRole('checkbox', { name: 'Auto-fit connection' })
    expect(autoFit).toBeChecked()
    expect(screen.queryByLabelText('units')).not.toBeInTheDocument()
    await screen.findByText(/4 features → 4 padded → 4 output; 1 hypercomplex units/)
    fireEvent.click(autoFit)
    expect(await screen.findByLabelText('units')).toHaveValue('8')
    await screen.findByText(/Node add_1: width 32 must match 4/)
    expect(autoFit).toBeEnabled()
    fireEvent.click(autoFit)
    await screen.findByText(/4 features → 4 padded → 4 output; 1 hypercomplex units/)
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    expect(autoFit).not.toBeChecked()
    fireEvent.click(screen.getByRole('button', { name: 'Redo' }))
    expect(autoFit).toBeChecked()
    fireEvent.click(screen.getByRole('tab', { name: 'Architecture' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(saved).toHaveLength(1))
    expect(saved[0].nodes.find((n: any) => n.kind === 'hyper_dense').params.shape_mode).toBe('preserve')
  })
  it('requires an insertion destination by default and offers disconnected drafts explicitly', async () => {
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Add' }))
    fireEvent.change(screen.getByLabelText('Search layer palette'), { target: { value: 'Dense' } })
    expect(screen.getByTitle('Add Dense')).toBeDisabled()
    fireEvent.click(screen.getByText('Advanced', { selector: 'summary' }))
    fireEvent.click(screen.getByRole('button', { name: 'Create unconnected layer' }))
    fireEvent.click(screen.getByTitle('Add Dense'))
    expect(await screen.findByRole('button', { name: /^Select dense-/ })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Inspect' })).toHaveAttribute('aria-selected', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    await waitFor(() => expect(screen.queryByRole('button', { name: /^Select dense-/ })).not.toBeInTheDocument())
  })
  it('replaces the selected layer from its type selector and restores the exact graph with undo', async () => {
    const originalFetch = fetch, saved: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      if (String(input).endsWith('/architectures') && options?.body) {
        const payload = JSON.parse(String(options.body)); saved.push(payload)
        return new Response(JSON.stringify({ id: 'saved', spec: payload }))
      }
      return originalFetch(input, options)
    }))
    renderApp()
    fireEvent.click(await screen.findByRole('button', { name: 'Expand diagram Model core' }))
    fireEvent.click((await screen.findAllByRole('button', { name: 'Select linear' }))[0])
    expect(screen.getByLabelText('Layer type')).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Clone to edit' }))
    fireEvent.change(screen.getByLabelText('Layer type'), { target: { value: 'dropout' } })
    await waitFor(() => expect(screen.getByLabelText('Layer type')).toHaveValue('dropout'))
    expect(screen.queryByRole('button', { name: 'Replace selected' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'Architecture' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(saved).toHaveLength(1))
    const replacement = saved[0].nodes[1]
    expect(replacement).toMatchObject({ kind: 'dropout', group: 'core', params: { p: .1 } })
    expect(saved[0].edges).toEqual([
      { source: 'input', target: replacement.id, port: 'x' },
      { source: replacement.id, target: 'output', port: 'args/0' },
    ])
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(saved).toHaveLength(2))
    expect(saved[1].nodes).toEqual(executableGraph.nodes)
    expect(saved[1].edges).toEqual(executableGraph.edges)
    fireEvent.click(screen.getByRole('button', { name: 'Redo' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(saved).toHaveLength(3))
    expect(saved[2].nodes).toEqual(saved[0].nodes)
    expect(saved[2].edges).toEqual(saved[0].edges)
  })
  it('reconnects with the input picker while rejecting cycles and preserving locked models', async () => {
    renderApp()
    fireEvent.click(await screen.findByRole('button', { name: 'Expand diagram Model core' }))
    fireEvent.click((await screen.findAllByRole('button', { name: 'Select linear' }))[0])
    fireEvent.click(screen.getByText('Input connections'))
    const picker = screen.getByLabelText('Source for args/0')
    expect(picker).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Clone to edit' }))
    await waitFor(() => expect(picker).toBeEnabled())
    fireEvent.change(picker, { target: { value: 'output' } })
    expect(await screen.findByText(/This connection would create a cycle/)).toBeInTheDocument()
    expect(picker).toHaveValue('input')
    fireEvent.change(picker, { target: { value: '' } })
    expect(picker).toHaveValue('')
    fireEvent.change(picker, { target: { value: 'input' } })
    expect(picker).toHaveValue('input')
  })
  it('switches Dense to HyperDense directly and saves matched units and algebra', async () => {
    const originalFetch = fetch
    const saved: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      const path = String(input), payload = options?.body ? JSON.parse(String(options.body)) : null
      if (path.endsWith('/architectures') && payload) { saved.push(payload); return new Response(JSON.stringify({ id: 'saved', spec: payload })) }
      const response = await originalFetch(input, options)
      if (!path.endsWith('/validate')) return response
      const result = await response.json(), node = payload.architecture.nodes.find((n: any) => n.id === 'linear')
      result.graph_nodes.linear = { ...graphInfo.linear, shape: [2, 10, 32], label: node.kind === 'source' ? 'Linear' : node.kind, category: node.kind === 'source' ? 'call_module' : 'custom', settings: node.kind === 'source' ? { out_features: 32, bias: false } : node.params }
      return new Response(JSON.stringify(result))
    }))
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Expand diagram Model core' }))
    fireEvent.click((await screen.findAllByRole('button', { name: 'Select linear' }))[0])
    const selector = await screen.findByRole('combobox', { name: 'Layer type' })
    await waitFor(() => expect(selector).toHaveValue('dense'))
    await waitFor(() => expect(screen.getByRole('option', { name: 'HyperDense', exact: true })).toBeEnabled())
    fireEvent.change(selector, { target: { value: 'hyper_dense' } })
    await waitFor(() => expect(selector).toHaveValue('hyper_dense'))
    const algebra = screen.getByRole('combobox', { name: 'Algebra' })
    await waitFor(() => expect(algebra).toBeEnabled())
    fireEvent.change(algebra, { target: { value: 'cl11' } })
    await waitFor(() => expect(algebra).toHaveValue('cl11'))
    fireEvent.click(screen.getByRole('tab', { name: 'Architecture' })); fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(saved).toHaveLength(1))
    expect(saved[0].nodes.find((n: any) => n.id === 'linear').params).toEqual({ units: 8, bias: false, algebra: 'cl11' })
    expect(saved[0].edges).toEqual([{ source: 'input', target: 'linear', port: 'x' }, { source: 'linear', target: 'output', port: 'args/0' }])
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Inspect' }))
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Layer type' })).toHaveValue('dense'))
  })
  it('keeps the original layer when a direct swap is incompatible', async () => {
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Expand diagram Model core' }))
    fireEvent.click((await screen.findAllByRole('button', { name: 'Select linear' }))[0])
    const selector = await screen.findByRole('combobox', { name: 'Layer type' })
    await waitFor(() => expect(selector).toHaveValue('dense'))
    await waitFor(() => expect(screen.getByRole('option', { name: 'HyperDense', exact: true })).toBeEnabled())
    fireEvent.change(selector, { target: { value: 'hyper_dense' } })
    expect(await screen.findByText(/must both be divisible by 4/)).toBeVisible()
    expect(selector).toHaveValue('dense')
  })
  it('does not apply a swap that changes the layer shape in another evaluation cell', async () => {
    const originalFetch = fetch
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      const response = await originalFetch(input, options)
      if (!String(input).endsWith('/validate')) return response
      const payload = JSON.parse(String(options?.body)), result = await response.json()
      const isHyper = payload.architecture.nodes.find((n: any) => n.id === 'linear').kind === 'hyper_dense'
      result.graph_nodes.linear = { ...graphInfo.linear, shape: [2, 10, isHyper && payload.horizon === 3 ? 16 : 32], settings: { out_features: 32, bias: true } }
      return new Response(JSON.stringify(result))
    }))
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(screen.getByRole('tab', { name: 'Run' }))
    fireEvent.click(screen.getByText('Evaluation settings'))
    fireEvent.change(screen.getByLabelText('Cells (window/horizon)'), { target: { value: '10/1, 10/3' } })
    fireEvent.click(await screen.findByRole('button', { name: 'Expand diagram Model core' }))
    fireEvent.click((await screen.findAllByRole('button', { name: 'Select linear' }))[0])
    const selector = await screen.findByRole('combobox', { name: 'Layer type' })
    await waitFor(() => expect(selector).toHaveValue('dense'))
    await waitFor(() => expect(screen.getByRole('option', { name: 'HyperDense', exact: true })).toBeEnabled())
    fireEvent.change(selector, { target: { value: 'hyper_dense' } })
    expect(await screen.findByText(/Original layer kept/)).toBeVisible()
    expect(selector).toHaveValue('dense')
  })
  it('organizes the sidebar into architecture, inspection, add and run sections', async () => {
    renderApp(); await screen.findByRole('button', { name: 'Clone to edit' })
    expect(screen.getByRole('combobox', { name: 'Load graph preset' })).toHaveValue('paper-quaternion')
    expect(screen.getAllByRole('tab')).toHaveLength(4)
    fireEvent.click(screen.getByRole('tab', { name: 'Add' }))
    expect(screen.getByRole('textbox', { name: 'Search layer palette' })).toBeVisible()
    expect(screen.queryByRole('textbox', { name: 'Search layers' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'Architecture' }))
    expect(screen.getByRole('textbox', { name: 'Search layers' })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Export YAML' })).toBeVisible()
  })
  it('edits actual layers on the same canvas and saves the executable graph', async () => {
    const originalFetch = fetch
    const requests: any[] = []
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      if (options?.body) requests.push({ path: String(input), body: JSON.parse(String(options.body)) })
      if (String(input).endsWith('/architectures') && options?.body) return new Response(JSON.stringify({ id: 'saved', spec: executableGraph }))
      return originalFetch(input, options)
    }))
    renderApp()
    fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Expand diagram Model core' }))
    fireEvent.click((await screen.findAllByRole('button', { name: 'Select linear' }))[0])
    const width = await screen.findByRole('textbox', { name: 'out_features' })
    fireEvent.change(width, { target: { value: '64' } }); fireEvent.blur(width)
    await waitFor(() => expect(requests.some(r => r.path.endsWith('/validate') && r.body.architecture.nodes.some((n: any) => n.params.out_features === 64))).toBe(true))
    expect(screen.getAllByTestId('diagram')).toHaveLength(1)
    expect(requests.some(r => r.path.includes('internal-graph'))).toBe(false)
    fireEvent.click(screen.getByRole('tab', { name: 'Architecture' })); fireEvent.click(screen.getByRole('button', { name: 'Save graph' }))
    await waitFor(() => expect(requests.some(r => r.path.endsWith('/architectures') && r.body.schema_version === 2 && r.body.view)).toBe(true))
  })
  it('preserves edits across navigation and supports undo and library loading', async () => {
    renderApp(); fireEvent.click(await screen.findByRole('button', { name: 'Clone to edit' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Graph name' }), { target: { value: 'My experiment' } }); fireEvent.blur(screen.getByRole('textbox', { name: 'Graph name' }))
    fireEvent.click(screen.getByRole('button', { name: /^Runs/ })); fireEvent.click(screen.getByRole('button', { name: /^Builder$/ }))
    expect(screen.getByRole('textbox', { name: 'Graph name' })).toHaveValue('My experiment')
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }))
    expect(screen.getByRole('textbox', { name: 'Graph name' })).toHaveValue('Paper Quaternion experiment')
    fireEvent.click(screen.getByRole('button', { name: 'Method collection' }))
    fireEvent.click(screen.getByRole('button', { name: 'Load Quaternion' }))
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Graph name' })).toHaveValue('Paper Quaternion'))
  })
  it('validates every cell and blocks training when any one fails', async () => {
    const originalFetch = fetch
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, options?: RequestInit) => {
      if (String(input).endsWith('/validate') && options?.body && JSON.parse(String(options.body)).horizon === 3) return new Response(JSON.stringify({ detail: 'Output shape mismatch' }), { status: 422 })
      return originalFetch(input, options)
    }))
    renderApp(); await screen.findByRole('button', { name: 'Clone to edit' })
    fireEvent.click(screen.getByRole('tab', { name: 'Run' }))
    await waitFor(() => expect(screen.getByRole('button', { name: /run validation/i })).toBeEnabled())
    fireEvent.click(screen.getByRole('tab', { name: 'Run' }))
    fireEvent.click(screen.getByText('Evaluation settings'))
    fireEvent.change(screen.getByLabelText('Cells (window/horizon)'), { target: { value: '10/1, 10/3' } })
    await screen.findByText(/Output shape mismatch/)
    expect(screen.getByRole('button', { name: /run validation/i })).toBeDisabled()
  })
  it('retains local and Modal execution controls', async () => {
    renderApp()
    fireEvent.click(await screen.findByRole('tab', { name: 'Run' }))
    const method = await screen.findByRole('combobox', { name: /run method/i })
    expect(method).toHaveValue('local')
    fireEvent.change(method, { target: { value: 'modal' } })
    expect(await screen.findByRole('combobox', { name: /modal gpu/i })).toHaveValue('L4')
    expect(await screen.findByText(/usage charges may apply/i)).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: /run validation/i })).toBeEnabled())
  })
  it('offers valid GCP GPU counts and resets count when changing GPU', async () => {
    renderApp()
    fireEvent.click(await screen.findByRole('tab', { name: 'Run' }))
    fireEvent.change(await screen.findByRole('combobox', { name: /run method/i }), { target: { value: 'gcp' } })
    const gpu = await screen.findByRole('combobox', { name: 'GCP GPU' })
    const count = screen.getByRole('combobox', { name: 'GCP GPU count' })
    expect(gpu).toHaveValue('L4')
    expect(count.querySelectorAll('option')).toHaveLength(5)
    fireEvent.change(count, { target: { value: '16' } })
    expect(count).toHaveValue('16')
    expect(screen.getByRole('option', { name: '16 × GPU · 2 VMs' })).toBeInTheDocument()
    fireEvent.change(gpu, { target: { value: 'A100-40GB' } })
    expect(count.querySelectorAll('option')).toHaveLength(4)
    expect(count).toHaveValue('1')
    await waitFor(() => expect(screen.getByRole('button', { name: /run validation/i })).toBeEnabled())
    expect(screen.getByText(/Billed VM/)).toBeInTheDocument()
  })
})
