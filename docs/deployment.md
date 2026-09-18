# Browser-only Hypercast deployment

The `deployment` branch publishes a static architecture playground on Vercel.
It has no API rewrite, server function, Modal workspace, account system, email
gate, cookies, or server-side storage.

Visitors can load all 25 presets, inspect their reference shapes and parameter
counts, clone and edit graphs, insert or move layers, configure HyperDense
auto-fit, undo and redo changes, import YAML/JSON, and export YAML. **Save graph**
stores up to five recent designs in that browser's local storage. Hypercast does
not receive those designs.

The checked-in `web/public/offline-manifest.json` contains the layer catalog,
method collection, converted preset graphs, and reference validation metadata
for window 10 / horizon 1. After an edit, the interface labels dimensions as
unvalidated. Training, final shape validation, run history, comparisons, and
weight inspection require a connected Hypercast installation.

## Build

Regenerate the static catalog after architecture or preset changes:

```sh
.venv/bin/python scripts/generate_offline_manifest.py
```

Build the normal connected frontend:

```sh
npm --prefix web run build
```

Build the static Vercel output:

```sh
node scripts/build-offline.mjs
```

The static build sets `VITE_OFFLINE_DEMO=true` only for that build. It emits
`.vercel/output/static` and a Build Output API configuration with SPA fallback
and security headers. It defines no `/api` route or external origin.

## Vercel project

Link the repository root to the Vercel project `hypercast` (or
`hypercast-w3joe` if that name is unavailable), select **Other** as the framework,
and use `deployment` as the production branch. The checked-in `vercel.json`
installs the frontend dependencies and runs the static build. No environment
variables are required.

Deploy from the repository root:

```sh
npx vercel@latest link --yes --project hypercast
npx vercel@latest --prod --yes
```

Verify the assigned production URL, a direct-route reload, preset switching,
editing, undo/redo, local save/reopen, import/export, the disabled Run and weight
controls, and a mobile-width layout. Network requests should include only the
site's static files, including `/offline-manifest.json`; there should be no
`/api` request.

## Checks

```sh
.venv/bin/python -m pytest tests/test_offline_manifest.py
npm --prefix web test
npm --prefix web run build
node scripts/build-offline.mjs
```
