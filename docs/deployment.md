# Browser-only Hypercast deployment

The static architecture playground is published on GitHub Pages at
<https://w3joe.github.io/hypercast/>.
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

Build the static GitHub Pages site:

```sh
node scripts/build-offline.mjs
```

The static build sets `VITE_OFFLINE_DEMO=true` only for that build. It emits
`build/pages` with the `/hypercast/` base path, a `404.html` SPA fallback, and
`.nojekyll`. It defines no `/api` route or external origin. Set `PAGES_BASE=/`
when serving from a custom domain.

## GitHub Pages

`.github/workflows/pages.yml` builds and deploys the site on every push to
`main`, and can also be started by hand from the Actions tab. In the repository
settings, **Pages → Source** must be **GitHub Actions**. No secrets or
environment variables are required.

Verify the Pages URL, a direct-route reload, preset switching,
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
