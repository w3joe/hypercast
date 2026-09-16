# Public Hypercast demo: Vercel + Modal

The `deployment` branch adds a public edition without changing the local research
server. Vercel serves the React interface and proxies `/api/*` to Modal. Modal
hosts a small CPU API, a bounded CPU model-checking worker, and one L4 worker.
Training uses durable asynchronous calls, so closing a browser does not cancel a
job. No training or background threads run on Vercel.

Visitors enter an email address and a one-time code. There are no passwords or
account-creation screens. An HttpOnly, Secure cookie identifies their seven-day
session. The server stores an HMAC of the normalized email rather than the email
itself. Resend receives the address to deliver the requested verification code.

## Restrictions

- Three attempted runs per verified email per UTC day; failures and cancellations count.
- One queued/running experiment per email and ten pending experiments globally.
- 200 attempted experiments per calendar month across the entire demo.
- One L4 worker globally; 120-second execution timeout; no configured retries.
- Five epochs maximum, one window/horizon, one seed, one chronological split.
- Window 2–60, horizon 1–20, fixed batch size 32, early stopping after two epochs without improvement.
- 512 original synthetic observations, four numeric signals; no uploads or private research data.
- Bounded model widths, layers, graph size and parameter count (`demo_policy.py`).
- No final tests, research tuning, replication, alternate GPUs or arbitrary code.
- Model checking runs in a separate CPU container with a 20-second timeout and 4 GiB hard memory limit.
- Per-email and shared quotas also cover model checks and verification emails.
- Twenty saved architectures per email; saved designs/results expire after seven days.
- Quota history expires after 62 days; expired codes and sessions are removed.

Limits are enforced on the API and again in the training worker. Email aliases
and multiple mailboxes can circumvent a per-email allowance; shared run/check
quotas and the Modal workspace budget provide the overall backstop. These are
demo results, not fully trained research benchmarks.

## 1. Create a separate Modal workspace

1. Open [Modal workspaces](https://modal.com/settings/workspaces), select **Create
   Workspace**, and create `hypercast-demo` (or another available name).
2. Modal may require a default payment method before creating another workspace.
   Complete the verification in your own browser.
3. In **that new workspace**, set the monthly **usage budget** to **$10** under
   Usage & billing. Usage budgets apply before credits. A **spend limit** instead
   applies after credits. Do not apply this limit to your research workspace.
4. Create a dedicated local token/profile without activating it globally:

   ```sh
   python -m modal token new --profile hypercast-demo --no-activate
   ```

   Choose the **demo workspace** in the browser authorization flow. Verify it:

   ```sh
   python scripts/deploy_demo.py --profile hypercast-demo --workspace hypercast-demo --check
   ```

The profile name is a local label; it does not determine which workspace a token
can access. The deployment script checks the actual authenticated workspace and
refuses a mismatch. It also rejects inherited `MODAL_TOKEN_*` overrides. The
normal research profile remains unchanged.

## 2. Prepare email delivery and the website address

Create a [Resend](https://resend.com/docs/dashboard/domains/introduction) sending
domain, verify its DNS records, and create a sending API key. Resend's sandbox
sender cannot deliver to arbitrary public visitors; use a verified domain.

Create a Vercel project linked to this repository with:

- Repository root as the root directory (not `web/`).
- Framework preset **Other**; the checked-in `vercel.json` supplies install/build commands.
- Production branch **deployment**.
- The stable production URL (or your custom domain) as the public demo address.

The first Vercel build will fail closed until `DEMO_API_ORIGIN` is configured in
step 4. Do not put any Modal tokens, Resend keys, or session secrets in Vercel's
frontend variables.

## 3. Configure and deploy Modal

Install the project with its Modal extra into a working Python environment:

```sh
python -m pip install -e '.[modal]'
cp deploy/demo.env.example deploy/demo.env
```

Fill `deploy/demo.env` locally:

- `DEMO_PUBLIC_ORIGIN`: exact HTTPS Vercel/custom-domain origin, with no path.
- `DEMO_SESSION_SECRET`: a random secret of at least 32 characters.
- `DEMO_EMAIL_FROM`: a sender on the verified domain.
- `RESEND_API_KEY`: the sending key.

This file is ignored by git. Upload it as a secret into the demo workspace:

```sh
python -m modal secret create hypercast-demo-config --from-dotenv deploy/demo.env --profile hypercast-demo --env main
python scripts/deploy_demo.py --profile hypercast-demo --workspace hypercast-demo
```

Record the HTTPS `api` endpoint printed by Modal. GPU and model-checking functions
are private Modal functions; they have no public HTTP endpoints. The public API
requires an email session for every model operation and private artifact read.

## 4. Finish Vercel deployment

Set **DEMO_API_ORIGIN** in the Vercel project's production environment to the
Modal API origin, e.g. `https://YOUR-WORKSPACE--hypercast-demo-api.modal.run`.
Use the actual URL from Modal rather than guessing it. Redeploy the `deployment`
branch.

The build script emits Vercel Build Output API files, including an external API
rewrite. Cookies stay on the website's origin, avoiding cross-site cookie issues.
The backend checks POST origins against `DEMO_PUBLIC_ORIGIN` and does not enable
cross-origin browser access. Preview deployments intentionally cannot submit to
the production backend; use a separate demo environment for authenticated preview
testing, or test through the stable production domain.

The normal `npm run build` still builds the unrestricted local UI. Only the
Vercel build script enables `VITE_PUBLIC_DEMO=true`; server limits do not depend
on this flag and cannot be bypassed by modifying the frontend.

## Storage and recovery

The API exclusively owns the `hypercast-demo-state` Modal Volume. It contains a
SQLite database for sessions, reservations and ownership, plus result artifacts.
The API must remain at **one container** and deploy with **strategy=recreate**;
SQLite on a shared Modal Volume is not a multi-replica database. Mutations are
serialized, committed, closed, then persisted to the Volume. GPU workers never
mount this volume or receive email-delivery secrets.

Calls and quota reservations persist before the response returns. A failed
dispatch is not refunded automatically, preventing retry abuse. The website
polls asynchronous results; logs arrive when training finishes. A platform
timeout reports a failed demo run rather than pretending training completed.
Results that cannot be retrieved within an hour expire. A deployment can briefly
interrupt the API, but the GPU calls continue independently.

For more traffic, migrate this state to a transactional hosted database before
increasing API replica count.

## Budget and verification

The 200-run cap is an allowance, not an exact dollar meter. CPU checks, startup,
idle container time, API traffic, platform retries and storage can also incur
charges. The **separate Modal workspace's $10 usage budget** is the actual cost
backstop. Vercel and email-provider charges are separate. No automatic paid plan
upgrade is needed by this implementation.

Before sharing the link, verify in production:

1. Email delivery, invalid-code rejection, sign-out and a fresh browser session.
2. A small model plus auto-fit HyperDense completes on the L4.
3. A second verified email cannot see the first visitor's jobs, designs, logs or exports.
4. A fourth daily run and oversized/long-running requests are rejected.
5. The Modal budget belongs to the demo workspace, with scale-to-zero enabled.

Local automated checks:

```sh
python -m pytest tests/test_demo.py
npm --prefix web test
DEMO_API_ORIGIN=https://example--hypercast-demo-api.modal.run node scripts/build-demo.mjs
```

References: [Modal budgets](https://modal.com/docs/guide/budgets),
[Modal asynchronous jobs](https://modal.com/docs/guide/job-queue),
[Vercel Build Output API](https://vercel.com/docs/build-output-api/configuration).
