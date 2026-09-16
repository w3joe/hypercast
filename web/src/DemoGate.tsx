import { useState, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import BrandLogo from './BrandLogo'
import { api } from './api'
import { ErrorNotice } from './BuilderControls'

export default function DemoGate({ children }: { children: ReactNode }) {
  const cache = useQueryClient()
  const [email, setEmail] = useState('')
  const [code, setCode] = useState('')
  const [challenge, setChallenge] = useState('')
  const session = useQuery({ queryKey: ['demo-session'], queryFn: api.demoSession, refetchInterval: 30_000, retry: false })
  const send = useMutation({ mutationFn: () => api.demoRequestCode(email), onSuccess: result => setChallenge(result.challenge_id) })
  const verify = useMutation({ mutationFn: () => api.demoVerify(challenge, code), onSuccess: async () => {
    cache.clear()
    await session.refetch()
    setCode(''); setEmail(''); setChallenge('')
  } })
  const logout = useMutation({ mutationFn: api.demoLogout, onSuccess: async () => { cache.clear(); await session.refetch() } })
  if (session.isLoading) return <div className="app-loading"><BrandLogo className="loading-logo" /><p>Opening the demo…</p></div>
  if (session.data?.authenticated) return <div className="demo-workspace">
    <div className="demo-banner" role="status"><span><strong>Public demo</strong> · Limited training on synthetic data · {session.data.remaining_runs} runs left today · Results kept 7 days</span><button onClick={() => logout.mutate()} disabled={logout.isPending}>Sign out</button></div>
    <ErrorNotice error={logout.error} />{children}
  </div>
  return <main className="demo-entry"><section className="demo-entry-card">
    <BrandLogo className="loading-logo" />
    <span className="eyebrow">Try Hypercast</span><h1>Build a model. See it learn.</h1>
    <p>Explore forecasting models, add HyperDense layers, and run a short experiment. Verify your email to start—no password or account setup.</p>
    <div className="demo-limits"><span>3 runs per day</span><span>Up to 2 minutes per run</span><span>Up to 5 epochs</span></div>
    <form onSubmit={event => { event.preventDefault(); challenge ? verify.mutate() : send.mutate() }}>
      {!challenge ? <label>Email address<input autoComplete="email" type="email" required value={email} onChange={event => setEmail(event.target.value)} placeholder="you@example.com" /></label>
        : <><p>Enter the code sent to <strong>{email}</strong>. It expires in 10 minutes.</p><label>Verification code<input autoComplete="one-time-code" inputMode="numeric" pattern="[0-9]{6}" maxLength={6} required value={code} onChange={event => setCode(event.target.value)} /></label></>}
      <ErrorNotice error={session.error || send.error || verify.error} />
      <button className="primary-button" disabled={send.isPending || verify.isPending || Boolean(session.error)}>{send.isPending || verify.isPending ? 'Please wait…' : challenge ? 'Open the demo' : 'Email me a code'}</button>
      {challenge && <button type="button" className="text-button" onClick={() => { setChallenge(''); setCode(''); send.reset(); verify.reset() }}>Use another email or request a new code</button>}
      {session.error && <button type="button" onClick={() => session.refetch()}>Retry connection</button>}
    </form>
    <p className="muted-copy">The demo uses 512 synthetic observations. Results illustrate the workflow and are not research benchmarks. Export your architecture to train it fully on your own machine.</p>
    <p className="muted-copy">Your email is sent to our email provider only to deliver your code. Hypercast keeps a protected identifier for your quota, a session cookie, and your saved designs and results. Designs and results expire after 7 days; quota records after 62 days.</p>
  </section></main>
}
