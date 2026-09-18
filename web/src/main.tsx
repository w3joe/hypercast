import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import DemoGate from './DemoGate'
import './styles.css'
import './workspace.css'
import './visualizations.css'
import './architecture.css'
import './brand.css'
import './demo.css'

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      {import.meta.env.VITE_PUBLIC_DEMO === 'true' && import.meta.env.VITE_OFFLINE_DEMO !== 'true' ? <DemoGate><App /></DemoGate> : <App />}
    </QueryClientProvider>
  </StrictMode>,
)
