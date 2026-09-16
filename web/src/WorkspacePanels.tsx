import { createContext, useContext } from 'react'
import { createPortal } from 'react-dom'
import type { ReactNode } from 'react'

export const WorkspacePanels = createContext<{ sidebar: HTMLElement | null; toolbar: HTMLElement | null; openSidebar?: () => void } | null>(null)

export function SidePanel({ children, active = true }: { children: ReactNode; active?: boolean }) {
  const panels = useContext(WorkspacePanels)
  if (!active) return null
  return panels ? panels.sidebar && createPortal(children, panels.sidebar) : <>{children}</>
}

export function TopbarTools({ children }: { children: ReactNode }) {
  const panels = useContext(WorkspacePanels)
  return panels ? panels.toolbar && createPortal(children, panels.toolbar) : <>{children}</>
}

export function useWorkspacePanels() { return useContext(WorkspacePanels) }
