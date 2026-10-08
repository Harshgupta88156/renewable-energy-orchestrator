import { useCallback, useEffect, useRef, useState } from 'react'

const MAX_LOG = 300

async function api(path, body, method) {
  const r = await fetch(`/api${path}`, {
    method: method || (body === undefined ? 'GET' : 'POST'),
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail || r.statusText)
  return data
}

/** One WebSocket to the backend; keeps everything the dashboard renders. */
export function useOrchestrator() {
  const [s, set] = useState({
    connected: false, status: null, catalog: null, chart: [], log: [], events: [], kpis: null,
    observation: null, planner: null, plannerHistory: [], instructions: [], baseline: null, error: null,
  })
  const wsRef = useRef(null)
  const runRef = useRef(null)

  const loadBaseline = useCallback(() => {
    api('/baseline').then((b) => set((p) => ({ ...p, baseline: b }))).catch(() => {})
  }, [])

  useEffect(() => {
    let closed = false
    let retry
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      wsRef.current = ws
      ws.onopen = () => set((p) => ({ ...p, connected: true, error: null }))
      ws.onclose = () => {
        set((p) => ({ ...p, connected: false }))
        if (!closed) retry = setTimeout(connect, 1500)
      }
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data)
        if (m.type === 'snapshot') {
          const newRun = runRef.current !== m.status?.run_id
          runRef.current = m.status?.run_id
          set((p) => ({
            ...p, status: m.status, catalog: m.catalog, chart: m.chart || [], log: (m.decisions || []).slice().reverse(),
            events: m.events || [], kpis: m.kpis, observation: m.observation, planner: m.planner,
            plannerHistory: m.planner ? [m.planner] : [], instructions: m.instructions || [],
            baseline: newRun ? null : p.baseline,
          }))
          if (newRun) loadBaseline()
        } else if (m.type === 'tick') {
          set((p) => {
            const fresh = m.planner && (!p.planner || p.planner.step !== m.planner.step)
            return {
              ...p, status: m.status, chart: [...p.chart, m.point], log: [m.log, ...p.log].slice(0, MAX_LOG),
              kpis: m.kpis, observation: m.observation, events: m.events, planner: m.planner || p.planner,
              plannerHistory: fresh ? [m.planner, ...p.plannerHistory].slice(0, 30) : p.plannerHistory,
            }
          })
        } else if (m.type === 'status') {
          set((p) => ({ ...p, status: m.status }))
        } else if (m.type === 'event') {
          set((p) => ({ ...p, events: m.events, observation: m.observation }))
        } else if (m.type === 'instruction') {
          set((p) => ({ ...p, instructions: [...p.instructions, m.text].slice(-5) }))
        } else if (m.type === 'error') {
          set((p) => ({ ...p, error: m.message }))
        }
      }
    }
    connect()
    return () => { closed = true; clearTimeout(retry); wsRef.current?.close() }
  }, [loadBaseline])

  const act = useCallback(async (path, body) => {
    try { return await api(path, body) } catch (e) { set((p) => ({ ...p, error: e.message })) }
  }, [])

  return { ...s, act, clearError: () => set((p) => ({ ...p, error: null })) }
}
