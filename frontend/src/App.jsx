import { useMemo, useState } from 'react'
import {
  Area, CartesianGrid, ComposedChart, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { useOrchestrator } from './useOrchestrator'

const L = (v) => (v == null ? '–' : `₹${(v / 1e5).toFixed(2)} L`)
const n0 = (v) => (v == null ? '–' : Math.round(v).toLocaleString('en-IN'))
const RISK = { low: 'good', elevated: 'warning', high: 'serious', critical: 'critical' }
const RISK_ICON = { low: '●', elevated: '▲', high: '▲', critical: '■' }
const ARB = { cons: 'conservative', norm: 'normal', aggr: 'aggressive' }

function planText(id) {
  const m = /^R(\d+)-(\w+)-(\d+)h$/.exec(id || '')
  if (!m) return id || '–'
  return `Battery reserve ${m[1]}% · ${ARB[m[2]] || m[2]} trading · ${m[3]} h look-ahead`
}

/* ------------------------------------------------------------------ header & controls */
function Controls({ st, act }) {
  const { status, catalog } = st
  const [seed, setSeed] = useState('')
  if (!status || !catalog) return null
  const running = status.state === 'running'
  const reset = (patch) => act('/sim/reset', {
    scenario: status.scenario, agent: status.agent, seed: seed === '' ? status.seed : Number(seed), ...patch,
  })
  return (
    <div className="controls">
      <label>Scenario
        <select value={status.scenario} onChange={(e) => reset({ scenario: e.target.value, seed: null })}>
          {catalog.scenarios.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>
      </label>
      <label>Agent
        <select value={status.agent || ''} onChange={(e) => act('/sim/agent', { agent: e.target.value })}>
          {catalog.agents.map((a) => <option key={a.name} value={a.name}>{a.label}</option>)}
        </select>
      </label>
      <label>Seed
        <input className="seed" placeholder={String(status.seed)} value={seed}
          onChange={(e) => setSeed(e.target.value.replace(/\D/g, ''))} />
      </label>
      <button onClick={() => reset({})}>Reset</button>
      {running
        ? <button className="primary" onClick={() => act('/sim/pause', {})}>Pause</button>
        : <button className="primary" disabled={status.state === 'finished'} onClick={() => act('/sim/start', {})}>Play</button>}
      <button onClick={() => act('/sim/step', { steps: 1 })} disabled={status.state === 'finished'}>Step</button>
      <label>Speed
        <select value={status.speed} onChange={(e) => act('/sim/speed', { speed: Number(e.target.value) })}>
          {[60, 150, 300, 600, 1200].map((v) => <option key={v} value={v}>{(900 / v).toFixed(1)} s/step</option>)}
          {![60, 150, 300, 600, 1200].includes(status.speed) && <option value={status.speed}>{status.speed}x</option>}
        </select>
      </label>
    </div>
  )
}

/* ------------------------------------------------------------------ KPI tiles */
function Kpis({ st }) {
  const k = st.kpis
  if (!k) return null
  const step = st.chart.length
  const rb = st.baseline?.agents?.rule_based?.objective_curve
  const vsRule = rb && step > 0 ? rb[Math.min(step, rb.length) - 1] - k.objective.total : null
  const tiles = [
    { label: 'Profit so far', value: L(k.money.profit) },
    { label: 'Saved vs rule-based', value: vsRule == null ? '–' : `${vsRule >= 0 ? '' : '−'}${L(Math.abs(vsRule)).slice(0)}`,
      tone: vsRule == null ? '' : vsRule >= 0 ? 'good' : 'critical', hint: 'same day, same weather' },
    { label: 'Unserved energy', value: `${k.energy.unserved_mwh.toFixed(1)} MWh`, tone: k.energy.unserved_mwh > 0.05 ? 'critical' : 'good' },
    { label: 'Clean share', value: `${k.percent.clean_share_pct.toFixed(1)}%` },
    { label: 'CO₂', value: `${n0(k.carbon.co2_t)} t` },
    { label: 'Emergency actions', value: k.agent.emergency_actions, hint: `${k.agent.fallbacks} fallbacks` },
  ]
  return (
    <div className="kpis">
      {tiles.map((t) => (
        <div className="tile" key={t.label}>
          <div className="tile-label">{t.label}</div>
          <div className={`tile-value ${t.tone || ''}`}>{t.value}</div>
          {t.hint && <div className="tile-hint">{t.hint}</div>}
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ charts */
const tipStyle = { background: 'var(--surface-2)', border: '1px solid var(--line)', borderRadius: 8, fontSize: 12 }
const axis = { stroke: 'var(--text-muted)', fontSize: 11, tickLine: false }

function eventBands(events) {
  return (events || []).filter((e) => ['warning', 'critical'].includes(e.severity) && e.start_step != null)
    .map((e) => ({ id: e.id, x1: e.start_step, x2: e.end_step ?? e.start_step + 4, title: e.title }))
}

function Legend({ items }) {
  return (
    <div className="legend">
      {items.map(([name, color, dashed]) => (
        <span key={name}><i style={{ background: dashed ? 'transparent' : color, borderTop: dashed ? `2px dashed ${color}` : 'none' }} />{name}</span>
      ))}
    </div>
  )
}

function ChartCard({ title, sub, legend, children, height = 220 }) {
  return (
    <section className="card chart-card">
      <header><h3>{title}</h3>{sub && <span className="sub">{sub}</span>}</header>
      {legend && <Legend items={legend} />}
      <div style={{ height }}><ResponsiveContainer>{children}</ResponsiveContainer></div>
    </section>
  )
}

function Charts({ st }) {
  const data = useMemo(() => st.chart.map((p) => ({
    ...p, x: p.step, bat_out: Math.max(0, p.battery_net), grid_in: Math.max(0, p.grid),
    soc1: p.soc_B1 != null ? Math.round(p.soc_B1 * 100) : null, soc2: p.soc_B2 != null ? Math.round(p.soc_B2 * 100) : null,
  })), [st.chart])
  const total = st.status?.total_steps || 96
  const bands = eventBands(st.events)
  const xProps = { dataKey: 'x', type: 'number', domain: [0, total], ...axis,
    ticks: Array.from({ length: Math.floor(total / 16) + 1 }, (_, i) => i * 16),
    tickFormatter: (v) => `${String(Math.floor((v % 96) / 4)).padStart(2, '0')}:00` }
  const timeLabel = (v) => data.find((d) => d.x === v)?.time || v
  const bandEls = bands.map((b) => <ReferenceArea key={b.id} x1={b.x1} x2={b.x2} fill="var(--event-band)" ifOverflow="hidden" />)

  const cost = useMemo(() => {
    const b = st.baseline?.agents
    const rows = []
    const len = b ? b.rule_based.objective_curve.length : 0
    for (let i = 0; i < Math.max(len, data.length); i++) {
      rows.push({ x: i + 1, agent: data[i]?.objective != null ? data[i].objective / 1e5 : null,
        rule: b ? b.rule_based.objective_curve[i] / 1e5 : null, naive: b ? b.naive.objective_curve[i] / 1e5 : null })
    }
    return rows
  }, [data, st.baseline])

  return (
    <div className="charts">
      <ChartCard title="Supply vs demand" sub="MW · shaded = storm / warning windows"
        legend={[['Solar', 'var(--solar)'], ['Wind', 'var(--wind)'], ['Battery out', 'var(--battery)'], ['Grid import', 'var(--grid)'], ['Demand', 'var(--text-primary)', true]]}
        height={250}>
        <ComposedChart data={data} margin={{ top: 6, right: 8, left: -12, bottom: 0 }}>
          <CartesianGrid stroke="var(--grid-line)" vertical={false} />
          <XAxis {...xProps} />
          <YAxis {...axis} />
          {bandEls}
          <Tooltip contentStyle={tipStyle} labelFormatter={timeLabel} formatter={(v, name) => [`${n0(v)} MW`, name]} />
          <Area type="monotone" dataKey="solar" name="Solar" stackId="1" stroke="var(--solar)" fill="var(--solar)" fillOpacity={0.55} strokeWidth={1.5} isAnimationActive={false} />
          <Area type="monotone" dataKey="wind" name="Wind" stackId="1" stroke="var(--wind)" fill="var(--wind)" fillOpacity={0.55} strokeWidth={1.5} isAnimationActive={false} />
          <Area type="monotone" dataKey="bat_out" name="Battery out" stackId="1" stroke="var(--battery)" fill="var(--battery)" fillOpacity={0.55} strokeWidth={1.5} isAnimationActive={false} />
          <Area type="monotone" dataKey="grid_in" name="Grid import" stackId="1" stroke="var(--grid)" fill="var(--grid)" fillOpacity={0.35} strokeWidth={1.5} isAnimationActive={false} />
          <Line type="monotone" dataKey="demand" name="Demand" stroke="var(--text-primary)" strokeDasharray="5 4" dot={false} strokeWidth={2} isAnimationActive={false} />
        </ComposedChart>
      </ChartCard>
      <div className="chart-row">
        <ChartCard title="Battery charge" sub="% state of charge" legend={[['B1', 'var(--battery)'], ['B2', 'var(--wind)', true]]} height={170}>
          <LineChart data={data} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="var(--grid-line)" vertical={false} />
            <XAxis {...xProps} />
            <YAxis {...axis} domain={[0, 100]} />
            {bandEls}
            <Tooltip contentStyle={tipStyle} labelFormatter={timeLabel} formatter={(v, name) => [`${v}%`, name]} />
            <Line type="stepAfter" dataKey="soc1" name="B1" stroke="var(--battery)" dot={false} strokeWidth={2} isAnimationActive={false} />
            <Line type="stepAfter" dataKey="soc2" name="B2" stroke="var(--wind)" strokeDasharray="6 4" dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ChartCard>
        <ChartCard title="Market price" sub="₹ / MWh" height={170}>
          <LineChart data={data} margin={{ top: 6, right: 8, left: -6, bottom: 0 }}>
            <CartesianGrid stroke="var(--grid-line)" vertical={false} />
            <XAxis {...xProps} />
            <YAxis {...axis} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
            {bandEls}
            <Tooltip contentStyle={tipStyle} labelFormatter={timeLabel} formatter={(v) => [`₹${n0(v)}`, 'Price']} />
            <Line type="monotone" dataKey="price" stroke="var(--grid)" dot={false} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ChartCard>
      </div>
      <ChartCard title="Running cost: this agent vs baselines" sub="₹ lakh · lower is better · same day replayed by the other agents"
        legend={[['This agent', 'var(--accent)'], ['Rule-based', 'var(--text-secondary)', true], ['Naive', 'var(--text-muted)', true]]} height={200}>
        <LineChart data={cost} margin={{ top: 6, right: 8, left: -6, bottom: 0 }}>
          <CartesianGrid stroke="var(--grid-line)" vertical={false} />
          <XAxis {...xProps} />
          <YAxis {...axis} />
          <Tooltip contentStyle={tipStyle} labelFormatter={(v) => `${String(Math.floor(((v - 1) % 96) / 4)).padStart(2, '0')}:${String(((v - 1) % 4) * 15).padStart(2, '0')}`}
            formatter={(v, name) => [v == null ? '–' : `₹${v.toFixed(2)} L`, name]} />
          <Line type="monotone" dataKey="naive" name="Naive" stroke="var(--text-muted)" strokeDasharray="3 4" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line type="monotone" dataKey="rule" name="Rule-based" stroke="var(--text-secondary)" strokeDasharray="6 4" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          <Line type="monotone" dataKey="agent" name="This agent" stroke="var(--accent)" dot={false} strokeWidth={2.5} isAnimationActive={false} connectNulls={false} />
        </LineChart>
      </ChartCard>
    </div>
  )
}

/* ------------------------------------------------------------------ agent brain */
function Brain({ st }) {
  const d = st.planner
  const agent = st.catalog?.agents.find((a) => a.name === st.status?.agent)
  if (!d) {
    return (
      <section className="card brain">
        <header><h3>Agent brain</h3></header>
        <p className="muted">{st.status?.agent === 'llm_planner'
          ? 'Press Play: the planner reviews the situation at the first step.'
          : `${agent?.label || 'This agent'} has no planning layer. Switch to the Agentic planner to see its reasoning.`}</p>
      </section>
    )
  }
  const chosen = d.plan
  return (
    <section className="card brain">
      <header>
        <h3>Agent brain</h3>
        <span className={`chip ${d.source === 'llm' ? 'llm' : ''}`}>{d.source === 'llm' ? 'LLM' : 'Auto (no LLM)'}</span>
      </header>
      <div className="brain-top">
        <span className={`risk ${RISK[d.risk_level]}`}>{RISK_ICON[d.risk_level]} Risk {d.risk_level}</span>
        <span className="muted small">{d.trigger} · step {d.step} · {d.latency_ms} ms</span>
      </div>
      <div className="plan">
        <div className="plan-id">{chosen}</div>
        <div className="small muted">{planText(chosen)}</div>
      </div>
      {d.assessment && <p className="assessment">{d.assessment}</p>}
      {d.reasons?.length > 0 && <ul className="reasons">{d.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>}
      <div className="twin">
        <div className="twin-head">Digital twin · plans tested over the next 12 h</div>
        <table>
          <thead><tr><th>Plan</th><th>Normal day</th><th>Bad case</th><th>Blackout risk</th></tr></thead>
          <tbody>
            {d.evaluated?.map((r) => (
              <tr key={r.plan_id} className={r.plan_id === chosen ? 'chosen' : ''}>
                <td>{r.plan_id === chosen ? '▶ ' : ''}{r.plan_id}</td>
                <td>₹{r.expected_lakh.toFixed(2)} L</td>
                <td>₹{r.stress_lakh.toFixed(2)} L</td>
                <td>{r.unserved_stress_mwh > 0 ? `${r.unserved_stress_mwh} MWh` : 'none'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {d.saving_vs_default_rs > 1000 && <div className="small good-text">Chosen plan beats the default by ₹{(d.saving_vs_default_rs / 1e5).toFixed(2)} L in the twin</div>}
      </div>
      <PlannerStats st={st} />
    </section>
  )
}

function PlannerStats({ st }) {
  const m = st.log.find((e) => e.agent === 'llm_planner')
  const label = st.catalog?.agents.find((a) => a.name === 'llm_planner')?.label || ''
  const prov = /\((\w+) \+/.exec(label)?.[1]
  return <div className="small muted stats">Model: {prov || '–'} · reviews every ~2 h or on new events · guardrails: risk can only go up, plan must match twin within 0.2%{m?.fallback ? ' · fallback used' : ''}</div>
}

/* ------------------------------------------------------------------ operator + events */
function Operator({ st, act }) {
  const [text, setText] = useState('')
  const isPlanner = st.status?.agent === 'llm_planner'
  const send = async () => {
    if (!text.trim()) return
    await act('/agent/instruction', { text })
    setText('')
  }
  const examples = ['Cyclone warning for tonight, keep batteries full', 'Hospital feeder C3 must not be cut today', 'Prices look low tomorrow morning, trade more']
  return (
    <section className="card">
      <header><h3>Talk to the agent</h3></header>
      {!isPlanner && <p className="small muted">Only the Agentic planner reads operator notes.</p>}
      <div className="note-row">
        <input value={text} placeholder="Plain-language guidance for the agent…" onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()} />
        <button className="primary" onClick={send} disabled={!text.trim()}>Send</button>
      </div>
      <div className="examples">{examples.map((x) => <button key={x} className="ghost" onClick={() => setText(x)}>{x}</button>)}</div>
      {st.instructions.length > 0 && <ul className="notes">{st.instructions.slice().reverse().map((x, i) => <li key={i}>“{x}”</li>)}</ul>}
    </section>
  )
}

function Events({ st, act }) {
  const types = st.catalog ? Object.values(st.catalog.events) : []
  const [type, setType] = useState('storm')
  const inject = () => act('/events', { type, start_in_steps: type === 'storm' ? 8 : 0, announce_in_steps: 0 })
  const evs = (st.events || []).slice().sort((a, b) => a.start_step - b.start_step)
  return (
    <section className="card">
      <header><h3>Events</h3></header>
      <div className="note-row">
        <select value={type} onChange={(e) => setType(e.target.value)}>
          {types.map((t) => <option key={t.type} value={t.type}>{t.label}</option>)}
        </select>
        <button onClick={inject} disabled={!st.status || st.status.state === 'finished'}>Inject</button>
      </div>
      <ul className="events">
        {evs.length === 0 && <li className="muted small">No events yet.</li>}
        {evs.map((e) => (
          <li key={e.id} className={e.status}>
            <span className={`dot ${e.severity}`} />
            <div>
              <div>{e.title}</div>
              <div className="small muted">{e.status}{e.source === 'operator' || e.source === 'api' ? ' · injected' : ''}</div>
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}

/* ------------------------------------------------------------------ assets + log */
function Assets({ st }) {
  const cur = st.observation?.current
  if (!cur) return null
  const items = [
    ...Object.entries(cur.solar).map(([k, a]) => ({ k, name: a.name, kind: 'Solar', v: `${n0(a.available_mw)} MW`, status: a.status })),
    ...Object.entries(cur.wind).map(([k, a]) => ({ k, name: a.name, kind: 'Wind', v: `${n0(a.available_mw)} MW`, status: a.status })),
    ...Object.entries(cur.batteries).map(([k, b]) => ({ k, name: b.name, kind: 'Battery', v: `${Math.round(b.soc * 100)}%`, status: b.status })),
    ...Object.entries(cur.grid.lines).map(([k, l]) => ({ k, name: l.name, kind: 'Line', v: `${n0(l.rating_mw)} MW`, status: l.status })),
  ]
  return (
    <div className="assets">
      {items.map((a) => (
        <div key={a.k} className={`asset ${a.status}`} title={`${a.name}: ${a.status}`}>
          <span className="asset-k">{a.k}</span><span className="asset-v">{a.v}</span>
          {a.status !== 'online' && <span className="asset-s">{a.status}</span>}
        </div>
      ))}
    </div>
  )
}

function Log({ st }) {
  const [onlyPlanner, setOnly] = useState(false)
  const rows = st.log.filter((e) => !onlyPlanner || e.reasons.some((r) => r.startsWith('[LLM') || r.startsWith('[Planner')))
  return (
    <section className="card log">
      <header>
        <h3>Decision log</h3>
        <label className="small check"><input type="checkbox" checked={onlyPlanner} onChange={(e) => setOnly(e.target.checked)} /> Planner reviews only</label>
      </header>
      <div className="log-body">
        {rows.slice(0, 120).map((e) => (
          <div key={`${e.step}-${e.agent}`} className={`log-row ${e.fallback ? 'fallback' : ''}`}>
            <span className="t">{e.time}</span>
            <span className="mode">{e.mode}</span>
            <span className="why">{e.reasons.slice(0, 3).map((r, i) => <span key={i} className={r.startsWith('[LLM') ? 'llm-line' : ''}>{r}</span>)}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ app */
export default function App() {
  const st = useOrchestrator()
  const s = st.status
  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <div className="logo">⚡</div>
          <div><h1>Sutradhar</h1><div className="small muted">Renewable Energy Orchestrator · agentic control room · 5 solar · 3 wind · 2 batteries · 2 tie-lines</div></div>
        </div>
        <div className="clock">
          <span className={`live ${st.connected ? (s?.state === 'running' ? 'on' : '') : 'off'}`} />
          <span className="time">{s?.label || '–'}</span>
          <span className="small muted">{st.connected ? s?.state : 'reconnecting…'}</span>
        </div>
      </header>
      <Controls st={st} act={st.act} />
      {st.error && <div className="error" onClick={st.clearError}>{st.error} (click to dismiss)</div>}
      <Kpis st={st} />
      <Assets st={st} />
      <main className="grid">
        <div className="left"><Charts st={st} /><Log st={st} /></div>
        <aside className="right"><Brain st={st} /><Operator st={st} act={st.act} /><Events st={st} act={st.act} /></aside>
      </main>
    </div>
  )
}
