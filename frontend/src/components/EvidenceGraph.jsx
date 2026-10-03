import { useEffect, useState } from 'react'

const graphWidth = 600
const graphHeight = 280

function displayName(value) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function riskColor(value) {
  if (value >= 0.7) return 'var(--forensic-risk-high)'
  if (value >= 0.4) return 'var(--forensic-risk-medium)'
  return 'var(--forensic-risk-low)'
}

function getModality(modalities, id) {
  if (Array.isArray(modalities)) return modalities.find((item) => (item.id ?? item.modality) === id)
  return modalities?.[id]
}

function getSignals(modality) {
  if (!modality) return []
  return [
    ...(Array.isArray(modality.signals) ? modality.signals : []),
    ...(Array.isArray(modality.claims) ? modality.claims : []),
    ...(modality.transcript ? [`Transcript: ${modality.transcript}`] : []),
  ]
}

function EvidenceGraph({ graph = {}, modalities = {} }) {
  const [selectedNode, setSelectedNode] = useState(null)
  const nodes = (graph.nodes ?? []).map((node, index) => ({
    ...node,
    value: Math.max(0, Math.min(1, Number(node.risk) || 0)),
    position: getPosition(index, graph.nodes.length),
  }))
  const edges = graph.edges ?? []
  const selectedModality = selectedNode ? getModality(modalities, selectedNode.id) : null
  const selectedSignals = getSignals(selectedModality)

  useEffect(() => {
    if (!selectedNode) return undefined
    function handleKeyDown(event) {
      if (event.key === 'Escape') setSelectedNode(null)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedNode])

  return (
    <section aria-labelledby="graph-title" className="overflow-hidden rounded-md border border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900">
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-slate-200 px-4 py-3 sm:px-5 dark:border-slate-800">
        <div>
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600 dark:text-slate-400">Signal topology</p>
          <h2 id="graph-title" className="text-sm font-semibold text-slate-900 dark:text-slate-100">Evidence graph</h2>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-slate-700 dark:text-slate-300">
          <span className="flex items-center gap-1.5"><i className="size-2 rounded-full bg-teal-700 dark:bg-cyan-400" />Modality</span>
          <span className="flex items-center gap-1.5"><i className="h-0.5 w-4 bg-rose-700 dark:bg-rose-500" />High conflict</span>
        </div>
      </div>
      {nodes.length ? (
        <div className="relative mx-auto aspect-[600/280] min-h-[210px] w-full max-w-3xl px-2 py-1 sm:min-h-[260px]">
          <svg className="absolute inset-0 size-full" viewBox={`0 0 ${graphWidth} ${graphHeight}`} role="img" aria-label="Connections between evidence modalities">
            <defs>
              <filter id="conflict-glow" x="-40%" y="-40%" width="180%" height="180%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>
            {edges.map((edge, index) => {
              const from = nodes.find((node) => node.id === edge.from)?.position
              const to = nodes.find((node) => node.id === edge.to)?.position
              if (!from || !to) return null
              const conflict = Math.max(0, Math.min(1, Number(edge.conflict) || 0))
              const highConflict = conflict >= 0.65
              return (
                <g key={`${edge.from}-${edge.to}-${index}`}>
                  <line
                    x1={from.x}
                    y1={from.y}
                    x2={to.x}
                    y2={to.y}
                    stroke={highConflict ? 'var(--forensic-edge-high)' : 'var(--forensic-graph-line)'}
                    strokeWidth={highConflict ? 3 : 1.5}
                    strokeDasharray={highConflict ? undefined : '5 6'}
                    strokeOpacity={highConflict ? 0.95 : 0.55}
                    filter={highConflict ? 'url(#conflict-glow)' : undefined}
                  />
                  <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 7} fill={highConflict ? 'var(--forensic-edge-high-label)' : 'var(--forensic-graph-label)'} textAnchor="middle" fontSize="10" fontFamily="monospace">
                    {Math.round(conflict * 100)}%
                  </text>
                </g>
              )
            })}
          </svg>
          {nodes.map((node) => {
            const isSelected = selectedNode?.id === node.id
            const modalityRisk = node.value
            return (
              <button
                key={node.id}
                type="button"
                className="absolute z-10 flex w-[76px] -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-1 rounded-md px-1 py-2 text-center transition-colors hover:bg-teal-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-700 sm:w-[92px] dark:hover:bg-slate-800 dark:focus-visible:ring-cyan-300"
                style={{ left: `${node.position.x / graphWidth * 100}%`, top: `${node.position.y / graphHeight * 100}%` }}
                onClick={() => setSelectedNode(isSelected ? null : node)}
                aria-label={`Open ${displayName(node.id)} findings, risk ${Math.round(modalityRisk * 100)} percent`}
                aria-expanded={isSelected}
                aria-haspopup="dialog"
              >
                <span className={`grid size-10 place-items-center rounded-full border bg-white font-mono text-[10px] font-semibold text-slate-900 shadow-[0_0_22px_rgba(15,23,42,0.16)] transition-colors sm:size-12 sm:text-xs dark:bg-slate-950 dark:text-slate-100 dark:shadow-[0_0_22px_rgba(0,0,0,0.5)] ${isSelected ? 'border-teal-700 dark:border-cyan-300' : 'border-slate-400 dark:border-slate-600'}`} style={{ boxShadow: `0 0 18px ${riskColor(modalityRisk)}30` }}>
                  {Math.round(modalityRisk * 100)}%
                </span>
                <span className="text-[10px] font-semibold text-slate-800 sm:text-xs dark:text-slate-200">{displayName(node.id)}</span>
                <span className="text-[9px] uppercase tracking-wider text-slate-600 dark:text-slate-400">Risk</span>
              </button>
            )
          })}
        </div>
      ) : <p className="px-5 py-8 text-sm text-slate-600 dark:text-slate-400">No evidence nodes are available for this case.</p>}

      {selectedNode && (
        <div className="fixed inset-0 z-50 flex justify-end" role="presentation">
          <button type="button" className="absolute inset-0 cursor-default bg-black/65" onClick={() => setSelectedNode(null)} aria-label="Close modality findings" />
          <aside role="dialog" aria-modal="true" aria-labelledby="modality-panel-title" className="relative h-full w-full max-w-sm overflow-y-auto border-l border-slate-300 bg-white p-5 shadow-2xl animate-[slide-in_180ms_ease-out] dark:border-slate-700 dark:bg-slate-900">
            <div className="flex items-start justify-between gap-4 border-b border-slate-200 pb-4 dark:border-slate-800">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600 dark:text-slate-400">Modality detail</p>
                <h2 id="modality-panel-title" className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-100">{displayName(selectedNode.id)}</h2>
              </div>
              <button type="button" className="rounded-sm border border-slate-300 px-2.5 py-1.5 text-xs text-slate-800 hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-700 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-800 dark:focus-visible:ring-cyan-300" onClick={() => setSelectedNode(null)}>Close</button>
            </div>
            <div className="mt-5 rounded-sm border border-slate-300 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-950">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-600 dark:text-slate-400">Modality risk</p>
              <p className="mt-1 font-mono text-3xl font-semibold tabular-nums" style={{ color: riskColor(selectedNode.value) }}>{Math.round(selectedNode.value * 100)}%</p>
            </div>
            <h3 className="mb-3 mt-6 text-xs font-semibold uppercase tracking-wider text-slate-700 dark:text-slate-300">Observed findings</h3>
            {selectedSignals.length ? (
              <ul className="space-y-3">
                {selectedSignals.map((signal, index) => <li key={`${selectedNode.id}-${index}`} className="rounded-sm border border-slate-300 bg-slate-50 p-3 text-sm leading-5 text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-200">{signal}</li>)}
              </ul>
            ) : <p className="text-sm leading-5 text-slate-600 dark:text-slate-400">No modality-specific findings were included in the response.</p>}
          </aside>
        </div>
      )}
    </section>
  )
}

function getPosition(index, count) {
  if (count <= 1) return { x: graphWidth / 2, y: graphHeight / 2 }
  if (count === 2) return { x: index === 0 ? 190 : 410, y: graphHeight / 2 }
  if (count === 3) return { x: 100 + index * 200, y: graphHeight / 2 }
  const columns = Math.ceil(count / 2)
  const column = index % columns
  const row = Math.floor(index / columns)
  const x = columns === 1 ? graphWidth / 2 : 100 + column * (400 / (columns - 1))
  return { x, y: row === 0 ? 72 : 208 }
}

export default EvidenceGraph