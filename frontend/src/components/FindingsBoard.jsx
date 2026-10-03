const riskFields = ['synthetic_prob', 'spoof_prob', 'manipulation_prob', 'risk', 'score']

function asEntries(modalities) {
  if (Array.isArray(modalities)) return modalities.map((item, index) => [item.id ?? item.modality ?? `modality-${index + 1}`, item])
  return Object.entries(modalities ?? {})
}

function displayName(value) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function getRisk(analysis) {
  const key = riskFields.find((field) => analysis?.[field] != null && Number.isFinite(Number(analysis[field])))
  return key ? Number(analysis[key]) : null
}

function formatScore(value) {
  return `${Math.round(value * 100)}%`
}

function FindingsBoard({ modalities, crossModal = [] }) {
  const entries = asEntries(modalities)

  return (
    <section aria-labelledby="findings-title" className="rounded-md border border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900">
      <div className="border-b border-slate-200 px-4 py-3 sm:px-5 dark:border-slate-800">
        <p className="mb-1 font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600 dark:text-slate-400">Supporting evidence</p>
        <h2 id="findings-title" className="text-sm font-semibold text-slate-900 dark:text-slate-100">Findings breakdown</h2>
      </div>
      <div className="divide-y divide-slate-200 dark:divide-slate-800">
        <div className="p-4 sm:p-5">
          <h3 className="mb-3 text-xs font-medium text-slate-800 dark:text-slate-200">Individual modality signals</h3>
          {entries.length ? (
            <div className="grid gap-3 sm:grid-cols-2">
              {entries.map(([name, analysis]) => {
                const risk = getRisk(analysis)
                const evidence = [
                  ...(Array.isArray(analysis.signals) ? analysis.signals : []),
                  ...(Array.isArray(analysis.claims) ? analysis.claims : []),
                ]
                return (
                  <article key={name} className="rounded-sm border border-slate-300 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-950">
                    <div className="mb-3 flex items-start justify-between gap-3">
                      <h4 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{displayName(name)}</h4>
                      {risk !== null && <span className="shrink-0 font-mono text-xs font-semibold tabular-nums text-rose-800 dark:text-rose-300">{formatScore(risk)} risk</span>}
                    </div>
                    {analysis.transcript && <p className="mb-3 text-xs leading-5 text-slate-700 dark:text-slate-300">Transcript: “{analysis.transcript}”</p>}
                    {evidence.length > 0 ? (
                      <ul className="space-y-2">
                        {evidence.map((signal, index) => <li key={`${name}-${index}`} className="flex gap-2 text-xs leading-5 text-slate-700 dark:text-slate-300"><span className="mt-1 size-1.5 shrink-0 rounded-full bg-teal-700 dark:bg-cyan-400" />{signal}</li>)}
                      </ul>
                    ) : !analysis.transcript && <p className="text-xs text-slate-600 dark:text-slate-400">No supporting signals provided.</p>}
                  </article>
                )
              })}
            </div>
          ) : <p className="text-sm text-slate-600 dark:text-slate-400">No modality findings in this response.</p>}
        </div>

        <div className="p-4 sm:p-5">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-xs font-medium text-slate-800 dark:text-slate-200">Cross-modal conflicts</h3>
            <span className="text-[10px] text-slate-600 dark:text-slate-400">Pairwise consistency signals</span>
          </div>
          {crossModal.length ? (
            <ul className="divide-y divide-slate-200 dark:divide-slate-800">
              {crossModal.map((item, index) => (
                <li key={`${item.pair ?? 'pair'}-${index}`} className="flex flex-wrap items-start justify-between gap-3 py-3 first:pt-0 last:pb-0">
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-medium text-slate-900 dark:text-slate-100">{displayName(item.pair ?? 'Evidence pair')}</p>
                    <p className="mt-1 text-xs leading-5 text-slate-700 dark:text-slate-300">{item.finding ?? 'No finding provided.'}</p>
                  </div>
                  {item.score != null && Number.isFinite(Number(item.score)) && <span className="shrink-0 rounded-sm border border-slate-300 bg-white px-2 py-1 font-mono text-[10px] font-medium tabular-nums text-slate-800 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-200">Signal {formatScore(Number(item.score))}</span>}
                </li>
              ))}
            </ul>
          ) : <p className="text-sm text-slate-600 dark:text-slate-400">No cross-modal conflicts in this response.</p>}
        </div>
      </div>
    </section>
  )
}

export default FindingsBoard