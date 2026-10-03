function titleCase(value) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function pairTypes(pair) {
  return pair.toLowerCase().split(/\s*[-↔]\s*/).map((type) => type.trim()).filter(Boolean)
}

function ExplainabilityPanel({ findings = [], onHighlight }) {
  return (
    <section aria-labelledby="explainability-title" className="overflow-hidden rounded-md border border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900">
      <div className="border-b border-slate-200 px-4 py-3 sm:px-5 dark:border-slate-800">
        <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600 dark:text-slate-400">Explainability</p>
        <h2 id="explainability-title" className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-100">Why these assets were flagged</h2>
      </div>
      {findings.length ? (
        <ul className="divide-y divide-slate-200 dark:divide-slate-800">
          {findings.map((item, index) => {
            const types = pairTypes(item.pair ?? '')
            const pairLabel = types.map(titleCase).join(' ↔ ') || 'Evidence comparison'
            const score = Number(item.score)
            const sentence = `${pairLabel} returned a ${Number.isFinite(score) ? `${Math.round(score * 100)}% comparison score` : 'comparison result'}. ${item.finding ?? 'The analysis identified an inconsistency.'}`
            return (
              <li key={`${item.pair ?? 'finding'}-${index}`}>
                <button
                  type="button"
                  className="group block w-full px-4 py-4 text-left transition-colors hover:bg-teal-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-teal-700 sm:px-5 dark:hover:bg-slate-800 dark:focus-visible:ring-cyan-400"
                  onMouseEnter={() => onHighlight(types)}
                  onMouseLeave={() => onHighlight([])}
                  onFocus={() => onHighlight(types)}
                  onBlur={() => onHighlight([])}
                  onClick={() => onHighlight(types)}
                  aria-label={`Highlight ${pairLabel} evidence: ${sentence}`}
                >
                  <span className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-slate-900 group-hover:text-teal-800 dark:text-slate-100 dark:group-hover:text-cyan-200">{pairLabel}</span>
                    {Number.isFinite(score) && <span className="font-mono text-[11px] tabular-nums text-rose-800 dark:text-rose-300">{Math.round(score * 100)}% comparison</span>}
                  </span>
                  <span className="mt-2 block text-xs leading-5 text-slate-700 dark:text-slate-300">{sentence}</span>
                </button>
              </li>
            )
          })}
        </ul>
      ) : <p className="px-5 py-4 text-xs text-slate-600 dark:text-slate-400">No cross-modal explanations are available for this case.</p>}
      <p className="border-t border-slate-200 px-4 py-2.5 text-[10px] text-slate-600 sm:px-5 dark:border-slate-800 dark:text-slate-400">Hover, focus, or select a comparison to locate its evidence.</p>
    </section>
  )
}

export default ExplainabilityPanel