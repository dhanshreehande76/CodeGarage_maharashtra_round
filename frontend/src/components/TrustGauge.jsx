function TrustGauge({ value = 0 }) {
  const score = Math.max(0, Math.min(1, Number(value) || 0))
  const circumference = 2 * Math.PI * 48
  const color = score >= 0.7 ? 'var(--forensic-risk-low)' : score >= 0.4 ? 'var(--forensic-risk-medium)' : 'var(--forensic-risk-high)'
  const label = score >= 0.7 ? 'Higher trust' : score >= 0.4 ? 'Mixed signals' : 'Low trust'

  return (
    <div className="flex items-center gap-4 xl:flex-col xl:gap-1" role="img" aria-label={`Trust score ${Math.round(score * 100)} percent, ${label}`}>
      <div className="relative size-28 shrink-0">
        <svg className="size-full -rotate-90" viewBox="0 0 112 112" aria-hidden="true">
          <circle cx="56" cy="56" r="48" fill="none" stroke="var(--forensic-gauge-track)" strokeWidth="8" />
          <circle
            cx="56"
            cy="56"
            r="48"
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - score)}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-2xl font-semibold tabular-nums text-slate-900 dark:text-slate-100">{Math.round(score * 100)}%</span>
          <span className="font-mono text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-600 dark:text-slate-400">Trust</span>
        </div>
      </div>
      <p className="text-xs font-semibold" style={{ color }}>{label}</p>
    </div>
  )
}

export default TrustGauge