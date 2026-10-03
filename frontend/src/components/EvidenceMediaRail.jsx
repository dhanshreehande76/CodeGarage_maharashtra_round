const mediaItems = [
  { id: 'image', label: 'Image evidence', kind: 'image' },
  { id: 'video', label: 'Video evidence', kind: 'video' },
]

function EvidenceMediaRail({ evidence, highlightedTypes = [] }) {
  const files = evidence?.files ?? {}
  const previews = evidence?.previews ?? {}
  const claim = evidence?.claim ?? ''

  function highlightClass(type) {
    return highlightedTypes.includes(type)
      ? 'border-teal-700 ring-2 ring-teal-600/40 shadow-[0_0_24px_rgba(15,118,110,0.12)] dark:border-cyan-300/80 dark:ring-cyan-400/70 dark:shadow-[0_0_24px_rgba(34,211,238,0.14)]'
      : 'border-slate-300 dark:border-slate-700'
  }

  return (
    <section className="space-y-4" aria-labelledby="evidence-rail-title">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600 dark:text-slate-400">Case evidence</p>
          <h2 id="evidence-rail-title" className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-100">Submitted materials</h2>
        </div>
        <span className="rounded-sm border border-slate-300 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-slate-700 dark:border-slate-700 dark:text-slate-300">Local</span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
        {mediaItems.map((item) => (
          <article key={item.id} data-evidence-type={item.id} className={`overflow-hidden rounded-md border bg-white transition-all duration-200 dark:bg-slate-900 ${highlightClass(item.id)}`}>
            <div className="flex items-center justify-between border-b border-slate-200 px-3 py-2.5 dark:border-slate-800">
              <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">{item.label}</h3>
              <span className="font-mono text-[10px] uppercase text-slate-600 dark:text-slate-400">{files[item.id] ? 'Attached' : 'No file'}</span>
            </div>
            {files[item.id] ? (
              item.kind === 'image' ? (
                <div className="grid min-h-40 place-items-center bg-slate-100 p-2 dark:bg-slate-950">
                  <img className="max-h-64 w-full object-contain" src={previews[item.id]} alt="Submitted image evidence" />
                </div>
              ) : (
                <video className="aspect-video w-full bg-slate-200 object-contain dark:bg-black" src={previews[item.id]} controls preload="metadata" aria-label="Submitted video evidence" />
              )
            ) : (
              <div className="grid min-h-32 place-items-center bg-slate-100 px-4 text-center dark:bg-slate-950">
                <p className="text-xs text-slate-600 dark:text-slate-400">No {item.id} supplied for this case.</p>
              </div>
            )}
            {files[item.id] && <p className="truncate border-t border-slate-200 px-3 py-2 font-mono text-[10px] text-slate-700 dark:border-slate-800 dark:text-slate-300">{files[item.id].name}</p>}
          </article>
        ))}
      </div>

      {files.audio && (
        <article data-evidence-type="audio" className={`rounded-md border bg-white p-3 transition-all duration-200 dark:bg-slate-900 ${highlightClass('audio')}`}>
          <div className="mb-2 flex items-center justify-between">
            <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">Audio evidence</h3>
            <span className="font-mono text-[10px] text-slate-700 dark:text-slate-300">{files.audio.name}</span>
          </div>
          <audio className="h-9 w-full" src={previews.audio} controls preload="metadata" aria-label="Submitted audio evidence" />
        </article>
      )}

      <article data-evidence-type="text" className={`rounded-md border bg-white p-4 transition-all duration-200 dark:bg-slate-900 ${highlightClass('text')}`}>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">Claim under review</h3>
          <span className="font-mono text-[10px] uppercase text-slate-600 dark:text-slate-400">Text</span>
        </div>
        <blockquote className="border-l-2 border-teal-700 pl-3 text-sm leading-6 text-slate-800 dark:border-cyan-400 dark:text-slate-300">
          {claim || 'No claim text supplied for this case.'}
        </blockquote>
      </article>
      <p className="text-[10px] leading-4 text-slate-600 dark:text-slate-400">Media previews are held in this browser session and are not uploaded by the demo.</p>
    </section>
  )
}

export default EvidenceMediaRail