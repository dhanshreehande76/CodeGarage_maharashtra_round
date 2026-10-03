import { useState } from 'react'

const uploadFields = [
  { id: 'image', label: 'Image', accept: 'image/*', hint: 'JPG, PNG, WEBP' },
  { id: 'video', label: 'Video', accept: 'video/*', hint: 'MP4, MOV, WEBM' },
  { id: 'audio', label: 'Audio', accept: 'audio/*', hint: 'WAV, MP3, M4A' },
]

function MediaUploader({ onAnalyze }) {
  const [selectedFiles, setSelectedFiles] = useState({})
  const [claim, setClaim] = useState('')
  const [notice, setNotice] = useState('')

  function handleSubmit(event) {
    event.preventDefault()
    const hasEvidence = Object.keys(selectedFiles).length > 0 || claim.trim().length > 0
    if (!hasEvidence) {
      setNotice('Add at least one media file or a claim to start an investigation.')
      return
    }
    onAnalyze({ files: selectedFiles, claim })
  }

  return (
    <section aria-labelledby="upload-title" className="rounded-md border border-slate-300 bg-white shadow-xl shadow-slate-900/5 transition-colors dark:border-slate-700 dark:bg-slate-900 dark:shadow-black/20">
      <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
        <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600 dark:text-slate-400">New evidence</p>
        <h2 id="upload-title" className="text-base font-semibold text-slate-900 dark:text-slate-100">Build an investigation</h2>
      </div>
      <form className="space-y-4 p-5" onSubmit={handleSubmit}>
        {uploadFields.map((field) => (
          <div key={field.id}>
            <label htmlFor={`upload-${field.id}`} className="mb-1.5 block text-xs font-medium text-slate-800 dark:text-slate-200">{field.label}</label>
            <label htmlFor={`upload-${field.id}`} className="flex min-h-16 cursor-pointer items-center justify-between gap-3 rounded-sm border border-dashed border-slate-400 bg-slate-50 px-3 py-2.5 transition-colors hover:border-teal-700 hover:bg-teal-50 dark:border-slate-600 dark:bg-slate-950 dark:hover:border-cyan-400 dark:hover:bg-slate-800">
              <span className="min-w-0">
                <span className="block truncate text-sm text-slate-900 dark:text-slate-100">{selectedFiles[field.id]?.name ?? 'Choose a file'}</span>
                <span className="mt-0.5 block text-xs text-slate-600 dark:text-slate-400">{selectedFiles[field.id] ? 'Ready to stage' : field.hint}</span>
              </span>
              <span className="shrink-0 rounded-sm border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-800 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200">Browse</span>
            </label>
            <input
              id={`upload-${field.id}`}
              className="sr-only"
              type="file"
              accept={field.accept}
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) setSelectedFiles((current) => ({ ...current, [field.id]: file }))
              }}
            />
          </div>
        ))}
        <div>
          <label htmlFor="claim" className="mb-1.5 block text-xs font-medium text-slate-800 dark:text-slate-200">Claim</label>
          <textarea
            id="claim"
            rows="3"
            value={claim}
            onChange={(event) => setClaim(event.target.value)}
            placeholder="What does this evidence claim to show?"
            className="w-full resize-y rounded-sm border border-slate-400 bg-white px-3 py-2.5 text-sm leading-5 text-slate-900 outline-none placeholder:text-slate-500 focus:border-teal-700 focus:ring-2 focus:ring-teal-200 dark:border-slate-600 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-500 dark:focus:border-cyan-400 dark:focus:ring-cyan-950"
          />
        </div>
        <button type="submit" className="w-full rounded-sm bg-teal-800 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-teal-900 focus:outline-none focus:ring-2 focus:ring-teal-700 focus:ring-offset-2 focus:ring-offset-white dark:bg-cyan-500 dark:text-slate-950 dark:hover:bg-cyan-400 dark:focus:ring-cyan-400 dark:focus:ring-offset-slate-900">
          Analyze evidence
        </button>
        {notice && <p role="status" className="text-xs leading-5 text-amber-800 dark:text-amber-300">{notice}</p>}
        <p className="text-xs leading-5 text-slate-600 dark:text-slate-400">Files remain in this browser. Results use a sample analysis response.</p>
      </form>
    </section>
  )
}

export default MediaUploader