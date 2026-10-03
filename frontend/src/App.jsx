import { useEffect, useState } from 'react'
import response from '../../docs/mock_response.json'
import EvidenceGraph from './components/EvidenceGraph.jsx'
import EvidenceMediaRail from './components/EvidenceMediaRail.jsx'
import ExplainabilityPanel from './components/ExplainabilityPanel.jsx'
import FindingsBoard from './components/FindingsBoard.jsx'
import MediaUploader from './components/MediaUploader.jsx'
import ThemeToggle from './components/ThemeToggle.jsx'
import TrustGauge from './components/TrustGauge.jsx'

const verdictLabels = {
  authentic: 'Authentic',
  manipulated: 'Manipulated',
  coordinated_synthetic: 'Coordinated synthetic',
  uncertain: 'Uncertain / requires verification',
  uncertain_requires_verification: 'Uncertain / requires verification',
  requires_verification: 'Uncertain / requires verification',
}

function formatPercent(value) {
  return `${Math.round(Number(value) * 100)}%`
}

function App() {
  const [stage, setStage] = useState('upload')
  const [evidence, setEvidence] = useState(null)
  const [highlightedTypes, setHighlightedTypes] = useState([])
  const verdict = verdictLabels[response.verdict] ?? response.verdict
  const verdictStyle = getVerdictStyle(response.verdict)

  useEffect(() => {
    if (stage !== 'loading') return undefined
    const timeout = window.setTimeout(() => setStage('results'), 1800)
    return () => window.clearTimeout(timeout)
  }, [stage])

  useEffect(() => () => {
    Object.values(evidence?.previews ?? {}).forEach((url) => URL.revokeObjectURL(url))
  }, [evidence])

  function handleAnalyze({ files, claim }) {
    const previews = Object.fromEntries(Object.entries(files).map(([type, file]) => [type, URL.createObjectURL(file)]))
    setEvidence({ files, previews, claim })
    setStage('loading')
  }

  function startNewInvestigation() {
    setEvidence(null)
    setHighlightedTypes([])
    setStage('upload')
  }

  return (
    <div className="min-h-screen bg-forensic-light-background text-forensic-light-text transition-colors duration-300 dark:bg-forensic-dark-background dark:text-forensic-dark-text">
      <header className="sticky top-0 z-30 border-b border-slate-300/90 bg-white/90 backdrop-blur dark:border-slate-800/90 dark:bg-slate-950/90">
        <div className="mx-auto flex max-w-[1680px] items-center justify-between gap-4 px-5 py-3 sm:px-8">
          <button className="flex items-center gap-3 text-left" onClick={startNewInvestigation} aria-label="Start a new TrustLayer investigation">
            <span className="grid size-9 place-items-center rounded-sm border border-teal-800 bg-teal-700 font-mono text-xs font-bold text-white dark:border-cyan-800 dark:bg-slate-900 dark:text-cyan-300">TL</span>
            <span>
              <span className="block text-sm font-semibold text-slate-900 dark:text-slate-100">TrustLayer</span>
              <span className="block text-[10px] uppercase tracking-[0.14em] text-slate-600 dark:text-slate-400">Forensic review workspace</span>
            </span>
          </button>
          <div className="flex items-center gap-3">
            <div className="hidden items-center gap-2 rounded-sm border border-amber-300 bg-amber-50 px-2.5 py-1.5 font-mono text-[9px] uppercase tracking-wider text-amber-900 sm:flex dark:border-amber-900 dark:bg-amber-950/60 dark:text-amber-200">
              <span className="size-1.5 rounded-full bg-amber-600 dark:bg-amber-400" />
              Sample analysis
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {stage === 'upload' && (
        <main className="mx-auto grid max-w-[1180px] items-start gap-10 px-5 py-8 sm:px-8 sm:py-14 lg:grid-cols-[minmax(0,1fr)_480px] lg:gap-16">
          <section className="pt-2 sm:pt-8">
            <p className="mb-3 font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-teal-800 dark:text-cyan-400">Digital authenticity review</p>
            <h1 className="max-w-xl text-3xl font-semibold leading-tight text-slate-900 sm:text-4xl dark:text-slate-100">Bring the evidence together.</h1>
            <p className="mt-4 max-w-lg text-sm leading-6 text-slate-700 dark:text-slate-300">Review images, video, audio, and the claim in one investigation. TrustLayer surfaces individual signals and inconsistencies between sources.</p>
            <div className="mt-9 border-l-2 border-teal-700 pl-4 dark:border-cyan-400">
              <p className="text-sm font-medium text-slate-800 dark:text-slate-200">Assessment, not a definitive finding</p>
              <p className="mt-1 text-xs leading-5 text-slate-600 dark:text-slate-400">Results highlight supporting evidence and confidence. Verify important conclusions independently.</p>
            </div>
          </section>
          <MediaUploader onAnalyze={handleAnalyze} />
        </main>
      )}

      {stage === 'loading' && (
        <main className="mx-auto flex min-h-[calc(100vh-57px)] max-w-3xl flex-col items-center justify-center px-5 py-12 text-center">
          <div className="grid size-16 place-items-center rounded-full border border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900">
            <span className="size-8 animate-spin rounded-full border-[3px] border-slate-300 border-t-teal-700 dark:border-slate-700 dark:border-t-cyan-400" />
          </div>
          <p className="mt-6 font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-teal-800 dark:text-cyan-400">Analysis in progress</p>
          <h1 className="mt-2 text-2xl font-semibold text-slate-900 dark:text-slate-100">Reviewing submitted evidence</h1>
          <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">Comparing modality signals and cross-modal consistency.</p>
          <div className="mt-8 w-full max-w-sm space-y-3 text-left">
            {['Inspecting media signals', 'Comparing claim and evidence', 'Preparing explainable result'].map((step, index) => (
              <div key={step} className="flex items-center gap-3 rounded-sm border border-slate-300 bg-white px-4 py-3 text-sm text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
                <span className={`grid size-5 place-items-center rounded-full font-mono text-[10px] font-semibold ${index === 0 ? 'bg-teal-100 text-teal-800 dark:bg-teal-950 dark:text-cyan-300' : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400'}`}>{index === 0 ? '✓' : index + 1}</span>
                {step}
              </div>
            ))}
          </div>
          <p className="mt-5 font-mono text-[10px] uppercase tracking-wider text-slate-600 dark:text-slate-400">Loading the local demonstration response</p>
        </main>
      )}

      {stage === 'results' && (
        <main id="dashboard" className="mx-auto max-w-[1680px] px-4 py-5 sm:px-6 lg:px-8">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="mb-1.5 font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-teal-800 dark:text-cyan-400">Investigation / local sample</p>
              <h1 className="text-xl font-semibold text-slate-900 sm:text-2xl dark:text-slate-100">Case assessment</h1>
            </div>
            <button className="rounded-sm border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-800 transition-colors hover:border-teal-700 hover:bg-teal-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-cyan-400 dark:hover:bg-slate-800 dark:focus-visible:ring-cyan-400" onClick={startNewInvestigation}>
              New investigation
            </button>
          </div>
          <p role="note" className="mb-5 rounded-sm border border-amber-300 bg-amber-50 px-4 py-2.5 text-xs leading-5 text-amber-950 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-200">
            Demonstration only: this assessment is loaded from a fixed sample response. Submitted files and claim are not analyzed.
          </p>

          <div className="grid items-start gap-5 lg:grid-cols-[minmax(290px,0.82fr)_minmax(0,1.18fr)] xl:gap-6">
            <div className="lg:sticky lg:top-[76px] lg:max-h-[calc(100vh-92px)] lg:overflow-y-auto lg:pr-1">
              <EvidenceMediaRail evidence={evidence} highlightedTypes={highlightedTypes} />
            </div>
            <div className="min-w-0 space-y-4">
              <section aria-labelledby="verdict-title" className={`overflow-hidden rounded-md border bg-white dark:bg-slate-900 ${verdictStyle.border}`}>
                <div className={`h-1 ${verdictStyle.bar}`} />
                <div className="grid gap-5 p-4 sm:p-5 xl:grid-cols-[108px_minmax(0,1fr)_150px] xl:items-center">
                  <TrustGauge value={response.trust_score} />
                  <div>
                    <div className="mb-2 flex flex-wrap items-center gap-2.5">
                      <span className={`rounded-sm border px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.12em] ${verdictStyle.badge}`}>{verdict}</span>
                      <span className="text-[10px] text-slate-600 dark:text-slate-400">Evidence-based assessment</span>
                    </div>
                    <h2 id="verdict-title" className="mb-1.5 text-sm font-semibold text-slate-900 dark:text-slate-100">Assessment summary</h2>
                    <p className="text-xs leading-5 text-slate-700 dark:text-slate-300">{response.explanation}</p>
                  </div>
                  <div className="rounded-sm border border-slate-300 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-950">
                    <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600 dark:text-slate-400">Confidence</p>
                    <p className="mt-1 font-mono text-2xl font-semibold tabular-nums text-slate-900 dark:text-slate-100">{formatPercent(response.confidence)}</p>
                    <div className="mt-2 h-1 overflow-hidden rounded-full bg-slate-300 dark:bg-slate-700">
                      <div className="h-full rounded-full bg-teal-700 dark:bg-cyan-400" style={{ width: formatPercent(response.confidence) }} />
                    </div>
                    <p className="mt-2 text-[9px] leading-4 text-slate-600 dark:text-slate-400">Confidence in this assessment, not proof.</p>
                  </div>
                </div>
              </section>

              <EvidenceGraph graph={response.evidence_graph} modalities={response.modalities} />
              <ExplainabilityPanel findings={response.cross_modal} onHighlight={setHighlightedTypes} />
              <FindingsBoard modalities={response.modalities} crossModal={response.cross_modal} />
            </div>
          </div>
          <p className="mt-6 border-t border-slate-300 pt-3 text-[10px] leading-4 text-slate-600 dark:border-slate-800 dark:text-slate-400">
            TrustLayer supports review by surfacing signals and conflicts across submitted evidence. Findings may be incomplete or incorrect; independently verify consequential conclusions.
          </p>
        </main>
      )}
    </div>
  )
}

function getVerdictStyle(verdict) {
  const normalizedVerdict = verdict.toLowerCase()
  if (normalizedVerdict === 'authentic') return { border: 'border-green-700 dark:border-green-500', bar: 'bg-green-600 dark:bg-green-400', badge: 'border-green-700 bg-green-100 text-green-900 dark:border-green-700 dark:bg-green-950 dark:text-green-300' }
  if (normalizedVerdict.startsWith('uncertain') || normalizedVerdict.includes('requires_verification') || normalizedVerdict.includes('requires verification')) return { border: 'border-amber-600 dark:border-amber-500', bar: 'bg-amber-500 dark:bg-amber-400', badge: 'border-amber-700 bg-amber-100 text-amber-950 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200' }
  return { border: 'border-rose-700 dark:border-rose-500', bar: 'bg-rose-700 dark:bg-rose-500', badge: 'border-rose-800 bg-rose-100 text-rose-950 dark:border-rose-700 dark:bg-rose-950 dark:text-rose-200' }
}

export default App