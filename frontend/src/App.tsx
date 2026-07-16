import { useState } from 'react'
import { Logo } from './components/Logo'
import { SearchBar } from './components/SearchBar'
import { ExampleChips } from './components/ExampleChips'
import { ResultCard } from './components/ResultCard'
import { ResultsSkeleton } from './components/ResultsSkeleton'
import { runQuery } from './lib/api'
import type { QueryResponse } from './lib/api'

function App() {
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSearch(submitted: string) {
    setQuery(submitted)
    setLoading(true)
    setError(null)
    try {
      const result = await runQuery(submitted)
      setResponse(result)
    } catch {
      setError('Could not reach the Strand API. Is the backend running on :8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-midnight text-off-white">
      <div className="mx-auto flex max-w-5xl flex-col items-center gap-10 px-6 py-16">
        <header className="flex flex-col items-center gap-4 text-center">
          <div className="flex items-center gap-3">
            <Logo size={40} />
            <h1 className="text-3xl font-semibold tracking-tight">Strand</h1>
          </div>
          <p className="text-cool-gray">Every detail, connected.</p>
        </header>

        <section className="flex w-full flex-col items-center gap-5">
          <SearchBar value={query} onChange={setQuery} onSubmit={handleSearch} loading={loading} />
          <ExampleChips onSelect={handleSearch} />
        </section>

        {error && (
          <p className="rounded-xl border border-amber/30 bg-amber/5 px-4 py-2 text-sm text-amber">
            {error}
          </p>
        )}

        {loading && <ResultsSkeleton />}

        {!loading && response && (
          <section className="flex w-full flex-col gap-6">
            <div className="rounded-2xl border border-white/10 bg-slate-surface/60 p-4 text-sm text-cool-gray">
              <span className="text-off-white">Parsed as: </span>
              {response.parsed.garments.length === 0 &&
                !response.parsed.scene &&
                !response.parsed.style && <span>no structured signal detected — falling back to dense match</span>}
              {response.parsed.garments.map((g, i) => (
                <span key={i} className="mr-1.5">
                  {g.color ? `${g.color} ` : ''}
                  {g.type} ({g.slot})
                  {i < response.parsed.garments.length - 1 ? ',' : ''}
                </span>
              ))}
              {response.parsed.scene && <span className="mr-1.5">scene: {response.parsed.scene}</span>}
              {response.parsed.style && <span className="mr-1.5">style: {response.parsed.style}</span>}
            </div>

            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
              {response.results.map((r) => (
                <ResultCard key={r.record.id} result={r} />
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  )
}

export default App
