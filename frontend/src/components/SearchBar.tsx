import { useState } from 'react'
import type { FormEvent } from 'react'

interface SearchBarProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (query: string) => void
  loading: boolean
}

export function SearchBar({ value, onChange, onSubmit, loading }: SearchBarProps) {
  const [localValue, setLocalValue] = useState(value)

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (localValue.trim()) {
      onSubmit(localValue.trim())
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex w-full max-w-2xl gap-2">
      <input
        type="text"
        value={localValue}
        onChange={(e) => {
          setLocalValue(e.target.value)
          onChange(e.target.value)
        }}
        placeholder="Describe an outfit, scene, or vibe…"
        className="flex-1 rounded-2xl border border-white/10 bg-slate-surface px-5 py-3.5 text-off-white placeholder:text-cool-gray/70 outline-none transition-shadow focus:ring-2 focus:ring-indigo/60"
      />
      <button
        type="submit"
        disabled={loading || !localValue.trim()}
        className="rounded-2xl bg-indigo px-6 py-3.5 font-medium text-off-white transition-opacity hover:opacity-90 disabled:opacity-40 cursor-pointer disabled:cursor-not-allowed"
      >
        {loading ? 'Searching…' : 'Search'}
      </button>
    </form>
  )
}
