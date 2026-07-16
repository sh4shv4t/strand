const EXAMPLE_QUERIES = [
  'A bright yellow raincoat',
  'Professional business attire inside a modern office',
  'Someone wearing a blue shirt sitting on a park bench',
  'Casual weekend outfit for a city walk',
  'A red tie and a white shirt in a formal setting',
]

export function ExampleChips({ onSelect }: { onSelect: (query: string) => void }) {
  return (
    <div className="flex flex-wrap justify-center gap-2">
      {EXAMPLE_QUERIES.map((q) => (
        <button
          key={q}
          type="button"
          onClick={() => onSelect(q)}
          className="rounded-full border border-white/10 bg-slate-surface/60 px-3.5 py-1.5 text-sm text-cool-gray transition-colors hover:border-indigo/60 hover:text-off-white cursor-pointer"
        >
          {q}
        </button>
      ))}
    </div>
  )
}
