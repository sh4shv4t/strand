export function ResultsSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="flex h-64 animate-pulse flex-col gap-4 rounded-2xl border border-white/10 bg-slate-surface p-5"
        >
          <div className="h-4 w-1/3 rounded bg-white/10" />
          <div className="flex gap-1.5">
            <div className="h-6 w-16 rounded-full bg-white/10" />
            <div className="h-6 w-20 rounded-full bg-white/10" />
          </div>
          <div className="h-3 w-full rounded bg-white/10" />
          <div className="h-3 w-2/3 rounded bg-white/10" />
          <div className="mt-auto h-1.5 w-full rounded-full bg-white/10" />
        </div>
      ))}
    </div>
  )
}
