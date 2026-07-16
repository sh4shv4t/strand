import { useState } from 'react'
import type { ScoredResult } from '../lib/api'

function GarmentChip({
  label,
  matched,
}: {
  label: string
  matched: boolean
}) {
  return (
    <span
      className={`rounded-full border px-2.5 py-1 text-xs ${
        matched
          ? 'border-emerald/50 bg-emerald/10 text-emerald'
          : 'border-white/10 bg-white/5 text-cool-gray'
      }`}
    >
      {label}
    </span>
  )
}

export function ResultCard({ result }: { result: ScoredResult }) {
  const { record, score, matched_fields } = result
  const [photoFailed, setPhotoFailed] = useState(false)
  const showPhoto = record.id.startsWith('fp_') && !photoFailed

  return (
    <div className="flex flex-col gap-4 rounded-2xl border border-white/10 bg-slate-surface p-5 transition-colors hover:border-indigo/40">
      {showPhoto && (
        <img
          src={`/api/images/${record.id}.jpg`}
          alt=""
          className="h-40 w-full rounded-xl object-cover"
          onError={() => setPhotoFailed(true)}
        />
      )}

      <div className="flex items-center gap-1.5">
        {record.swatch.map((hex, i) => (
          <span
            key={i}
            className="h-4 w-4 rounded-full border border-white/20"
            style={{ background: hex }}
          />
        ))}
        <span className="ml-auto font-[var(--font-code)] text-xs text-cool-gray">
          {record.id}
        </span>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {record.garments.map((g) => {
          const label = `${g.color ? g.color + ' ' : ''}${g.type}`
          return (
            <GarmentChip key={g.slot} label={label} matched={matched_fields.includes(label)} />
          )
        })}
        {record.scene && (
          <span className="rounded-full border border-emerald/30 bg-emerald/5 px-2.5 py-1 text-xs text-emerald">
            {record.scene}
          </span>
        )}
        {record.style && (
          <span className="rounded-full border border-amber/30 bg-amber/5 px-2.5 py-1 text-xs text-amber">
            {record.style}
          </span>
        )}
      </div>

      <p className="text-sm italic text-cool-gray">{record.caption}</p>

      <div className="mt-auto">
        <div className="mb-1 flex justify-between text-xs text-cool-gray">
          <span>match score</span>
          <span>{Math.round(score * 100)}%</span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-gradient-to-r from-indigo to-emerald"
            style={{ width: `${Math.round(score * 100)}%` }}
          />
        </div>
      </div>
    </div>
  )
}
