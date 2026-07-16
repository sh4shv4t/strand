export interface Garment {
  slot: string
  type: string
  color?: string | null
}

export interface ImageRecord {
  id: string
  garments: Garment[]
  scene?: string | null
  style?: string | null
  notable: string[]
  caption: string
  swatch: string[]
}

export interface ParsedQuery {
  raw_query: string
  garments: Garment[]
  scene?: string | null
  style?: string | null
  confidence: number
}

export interface ScoredResult {
  record: ImageRecord
  score: number
  symbolic_score: number
  dense_score: number
  matched_fields: string[]
}

export interface QueryResponse {
  parsed: ParsedQuery
  results: ScoredResult[]
}

export async function runQuery(query: string, topK = 6): Promise<QueryResponse> {
  const res = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
  })
  if (!res.ok) {
    throw new Error(`Query failed with status ${res.status}`)
  }
  return res.json()
}
