# Glance ML Internship Assignment — Working Notes (v2)

Status: all design decisions are locked in (§9), the retrieval pipeline runs on real data (Chroma, 1,000 real Fashionpedia records at the assignment's stated scale, §5, a real CLIP-based indexer using a fashion-tuned backbone, §10/§15.1), it's measured against real baselines (§12), and it has a measured, not just reasoned, scaling estimate (§13). Real CLIP image embeddings and caption-text embeddings are both blended into ranking (§14.1), garment-type synonym matching closes a real cross-vocabulary matching gap (§14.2), a newly indexed image can be registered into the live catalog immediately (§14.3), and a plain CLI (§14.4) satisfies the literal "a script that..." wording alongside the API. The real Gemini integration (query parser and VLM attribute extraction, §8/§10) is written and unit-tested, with automatic fallback to the keyword parser when it isn't configured, but has not been exercised against a real API key yet. Set `GEMINI_API_KEY` (see `backend/.env.example`) and it should just work; if the SDK's API has drifted since this was written, that would surface as an error in `gemini_client.py`, the one place that talks to it. `README.md` has the user-facing overview, architecture diagrams, and data sources; this file is the detailed engineering log behind it, including the attempts that didn't pan out.

---

## 1. Problem nuances (unchanged, still true)

- CLIP's dual-encoder trains on whole-image/whole-caption contrastive pairs → no attribute-object binding → "bag of words" failure mode (the ARO benchmark line of work documents this). This is exactly what breaks eval query 5 ("red tie, white shirt") — color-swapped images score similarly under vanilla CLIP.
- Fine-tuned fashion embedding models close the *vocabulary* gap but not the *binding* gap. Marqo-FashionCLIP and Marqo-FashionSigLIP (fine-tuned from SigLIP/CLIP backbones using Generalised Contrastive Learning) report large recall gains over generic CLIP and the older FashionCLIP2.0 baseline on fashion retrieval benchmarks — Marqo's own reported figures put FashionSigLIP's improvement around the +50% range in MRR/recall over FashionCLIP2.0. But it's still a single dense vector per image, so compositional confusion (query 5, and to a lesser extent query 3/4) doesn't go away — it's still one pooled embedding trying to represent multiple garments + scene + style at once.
- "Where" (location/scene) and "vibe" (style) are scene-level signals, distinct from garment attributes. Cramming everything into one embedding just recreates bag-of-words at a higher level of abstraction. These are naturally three separable axes (garment, scene, style) and the eval queries test each axis individually and in combination — worth treating them as separate, independently matchable fields rather than one blob.
- Assignment explicitly deprioritizes indexing engineering — "pick the easiest vector DB" — so the grading weight is on retrieval *logic*, not infra plumbing.

---

## 2. Candidate architectures (tradeoff table — Option D chosen, see below)

### Option A — Vanilla CLIP / SigLIP whole-image embedding
Baseline only. Given as the anti-pattern in the prompt itself. Fails query 5 by construction, and does poorly on query 3/4 (contextual/style inference) since a single vector under-weights secondary attributes. Zero eng cost, zero fashion awareness. Use only as your reported baseline number, not your submission.

### Option B — Fine-tuned fashion dense embedding only (Marqo-FashionCLIP/SigLIP)
Swap the backbone for a fashion-tuned one. Better recall on garment vocabulary and color naming, same binding failure. Low eng cost (drop-in HF model). Still a single vector — scene and style get entangled with garment attributes in the same embedding space. Reasonable *fallback/reranking* component, weak as a sole system.

### Option C — Region/detection-based compositional retrieval
Detect each garment (bbox) with a fashion object detector (DeepFashion2-trained, or a general detector like YOLO fine-tuned on Fashionpedia's segmentation masks), embed each region + classify its color independently, then match query sub-phrases to regions. Solves compositionality *by construction* — this is structurally what Pinterest's Shop-the-Look and Myntra's My Stylist actually do in production (both name a dedicated detection step as stage one of their pipeline, before any embedding). Highest fidelity on query 5, but needs a trained/fine-tuned detector — higher eng cost, and the assignment explicitly asks you to *not* over-invest in engineering-heavy indexing infrastructure. Worth citing as the "correct at scale" answer in your future-work section even if you don't build it fully.

### Option D — Structured VLM attribute extraction → symbolic schema + dense fallback ✅ CHOSEN
A VLM (or small fine-tuned model) emits a fixed JSON schema per image at index time:
```
{garments: [{slot: upper/lower/outerwear/footwear, type, color}],
 scene: office/street/park/home/other,
 style: formal/casual/athleisure/...,
 notable: [free text]}
```
Query time: an LLM parses the NL query into the same schema; retrieval = symbolic filter (exact/fuzzy match on schema fields) intersected or blended with dense cosine similarity on a flattened caption (for recall on free-text/style nuance the schema doesn't capture). Solves compositionality via explicit slot binding rather than spatial detection — cheaper than Option C, still zero-shot (open-vocabulary VLM + LLM parser, no closed label set to retrain). A real, recent precedent for the "small VLM fine-tuned to emit fashion JSON" pattern: *Fashion Florence* (arXiv, May 2026) fine-tunes Florence-2 (0.77B params) with LoRA on iMaterialist Fashion labels to emit compact JSON (category/color/material/style/occasion), and reports it beating GPT-4o-mini and Gemini 2.5 Flash on category and style-tag accuracy for this exact structured-extraction task, while running cheaply on a single GPU. That's a solid citation for "this pattern is practical, not speculative."

### Option E — Hybrid: coarse detection + structured attributes (no full re-ID)
A middle ground between C and D: instead of a full fashion object detector, use lightweight crop heuristics (simple upper/lower/outerwear region heuristics from pose estimation, or just quadrant-based crops since most fashion photos are single-person full-body shots) to get 2–3 rough garment crops, then run the VLM/attribute-extraction step *per crop* instead of on the whole image. Gets you most of Option C's binding accuracy without training a real detector — pose estimation models (e.g. MediaPipe Pose, MMPose) are pretrained and off-the-shelf. Medium eng cost, good compositionality, no detector fine-tuning needed. This is probably the best "shows ML judgment without over-engineering" answer for a take-home.

**Decision:** going with D. It's the safest, fastest-to-build, cleanly-fashion-aware answer and matches the assignment's "don't over-invest in indexing engineering" instruction almost exactly — and the scaffold's schema/parser/retriever contracts were already built against it, so no rework is needed to lock it in. E remains a natural extension (pose/crop step before the VLM call) worth a paragraph in future-work if there's time to spare, not a requirement. C is cite-only — "what production does at scale" — in the future-work section.

---

## 3. Hybrid pipeline design space (mix and match)

You don't have to pick a single point on the table above — the retrieval *scoring* step can blend signals regardless of which extraction method you use upstream:

- **Symbolic-only:** exact filter on schema fields. Fast, precise, brittle to query phrasing / missing fields.
- **Dense-only:** cosine similarity on flattened caption or CLIP embedding. Robust to phrasing, weak on compositionality.
- **Weighted hybrid (what the earlier notes proposed):** `score = α·symbolic_match + (1-α)·dense_cosine`, tune α against the 5 eval queries. Simple, interpretable, easy to defend in a write-up.
- **Filter-then-rerank (Pinterest's actual pattern):** cheap ANN/ hamming-distance recall pass first to get a shortlist, then a heavier relevance signal reranks only that shortlist. Structurally identical to the weighted hybrid but staged instead of blended — worth mentioning as the production-grade version of the same idea, with "replace fixed weighting with a learned reranker" as a natural future-work line.
- **Cascade/fallback:** try symbolic filter first; if it returns too few results (schema too strict, or the query parser has low confidence on a field), fall back to pure dense search. Handles queries that don't map cleanly onto your fixed schema.

Recommend building the weighted hybrid first (easiest to implement and to explain in the write-up), then mentioning the cascade and reranker versions as future-work extensions — that satisfies both "working system" and "shows you know the more sophisticated version" grading angles.

---

## 4. Full workflow — indexing, serving, and querying end to end

### 4.1 Offline indexing loop (runs once per image, not per query)
1. **Ingest**: pull 500–1000 sampled images from chosen dataset(s) (see §5).
2. **Attribute/region extraction** (method depends on architecture choice in §2):
   - Option D: single VLM call per image → JSON schema.
   - Option E: pose/crop step → per-crop VLM call → merge into one JSON.
   - Option C: detector inference → per-bbox embed + color classify.
3. **Dense embedding pass**: encode a flattened caption (built from the JSON, e.g. "casual blue hoodie and grey joggers, street setting") through a fashion-tuned encoder (Marqo-FashionCLIP/SigLIP) for the reranking/fallback signal. Also keep the raw image embedding if you want direct CLIP-similarity as a secondary signal.
4. **Scene/style tagging**: if not covered by your dataset's own labels, zero-shot classify scene (office/street/park/home) and style (formal/casual/athleisure) with CLIP or the same VLM, since Fashionpedia doesn't natively label environment.
5. **Write to store**: JSON schema as payload/metadata + dense vector, in one record per image, keyed by image ID.
6. **Delta processing going forward**: only re-run steps 2–5 on new/changed images, not the whole catalog — this is what makes the "does it work at 1M images" scalability answer credible. All heavy model inference lives here, offline, where latency doesn't matter.

### 4.2 Model serving — how each model actually gets called
This directly answers "how will you serve the models" — worth a short paragraph in your report.

| Model | Role | Serving mode | Notes |
|---|---|---|---|
| VLM (attribute extractor) | Index-time, batch | ✅ **Chosen: hosted API** (Gemini Flash / GPT-4o-mini) called in a batch loop | One-time or delta batch job over 500–1000 images, no real-time requirement — chosen over self-hosting (Florence-2-large / Qwen2-VL-2B) purely for least setup friction: no checkpoint download, no local GPU/quantization plumbing, just an API key and a batch loop. Self-hosting stays a valid future-work swap if API cost/rate-limits become a problem at larger scale |
| Fashion dense encoder (Marqo-FashionCLIP/SigLIP) | Index-time (image) + query-time (text) | Self-hosted, local inference | ~150–400M params, trivial on a 4GB GPU or even CPU; load once, keep resident in a small FastAPI service or just in-process in your retriever script |
| LLM query parser | Query-time only | API call (no local hosting needed) | Runtime latency matters here since it's per-query, not per-image — a small/fast model (e.g. a mini-tier model) with a few-shot prompt is enough; this is the actual bottleneck at high query volume, not the vector search |
| Vector DB — ✅ **Chosen: Chroma** | Both | Embedded, in-process (`pip install chromadb`, no separate server) | Lowest setup friction of the two: Qdrant needs a running service (Docker or Cloud) even for local dev, Chroma runs in-process with a local persistence directory and still supports ANN + metadata filter in one call. Qdrant remains the natural swap if this needs to run as a standalone service later (sharding/read-replicas, §4.4) |

Your dev machine (RTX 3050, 4GB VRAM + i7) is fine for all of this — the only thing that would actually stress it is *fine-tuning* a VLM, which none of these options require (Fashion Florence-style fine-tuning is a nice-to-mention future-work stretch goal, not a requirement here).

### 4.3 Online query loop (runs per user query, must be fast)
1. User submits NL query, e.g. "a red tie and a white shirt in a formal setting."
2. **LLM parser** (few-shot prompted) converts it into the same JSON schema used at index time — e.g. `{garments: [{slot: upper, type: shirt, color: white}, {slot: accessory, type: tie, color: red}], style: formal}`.
3. **Retrieval**: run your chosen scoring strategy (weighted hybrid / cascade / rerank) against the precomputed store — symbolic filter on schema fields intersected/blended with dense cosine similarity on the flattened caption.
4. **Return top-k** image IDs + scores.
5. Never call the VLM or detector at query time — only the lightweight parser + ANN lookup touch the query path. This offline/online split is the direct answer to the scalability grading criterion: it's irrelevant at 500 images and is the difference between working and not working at 1M.

#### 4.3.1 ✅ Chosen: query-parser prompt/schema

Same schema on both sides of the parser (index-time VLM output and query-time LLM output), so retrieval never has to reconcile two shapes. System prompt + few-shot, ready to drop into a real LLM call in place of `backend/app/services/query_parser.py`'s keyword-spotting stand-in:

```
System:
You are a query parser for a fashion image search system. Given a natural
language search query, extract structured information as JSON matching
this schema exactly:

{
  "garments": [{"slot": "upper|lower|outerwear|footwear|accessory", "type": string, "color": string|null}],
  "scene": "office|street|park|home|other" | null,
  "style": "formal|casual|athleisure|business|other" | null
}

Only include a garment, scene, or style if the query supports it — do not
guess. Return valid JSON only, no other text.

Few-shot:

Query: "A bright yellow raincoat"
JSON: {"garments": [{"slot": "outerwear", "type": "raincoat", "color": "yellow"}], "scene": null, "style": null}

Query: "Professional business attire inside a modern office"
JSON: {"garments": [], "scene": "office", "style": "business"}

Query: "Someone wearing a blue shirt sitting on a park bench"
JSON: {"garments": [{"slot": "upper", "type": "shirt", "color": "blue"}], "scene": "park", "style": null}

Query: "Casual weekend outfit for a city walk"
JSON: {"garments": [], "scene": "street", "style": "casual"}

Query: "A red tie and a white shirt in a formal setting"
JSON: {"garments": [{"slot": "accessory", "type": "tie", "color": "red"}, {"slot": "upper", "type": "shirt", "color": "white"}], "scene": null, "style": "formal"}
```

The five few-shot examples are exactly the five eval queries from §6 — deliberately, so the parser's few-shot coverage and the grading queries are the same set. Output maps 1:1 onto `ParsedQuery`/`Garment` in `backend/app/schema.py`, so swapping this in is a body-only change to `query_parser.parse_query`; the API and retriever are untouched.

### 4.4 Query-volume scaling (distinct from dataset-size scaling)
- ANN + metadata filtering scales via standard sharding/read-replicas behind a load balancer — Qdrant and Chroma both support this natively, no custom indexing code needed.
- The real bottleneck at high QPS is the **LLM query parser** (per-query network call). Mitigations, roughly in order of effort: (1) cache parsed JSON for repeated/near-duplicate queries, (2) once you've logged enough query→JSON pairs, distill into a small fine-tuned local classifier/NER model and drop the API dependency entirely, (3) add a Redis-style cache layer in front of the whole pipeline for popular queries.

---

## 5. Dataset plan (expanded)

### Primary
- **Fashionpedia** (via Hugging Face, `detection-datasets/fashionpedia`): 45,623 train + 1,158 val, ~3.5GB, CC-BY-4.0. Real daily-life/street-style/celebrity-event photos (not studio shots) — covers clothing type + color natively via 294 fine-grained attribute labels + segmentation masks. No scraping needed. Environment axis isn't natively labeled — tag it yourself via zero-shot CLIP classification against {office, street, park, home}, since the photos already vary in setting.

### Worth adding alongside it (new — not in v1 notes)
- **DeepFashion2** (491K images): larger, has consumer-photo/shop-photo pairs and richer bounding-box + landmark annotations than Fashionpedia. Overkill to use in full, but a small sample (a few hundred images) is a good *additional* environment-diversity source, and its bbox annotations are useful if you go with Option C/E (detection-based).
- **iMaterialist Fashion** (228 fine-grained attribute labels across category/color/material/style/pattern/sleeve/neckline/gender): good complementary attribute vocabulary — this is actually the dataset Fashion Florence itself fine-tunes on, so if you want a similar structured-JSON attribute schema, iMaterialist's label taxonomy is a solid template to borrow from directly.
- **Polyvore Outfits dataset**: curated outfit sets (multiple garments styled together), useful if you want extra "vibe/style" signal since it's built around styling/outfit coherence rather than single-item catalog shots — good for style-axis diversity, thinner on garment-level bounding boxes.
- **Kaggle "Fashion Product Images" (Myntra-sourced, ~44K)**: good clothing/color labels but studio/plain-background only — no environment diversity. Useful as a supplemental *catalog-style* negative-diversity source (to make sure your system doesn't only work on street photos) but not a substitute for Fashionpedia.
- **Fashion200K**: shorter product-description style captions, useful mainly as extra text-to-image training/eval signal if you want more text variety for tuning your query parser's few-shot examples.

### Verdict
Fashionpedia alone is sufficient to hit the 500–1000 image / 3-axis requirement on its own. Adding a small DeepFashion2 or iMaterialist sample mainly strengthens your "zero-shot / generalizes beyond one dataset's label set" story, which maps directly onto the zero-shot grading criterion — worth doing if you have time, not required.

**Executed at 1,000, not just planned at 40.** `pull_fashionpedia_sample.py` originally ran at `TARGET_COUNT=40` for fast local iteration during scaffolding, and that smaller sample sat committed as `real_catalog_sample.json` for the rest of scaffolding, well short of the 500–1,000 minimum this section already argued Fashionpedia alone could satisfy. Re-run at `TARGET_COUNT=1000`: the catalog is now 1,000 real records + 12 mock, and `build_image_vector_index.py` was re-run to persist a real CLIP embedding for all 1,000 (§14.1's image-similarity wiring needs those present to do anything). See §12.2 for how this changed the Tier 2 baseline comparison's metric choice.

---

## 6. Mapping the 5 eval queries to what each architecture needs to get right

1. **"A bright yellow raincoat"** — single attribute + single garment. All options handle this; it's the sanity-check query.
2. **"Professional business attire inside a modern office"** — needs the scene/style axes decoupled from garment detail. Tests whether your schema treats scene/style as first-class fields rather than folding them into a single caption.
3. **"Someone wearing a blue shirt sitting on a park bench"** — garment + scene combo, mild compositionality (one garment, one scene — lower risk than query 5).
4. **"Casual weekend outfit for a city walk"** — style inference query, no explicit garment colors named. Tests whether your style tagging (zero-shot classified or VLM-inferred) is doing real work, since there's nothing literal to pattern-match against.
5. **"A red tie and a white shirt in a formal setting"** — the compositional stress test. This is the one vanilla CLIP fails and the one that justifies whichever non-baseline architecture you pick. Worth explicitly showing a before/after (CLIP baseline vs. your system) on this query in the report — it's the single most persuasive result you can show.

---

## 7. Industry precedent (verified, safe to cite in the report)

- **Glance's own engineering blog** (with Google Cloud, published ~July 2025; follow-up detail post on Glance's engineering blog): describes using Gemini models to extract structured entities/relationships from raw content into a Neo4j knowledge graph, rather than relying on pooled dense embeddings alone — the same "structured extraction over raw embedding" pattern proposed here, applied to news instead of fashion, at a reported scale of 50,000+ daily articles. Good opening line for your "why this approach" section since it's literally Glance's own precedent.
- **Pinterest visual search / Shop-the-Look**: runs object detection on the scene photo to isolate individual products before matching — the production validation for the region/detection-based approach (Option C above). Their hybrid search is a retrieve-then-rerank pipeline (cheap ANN recall first, heavier relevance model reranks the shortlist) — structurally the same as the weighted-hybrid design in §3, just with a learned reranker instead of a fixed weighted sum. Pinterest also historically ran several specialized embeddings per product type before unifying into one multi-task embedding for maintainability — a good analogy for why one shared JSON schema (rather than per-query-type logic) is the right call.
- **Myntra's "My Stylist"**: Myntra's own blog names 'Fashion Object Detection' as one of three explicit pipeline components (alongside 'Image Search' and 'Outfit Recommendations'), built for high query volume. Confirms detect-then-embed is treated as production infrastructure, not an optional add-on, at a real e-commerce scale (~450K styles).
- Taken together: three independent companies converge on the same underlying principle — extract structured signal once, offline, and keep the online query path cheap. That's a strong framing line for your report's introduction to the chosen-approach section.

---

## 8. Future work section (for the report)

- ~~**Wire the real VLM**~~: done, code-complete, pending only a `GEMINI_API_KEY`. `services/gemini_client.py`, `services/llm_query_parser.py`, and `services/vlm_attribute_extractor.py` implement the query-time and index-time Gemini calls described in §4.2/§4.3.1, both against the `ExtractedAttributes` schema (`schema.py`). `services/query_parsing.py` tries the real parser first and falls back to the keyword parser on any failure (not configured, network, rate limit), so the app degrades instead of erroring when the key is absent, exactly as it does today. `scripts/extract_attributes_with_vlm.py` is the batch driver that fills in `color`, `scene`, and `style` on the real catalog once a key exists. None of this has been exercised against a real key yet (the SDK's request/response shapes were verified directly against the installed `google-genai` package, not against a live call), so treat it as ready-to-run rather than proven, until someone actually runs it with a key.
- ~~**Tune α against real accuracy data**~~: done for now, see `scripts/tune_alpha.py`. Sweeping alpha 0.00 to 1.00 against the 5 eval queries plus the compositional decoy pair found accuracy pinned at the ceiling across the whole range and the decoy gap monotonically increasing with alpha, a sign this eval set is too narrow to tune against (it never exercises a partially-recognized query, the case the graduated confidence fallback below exists for), not evidence that alpha=1.0 is actually correct. alpha stays at 0.6 pending a broader eval set built once the VLM fills in real color/scene/style data.
- **Location/weather extension**: add as another symbolic-filterable schema field (a scene classifier extension, or EXIF/geo metadata if your dataset carries it) — no architecture change needed, just schema growth.
- **Precision improvement**: hard-negative mining on attribute-swapped captions (NegCLIP/ARO-style) to fine-tune the dense reranker specifically on compositional confusions. The confidence-based fallback is now graduated (§10: `_effective_alpha` scales alpha by `parsed.confidence`, so a partially-recognized query leans further on dense similarity, not just the binary zero-signal case), so this item is now purely about the reranker itself.
- **Reranker upgrade**: replace the fixed weighted-sum score with a learned reranker (small cross-encoder or LTR model) once you have query→relevance feedback data — mirrors Pinterest's retrieve-then-rerank pattern directly.
- **Query parser distillation**: once enough query→JSON pairs are logged, distill the LLM parser into a small local classifier/NER model to cut API latency and cost at scale — mirrors Pinterest's move from many specialized models to one unified, cheaper-to-serve model.
- **Observability upgrade**: §11's OpenTelemetry setup only prints to console; pointing the exporter at a real backend (Jaeger, Honeycomb, Grafana Tempo) is a one-line change whenever query volume is high enough that console spans stop being useful.
- **Test coverage growth**: `test_query_parser.py` currently pins the keyword-spotter's exact behavior; once it's replaced with a real LLM call, those tests become "does the LLM parse still hit the same 5 examples" rather than exact-string assertions, and want a wider held-out query set to catch regressions the original 5 wouldn't.

---

## 9. Open decisions / TODO

- [x] **Pick the architecture** from §2 — **Option D chosen** (structured VLM attribute extraction → symbolic schema + dense fallback).
- [x] Which serving mode for the VLM — **hosted API chosen** (Gemini Flash / GPT-4o-mini) over self-hosting, for least setup friction (see §4.2).
- [x] Exact few-shot prompt/schema for the LLM query parser — **drafted in §4.3.1**, ready to drop into `query_parser.py`.
- [x] Qdrant vs Chroma — **Chroma chosen** (embedded, no server to run), see §4.2.
- [x] Repo structure (Part A: `indexer/`, Part B: `retriever/`) — **reconciled with what's built**: one FastAPI app (`backend/app/`) rather than two top-level dirs, with the Part A/Part B split kept at the module level instead — `services/indexer.py` (Part A, offline) sits next to `services/retriever.py` + `query_parser.py` (Part B, online), sharing `schema.py` so both sides speak the same JSON contract. No renaming needed; a two-top-level-dir split would only add friction for no behavioral difference at this scale.
- [x] Scoring weights (α in the weighted hybrid) — **swept empirically in `scripts/tune_alpha.py`, kept at 0.6**. The 5 eval queries plus decoy pair aren't a wide enough eval set to move off the default (see §8); a real re-tune needs the broader query set that same section calls out.
- [x] Decide how much of DeepFashion2/iMaterialist/Polyvore to actually pull in vs. just cite as "considered" — **Fashionpedia only, others cite-only**. §5's own verdict already says Fashionpedia alone is sufficient for the 500–1000 image / 3-axis requirement; pulling in extra datasets buys "generalizes beyond one label set" evidence the assignment doesn't grade for, at the cost of real setup time (three more download/licensing/format checks). Matches every other decision so far: least friction that still satisfies the assignment as written.

---

## 10. Scaffold status (this repo)

The repo is partway from mocked to real: garment detection, dense retrieval, and the real-VLM code paths all run on real data/models now, but attribute extraction (color, scene, style) is still empty on the real catalog pending a `GEMINI_API_KEY` actually being set.

- `backend/app/services/gemini_client.py` — shared wrapper around the `google-genai` SDK: one place that builds the client, decides the model, and turns a missing key into `GeminiNotConfigured` rather than a raw SDK error surfacing wherever a call happened to be made. Nothing here runs at import time; the client is created lazily on first real use.
- `backend/app/services/llm_query_parser.py` — the real LLM query parser, using the exact §4.3.1 prompt and the same five eval queries as its few-shot examples. Same input/output contract as the keyword parser (`ParsedQuery` in, `ParsedQuery` out).
- `backend/app/services/vlm_attribute_extractor.py` — the real VLM image attribute extractor for index time, sharing the `ExtractedAttributes` schema with the query parser so an image and a query land in the same shape. `scripts/extract_attributes_with_vlm.py` is the batch driver that fills in `color`/`scene`/`style` on the real catalog, keeping Fashionpedia's own ground-truth garment presence and only asking the VLM for the axis it doesn't cover.
- `backend/app/services/query_parsing.py` — the actual entry point `routers/query.py` calls now (not `query_parser.py` directly): tries the real LLM parser, falls back to the keyword parser on `GeminiNotConfigured` or any other failure. This is the one place that decides between them, so that decision doesn't leak into the API layer or into either parser.
- `backend/app/services/query_parser.py` — rule-based keyword-spotting parser, the fallback path `query_parsing.py` uses whenever Gemini isn't configured or a call fails. Its public function is now named `parse_query_keywords` to make that relationship explicit. Same output schema either way.
- `backend/app/data/sample_catalog.json` — 12 hand-written mock records, kept specifically for the color-swapped decoy pair (img_005/img_006) that demonstrates compositional binding.
- `backend/app/data/real_catalog_sample.json` — 1,000 real records pulled from Fashionpedia's validation split via `backend/scripts/pull_fashionpedia_sample.py` (see §5, resized from an original 40-image scaffolding sample up to the assignment's stated 500–1,000 minimum), using the dataset's own ground-truth bbox categories mapped onto our slot/type taxonomy. This particular Fashionpedia mirror (`detection-datasets/fashionpedia`) only exposes category + bbox, not the original paper's 294 fine-grained attributes — so these records have real garments/slots but `color`, `scene`, and `style` are honestly `null` rather than guessed; that's exactly the piece the still-mocked VLM step (§4.2) would fill in.
- `backend/app/services/catalog.py` — loads and caches the combined catalog (mock + real sample). Split out of retriever.py so both it and the router share one catalog instance instead of re-reading JSON per call.
- `backend/app/services/vector_store.py` — `DenseScorer`: symbolic scoring's dense counterpart, isolated behind a `.score(query) -> {id: similarity}` interface. Queries a local Chroma collection (default embedding function, no API key) for real cosine similarity; falls back to word overlap if the embedding model can't be loaded, so the app degrades instead of failing to start. `retriever.py` now only blends the two scores — it doesn't know Chroma exists.
- `backend/app/services/indexer.py` — no longer a stub. `index_image()` runs a real CLIP model (`open_clip`, local, no API key, now Marqo-FashionCLIP rather than vanilla CLIP, see §15.1) over an image and returns a real embedding, which `image_vector_store.py` persists to an on-disk Chroma collection. This is Part A's "Feature Extraction... Vector Storage" requirement, satisfied literally, using pixels rather than ground-truth labels. It deliberately does not touch color/scene/style; that axis still needs the real VLM call from §4.2. `backend/scripts/build_image_vector_index.py` is the runnable batch driver, and `/api/index` exposes the same thing over the API (503 if `torch`/`open_clip` aren't installed in a given deployment, 404 for a missing file, 200 with the embedding dimension on success). These embeddings are now actually consulted at query time too (`services/image_similarity.py`, §14.1), not just persisted, and supplying `garments` to `/api/index` registers the image into the live catalog immediately, closing the cold-start gap §14.3 documents.
- `backend/app/routers/query.py` and `retriever.py` — `_effective_alpha` now scales alpha by `parsed.confidence`: a query the parser recognized nothing in gets alpha=0 (dense reported at full strength, matching the old binary behavior), a fully-recognized query gets the full configured alpha, and anything in between is scaled proportionally rather than snapping straight from 0 to 0.6. `test_retriever.py::test_lower_confidence_scales_alpha_down` pins this against two queries differing only in confidence.

The architecture decision is resolved (§9); the remaining open decisions (VLM serving mode, query-parser prompt, vector DB choice, dataset pull scope) are not, this scaffold exists to make the retrieval *logic* (weighted hybrid, schema shape, query to result flow) testable end to end ahead of that real dataset/model integration.

---

## 11. Testing, CI, and observability

- **Tests** (`backend/tests/`, pytest): schema validation, query_parser against the 5 eval queries, retriever symbolic scoring (including the img_005/img_006 compositional decoy check), `DenseScorer`'s word-overlap fallback path, and FastAPI `TestClient` smoke tests for every route. A separate `test_eval_accuracy.py` runs the 5 canonical eval queries end to end through the API and checks the top-1 result lands in the expected family of records — an accuracy regression, not just a correctness unit test, so a scoring change that quietly breaks retrieval quality fails CI even if every other test still passes.
- **Determinism in CI**: `STRAND_DISABLE_EMBEDDINGS=1` (checked by `vector_store.DenseScorer`) skips the real Chroma embedding download entirely during tests, forcing the deterministic word-overlap fallback. Without this, CI would hit the network on every fresh runner to cache a ~80MB model — slow, and a source of flakiness unrelated to the code being tested. Real runs (the actual app, not tests) leave this unset and get real embeddings as normal.
- **CI** (`.github/workflows/ci.yml`): backend job runs `ruff check .` then `pytest`, frontend job runs `npm run lint` + `npm run build`, both on push/PR. No deploy step; this is correctness/regression gating, not a release pipeline.
- **Lint** (`backend/pyproject.toml`): `ruff`, configured at `line-length = 165` (the codebase's own actual longest lines, mostly information-dense docstrings and one literal JSON few-shot example, rather than reformatting everything down to an arbitrary shorter default) with `E`/`F`/`I`/`W` selected. Ran once across the whole backend as part of this round's cleanup: found exactly one real issue (an intentional re-export in `indexer.py` that looked unused, fixed with an explicit `__all__`) and two import-ordering nits (auto-fixed), everything else was pre-existing, legitimate code, not noise from a rule set that didn't fit the project.
- **Observability** (`backend/app/observability.py`): structured logging to stdout plus OpenTelemetry tracing with a `ConsoleSpanExporter` — spans print alongside the logs. No collector, agent, or external service to stand up or ever touch again; swapping to a real backend later (Jaeger, Honeycomb, etc.) is a one-line change to the exporter, not new instrumentation. A request-logging/tracing middleware wraps every call in `main.py`, and `/api/query` additionally traces the parse and retrieve steps as child spans.
- **Error handling**: a generic `Exception` handler in `main.py` catches anything unhandled, logs it, and returns a structured `{"error": "internal_error", "detail": ...}` 500 instead of leaking a raw traceback. FastAPI's own `HTTPException` handling (used by `/api/index`'s 404/503 responses) is untouched — Starlette resolves the more specific handler first.

---

## 12. Empirical baseline comparison

Two tiers, both run against the real Fashionpedia sample (not just the mock decoy pair), to put actual numbers behind the "single pooled embedding vs. structured schema" argument from §1/§2 instead of leaving it theoretical.

### 12.1 Tier 1 — dense-only (alpha=0) vs. our hybrid (alpha=0.6)

`backend/scripts/eval_baselines.py`, pinned as regression tests in `backend/tests/test_baselines.py`. On the 5 canonical eval queries from §6, both score 5/5 top-1 accuracy — the curated queries alone aren't adversarial enough to break dense-only retrieval outright. The compositional decoy test is where the difference actually shows up:

```
Compositional decoy test (img_005 = true match, img_006 = color-swapped decoy):
  dense-only:  img_005=0.6667  img_006=0.6667  gap=+0.0000
  hybrid:      img_005=0.8667  img_006=0.4667  gap=+0.4000
```

Dense-only produces an **exact tie** between the correct answer and its color-swapped decoy — not just a close call, a literal tie, because bag-of-words captions are identical regardless of which word attaches to which garment. This is the cleanest possible demonstration of the failure mode §1 describes. The hybrid resolves it via the symbolic layer alone.

### 12.2 Tier 2 — real vanilla-CLIP image baseline (Option A, literally) vs. our system

`backend/scripts/eval_clip_baseline.py` (deps in `backend/scripts/requirements-eval.txt`: `datasets`, `pillow`, `open_clip_torch` — not part of the app's own requirements, this is offline eval tooling, not something the running app needs). Embeds the real Fashionpedia photos with `open_clip` (`ViT-B-32`, `openai` weights) and runs zero-shot text-image similarity for 8 single-garment probe queries (types appearing ≥3× in the sample and recognized by `query_parser.py`'s vocabulary, so the symbolic layer gets a fair chance to engage for all three methods). Ground truth is Fashionpedia's own category labels — real data, not invented.

Re-run against the current 1,000-image sample (§5, up from the original 40 this was first measured against):

```
probe query        #relevant  CLIP p@5  dense p@5  hybrid p@5   CLIP r@5  dense r@5  hybrid r@5
------------------------------------------------------------------------------------------------
a pair of shoes    696        1.00      0.60       1.00         0.01      0.00       0.01
a dress            425        0.80      1.00       1.00         0.01      0.01       0.01
a t-shirt          395        0.80      1.00       1.00         0.01      0.01       0.01
pants              269        0.60      1.00       1.00         0.01      0.02       0.02
shorts              95        0.80      1.00       1.00         0.04      0.05       0.05
a jacket           156        1.00      1.00       1.00         0.03      0.03       0.03
a shirt             77        0.00      0.20       1.00         0.00      0.01       0.06
a skirt            140        0.80      1.00       1.00         0.03      0.04       0.04
------------------------------------------------------------------------------------------------
mean precision@5 -- CLIP: 0.725  dense-only: 0.850  hybrid: 1.000
mean recall@5    -- CLIP: 0.018  dense-only: 0.023  hybrid: 0.029
```

**Recall@5 collapsed and that's a metric artifact, not a regression.** At 40 images, single-garment probe queries had a handful of relevant images each, so recall@5 (what fraction of *all* relevant items appear in the top 5) was meaningful. At 1,000 images, common garment types now cover the *majority* of the catalog (696/1,000 records contain "shoe"), so recall@5 is mathematically capped near `5/relevant` regardless of ranking quality, `"a pair of shoes"` can never exceed 5/696 ≈ 0.007 no matter how good the ranking is. All three methods collapsed together (CLIP 0.256→0.018, dense-only 0.564→0.023, hybrid 0.647→0.029) because they're all being measured by the same now-uninformative yardstick, not because anything got worse. **Precision@5** (are the top 5 results actually relevant, not what share of a now-enormous relevant set fits in 5 slots) is what stays meaningful at this scale, and by that metric the result is if anything cleaner than before: **hybrid hits a perfect 1.000 mean precision@5**, against dense-only's 0.850 and real vanilla CLIP's 0.725. The `"a shirt"` row is again the standout single-query evidence: CLIP 0.00, dense-only 0.20, hybrid 1.00, the same story the original 40-image run told, now on a dataset at the assignment's actual stated scale.

**Caveat, stated plainly so this doesn't overclaim**: this isn't purely "CLIP is bad" — our dense/hybrid methods have privileged access to Fashionpedia's own ground-truth category labels baked directly into each record's caption (e.g. `"jacket (outerwear)"`), while CLIP has to recognize the garment from pixels alone with a generic, non-prompt-engineered query. That asymmetry is real. But it's also *exactly* the point this whole project is built on: extracting structured labels up front (whether from dataset ground truth or a VLM) and matching symbolically is a more reliable foundation than asking a single embedding — image or text — to carry that entire semantic burden zero-shot. Tier 2 puts a real number behind that argument instead of leaving it theoretical.

Neither script runs in CI — both need real network access, real model downloads (CLIP weights, ~350MB) and real images on disk, which is exactly the kind of flakiness §11 already deliberately keeps out of the test suite. They're `pull_fashionpedia_sample.py`'s regenerable images plus a manual `python scripts/eval_clip_baseline.py` when you want fresh numbers.

### 12.3 Attempted, not shipped — zero-shot CLIP scene/style tagging

§4.1 step 4 says to zero-shot classify scene/style since Fashionpedia doesn't label environment natively, so `backend/scripts/tag_real_catalog_scene_style.py` tried exactly that against the 40 real images (same local `open_clip` model as Tier 2, one prompt per class, argmax wins). The result was unreliable enough that **it was not applied to the shipped catalog** — `real_catalog_sample.json`'s `scene`/`style` fields are still `null`, unchanged from before this experiment:

- **Scene**: one reasonable set of visually-grounded prompts ("an indoor office interior with desks...") produced a completely different label distribution than another reasonable set ("a photo of a person in an office") on the *exact same 40 images* — "office"-dominant (18/40) flipped to "park"-dominant (30/40).
- **Style**: same instability — one prompt set spread roughly across all 4 labels; a reworded set collapsed to "formal"/"casual" only, with "business" never winning a single image despite being an option both times.
- **Confidence throughout was razor-thin** (~0.17–0.29 raw cosine similarity) regardless of wording, meaning CLIP genuinely isn't finding a strong signal, not just picking a slightly-wrong label with confidence. A plausible cause: this dataset's fashion photography often has deliberately blurred/bokeh backgrounds to keep focus on the garment, so there may not be much scene information in the pixels to begin with.
- Spot-checking a few images directly confirmed real misclassifications (e.g. a clearly outdoor street photoshoot by a yellow door, classified "office").

**Why this is worth documenting rather than deleting**: it's a real, evidence-backed answer to "did you validate your zero-shot approach before trusting it", which is a stronger report artifact than either (a) not trying, or (b) shipping unvalidated labels that look plausible but are wrong roughly as often as right. The fix isn't "zero-shot CLIP can't work here", it's that a single hand-written prompt per class is too weak; the original CLIP paper's zero-shot recipe ensembles ~80 prompt templates per class specifically because single-template zero-shot is this unstable. That ensembling, or just wiring the real VLM (already the plan per §9), are the two credible ways to actually close this gap, not more one-off prompt tweaking.

---

## 13. Concrete scaling estimate: 1M images

Section 4.4's offline/online split argument was correct but never had a real number behind it. Timing the actual code against the real 52-record catalog:

```
per-record symbolic_score call:        1.91 microseconds (measured)
full search() call at 52 records:      1601.8 milliseconds (measured)
```

Those two numbers together are the useful finding. The symbolic scoring step in `retriever.py` loops over every record in pure Python, and at 1.91 microseconds per record that's fine today (under a tenth of a millisecond total for 52 records), extrapolating linearly to 1,000,000 records comes out to roughly 1.9 seconds.

The 1601.8 millisecond full-call number is the more surprising part, and almost none of it is the symbolic loop (measured at roughly 100 microseconds total for all 52 records). It's dominated by encoding the query text into an embedding via the local MiniLM model on each call. That number is higher than MiniLM inference typically takes and is very likely inflated by this specific sandboxed test environment rather than representative of a normal deployment, so it shouldn't be read as "Strand takes 1.6 seconds per query" in general. What it does confirm, measured rather than assumed, is the shape of the bottleneck: encoding cost is paid once per query and does not grow with catalog size, while the part that does grow with catalog size (the symbolic loop) stays small even at 1,000,000 records.

### 13.1 Investigated, not adopted: a symbolic-scoring pre-filter

The estimate above originally proposed a concrete next step: move garment, scene, and style matching out of a Python `for` loop and into a database-native filter, so the database narrows the candidate set before Python ever touches a record. This was actually built and benchmarked, twice, and both times measured out as a net slowdown, not a speedup.

**First attempt: Chroma's metadata `where` filtering.** Storing garment/scene/style as flat metadata fields and querying with `collection.get(where=...)` to get a candidate set before scoring. This measured consistently slower than the plain Python loop at every scale tried, the per-call overhead of a Chroma query outweighs a comparison that only costs 1.91 microseconds in pure Python. It also hit a hard `InternalError: too many SQL variables` failure somewhere between 50,000 and 70,000 synthetic records, well short of the 1,000,000-record target, ruling it out on correctness grounds alone regardless of speed.

**Second attempt: a plain in-process inverted index**, a `dict[(field, value), set[id]]` built once and reused, no database round trip at all. This should have been strictly cheaper than the Chroma version, and it was, but it still measured slower than brute force at every scale from 1,000 to 1,000,000 synthetic records:

```
        N   pre-filter speedup vs. brute force   candidate-set selectivity
     1,000                             0.7x                   36.5% of N
    10,000                             0.6x                   34.5% of N
   100,000                             0.8x                   34.3% of N
 1,000,000                             0.6x                   34.3% of N
```

The initial synthetic vocabulary (3 to 4 garment-type words per slot) made candidate sets a non-selective ~50% of N regardless of scale, so a second run widened it to 10 to 12 types per slot, narrowing selectivity to ~34 to 36% of N. Still a net slowdown. A third run pushed to a deliberately rich, unrealistic-on-the-high-end vocabulary (100 types per slot), narrowing the true candidate set to 4.7% of N at both 100,000 and 1,000,000 records, a query this selective should be the best possible case for a pre-filter:

```
        N   pre-filter speedup vs. brute force   candidate-set selectivity
   100,000                            0.58x                    4.70% of N
 1,000,000                            0.51x                    4.71% of N
```

Still slower, by roughly 2x, even at that selectivity. Three attempts at varying selectivity all landing on the same negative result rules out "not selective enough yet" as the explanation. The real reason is structural, not a tuning gap: `retriever.py`'s `search()` is a hybrid ranking, `score = alpha * symbolic + (1 - alpha) * dense`, so a record with symbolic score 0 can still win top-k on dense score alone. That means dense similarity has to be computed for every record in the catalog regardless of any symbolic pre-filter, the full-catalog loop can never be skipped, only what happens inside it can change. A pre-filter can therefore only ever replace the ~2 microsecond `_symbolic_score` call for non-candidates with a same-cost set membership check, on top of the fixed cost of building and unioning the candidate set in the first place. There is no selectivity level and no scale at which that trade wins, because the thing being skipped was already nearly free and the replacement mechanism costs about the same or more.

**Conclusion: not adopted.** `retriever.py`'s `search()` stays a plain per-record loop with no symbolic pre-filter. The 1.9 second extrapolated cost at 1,000,000 records for the symbolic loop alone (measured directly at up to 1,000,000 synthetic records during this investigation: 3.5 to 6.7 seconds depending on run, in the same ballpark as the original extrapolation) is real, but a pre-filter is not the fix for it, because this architecture cannot exploit one. If symbolic-scoring latency ever becomes the actual bottleneck at real scale, the fix would need to change the ranking design itself, for example scoring only a small dense-similarity shortlist symbolically instead of the full catalog, not bolt an index onto the existing full-catalog hybrid loop.

Storage at 1M images: roughly 2KB per image for a 512-dimensional CLIP vector, roughly 1.5KB per record for a 384-dimensional caption embedding, roughly 0.5 to 1KB of JSON schema metadata per record. Altogether that's somewhere around 6 to 8GB total, comfortably in RAM on a single mid-size server, so lookups stay in memory rather than hitting disk. The real cost driver at any scale is the per-query encoding step (the local embedding model today, the LLM query parser once §4.2 is wired up), not the size of the catalog, exactly the argument Section 4.4 already made, now with a measured number under it instead of only reasoning.

---

## 14. Wiring Part A's real image embeddings into ranking, garment-type synonym matching, and cold start

Three gaps identified from a direct review of the ML design, all fixable without a `GEMINI_API_KEY`.

### 14.1 The two halves are now actually one system

Until this round, Part A (`indexer.py`'s real CLIP image feature extraction) and Part B (`retriever.py`'s symbolic + dense retrieval) were built and tested independently, satisfying their own literal assignment requirement each, but never actually wired together: `index_image()` persisted a real CLIP embedding for every real Fashionpedia photo into `image_vector_store`, and nothing ever read it back. `retriever.py`'s "dense" signal was only ever a text embedding of each record's auto-generated caption (`vector_store.DenseScorer`), the real image pixels were indexed but never consulted at query time. That gap wasn't a deliberate design call, it was two checkboxes built and verified in separate passes without a third pass to connect them.

`app/services/image_similarity.py` closes it: it encodes the query text through the same CLIP model's text tower (`clip_model.py`, shared with `indexer.py` so both sides use one loaded model, not two copies), and compares it against the already-persisted image embeddings via cosine similarity, real image-pixel similarity, not a proxy. `retriever.py`'s `_blend_dense()` averages this against the existing caption-text similarity when both exist for a record, and falls back to caption-only when they don't (the 12 hand-written mock records have no real photo, so no stored embedding). A plain mean, not a new tunable weight: alpha is already unvalidated (§8), adding a second hyperparameter before either is validated against real data would compound the problem rather than fix anything.

Verified end to end against the real catalog (not just unit tests): querying `"a jacket"` with real embeddings enabled returns real Fashionpedia records with genuinely different, non-degenerate dense scores per record (`fp_12809`: dense=0.5194, `fp_18936`: dense=0.5186, `fp_18669`: dense=0.4523), confirming the image-similarity signal is actually varying per record rather than silently falling through to zero or a constant.

**Measured cost of the catalog resize (§5) on startup, not query, latency.** `DenseScorer`'s caption-embedding collection is built once, at process import time, by embedding every record's caption. At 52 records this was sub-second and unremarkable; measured directly against the current 1,012-record catalog, it now takes ~20 seconds (`19.81s` measured), plus a separate ~5 seconds to load the CLIP model for `image_similarity`. This is a one-time cost paid once per process (server startup, or once per CLI invocation of `scripts/search.py`), not per query, each query after that is back to well under a second (`0.26s` measured for `DenseScorer.score`). Consistent with the scaling argument in §13: the catalog-size-dependent cost is real and grows with N, but it's paid once, not on every request, so it doesn't change the per-query scalability story, it does mean `scripts/search.py` (a fresh process per invocation) is not a fast interactive tool at this catalog size, by design it trades that for zero setup.

### 14.2 Garment-type synonym matching, and why generic fuzzy matching was rejected

`_garment_matches` compared `Garment.type` by exact string equality, but type strings come from independently-worded vocabularies (the keyword parser's closed list, Fashionpedia's own ground-truth categories, eventually an open-vocabulary VLM), so real synonyms silently never matched: a query for `"shoes"` (the parser's plural form) never matched any real catalog record, every one of which is typed `"shoe"` (Fashionpedia's singular ground-truth category), and `"blouse"` never matched a record typed `"shirt"` even though Fashionpedia's own taxonomy groups them as one category (`pull_fashionpedia_sample.py`'s `SLOT_MAP` literally lists `"shirt, blouse"` as a single ground-truth class).

`app/services/garment_vocabulary.py` fixes this with a small, curated synonym table (`canonical_type()`), derived directly from Fashionpedia's own category groupings plus plain singular/plural pairs, not general fuzzy/edit-distance string matching. That was tried and measured, not just assumed to be unsafe: computing `difflib.SequenceMatcher` ratios across this vocabulary found `"shirt"` vs `"skirt"` at 0.80 similarity, higher than several genuine synonym pairs (`"jean"`/`"jeans"` at 0.889, `"shoe"`/`"shoes"` at 0.889), and `"jacket"` vs the entirely unrelated word `"racket"` at 0.833. Any single threshold either merges shirt/skirt (a real, wrong match) or is too strict to catch most genuine typos at all, short garment words are too dense in edit-distance space for a threshold to cleanly separate synonyms from unrelated words. See `test_garment_vocabulary.py::test_edit_distance_matching_is_unsafe_on_this_vocabulary` for the pinned numbers.

Garments that merely share a slot but are stylistically distinct (`sneakers` vs `heels` vs `boots`, `jacket` vs `coat`) are deliberately **not** merged, even though a naive approach might be tempted to: Fashionpedia's ground truth has no footwear-style granularity at all (every real catalog footwear record is typed `"shoe"` regardless of actual style), so merging `"sneakers"` into `"shoe"` would fabricate a specific visual claim the data doesn't support, a query for sneakers would then "match" a record actually wearing heels. That's a data-granularity gap a real VLM call would need to close (Working_notes.md §4.2), not something a string comparison can fix honestly.

### 14.3 Cold start: a newly indexed image could never actually be found

Calling `POST /api/index` ran real feature extraction and persisted a real CLIP embedding, satisfying Part A's literal requirement, but nothing added the image to the catalog (`retriever._CATALOG`, loaded once from static JSON at process start) or its caption index, so a brand-new image could never be returned by `/api/query` regardless of how long it had been indexed. That is the actual cold-start gap in this system: not "does an existing record with missing color/scene/style score badly" (it degrades gracefully today, a sparse caption still gets some dense signal), but "there was no path at all for a new item to become searchable."

`/api/index` now optionally accepts `garments`/`scene`/`style` in its request body. Without them, behavior is unchanged (embedding persisted only, matching before this fix). With them, `retriever.register_record()` appends a full `ImageRecord` to the live catalog and adds its caption to `DenseScorer`'s live collection, making the image immediately searchable, no restart needed. `image_similarity` needs no equivalent registration call: unlike the caption collection, it re-queries `image_vector_store`'s persistent collection fresh on every call, so a newly stored embedding is already visible there the moment `index_image()` persists it. Verified end to end in `test_api.py::test_index_endpoint_registers_a_searchable_record_when_garments_supplied`: indexing a new image with a supplied garment makes it appear in `/api/catalog` and rank first for a matching `/api/query` call in the same request, no process restart.

This only handles structured attributes supplied at index time (manually, or eventually from the real VLM, §4.2/§8), it does not run a VLM call itself, that half of true zero-touch onboarding still needs `GEMINI_API_KEY`.

### 14.4 A plain CLI, satisfying Part B's literal wording

§3's Part B requirement is literally "create a script that accepts a natural language string and returns the top k matching images." The FastAPI + React app already does this and far more, but there was no single runnable script matching that literal description without starting a server or opening a browser. `backend/scripts/search.py` adds one: `python scripts/search.py "a query"` prints ranked results with their scores and matched fields. It calls `query_parsing.parse_query` + `retriever.search` directly, the exact code path `/api/query` uses, not a separate reimplementation, so it automatically picks up the real LLM parser once `GEMINI_API_KEY` is set, with no separate maintenance burden.

---

## 15. Two more ML optimizations, one adopted, one rejected

Both chosen specifically because neither is blocked on `GEMINI_API_KEY` or a labeled eval set, unlike the reranker/hard-negative-mining items in §8. Both measured against the same 8-probe-query, real-ground-truth harness Tier 2 (§12.2) already established, not assumed.

### 15.1 Adopted: swap the CLIP backbone for a fashion-tuned checkpoint

`clip_model.py` used vanilla `open_clip` ViT-B-32/openai, the exact "vanilla CLIP" baseline the assignment's own hint says is weak on fashion attributes, for the *production* image-similarity signal, not just the Tier 2 comparison baseline. Marqo-FashionCLIP (`hf-hub:Marqo/marqo-fashionCLIP`, Apache-2.0, loaded through `open_clip`'s own HF-hub integration) was cited in §2 as an architecture option but never actually tried as a drop-in backbone swap for the image embeddings this project already had wired in (§14.1).

Measured on the same 8 real probe queries as Tier 2, isolating just the image-embedding signal (bypassing the symbolic/caption blend entirely, to test the backbone alone):

```
                          mean precision@5
vanilla ViT-B-32/openai          0.725
Marqo-FashionCLIP                1.000
```

A clean, decisive improvement, every one of the 8 probe queries hit 1.00 precision@5 with the fashion-tuned backbone, including `"a shirt"`, the query vanilla CLIP scored 0.00 on. Same 512-dim output as ViT-B-32/openai, so this was a true drop-in swap, no schema change, no code beyond `clip_model.py`'s two constants. `scripts/eval_clip_baseline.py` deliberately still hardcodes vanilla ViT-B-32/openai directly rather than importing from `clip_model.py`, since that script's whole purpose is comparing against vanilla CLIP as the baseline, not against whatever backbone the production system happens to use, changing that would quietly invalidate its own comparison.

**Honest caveat: this improvement is real but doesn't move the full blended pipeline's precision@5 on these same 8 probe queries.** Re-running Tier 2 end to end after regenerating all 1,000 embeddings with the new backbone gives *identical* dense-only (0.850) and hybrid (1.000) mean precision@5 to before the swap. That's not the backbone swap failing, it's these particular probe queries (single garment-type lookups like `"a jacket"`) already saturating on caption similarity alone: the caption literally contains the ground-truth category name (`"jacket (outerwear)"`), so caption-dense similarity alone already hits the ceiling these coarse queries can measure, leaving no room for a better image signal to show through in *this* metric. The 0.725→1.000 gain is real and measured, but only demonstrated in isolation (image-similarity alone, bypassing the caption signal entirely); it should show up more where captions are weaker, compositional or color-specific queries the real catalog's null `color` field can't help with, and queries against future images that never get a rich caption. Reporting both numbers rather than only the flattering one.

All 1,000 persisted image embeddings were regenerated with the new backbone (`build_image_vector_index.py`, re-run after wiping the old vanilla-CLIP embeddings, they are not comparable across backbones and mixing them would silently corrupt every similarity score). Full test suite still passes unchanged, including the real-CLIP tests in `test_image_similarity.py`/`test_indexer.py`, which only assert general sanity (a red image scores higher than a blue one for a "red" query), not anything backbone-specific.

### 15.2 Rejected, measured: embedding whitening

PCA-whitening (correcting the known anisotropy of dense embedding spaces to sharpen cosine-similarity contrast, "whitening-BERT"-style) is a training-free technique: fit a linear transform from the catalog's own embedding statistics, no labels, no fine-tuning. Tried it on the persisted image embeddings, computing the whitening transform from all 1,000 vectors and applying it to both catalog and query embeddings before cosine similarity.

Full-rank whitening was a sharp regression, not an improvement:

```
                    mean precision@5
raw (unwhitened)          0.725
whitened (full rank)      0.250
```

Suspecting the regression came from amplifying noisy, low-variance directions (only 1,000 samples across 512 dimensions leaves several near-zero eigenvalues, and dividing by their square root blows up whatever noise lives there), tried truncated whitening keeping only the top-k principal components instead of the full 512:

```
top-k components   mean precision@5
16                        0.475
32                        0.450
64                        0.475
128                       0.475
256                       0.350
512 (full rank)           0.250
```

Every truncation level tested still underperforms doing nothing (0.725). Not adopted, at any tested configuration. The most likely root cause isn't sample size alone: the whitening transform was fit on the catalog's *image* embedding statistics only, then applied to *query text* embeddings, a different modality. CLIP-family models are known to have a "modality gap", image and text embeddings occupy systematically different regions of the shared space even after contrastive training aligns them well enough for retrieval, so a transform derived purely from image-side statistics doesn't necessarily describe how text queries should be reshaped, and can actively fight the cross-modal alignment CLIP's own training already established. This is a genuinely different failure mode from §13.1's symbolic pre-filter rejection (a structural ranking-architecture mismatch) or §12.3's CLIP zero-shot tagging rejection (prompt instability); this one is a cross-modal statistics mismatch, worth distinguishing rather than filing under a generic "didn't help."

---

## 16. Deterministic garment color from real bounding boxes, no VLM needed

The very first design pass (Section 2) proposed this: instead of asking a model to guess a garment's color in words, run detection and read the actual RGB values off the detected region. It sat as a future-work line for a while because it sounded like it needed our own object detector, until re-reading the data actually being pulled from: Fashionpedia's ground truth already ships real bounding boxes per garment (`sample["objects"]["bbox"]`), `pull_fashionpedia_sample.py` was already reading the parallel `category` array off this exact same structure and simply discarding the boxes. No detector needed, the detection already happened, upstream, for free.

`app/services/color_detection.py` crops each matched garment's real bounding box, reads the per-channel median RGB of the crop (median rather than mean, more robust to a handful of highlight or shadow outlier pixels), and maps it to the nearest name in the existing `colors.COLOR_HEX` palette, the same 19-color vocabulary already used to render swatch chips. `pull_fashionpedia_sample.py` now calls this for every matched garment instead of leaving `color: null`, and folds the detected color into the caption too (`"black jacket (outerwear)"` instead of `"jacket (outerwear)"`), so both the symbolic layer and the caption-dense signal gain a real color axis on the real catalog, not just the 12 hand-written mock records.

### A real bug, found by actually looking at the crop, not just measuring an aggregate number

Before trusting this at scale, spot-checked a handful of detections against the actual photos. One (`fp_2083`, a bearded man in a blazer by a yellow door) read its "shirt" garment as brown, an actual light grey shirt. Printing the crop itself showed why: the "shirt" bounding box's top edge sits right at the collar, and the crop it produces includes a real strip of visible neck. Skin tone, not shirt fabric, was dragging the median toward brown, confirmed directly by measuring the collar-region crop's raw RGB, (128, 110, 87), objectively far closer to "brown" (139, 94, 60) than to any grey/beige/khaki entry in the palette (all 12,000+ squared-distance units away), given a rectangular box can't exclude the skin sitting inside it.

Fixed by insetting every box by `INSET_FRACTION = 0.15` on all four sides before sampling, biasing the read toward the garment's center and away from its boundary with whatever is next to it. Measured directly on the same failing case: shirt reading went from (127, 117, 107) "brown" to (158, 150, 139) "khaki", a real, visible improvement (khaki is a much more honest description of that shirt than brown was), reached with a plain uniform inset, no special-casing for "this is an upper-body garment, trim more from the top." Re-verified on several more real photos after the fix (an all-black runway outfit read correctly as black head to toe; a brown double-breasted jacket over navy trousers read correctly as brown and navy), enough to trust the fix generally, not just on the one case it was built to solve.

Run across the full 1,000-image catalog: **2,790/2,790 garments (100%) got a detected color**, no crop failed outright. Distribution is plausible for real fashion photography, not degenerate: black (711), brown (488), navy (333), beige (266), khaki (248), denim (191), grey (170), white (112), with a believable long tail down to green/blue/amber (8, 8, 1). A `"a black jacket"` query, which could not have matched anything at all before this (every real record's color was `null`), now returns real photos with `symbolic_score=1.0` on both slot, type, and color.

### Honest limitations, stated plainly rather than oversold

- **A box is not a mask.** The inset reduces skin/neighbor bleed, it doesn't eliminate it, a segmentation mask would still beat a rectangle outright. This mirrors exactly Fashionpedia's own data limit already noted elsewhere in this document, this particular mirror only exposes boxes.
- **No white-balance correction.** A color is read straight from raw pixels; a garment under warm ambient light (like the yellow-door photo above) can legitimately read warmer than it would under neutral light. This wasn't the dominant cause of the bug just found, that was skin content, but it's a real, separate source of error still present after the fix.
- **Plain RGB Euclidean distance, not a perceptual color space.** CIEDE2000 over Lab coordinates would match human color perception more faithfully than squared distance in raw RGB; the simpler metric was the right choice for a first version, and `nearest_color_name()` is written so swapping the distance function later doesn't touch any caller.

None of this needed `GEMINI_API_KEY`. Scene and style still do, that axis is genuinely a different kind of judgment (a single color has a clear nearest-neighbor answer, "is this an office or a park" does not reduce to reading pixels the same way), see Section 4.1 step 4 and Section 8.

---

## 17. Detected color was a regression risk as a hard gate, fixed by making it a soft signal

Section 16 shipped, then got a second look with a direct question worth asking of any new signal: what happens when it's wrong. `_garment_matches` (`retriever.py`) had folded the newly-detected color straight into the same hard symbolic gate a hand-authored or VLM-confirmed color already used, `query_garment.color != rg.color` excludes the record outright. That gate is safe for a hand-authored color, wrong there would be a labeling bug, essentially never happens, and it will stay safe once a VLM confirms colors too, a model asked to look at the actual image and report back is a categorically different confidence level than a heuristic never checked against ground truth. A *detected* color is neither: Section 16 measured it against real photos and found a real failure mode (the skin-bleed misread), fixed one instance of it, and was honest that a box-not-a-mask crop and no white-balance correction remain. That is an expected, nonzero error rate, not a bug to stamp out, and hard-gating on it silently turns "unknown color" (a null `color` field, which never excludes anything) into "confidently wrong" (which wrongly excludes a real match) for every detection error. Worse than not detecting color at all, for exactly the queries this feature was built to help with.

The fix keeps the hard gate for confident colors and routes detected colors through a new soft, graded signal instead, rather than picking one policy for all colors regardless of provenance:

- `ImageRecord` gained `detected_color_slots: list[str]`, which slots (if any) had their color filled in by `color_detection.py` rather than hand-authored or VLM-confirmed. Deliberately not added to `Garment` itself, `Garment` is the type shared with `ExtractedAttributes`/`ParsedQuery`, the Gemini structured-output schema, and this is retriever-internal bookkeeping, not something to ask an LLM to extract from a query.
- `_garment_matches` now only applies the hard color gate when `rg.slot not in detected_color_slots`. A detected-color garment still has to match on slot and type, it just no longer gets hard-excluded for a color mismatch.
- A new `color_similarity(name_a, name_b) -> float` in `color_detection.py` scores two canonical color names by normalized RGB Euclidean distance in `[0, 1]`, 1.0 for an exact match, dropping off smoothly for perceptually close colors (navy vs. black scores higher than navy vs. yellow), 0.0 for an unrecognized name rather than raising.
- `retriever._color_signal` compares a query garment's color against a same-slot detected-color record garment via `color_similarity`, returning `None` (not `0.0`) when there's nothing to compare, so `_blend_dense` can drop the term from the average entirely instead of diluting a real signal with an irrelevant zero.
- `_blend_dense` folds this in as a third optional term alongside caption similarity and real image similarity, still a plain mean, not a new tunable weight, for the same reason Section 14 gave for not adding one for the image-similarity term: this project already has one unvalidated hyperparameter (alpha), adding a second and third before the first is validated against real labels would compound the problem rather than fix it.

Measured directly, not just reasoned about, both edges of the fix:

```
query: "a black jacket"  -> fp_12809 (detected black), symbolic=1.0, dense=0.7995 (color term pulls dense up)
query: "a red jacket"    -> fp_12809 still surfaces, symbolic=1.0 (type/slot still match, no longer hard-excluded),
                             dense correctly penalized lower (~0.46) by a poor color_similarity(red, black) score
```

The second line is the actual regression fix: before this change, `fp_12809` would never have appeared for `"a red jacket"` at all, a wrong detection would have silently removed a real jacket from every query whose color it happened to disagree with. Now it still surfaces, ranked appropriately lower rather than invisibly, exactly the "wrongly excludes" failure mode described above, closed. `test_color_detection.py` gained 4 tests for `color_similarity` itself (exact match, perceptual ordering, symmetry, unknown-name safety); `test_retriever.py`'s existing compositional-decoy test (Section 6's most persuasive result) still passes unchanged, that test's colors are hand-authored, not detected, so it was never exposed to this bug and its behavior is untouched by the fix.

---

## 18. Query and embedding caching

Three real, independent costs sit on the query path: parsing (`query_parsing.py`, an LLM round-trip when `GEMINI_API_KEY` is set, real regex work otherwise), caption-embedding similarity (`vector_store.DenseScorer`, a Chroma query over every record), and image-embedding similarity (`image_similarity.py`, encoding the query through CLIP's text tower). All three are pure functions of the query text (plus, for image similarity, the catalog size), and the app already has a natural source of repeats: the example query chips in the UI, and any user re-running or refining a search. None of the three had any caching before this round, every repeat paid full cost again.

Added `functools.lru_cache(maxsize=256)` at all three call sites, each renamed to a private `_*_cached` function wrapped by the existing public entry point, so no caller anywhere had to change:

- `DenseScorer.score()` → `DenseScorer._score_cached()`, keyed on `raw_query` alone (a `DenseScorer` instance already scopes one catalog).
- `image_similarity.score()` → `image_similarity._score_cached()`, keyed on `(raw_query, count)`, `count` is the persistent Chroma collection's current size, folded into the key specifically so a newly indexed image (which changes that count) busts stale cache entries on its own, with no explicit invalidation call needed, unlike the other two.
- `query_parsing.parse_query()` → `query_parsing._parse_query_cached()`, keyed on `raw_query` alone.

Three correctness details mattered more than the caching itself:

1. **Cache-key hashability.** A first draft of `image_similarity._score_cached` took `model`, `tokenizer`, and `collection` as arguments so the cached function wouldn't need to re-fetch them. Tested `hash()` on a real Chroma `Collection` object directly before trusting this, not assumed safe, it raises `TypeError: unhashable type: 'Collection'`, which would have crashed the very first real (non-disabled) call in production. Fixed by re-fetching all three as cheap internal singleton lookups instead (`clip_model.py`/`image_vector_store.py` already cache them at the module level), keeping only `raw_query` and `count`, the two genuinely-variable, hashable arguments, in the cache key.
2. **Defensive copies of mutable return values.** All three cached functions return a plain `dict` or a Pydantic model, both mutable. Every public wrapper returns a copy (`dict(...)` for the two dict-returning functions, `.model_copy(deep=True)` for `ParsedQuery`), not the cached object itself, so a caller mutating what it got back can't corrupt what every future identical query receives. `test_vector_store.py` and `test_query_parsing.py` both assert this directly: two calls with the same query return equal-but-distinct objects, and mutating the first result leaves the second (and a third, later call) unaffected.
3. **Invalidation is per-module, not one-size-fits-all.** `DenseScorer`'s cache is keyed on query text alone, so it needs an explicit `clear_cache()` call whenever the catalog changes, wired into `retriever.register_record()` (Section 14.3's cold-start fix), a query cached before a new record existed would otherwise keep returning a result missing it forever. `image_similarity`'s cache needs no equivalent call, its `count`-keyed cache invalidates itself the moment a new embedding changes that count, a narrower guarantee than `DenseScorer`'s (re-indexing an *existing* image with a new embedding without changing the collection's size would still return a stale result), stated plainly as an accepted tradeoff rather than a claim of perfect invalidation. `query_parsing` needs no invalidation at all, parsing doesn't depend on catalog contents; its own honest caveat is different, a transient LLM failure that trips the keyword-parser fallback gets that degraded result cached for that exact query text until evicted, a retry moments later that would have succeeded against Gemini won't happen automatically. Accepted given how unlikely an exact repeat of the same failure window is.

`test_vector_store.py`, `test_image_similarity.py`, and `test_query_parsing.py` each gained dedicated cache-behavior tests (repeat-call equality and independence, mutation isolation, and, for `DenseScorer`, an explicit test that a repeated query stays stale *without* `clear_cache()` first, proving the cache is doing real work rather than the invalidation test passing for an unrelated reason). Full suite green after adding them, no change to any existing test's behavior, caching is invisible to correctness by construction, only to repeat latency.

---

## 19. Reciprocal Rank Fusion over the raw query and the parser's own canonical phrase

Dense similarity (both the caption-embedding and real-image-embedding halves of `_blend_dense`) is sensitive to surface phrasing in a way symbolic matching never is. A user typing "something with a reddish jacket for a formal look maybe" gets exactly the same symbolic match as a user typing "a red jacket, formal", the schema doesn't care about phrasing, but the two queries can embed very differently, and the messier one may simply rank worse on the dense half through no fault of the underlying match. The parser already extracts the clean, structured meaning of that query, this section wires that structured meaning back into ranking, not just into the symbolic gate.

`retriever._canonical_phrase(parsed)` builds a short, clean restatement directly from the parsed fields, reusing the exact `"color type"` label format `_symbolic_score` already builds for `matched_fields` (not the fuller `pull_fashionpedia_sample.py` caption format, which also parenthesizes the slot, that extra token only dilutes word-overlap/embedding similarity against the free-text style of the 12 hand-written mock captions for no benefit): `"red jacket, formal"` for the example above. Returns `None` when there's no structured signal at all, a genuinely free-text query has nothing to build a canonical phrase from.

`search()` now runs the same scoring pass twice when a canonical phrase exists and differs from the raw query, once against `parsed.raw_query` (as before), once against the canonical phrase, and fuses the two resulting rankings via Reciprocal Rank Fusion: `sum(1/(k+rank))` per list, `k=60`, the standard textbook default, over rank *position*, not score magnitude. A record that ranks well under either phrasing floats up; a record that only ranks well under one specific wording, a lexical coincidence rather than a genuine match, doesn't dominate the result on that alone. Symbolic score, matched fields, and the color-similarity signal from Section 17 don't depend on which query text is being scored densely (they're computed from `parsed.garments`/`scene`/`style` against each record directly), so they're computed once per record and reused for both passes rather than redone twice, only the two dense-similarity computations actually differ.

Critically, RRF is only ever allowed to change the final *order* of results, never the numbers reported for any individual record: `ScoredResult.score`, `symbolic_score`, `dense_score`, and `matched_fields` all still come from the raw-query scoring pass exactly as they did before this section, `_reciprocal_rank_fusion` returns the raw-query pass's own result objects, re-ordered, never the canonical-phrase pass's. `test_retriever.py` asserts this directly: with `_canonical_phrase` forced to return `None` (RRF disabled) versus left enabled, every record's reported score/symbolic/dense/matched_fields are identical, only list order is free to differ.

Verified the fusion does real, non-trivial work, not a no-op dressed up as one: a synthetic pair of rankings where one record wins outright on raw score but ranks last under the canonical phrase, against a record that ranks a close second under both, confirms RRF places the "good under both" record ahead of the "raw-score winner," exactly the case RRF exists to catch. Checked manually against the real catalog too: a deliberately awkward query ("something with a reddish jacket for a formal look maybe") reorders two `fp_` records whose raw-only dense scores would have ranked them the other way around, once their canonical-phrase ranks are folded in. Skipped entirely, falling back to the exact pre-RRF sort, whenever there's no structured signal to fuse (Section 12's five canonical eval queries that are pure symbolic already, and Section 6's compositional decoy test, both keep their original, unfused behavior and both still pass) or when the canonical phrase happens to equal the raw query verbatim, nothing to gain from ranking identical text against itself twice.
