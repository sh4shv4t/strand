# Glance ML Internship Assignment — Working Notes (v2)

Status: all design decisions are locked in (§9), the retrieval pipeline runs on real data (Chroma, real Fashionpedia records, a real CLIP-based Part A indexer, §10), it's measured against real baselines (§12), and it has a measured, not just reasoned, scaling estimate (§13). The one thing still genuinely open is wiring a real vision-language model to fill in color, scene, and style on the real catalog and to replace the keyword-spotting query parser with something truly zero-shot; everything else in the repo is already built against exactly the shape that will take.

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

- **Wire the real VLM**: swap `query_parser.py`'s keyword-spotter for the §4.3.1 prompt against a real hosted API (Gemini Flash/GPT-4o-mini), and run `real_catalog_sample.json`'s records through the same model at index time to fill in `color`, `scene`, and `style`, which are honestly `null` right now. This is the one piece everything else in the repo is scaffolded around and waiting for.
- **Tune α against real accuracy data**: once the VLM fills in the missing fields, re-run `test_eval_accuracy.py`'s 5 queries while sweeping α and pick the value that maximizes top-1 accuracy, instead of the current unvalidated default of 0.6.
- **Location/weather extension**: add as another symbolic-filterable schema field (a scene classifier extension, or EXIF/geo metadata if your dataset carries it) — no architecture change needed, just schema growth.
- **Precision improvement**: hard-negative mining on attribute-swapped captions (NegCLIP/ARO-style) to fine-tune the dense reranker specifically on compositional confusions. The all-or-nothing version of parser-confidence fallback is done (§10: zero recognized signal now uses alpha=0 automatically), but `parsed.confidence` is still only used for the frontend's display value, not for a graduated fallback when the parser recognizes *some* but not all of a query, that's the remaining piece of this item.
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
- [ ] Scoring weights (α in the weighted hybrid) — tune against the 5 eval queries directly, that's your actual eval loop. Left open deliberately: needs a real dataset/VLM in the loop to tune against, not a design call to make ahead of time.
- [x] Decide how much of DeepFashion2/iMaterialist/Polyvore to actually pull in vs. just cite as "considered" — **Fashionpedia only, others cite-only**. §5's own verdict already says Fashionpedia alone is sufficient for the 500–1000 image / 3-axis requirement; pulling in extra datasets buys "generalizes beyond one label set" evidence the assignment doesn't grade for, at the cost of real setup time (three more download/licensing/format checks). Matches every other decision so far: least friction that still satisfies the assignment as written.

---

## 10. Scaffold status (this repo)

The repo is partway from mocked to real: garment detection and dense retrieval now run on real data/models, but attribute extraction (color, scene, style) is still mocked pending a VLM API key.

- `backend/app/services/query_parser.py` — rule-based keyword-spotting parser standing in for the LLM query parser in §4.3. Same output schema, swap the implementation later without touching the API surface.
- `backend/app/data/sample_catalog.json` — 12 hand-written mock records, kept specifically for the color-swapped decoy pair (img_005/img_006) that demonstrates compositional binding.
- `backend/app/data/real_catalog_sample.json` — 40 real records pulled from Fashionpedia's validation split via `backend/scripts/pull_fashionpedia_sample.py`, using the dataset's own ground-truth bbox categories mapped onto our slot/type taxonomy. This particular Fashionpedia mirror (`detection-datasets/fashionpedia`) only exposes category + bbox, not the original paper's 294 fine-grained attributes — so these records have real garments/slots but `color`, `scene`, and `style` are honestly `null` rather than guessed; that's exactly the piece the still-mocked VLM step (§4.2) would fill in.
- `backend/app/services/catalog.py` — loads and caches the combined catalog (mock + real sample). Split out of retriever.py so both it and the router share one catalog instance instead of re-reading JSON per call.
- `backend/app/services/vector_store.py` — `DenseScorer`: symbolic scoring's dense counterpart, isolated behind a `.score(query) -> {id: similarity}` interface. Queries a local Chroma collection (default embedding function, no API key) for real cosine similarity; falls back to word overlap if the embedding model can't be loaded, so the app degrades instead of failing to start. `retriever.py` now only blends the two scores — it doesn't know Chroma exists.
- `backend/app/services/indexer.py` — no longer a stub. `index_image()` runs a real CLIP model (`open_clip`, local, no API key) over an image and returns a real embedding, which `image_vector_store.py` persists to an on-disk Chroma collection. This is Part A's "Feature Extraction... Vector Storage" requirement, satisfied literally, using pixels rather than ground-truth labels. It deliberately does not touch color/scene/style; that axis still needs the real VLM call from §4.2. `backend/scripts/build_image_vector_index.py` is the runnable batch driver, and `/api/index` exposes the same thing over the API (503 if `torch`/`open_clip` aren't installed in a given deployment, 404 for a missing file, 200 with the embedding dimension on success).
- `backend/app/routers/query.py` and `retriever.py` — when the query parser recognizes no structured signal at all (a genuinely zero-shot query relative to its keyword vocabulary), `search()` now uses alpha=0 for that query instead of the fixed default, so the dense similarity score is reported at full strength rather than silently discounted by a weight meant for a symbolic signal that doesn't exist in that case. Ranking was already unaffected either way; this fixes what the match percentage honestly communicates.

The architecture decision is resolved (§9); the remaining open decisions (VLM serving mode, query-parser prompt, vector DB choice, dataset pull scope) are not, this scaffold exists to make the retrieval *logic* (weighted hybrid, schema shape, query to result flow) testable end to end ahead of that real dataset/model integration.

---

## 11. Testing, CI, and observability

- **Tests** (`backend/tests/`, pytest): schema validation, query_parser against the 5 eval queries, retriever symbolic scoring (including the img_005/img_006 compositional decoy check), `DenseScorer`'s word-overlap fallback path, and FastAPI `TestClient` smoke tests for every route. A separate `test_eval_accuracy.py` runs the 5 canonical eval queries end to end through the API and checks the top-1 result lands in the expected family of records — an accuracy regression, not just a correctness unit test, so a scoring change that quietly breaks retrieval quality fails CI even if every other test still passes.
- **Determinism in CI**: `STRAND_DISABLE_EMBEDDINGS=1` (checked by `vector_store.DenseScorer`) skips the real Chroma embedding download entirely during tests, forcing the deterministic word-overlap fallback. Without this, CI would hit the network on every fresh runner to cache a ~80MB model — slow, and a source of flakiness unrelated to the code being tested. Real runs (the actual app, not tests) leave this unset and get real embeddings as normal.
- **CI** (`.github/workflows/ci.yml`): two jobs on push/PR — backend (`pytest`) and frontend (`npm run lint` + `npm run build`). No deploy step; this is correctness/regression gating, not a release pipeline.
- **Observability** (`backend/app/observability.py`): structured logging to stdout plus OpenTelemetry tracing with a `ConsoleSpanExporter` — spans print alongside the logs. No collector, agent, or external service to stand up or ever touch again; swapping to a real backend later (Jaeger, Honeycomb, etc.) is a one-line change to the exporter, not new instrumentation. A request-logging/tracing middleware wraps every call in `main.py`, and `/api/query` additionally traces the parse and retrieve steps as child spans.
- **Error handling**: a generic `Exception` handler in `main.py` catches anything unhandled, logs it, and returns a structured `{"error": "internal_error", "detail": ...}` 500 instead of leaking a raw traceback. FastAPI's own `HTTPException` handling (used by the `/api/index` 501 stub) is untouched — Starlette resolves the more specific handler first.

---

## 12. Empirical baseline comparison

Two tiers, both run against the real 40-image Fashionpedia sample (not just the mock decoy pair), to put actual numbers behind the "single pooled embedding vs. structured schema" argument from §1/§2 instead of leaving it theoretical.

### 12.1 Tier 1 — dense-only (alpha=0) vs. our hybrid (alpha=0.6)

`backend/scripts/eval_baselines.py`, pinned as regression tests in `backend/tests/test_baselines.py`. On the 5 canonical eval queries from §6, both score 5/5 top-1 accuracy — the curated queries alone aren't adversarial enough to break dense-only retrieval outright. The compositional decoy test is where the difference actually shows up:

```
Compositional decoy test (img_005 = true match, img_006 = color-swapped decoy):
  dense-only:  img_005=0.6667  img_006=0.6667  gap=+0.0000
  hybrid:      img_005=0.8667  img_006=0.4667  gap=+0.4000
```

Dense-only produces an **exact tie** between the correct answer and its color-swapped decoy — not just a close call, a literal tie, because bag-of-words captions are identical regardless of which word attaches to which garment. This is the cleanest possible demonstration of the failure mode §1 describes. The hybrid resolves it via the symbolic layer alone.

### 12.2 Tier 2 — real vanilla-CLIP image baseline (Option A, literally) vs. our system

`backend/scripts/eval_clip_baseline.py` (deps in `backend/scripts/requirements-eval.txt`: `datasets`, `pillow`, `open_clip_torch` — not part of the app's own requirements, this is offline eval tooling, not something the running app needs). Embeds the actual 40 real Fashionpedia photos with `open_clip` (`ViT-B-32`, `openai` weights) and runs zero-shot text-image similarity for 8 single-garment probe queries (types appearing ≥3× in the sample and recognized by `query_parser.py`'s vocabulary, so the symbolic layer gets a fair chance to engage for all three methods). Ground truth is Fashionpedia's own category labels — real data, not invented.

```
probe query        #relevant  CLIP r@5   dense-only r@5   hybrid r@5
----------------------------------------------------------------------
a pair of shoes    25         0.12       0.12             0.12
a dress            19         0.26       0.26             0.26
a t-shirt          15         0.27       0.33             0.33
pants              8          0.00       0.62             0.62
shorts             6          0.67       0.83             0.83
a jacket           5          0.40       1.00             1.00
a shirt            3          0.33       0.33             1.00
a skirt            3          0.00       1.00             1.00
----------------------------------------------------------------------
mean recall@5 -- CLIP: 0.256  dense-only: 0.564  hybrid: 0.647
```

Real vanilla CLIP scores worse than even our word-overlap dense fallback, and well below the hybrid. **Caveat, stated plainly so this doesn't overclaim**: this isn't purely "CLIP is bad" — our dense/hybrid methods have privileged access to Fashionpedia's own ground-truth category labels baked directly into each record's caption (e.g. `"jacket (outerwear)"`), while CLIP has to recognize the garment from pixels alone with a generic, non-prompt-engineered query. That asymmetry is real. But it's also *exactly* the point this whole project is built on: extracting structured labels up front (whether from dataset ground truth or a VLM) and matching symbolically is a more reliable foundation than asking a single embedding — image or text — to carry that entire semantic burden zero-shot. Tier 2 puts a real number behind that argument instead of leaving it theoretical; the `"a shirt"` row (dense-only 0.33 → hybrid 1.00) is the cleanest single-query evidence that the symbolic layer earns its keep beyond just the decoy test.

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

Those two numbers together are the useful finding. The symbolic scoring step in `retriever.py` loops over every record in pure Python, and at 1.91 microseconds per record that's fine today (under a tenth of a millisecond total for 52 records), extrapolating linearly to 1,000,000 records comes out to roughly 1.9 seconds. That's the part of the system that actually scales with catalog size, and the part that would need to change before going much past a few thousand records: move garment, scene, and style matching out of a Python `for` loop and into Chroma's own metadata `where` filtering (storing them as flat metadata fields instead of only folding them into the caption string), so the database does the filtering natively instead of Python iterating every record on every request. This is the one concrete code change the system needs before real scale, not a rewrite.

The 1601.8 millisecond full-call number is the more surprising part, and almost none of it is the symbolic loop (measured at roughly 100 microseconds total for all 52 records). It's dominated by encoding the query text into an embedding via the local MiniLM model on each call. That number is higher than MiniLM inference typically takes and is very likely inflated by this specific sandboxed test environment rather than representative of a normal deployment, so it shouldn't be read as "Strand takes 1.6 seconds per query" in general. What it does confirm, measured rather than assumed, is the shape of the bottleneck: encoding cost is paid once per query and does not grow with catalog size, while the part that does grow with catalog size (the symbolic loop) is small enough to stay a rounding error even at 1,000,000 records, and smaller still once moved into Chroma's own filtering.

Storage at 1M images: roughly 2KB per image for a 512-dimensional CLIP vector, roughly 1.5KB per record for a 384-dimensional caption embedding, roughly 0.5 to 1KB of JSON schema metadata per record. Altogether that's somewhere around 6 to 8GB total, comfortably in RAM on a single mid-size server, so lookups stay in memory rather than hitting disk. The real cost driver at any scale is the per-query encoding step (the local embedding model today, the LLM query parser once §4.2 is wired up), not the size of the catalog, exactly the argument Section 4.4 already made, now with a measured number under it instead of only reasoning.
