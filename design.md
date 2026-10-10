# Design Document — RAG Ablation: Baseline vs. Re-Ranking

## 1. Objective

Implement and contrast two retrieval pipelines over the **same chunked corpus** (ITF *Rules of Tennis*, 2026 edition):

| Pipeline | Flow |
|---|---|
| **Baseline** | Bi-Encoder → FAISS Top-10 → take Top-3 → LLM → Answer 1 |
| **Re-ranking** | Bi-Encoder → FAISS Top-10 → Cross-Encoder re-sort → Top-3 → LLM → Answer 2 |

The goal is to demonstrate, with **measured evidence**, that a Cross-Encoder re-ranker
rescues queries where dense retrieval buries the correct chunk outside the Top-3.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PDF (2026-rules-of-tennis-english.pdf)                 │
│  → pypdf extract → clean → section split → chunk        │
└──────────────────────────┬──────────────────────────────┘
                           ▼
              Bi-Encoder (all-MiniLM-L6-v2)
              384-d normalized embeddings
                           ▼
                   FAISS IndexFlatIP
                   (exact cosine similarity)
                           ▼
              ┌────── Top-10 candidates ──────┐
              ▼                                ▼
     Pipeline 1 (Baseline)            Pipeline 2 (Re-ranking)
     Take Top-3 directly              Cross-Encoder scores
              │                        all 10 (query, chunk) pairs
              │                                │
              │                        Re-sort by CE score
              │                                │
              │                        Take Top-3
              ▼                                ▼
     ┌──────────────── LLM prompt (identical) ──────────┐
     │  Gemini (gemini-2.5-flash, temp=0)               │
     └───────────────────────────────────────────────────┘
              ▼                                ▼
          Answer 1                          Answer 2
```

### Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Bi-Encoder | `sentence-transformers/all-MiniLM-L6-v2` | Fast, 384-d, strong general-purpose dense retrieval |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Named in the assignment; MSMARCO-trained |
| Vector store | FAISS `IndexFlatIP` | Exact search → perfect recall; isolates *ranking* as the only variable |
| Normalization | L2-normalize embeddings | Inner product ≡ cosine similarity |
| Device | CPU (configurable) | Tiny models; stable latency numbers; no GPU memory pressure |
| LLM | Gemini API (`gemini-2.5-flash`) | Free tier, fast, deterministic at temp=0 |
| Chunking | Rule/case-aligned blocks | Legal text: the *exception* must be retrievable in isolation |

---

## 3. Chunking Strategy

Blind token-window chunking would split a rule from its exception. Instead:

1. **Section split** — detect `1. THE COURT` … `31. PLAYER ANALYSIS TECHNOLOGY`,
   `APPENDIX I` … `APPENDIX XII`, and named addenda (Wheelchair Tennis, Beach Tennis, etc.).
2. **Block split** — within each section, split at `Case N:` and `Note:` boundaries so every
   case/decision becomes its own retrievable unit.
3. **Merge** — adjacent blocks smaller than `min_chunk_words` are merged up to `chunk_size_words`.
4. **Oversize split** — blocks exceeding `chunk_size_words` are split on sentence boundaries
   with `chunk_overlap_words` of overlap.

All parameters live in `config.yaml` — no code changes needed to retune.

---

## 4. Failure-Mode Query Design

At least two queries where baseline retrieval stumbles but re-ranking recovers:

| # | Mode | Query (example) | Target chunk | Expected failure |
|---|---|---|---|---|
| 1 | **Keyword trap** | "Can a player keep using a racket after the strings break during a point?" | Rule 4 Case 4 | Distractors: Rule 4 body, Appendix II (high lexical overlap on "racket"/"strings") |
| 2 | **Keyword trap** | "Is a bird hitting the ball in play considered a hindrance?" | Rule 26 Case 3 | Synonym gap: "bird" vs. "hindrance" vocabulary |
| 3 | **Exception / corner case** | "If a let is called on a second service, does the whole point get replayed?" | Rule 22 final sentence | Baseline grabs Rule 23 ("whole point replayed") — misses the exception |
| 4 | **Exception / corner case** | "Is a ball that is soft but not broken at the end of a point replayed?" | Rule 3 Case 1 | Baseline grabs Rule 3 body ("broken → replayed") — misses the case exception |
| 5 | **Negation / constraint** | "In a singles match, may the server stand behind the part of the baseline between the singles and doubles sidelines?" | Rule 18 Case 1 | Baseline retrieves Rule 16/17 (positive framing) — misses the prohibition |
| 6 | **Negation / constraint** | "Is it a foot fault if the server has both feet off the ground while serving?" | Rule 18 Case 2 | Baseline retrieves Rule 18 body (foot fault conditions) — misses the case exception |
| 7 | **Buried chunk** | "How many times may the ball bounce before a wheelchair tennis player loses the point?" | Wheelchair Tennis — Two Bounce Rule | Baseline retrieves Rule 24(b) (able-bodied, opposite answer) |
| 8 | **Buried chunk** | "How many unsuccessful Electronic Review appeals is a player allowed per set?" | Appendix VII — Electronic Review §6 | Buried in a long appendix; low lexical overlap with query |

The notebook runs all queries, records the actual rank shifts, and the report documents
the ones that genuinely fail (with evidence).

---

## 5. Latency Profiling

| Measurement | What is timed |
|---|---|
| Bi-Encoder query encode | `encoder.encode([query])` |
| FAISS search | `INDEX.search(qv, k)` |
| Cross-Encoder scoring | `ce.predict([(q, chunk) × 10])` |
| Full pipeline (retrieval only) | encode + search + CE |
| LLM call | Gemini API round-trip (reported separately) |

- Warmup runs excluded; `timed_runs` repetitions per measurement.
- Report: mean, median, p95, std in a pandas table.
- LLM latency is reported separately (network-bound, not comparable to local inference).

---

## 6. Scaling Analysis

**Question:** *Why not run the Cross-Encoder over the entire corpus from the start?*

**Empirical demonstration in the notebook:**

1. Measure CE cost per `(query, chunk)` pair on the actual corpus.
2. Extrapolate to 1K, 10K, 100K chunks (configurable in `config.yaml`).
3. Compare with FAISS search time on a synthetic 100K-vector index.
4. Show that CE is **O(N)** per query (every chunk must be scored) while ANN retrieval is
   approximately **O(log N)** — at 100K chunks, CE would take ~minutes per query vs. ~1 ms for FAISS.

---

## 7. Deliverables

| File | Purpose |
|---|---|
| `rag_assignment.ipynb` | Single Jupyter Notebook — all code, all experiments, well-commented |
| `config.yaml` | All tunable parameters (chunk size, top-k, models, device, profiling) |
| `.env.example` | Template for `GEMINI_API_KEY` |
| `requirements.txt` | Python dependencies (Python 3.14 compatible) |
| `README.md` | Setup and run instructions |
| `results/` | Exported CSV/JSON tables (rank shifts, latency, scaling) |
| `report.md` / `report.docx` | Written report (authored by the student, not AI-generated) |

---

## 8. Notebook Structure

```
Section 1: Setup & Environment
Section 2: PDF Loading & Cleaning
Section 3: Section Detection & Chunking
Section 4: Bi-Encoder Index (FAISS)
Section 5: Retrieval Pipelines (Baseline + Re-ranking)
Section 6: LLM Answer Generation (Gemini, identical prompt)
Section 7: Failure-Mode Analysis (rank shifts, Answer 1 vs Answer 2)
Section 8: Latency Profiling
Section 9: Scaling Analysis
Section 10: Export Results
```

---

## 9. Reproducibility

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # add GEMINI_API_KEY
jupyter nbconvert --execute rag_assignment.ipynb --to notebook
```

- The notebook runs end-to-end **without** an API key (LLM cells show a placeholder).
- With a key, the same cells produce real Gemini answers.
- All random seeds fixed; `temperature=0` for deterministic LLM output.
