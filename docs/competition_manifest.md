# Competition Manifest: Google - Agent Harness on Physical-Engineering's Last Exam (PLE)

- **Competition Slug:** `physical-engineerings-last-exam`
- **URL:** https://www.kaggle.com/competitions/physical-engineerings-last-exam
- **Official Kit & Benchmark Repo:** https://github.com/google-research/ple
- **Public Dev Dataset:** `google/physical-engineerings-last-exam` (Hugging Face, 29 tasks, revision `b3857a3fea39a695e18f845d4a88f29442e96e75`)
- **Paper:** Han et al., *PLE: Physical-Engineering's Last Exam* (`arXiv:2610.30537`)
- **Competition Type:** Long-Horizon Autonomous Engineering Agent Harness (Hackathon — no tabular CSV dataset; `NOTE.md`: *"This is a Hackathon with no provided dataset."*)
- **Submission Format:** GitHub commit containing `submission.yaml`, WebSocket harness (`serve.py` + `taskbridge.py` + dependencies), one of the 8 Google-sponsored `google-cloud/<id>` models, and `artifacts/` exported from `eval/run.py` on the 29 public dev tasks.
- **Daily Submission Limit (`MAX_DAILY_SUBMISSIONS`):** Hackathon / Stage-based evaluation (no daily Kaggle CSV quota; `5` tracked as nominal cycle cap in orchestrator telemetry).

---

## 1. Benchmark & Discipline Breakdown (85 Total Tasks)

| Discipline | Total Tasks | Public Dev (29) | Hidden Test (56) | Dev % | Test % | Typical Deliverables |
|---|--:|--:|--:|--:|--:|---|
| **Mechanical engineering** | 30 | 7 | 23 | 24.1% | 41.1% | Cooling & controls, clash-free MEP routing (`solution.py`), equipment equivalency, fire-test validation |
| **Electrical engineering** | 27 | 9 | 18 | 31.0% | 32.1% | Load/capacity calculations, solar interconnection memos, cable sizing, energization checks |
| **Civil engineering** | 18 | 9 | 9 | 31.0% | 16.1% | Settlement & bearing capacity, foundation strategy, IFC site models (`model.ifc`), cost estimates |
| **Structural engineering** | 10 | 4 | 6 | 13.8% | 10.7% | Gravity/wind/seismic checks, EOR review corrections (`submission.zip`), member/connection checks |

### Grading Modalities in Public Dev (29 Tasks)
- `rubric` (LLM Judge): **14 tasks** (`final_message: true` on all 14)
- `checks` (Automated deterministic code/constraint checks): **12 tasks**
- `checks-judge` (Hybrid automated checks + LLM judge): **3 tasks**

---

## 2. Official Evaluation Metric & Hard-Gate Cliff

For constraint-graded tasks (`tests/paper_score.py` in `docs/rules.md`), each deliverable item scores:

$$s = 0.4\,p + 0.6\,G\,O', \qquad O' = \operatorname{clip}\!\left(\frac{O - O_{\text{naive}}}{1 - O_{\text{naive}}},\,0,\,1\right)$$

- $p \in [0, 1]$: fraction of soft/individual constraints satisfied.
- $G \in \{0, 1\}$: hard feasibility gate (`1` iff **every** hard safety/feasibility/format constraint passes, `0` otherwise). Any single hard-gate violation caps the item score at **0.40**.
- $O'$: normalized design objective relative to an unoptimized compliant baseline $O_{\text{naive}}$.
- Run score: mean across all tasks in the set (missing task = `0.0`).

---

## 3. Execution Protocol & Constraints

- **Transport:** WebSocket server (`serve.py`) speaking Serve Protocol v1 (`task`, `exec`, `read_file`, `write_file`, `http`, `log`, `usage`, `done`, `error`).
- **Task Timeout:** Up to **6 hours** (`21600s`) per task; `exec` capped at `600s`.
- **Network Isolation:** Task container has no external internet; harness egress proxy permits **only** the declared `google-cloud/<id>` model endpoint (no web search, no external APIs).
- **Concurrency:** Up to 8 concurrent WebSocket requests; `ping_interval=None` required in `websockets`.
- **Fair Play:** Zero hard-coding of task IDs, zero inspection of grading scripts or `artifacts/` at runtime, zero web search.
