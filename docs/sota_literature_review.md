# SOTA Literature Review & Architectural Strategy (PLE Benchmark)

## 1. Benchmark Baseline & Failure Modes (arXiv:2610.30537)
- **Paper:** Han et al. (2026), *PLE: Physical-Engineering's Last Exam* (`arXiv:2610.30537`).
- **Headline Result:** Frontier models achieve `90%+` on textbook engineering exams, yet peak at `50.6%` overall (`48.3%` on the 56 private test tasks) using `Claude Opus 5.5` inside `Claude Code` (~`27M` tokens and `63` min/task).
- **Primary Failure Modes Observed in Public Kernels & Paper:**
  1. **Hard-Gate Cliff ($G = 0$):** Over 48% of failures occur because a single hard safety, clearance, or schema constraint fails, capping the task score at `0.40` even when `95%` of soft clauses pass.
  2. **Superseded Revision Neglect:** Engineering corpuses average ~61 files (up to 207) across `Preliminary`, `Permit`, `Bid`, `IFC`, `Field_Directive`, `RFI_Response`, `Addendum`, and `AHJ_Letter`. Agents that read base drawings without enforcing **Revision Authority Hierarchy** optimize against obsolete specs.
  3. **3D Spatial & MEP Clashes:** Missing clearance/routing checks in `solution.py` or `model.ifc` deliverables.
  4. **Unit & Datum Drift:** Mixing imperial/SI units (`ft`/`in`/`m`, `kip`/`kN`, `kVA`/`kW`) or dual site datums across PDFs, XLSX schedules, and CAD/BIM files.
  5. **Deliverable Path / Schema Errors:** Failing to write to the exact expected deliverable path (e.g., `/app/files/.../output/solution.py` or `submission.zip`) or returning an incomplete `final_message` on `rubric` tasks.

---

## 2. Winning Architectural Patterns
1. **Revision-Authority-First Ingestion:** Index all workspace files and rank documents by authority (`Preliminary=10 < Permit=20 < Bid=30 < IFC=40 < Field_Directive=50 < RFI_Response=60 < Addendum=70 < AHJ_Letter=80`) before any engineering calculation.
2. **Code-Gated Verification Loop (`plan-verify` + Deterministic Solvers):**
   - **Structural (AISC 360-16 / ASCE 7):** LRFD load combinations (`1.2D + 1.6L`), flexural/shear DCR $\le 1.0$, deflection $\le L/360$ or $L/240$.
   - **Civil / Geotechnical:** Terzaghi/Meyerhof bearing capacity $q_u / FS \ge q_{\text{applied}}$, settlement $\delta \le \delta_{\text{allow}}$, cut/fill & grading slope bounds.
   - **Electrical (NEC / IEEE):** Ampacity derating, voltage drop ($\le 3\%$ branch / $\le 5\%$ feeder), short-circuit & solar interconnection thermal limits.
   - **Mechanical (ASHRAE / NFPA):** Psychrometric & cooling load balance, static pressure drop, 3D AABB MEP clash detection.
3. **Evolution-in-Code (FunSearch-style Prompt & Verifier Refinement):**
   - Iteratively mutate system prompts, verification checklists, and closed-form solver guards, accepting mutations only when all unit tests and dev-set proxies improve.
4. **Covariate-Shift Awareness:**
   - Mechanical (`24.1% -> 41.1%`) and Electrical (`31.0% -> 32.1%`) are heavily over-represented in the 56 hidden test tasks compared to the 29 dev tasks. Optimizing Mechanical MEP/HVAC and Electrical NEC/solar verifiers yields the highest expected test-set lift.
