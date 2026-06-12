---
name: dcn-project
description: >
  Reference and guardrail skill for the Dynamic Concept Networks (DCN) research project:
  an interpretable architecture where nodes are static concept subspaces with discrete
  codebook states selected by a controller (state superposition with collapse), plus
  symbolic rules over states. Use whenever the user mentions DCN, concept nodes or
  subspaces, codebooks/anchors, the controller, superposition/collapse, the k=1
  kill-test, concept purity/NMI, rule violation, intervention faithfulness, subspace
  containment, the synthetic shape/color/size/material task, dcn_mvp.py, or the project
  roadmap/phases. Also trigger when the user wants to run experiments, modify the model,
  add losses, analyze results, or asks "what's next" on this project — even without
  saying "DCN". Consult this skill BEFORE changing any architecture choice, loss, or
  experiment design: it holds frozen decisions, standing rules, and pre-committed
  fallbacks that must not be silently violated.
---

# Dynamic Concept Networks (DCN) — Project Skill

## Purpose of this skill

Keep the project in check. Every session that touches DCN must respect three things:
the **frozen v1 design** (don't redesign mid-experiment), the **standing rules**
(experimental hygiene), and the **pre-committed decision rules** (what to do when a
result or problem appears). When the user proposes a change that conflicts with these,
say so explicitly and point to the relevant rule — then let the user overrule
consciously if they want. The point is no *silent* drift.

## Core claim being tested

Nodes built as *concept subspaces* (static basis M_i) with *discrete canonical states*
(codebook V_i) selected by a controller give a network whose computation is readable at
the concept level — at no meaningful accuracy cost.

One-paragraph pitch: "Interpretability research observes concept subspaces and sparse
features inside trained networks; we make them the explicit computational primitive.
Symbolic structure becomes geometry (hierarchy = subspace containment) and exact logic
over discrete states."

## Frozen v1 design (do NOT change without explicit user sign-off)

| Component | Decision | Rationale (short) |
|---|---|---|
| Node | static matrix M_i ∈ R^{d×k}, orthonormal-ish basis of a concept subspace | static = no decoherence; identity is the subspace |
| States | codebook V_i of m anchors in R^k per node | discrete states solve controller identifiability |
| Controller | encoder → per-node distribution over anchors ("state superposition") | uncertainty readout for free; collapse = firing |
| Collapse | softmax temperature annealed (2.0 → 0.5), collapse at output | per-layer collapse deferred to future work |
| Expressed concept | z_i = (p_i V_i) M_iᵀ | point in the subspace |
| Routing | exactly 1 attention round over superposed z; attention matrix = the graph | more rounds only after Phase 3 validates causality |
| Grounding | anchor↔value CE on 5% of train samples | weak supervision is an accepted, stated assumption |
| Symbolic rule | soft product penalty on forbidden joint states | exact logic over discrete states |
| Core losses (only 3 at launch) | task CE + grounding + symbolic | everything else (usage, confidence, ortho) is auxiliary and must be ablated, not assumed |
| MVP dims | d=32, k=8, m=6, 4 nodes, hidden=64 | k and m are SWEPT in Phases 1–2, then re-frozen |

Explicitly REJECTED designs (do not reintroduce without new evidence): dynamic/drifting
node matrices (decoherence), free continuous controller coordinates (identifiability),
hypernetwork controllers (uninterpretable), per-node k variation in v1 (scope),
localStorage of older ideas like "3 fixed role columns" (superseded by subspace+codebook).

## Vocabulary (use these terms consistently)

- **Node / concept subspace**: M_i. - **Anchor / canonical state**: row of V_i.
- **State superposition**: the distribution p_i before collapse (NOT the Anthropic
  "features in superposition" sense — flag this distinction in any write-up).
- **Collapse / firing**: argmax or hard sample of p_i.
- **Grounding**: weak anchor↔value supervision. - **Purity**: NMI(collapsed state,
  true attribute). - **Faithfulness**: intervention effects match graph-predicted effects.

## Repository layout

```
dcn/
├── dcn_mvp.py      # full MVP: data gen, DCN, MLP baseline, train, evaluate
├── results.json    # latest evaluation output (machine-readable)
├── README.md       # design summary + MVP results + caveats
└── ROADMAP.md      # phases, targets, problems, fallbacks  ← canonical plan
```

A full copy of the roadmap is bundled at `references/ROADMAP.md` in this skill.
Read it before planning any experiment; the condensed phase table below is for
orientation only.

## Phase status (UPDATE THIS TABLE whenever a phase advances)

| Phase | What | Status | Headline target |
|---|---|---|---|
| 0 | MVP | ✅ done | parity + purity — ACHIEVED (see results below) |
| 1 | k=1 kill-test (k ∈ {1,2,4,8,16}, 5 seeds, param-matched) | 🔜 next | purity gap k=4 vs k=1 ≥ 5 NMI pts, else PIVOT |
| 2 | Codebook health (thresholded usage, EMA, dead-code reset, m sweep) | ⬜ | alive anchors == true cardinality on ≥3/4 nodes |
| 3 | Causal validation (clamp/ablate/swap interventions) | ⬜ | faithfulness ≥0.9 owned, ≤0.05 spillover |
| 4 | Grounding ablation 5% → 0% | ⬜ | purity within 10 NMI pts of 5% (or honest curve) |
| 5 | Hierarchy = subspace containment | ⬜ | implication >99%, acc drop <0.5 pts |
| 6 | Real CLEVR → CUB vs CBM baseline | ⬜ | within 3 acc pts of CBM, strictly better faithfulness |
| 7 | Write-up | ⬜ | three known objections pre-answered |

## Current results of record (Phase 0, synthetic task, 5 seeds NOT yet run — single seed)

- DCN test acc **0.993** vs param-matched MLP **0.996** (~17k params each)
- Node→attribute NMI: shape 0.90, color 0.94, size 0.77, material 0.84 (off-diag ≤ 0.08)
- State accuracy: 0.90–0.97 per node
- Rule "sphere ∧ yellow": hard violations **0.0**, soft mass 5e-5
- Known issues: slack anchors don't die cleanly (→ Phase 2); routing graph readable but
  not causally validated (→ Phase 3); single-seed numbers (re-run with 5 seeds before
  citing anywhere)

## Standing rules (enforce on every experiment — push back if violated)

1. **One change at a time.** Never tune two mechanisms in one run.
2. **5 seeds minimum** for any reported number. Medians + IQR, not means.
3. **Parameter-match every comparison** (shrink/grow elsewhere to equalize), or the
   result is invalid and must not be cited.
4. **Negative results get logged**, not discarded — append to results log with date,
   config, and outcome.
5. **Budget cap:** if a fix fails after ~3 focused attempts, take the documented
   fallback in ROADMAP.md instead of grinding.
6. **Kill-criteria are pre-committed.** If Phase 1 triggers its kill (k=1 ties on the
   hardened task), pivot per roadmap — do not rationalize the result away. The project
   does not survive Phase 1 kill + Phase 3 kill together; fallback is the smaller
   codebook-identifiability + exact-rules paper.
7. **Don't add losses casually.** Any new loss term needs: stated purpose, ablation
   showing it earns its place, and an entry in the frozen-design table.
8. **Before citing a metric, check its definition** in ROADMAP.md (e.g. anchor "alive"
   = wins >1% of samples; argmax-unique counting is deprecated as of Phase 0).

## When the user proposes something — decision guide

- **New architecture idea** → compare against the rejected-designs list and frozen
  table first. If genuinely new, suggest parking it in an IDEAS section of ROADMAP.md
  rather than derailing the current phase.
- **"Let's skip to real data"** → Phases 1–3 exist to kill the project cheaply; warn
  that skipping them risks weeks of work on an unvalidated mechanism. Proceed only if
  the user explicitly accepts that.
- **A result contradicts a target** → look up that phase's problem→action table in
  references/ROADMAP.md and apply the pre-committed fallback.
- **Writing anything for the paper** → use Phase 7 framing; pre-empt the three known
  objections ("this is capsules", "toy data", "loss zoo").

## Implementation conventions

- Single-file experiments extend `dcn_mvp.py` via config/flags; split into a package
  only when Phase 3 lands (interventions justify a module split: data / model / train /
  interventions / metrics).
- Every run writes a machine-readable JSON (config + seed + all metrics) — append-only
  results log, one file per run, named `results_<phase>_<desc>_<seed>.json`.
- Seeds: torch + numpy both seeded; fixed value-embedding RNG (seed 0) so the dataset
  is identical across model seeds.
- Keep CPU-runnable until Phase 6; if a sweep exceeds ~30 min CPU, batch via simple
  multiprocessing before reaching for GPU.

## Maintaining this skill

After completing any phase or making any design decision: update the Phase status
table, the results of record, and (if a frozen choice changed) the frozen-design table
with the new decision AND the evidence that justified the change. Stale project skills
are worse than none — they enforce outdated rules with confidence.
