# Dynamic Concept Networks — Research Roadmap

**Core claim being tested:** Nodes built as *concept subspaces* (static basis M_i) with
*discrete canonical states* (codebook V_i) selected by a controller give a network whose
computation is readable at the concept level — at no meaningful accuracy cost.

Every phase below has: what to do → what result we want → what can go wrong → what we do then.
A phase is DONE when its success criteria are met or its kill-criterion has triggered and
been resolved. Do not skip ahead: phases 1–3 decide whether the architecture is worth scaling.

**Status legend:** ✅ done · 🔜 next · ⬜ pending

---

## Phase 0 — MVP ✅ (done)

Built and validated on synthetic shape/color/size/material task (48 classes).

Achieved: accuracy parity with parameter-matched MLP (99.3% vs 99.6%), unsupervised node→
attribute specialization (NMI 0.77–0.94), readable discrete states (90–97% state accuracy),
zero hard rule violations, uncertainty readout, extractable routing graph.

Open issues carried forward: slack anchors don't die cleanly (P0-a), routing graph not
causally validated (→ Phase 3), friendly synthetic data (→ Phase 6).

---

## Phase 1 — The k=1 kill-test 🔜 (highest priority, ~1 day)

**Why first:** This is the load-bearing ablation. If a single vector per node does
everything the subspace does, the architectural contribution evaporates and the project
pivots before any more effort is spent.

**What to do**
- Train k ∈ {1, 2, 4, 8, 16}, all parameter-matched (shrink/grow elsewhere to equalize).
- 5 seeds per setting (variance matters more than the mean at this scale).
- Record: test accuracy, purity (NMI), state accuracy, rule violation, gradient stability.

**Result we are looking for**
- An interior optimum: k=1 measurably worse on purity and/or rule satisfaction
  (target: ≥5 NMI points or ≥10x soft rule mass), large k plateauing or degrading.
- Accuracy roughly flat across k (capacity is matched, so accuracy differences would
  indicate optimization effects, not capacity).

**Expected problems → what we do**

| Problem | Diagnosis | Action |
|---|---|---|
| k=1 matches k>1 on everything | Task too easy — attributes are linearly separable, one direction suffices | Don't kill yet: harden the task first (raise noise, add correlated attributes, add a compositional concept that genuinely needs >1 dim, e.g. "position" as continuous 2D). Re-run. If k=1 STILL matches → **pivot**: the contribution is the codebook/controller mechanism, not subspaces. Reframe paper around discrete-state identifiability + symbolic rules; drop subspace claims. |
| High seed variance swamps differences | Underdetermined optimization | Increase grounding to 10%, fix M_i init across seeds, report medians + IQR. |
| Large k degrades accuracy | Coordinates underconstrained, controller overfits | Add weight decay on V, reduce anchor norm; note as evidence FOR the interior-optimum story. |

**Kill-criterion:** k=1 ties on all metrics on the hardened task across seeds → subspace
framing is dropped (pivot above), Phases 5's containment work is redesigned or cut.

---

## Phase 2 — Codebook health & anchor death (P0-a fix, ~1–2 days)

**What to do**
- Replace argmax-usage metric with thresholded usage (anchor "alive" if it wins >1% of samples).
- Tune usage-entropy weight down; add EMA codebook updates + dead-code re-initialization.
- Sweep m ∈ {2, 4, 6, 10} per node against the true cardinalities (3/4/2/2).

**Result we are looking for**
- Alive-anchor count converges to true value count per attribute when m has slack
  ("the model discovers concept cardinality"). This is a headline plot if it works.

**Expected problems → what we do**

| Problem | Action |
|---|---|
| Codebook collapse (too few anchors) | Standard VQ toolkit in order: lower commitment/confidence weight → EMA updates → dead-code resets → entropy bonus floor. If still collapsing, anneal temperature more slowly. |
| Anchors never die (too many alive) | Add L0-style per-anchor gate with small penalty; reduce usage-entropy bonus to 0 after warmup epochs. |
| Alive count is right but anchors are duplicates | Add repulsion term between anchors of same node (min-distance penalty); report anchor cosine matrix. |
| m smaller than true cardinality | Verify graceful degradation: two values share an anchor → purity drop should be localized. If global collapse instead, controller is brittle — investigate before scaling. |

---

## Phase 3 — Causal validation of the graph (~2–3 days)

**Why:** Interpretability is currently asserted via probes. Reviewers (rightly) demand
interventions. This phase is what separates "readable" from "decorative."

**What to do**
- Clamp interventions: force node state (e.g. shape := cube) post-controller, measure
  output change vs the change predicted by reading the routing graph.
- Ablation interventions: zero a node's gate, measure which classes degrade.
- Swap test: exchange two inputs' states on ONE node, check output flips only on the
  attributes that node owns.
- Metric: intervention faithfulness = fraction of interventions whose effect direction
  matches graph-predicted effect.

**Result we are looking for**
- Faithfulness ≥ 0.9 on owned attributes; near-zero spillover onto non-owned attributes
  (this is the leakage test from concept-bottleneck literature).

**Expected problems → what we do**

| Problem | Action |
|---|---|
| Spillover: clamping shape changes color predictions | Information leaking through routing round or output head. First check: train a no-routing variant — if leakage vanishes, routing is the channel → add per-edge sparsity (L1 on attention) or hard top-1 routing. If leakage persists, it's the concat output head → switch to per-node readout heads summed into logits. |
| Interventions break outputs entirely (off-manifold) | Clamp to anchor states only (on-manifold by construction), never to arbitrary coordinates. |
| Graph predicts nothing (attention ≈ uniform) | Routing isn't being used. Either remove routing for v1 (honest simplification) or make the task require relations (Phase 6's relational variant) so routing has a job. Do NOT keep a decorative component. |

**Kill-criterion (soft):** faithfulness < 0.6 after fixes → interpretability claim must be
downgraded to "state-level readable, graph-level not validated"; paper survives but the
graph story is cut.

---

## Phase 4 — Grounding ablation: 5% → 0% (~1–2 days)

**What to do**
- Sweep anchor-grounding supervision: {5%, 1%, 0.1%, 0%}.
- At 0%, measure purity with post-hoc Hungarian matching of anchors to values.

**Result we are looking for**
- Graceful degradation; ideally purity at 0% within ~10 NMI points of 5%, proving the
  usage + symbolic + task losses alone identify states (strong identifiability claim).

**Expected problems → what we do**

| Problem | Action |
|---|---|
| 0% purity collapses (nodes entangle attributes) | Expected — full unsupervised identifiability is hard. Options in order: (1) add orthogonality between *expressed* concepts across nodes, (2) information-bottleneck per node (limit logit capacity), (3) accept weak supervision as a stated requirement — 5% labels is a cheap, honest assumption; report the curve and move on. Do not sink weeks here; the curve itself is the result. |
| Symbolic loss meaningless at 0% (anchor indices unbound) | Apply symbolic loss only after epoch E with anchors bound by clustering warm-start; or keep rules for grounded settings only and say so. |

---

## Phase 5 — Hierarchy as subspace containment (~3–5 days)

**Why:** This is the most distinctive single idea (rules as geometry:
span(M_child) ⊆ span(M_parent)) and the strongest novelty hook — IF Phase 1 kept subspaces.

**What to do**
- Add derived parent concepts (e.g. "metal-thing", "large-thing") as extra nodes.
- Containment loss: ‖(I − P_parent) M_child‖² for declared parent–child pairs.
- Evaluate: containment residual at convergence, whether parent states stay consistent
  with child states at inference (implication satisfaction rate), accuracy impact.

**Result we are looking for**
- Containment residual → ~0 without accuracy loss; implication satisfaction > 99%;
  visualization of nested subspaces (project to 2D) as the paper's signature figure.

**Expected problems → what we do**

| Problem | Action |
|---|---|
| Containment fights orthogonality penalty | They're contradictory if applied naively (child inside parent can't be orthogonal to it). Scope orthogonality to non-related node pairs only; document the loss interaction explicitly. |
| Parent subspace inflates to contain everything | Add rank/dimension penalty on parents, or fix parent k larger than child k by design and cap it. |
| Geometric containment holds but behavioral implication fails | Geometry ≠ usage. Add a state-level implication loss (if child fired state s, parent distribution must respect mapping). If geometry alone never suffices, that's a finding: report both losses and their gap. |
| Phase 1 killed subspaces | Replace this phase with hierarchy over discrete states (implication tables over codebooks) — weaker geometrically, still publishable as exact neuro-symbolic constraints. |

---

## Phase 6 — Escape the toy world (~1–2 weeks)

**What to do**
- Step 1: real CLEVR — frozen CNN/CLIP features per object (use ground-truth boxes first;
  detection is not our problem). Same node setup + add relational queries so routing earns
  its place (Left-of antisymmetry rule: Left(A,B) ⇒ ¬Left(B,A)).
- Step 2: one messy dataset — CUB-200 with attribute annotations (concepts are noisy,
  correlated, human-labeled — exactly the unfriendly regime).

**Result we are looking for**
- CLEVR: maintain purity within ~15 NMI points of synthetic; relational rule satisfied;
  routing now causally faithful (re-run Phase 3 tests here).
- CUB: DCN within 2–3 accuracy points of a concept-bottleneck baseline (CBM) while
  beating it on intervention faithfulness / leakage.

**Expected problems → what we do**

| Problem | Action |
|---|---|
| Purity craters on real features | Concepts in CLIP space aren't axis-aligned with our subspaces. Initialize M_i from SAE features trained on the backbone (this was always the plan — now it's required). If SAE init rescues it, that's a positive result about the SAE↔DCN connection. |
| Node vocabulary insufficient (unknown concepts in data) | v1 limitation, stated openly: fixed vocabulary. Add a small number of "free" ungrounded nodes and inspect what they capture; full open-vocabulary nodes = future work, do not attempt now. |
| Multiple objects per scene break the one-node-per-concept design | Use per-object node instantiation (slot-style) but keep SHARED M_i/V_i across instances — identity lives in shared parameters, instances are bindings. This is the key design move for scenes; prototype it on 2-object scenes before full CLEVR. |
| Compute blows up | Profile first. Routing is O(N²); cap N, use top-j neighbor routing. If DCN needs >2x baseline FLOPs for parity, report it honestly as the price of interpretability and measure the trade. |

---

## Phase 7 — Write-up & packaging (~1–2 weeks, overlaps 5–6)

**What to do**
- Frame: "Interpretability research observes concept subspaces and sparse features in
  trained networks; we make them the explicit computational primitive."
- Related-work table contrasting: Capsules, Slot Attention, CBMs, VQ-VAE, GNN structure
  learning, TPRs, concept-erasure subspaces — one column = "what we take", one = "what differs".
- Core results: Phase 1 curve, Phase 2 cardinality discovery, Phase 3 faithfulness,
  Phase 5 containment figure, Phase 6 CBM comparison.
- Pre-empt the three reviews we know are coming: "this is capsules" (answer: static
  subspaces + discrete states + exact rules, with the Phase 1/3 evidence), "toy data"
  (answer: CUB), "loss zoo" (answer: only 3 core losses at launch — task, grounding,
  symbolic; everything else ablated in appendix).

---

## Standing rules (apply to every phase)

1. **One change at a time.** Never tune two mechanisms in the same run; attribution dies.
2. **5 seeds minimum** for any number that goes in a table. Medians + IQR, not means.
3. **Parameter-match every comparison** or the result is invalid.
4. **Negative results get written down** in results.json + a log, not discarded — the
   k=1 tie and the 0%-grounding collapse are publishable findings if they happen.
5. **Budget cap per problem:** if a fix isn't working after ~3 focused attempts, take the
   documented fallback in its table instead of grinding. The fallbacks were chosen so the
   project survives every individual failure.
6. **The one failure the project does NOT survive:** Phase 1 kill + Phase 3 kill together
   (no subspace benefit AND no faithful graph). If both trigger, the remaining asset is the
   codebook-identifiability + exact-symbolic-rules mechanism — write that smaller paper
   and stop.

## Result targets at a glance

| Phase | Primary number | Target |
|---|---|---|
| 1 | Purity gap, k=4 vs k=1 | ≥ 5 NMI pts (else pivot) |
| 2 | Alive anchors vs true cardinality | exact match on ≥ 3/4 nodes |
| 3 | Intervention faithfulness | ≥ 0.9 owned / ≤ 0.05 spillover |
| 4 | Purity at 0% grounding | within 10 NMI pts of 5% (or honest curve) |
| 5 | Implication satisfaction | > 99%, accuracy drop < 0.5 pts |
| 6 | vs CBM on CUB | accuracy within 3 pts, faithfulness strictly better |
