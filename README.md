# Dynamic Concept Networks — MVP

Minimal end-to-end implementation of the DCN v1 design:
**static concept subspaces → discrete canonical states (codebooks) → controller as
distribution over states (superposition) → annealed collapse → symbolic rule over states.**

## Run

```bash
pip install torch scikit-learn numpy
python dcn_mvp.py        # ~2 min on CPU; writes results.json
```

## Architecture (frozen for v1)

| Component | Choice |
|---|---|
| Nodes | 4 (one expected per attribute, assignment NOT given) |
| Subspace basis M_i | 32 x 8, static, orthogonal init + ortho penalty |
| Codebook V_i | 6 anchors per node (true value counts are 3/4/2/2 → slack on purpose) |
| Controller | shared MLP encoder → per-node logits over anchors |
| Routing | 1 attention round over superposed concepts, refines state logits |
| Collapse | softmax temperature annealed 2.0 → 0.5 |
| Grounding | anchor↔value CE on 5% of training samples |
| Symbolic rule | "sphere ∧ yellow = impossible", soft product penalty |

Task: 48-way classification (shape×color×size×material) from noisy feature
encodings; the rule also holds in the data (no yellow spheres generated).

## MVP results (parameter-matched, ~17k params each)

| Metric | DCN | MLP baseline |
|---|---|---|
| Test accuracy | **0.993** | 0.996 |

Interpretability metrics (MLP offers none of these):

- **Concept purity (NMI, best match):** node_0→shape 0.90, node_1→color 0.94,
  node_2→size 0.77, node_3→material 0.84. Each node specialized to exactly one
  attribute; off-diagonal NMI ≤ 0.08. **Identifiability worked.**
- **State accuracy** (collapsed anchor == true value): shape 0.93, color 0.97,
  size 0.90, material 0.93 — the discrete states are readable concept labels.
- **Rule satisfaction:** hard violation rate **0.0000**, soft joint mass 5e-5.
- **Uncertainty readout:** superposed p exposes calibrated ambiguity per node
  before collapse (see `uncertainty_demo` in results.json).
- **Routing graph:** mean attention matrix extractable per input (B×N×N).

### Honest caveats
1. Accuracy parity, not superiority — expected; the claim is *interpretability at
   no accuracy cost*, and that's what was measured.
2. Slack anchors did not fully die (argmax-usage > true value count). The batch
   usage-entropy bonus is slightly too strong; needs a usage threshold metric and
   dead-code pruning. Known VQ issue, fix is standard.
3. Routing graph is not yet validated causally (no intervention test).
4. Synthetic data with linear-ish structure — friendly territory by design.

## Expansion roadmap (in order)

1. **k=1 kill-test** — replace M_i (d×k) with a single vector per anchor
   (equivalently k=1): does purity/rule satisfaction degrade? This is the
   load-bearing ablation for the subspace claim.
2. **Sweep k ∈ {1,2,4,8,16} and m**, parameter-matched; look for the interior
   optimum and whether discovered m matches true attribute cardinality.
3. **Intervention test** — clamp a node's state (force shape=cube), verify
   downstream output changes as the routing graph predicts (causal
   interpretability, not just probes).
4. **Remove grounding** (0% supervision) — test whether usage + symbolic losses
   alone identify anchors; measure purity drop.
5. **Hierarchy as subspace containment** — add a derived concept (e.g. "metal
   object") with loss ||(I − P_parent) M_child||²; this is the most distinctive
   contribution of the architecture.
6. **Real CLEVR** — frozen CNN/CLIP features per object instead of synthetic
   encodings; then one messy dataset (CUB) to escape toy-world objections.
