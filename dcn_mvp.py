"""
Dynamic Concept Networks — MVP + Phase 1 (k=1 kill-test)
========================================================
Architecture under test (v1, frozen design):
  - N concept nodes. Each node i owns:
      M_i  in R^{d x k}   : STATIC learned concept-subspace basis (orthonormal columns)
      V_i  in R^{m x k}   : codebook of canonical states (anchors) inside that subspace
  - Controller: encoder -> per-node logits over anchors -> distribution p_i
      ("state superposition": expressed concept is the expectation over anchors)
  - One routing round: nodes attend over each other's superposed concepts and
    refine their state distributions (the attention matrix = readable concept graph).
  - Collapse at output: temperature annealed toward hard anchor selection.
  - Expressed concept: z_i = (p_i @ V_i) @ M_i^T  in R^d
  - Output head reads gated concat of z_i.

Losses:
  task CE  +  anchor grounding CE (on 5% of samples)  +  symbolic rule penalty
  +  codebook usage (batch-entropy)  +  per-sample confidence (entropy anneal)
  +  subspace orthogonality

Task (synthetic CLEVR-like):
  Objects with shape(3) x color(4) x size(2) x material(2) -> 48-way classification
  from noisy continuous feature encodings.
  Symbolic rule in the data: spheres are NEVER yellow.
  Symbolic loss: penalize E[ p_shape(sphere) * p_color(yellow) ].

Baseline: parameter-matched MLP.

Metrics: accuracy, concept purity (NMI node-state vs ground-truth attributes),
codebook usage, rule violation rate, routing-graph readout.

Phase 1 extension (single-file per project convention — config/flags only, the
architecture, losses and training schedule above are UNCHANGED):
  - k sweep {1,2,4,8,16}, parameter-matched by adjusting controller hidden width.
  - 5 seeds per setting; per-run JSON in results/ (append-only log); medians + IQR.
  - --hardened flag (noise 0.35 -> 0.7 + size<->material correlation), used ONLY
    if k=1 ties on the base task, per ROADMAP.md Phase 1.

Usage:
  python dcn_mvp.py --phase 0                  # 5-seed MVP rerun (DCN + MLP)
  python dcn_mvp.py --phase 1                  # k in {1,2,4,8,16} x 5 seeds
  python dcn_mvp.py --phase 1 --hardened       # only if base-task k=1 ties
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score

DEVICE = "cpu"

# -----------------------------------------------------------------------------
# 1. Synthetic dataset
# -----------------------------------------------------------------------------
ATTRS = {
    "shape":    ["sphere", "cube", "cylinder"],
    "color":    ["red", "green", "blue", "yellow"],
    "size":     ["small", "large"],
    "material": ["rubber", "metal"],
}
ATTR_NAMES = list(ATTRS.keys())
N_VALUES = [len(ATTRS[a]) for a in ATTR_NAMES]          # [3, 4, 2, 2]
EMB_PER_ATTR = 8
INPUT_DIM = EMB_PER_ATTR * len(ATTR_NAMES)              # 32
N_CLASSES = int(np.prod(N_VALUES))                      # 48
NOISE = 0.35
NOISE_HARD = 0.7                                        # hardened task (Phase 1 fallback)

SPHERE_IDX, YELLOW_IDX = 0, 3                            # rule: no yellow spheres

# fixed value-embedding RNG (seed 0) so the dataset is identical across model seeds
_emb_rng = np.random.default_rng(0)
VALUE_EMB = {a: _emb_rng.normal(0, 1, (len(v), EMB_PER_ATTR)).astype(np.float32)
             for a, v in ATTRS.items()}

def sample_object(rng, hardened):
    while True:
        vals = [rng.integers(0, n) for n in N_VALUES]
        if hardened:                                     # correlated attributes:
            p_metal = 0.8 if vals[2] == 1 else 0.2       # large things tend to be metal
            vals[3] = int(rng.random() < p_metal)
        if not (vals[0] == SPHERE_IDX and vals[1] == YELLOW_IDX):  # data obeys rule
            return vals

def encode(vals, rng, noise):
    parts = [VALUE_EMB[a][v] for a, v in zip(ATTR_NAMES, vals)]
    x = np.concatenate(parts) + rng.normal(0, noise, INPUT_DIM).astype(np.float32)
    return x.astype(np.float32)

def label(vals):
    idx = 0
    for v, n in zip(vals, N_VALUES):
        idx = idx * n + v
    return idx

def make_split(n, sample_seed, hardened=False):
    rng = np.random.default_rng(sample_seed)
    noise = NOISE_HARD if hardened else NOISE
    V = np.array([sample_object(rng, hardened) for _ in range(n)])
    X = np.stack([encode(v, rng, noise) for v in V])
    y = np.array([label(v) for v in V])
    return (torch.tensor(X), torch.tensor(y, dtype=torch.long),
            torch.tensor(V, dtype=torch.long))

# -----------------------------------------------------------------------------
# 2. Model
# -----------------------------------------------------------------------------
class DCN(nn.Module):
    def __init__(self, n_nodes=4, d=32, k=8, m=6, hidden=64):
        super().__init__()
        self.n_nodes, self.d, self.k, self.m = n_nodes, d, k, m
        # static concept-subspace bases (orthogonal init + ortho penalty in loss)
        self.M = nn.Parameter(torch.stack(
            [nn.init.orthogonal_(torch.empty(d, k)) for _ in range(n_nodes)]))
        # codebooks of canonical states inside each subspace
        self.V = nn.Parameter(torch.randn(n_nodes, m, k) * 0.5)
        # controller
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = nn.Linear(hidden, n_nodes * m)
        # one routing round (graph): refine logits from attended context
        self.att_q = nn.Linear(d, 16, bias=False)
        self.att_k = nn.Linear(d, 16, bias=False)
        self.refine = nn.Linear(hidden + d, m)
        # output
        self.gates = nn.Parameter(torch.zeros(n_nodes))
        self.out = nn.Linear(n_nodes * d, N_CLASSES)

    def express(self, p):
        """p: (B, N, m) state distributions -> z: (B, N, d) expressed concepts."""
        c = torch.einsum("bnm,nmk->bnk", p, self.V)          # coords in subspace
        z = torch.einsum("bnk,ndk->bnd", c, self.M)          # point in R^d
        return z

    def forward(self, x, temp=1.0):
        B = x.shape[0]
        h = self.encoder(x)                                   # (B, H)
        logits0 = self.heads(h).view(B, self.n_nodes, self.m)
        p0 = F.softmax(logits0 / temp, dim=-1)                # superposition
        z0 = self.express(p0)

        # routing round: attention over superposed concepts
        q, kk = self.att_q(z0), self.att_k(z0)
        att = F.softmax(q @ kk.transpose(1, 2) / math.sqrt(16), dim=-1)  # (B,N,N)
        ctx = att @ z0                                        # (B, N, d)
        logits = logits0 + self.refine(
            torch.cat([h.unsqueeze(1).expand(-1, self.n_nodes, -1), ctx], -1))
        p = F.softmax(logits / temp, dim=-1)                  # refined states
        z = self.express(p)

        g = torch.sigmoid(self.gates).view(1, -1, 1)
        out = self.out((g * z).reshape(B, -1))
        return out, p, att

    def ortho_penalty(self):
        eye = torch.eye(self.k)
        MtM = torch.einsum("ndk,ndj->nkj", self.M, self.M)
        return ((MtM - eye) ** 2).mean()


class MLPBaseline(nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_CLASSES))
    def forward(self, x):
        return self.net(x)


def n_params(model):
    return sum(q.numel() for q in model.parameters())

# parameter matching (standing rule 3: equalize by shrinking/growing elsewhere)
PARAM_TARGET = n_params(DCN(k=8, hidden=64))             # 16850, the v1 reference

def match_dcn_hidden(k, target=PARAM_TARGET):
    return min(range(16, 256), key=lambda h: abs(n_params(DCN(k=k, hidden=h)) - target))

def match_mlp_hidden(target=PARAM_TARGET):
    return min(range(16, 256), key=lambda h: abs(n_params(MLPBaseline(h)) - target))

# -----------------------------------------------------------------------------
# 3. Losses & training
# -----------------------------------------------------------------------------
def symbolic_loss(p):
    """Rule: sphere AND yellow is impossible. p: (B, N, m)."""
    return (p[:, 0, SPHERE_IDX] * p[:, 1, YELLOW_IDX]).mean()

def grounding_loss(p, attrs):
    """CE between node-state distribution and true attribute value (anchor j <-> value j)."""
    loss = 0.0
    for n, nv in enumerate(N_VALUES):
        loss = loss + F.nll_loss(torch.log(p[:, n, :nv] + 1e-9), attrs[:, n])
    return loss / len(N_VALUES)

def usage_loss(p):
    """Maximize batch-level anchor usage entropy per node (anti codebook-collapse)."""
    mean_p = p.mean(0)                                       # (N, m)
    ent = -(mean_p * (mean_p + 1e-9).log()).sum(-1)          # (N,)
    return -ent.mean()

def confidence_loss(p):
    """Minimize per-sample entropy (push toward collapse)."""
    return -(p * (p + 1e-9).log()).sum(-1).mean()

def train_dcn(seed, k=8, hidden=64, hardened=False,
              epochs=20, bs=128, lr=2e-3, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    X_tr, y_tr, A_tr = make_split(10000, sample_seed=1000 + seed, hardened=hardened)
    # 5% anchor-grounding mask (weak supervision used to pin anchor identities)
    ground_mask = torch.zeros(len(X_tr), dtype=torch.bool)
    ground_mask[torch.randperm(len(X_tr))[: int(0.05 * len(X_tr))]] = True

    model = DCN(k=k, hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X_tr)
    for ep in range(epochs):
        temp = max(0.5, 2.0 - 1.5 * ep / (epochs - 1))       # 2.0 -> 0.5 anneal
        conf_w = 0.02 * ep / (epochs - 1)                    # confidence ramps up
        perm = torch.randperm(n)
        tot, correct = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x, y, a, gm = X_tr[idx], y_tr[idx], A_tr[idx], ground_mask[idx]
            out, p, att = model(x, temp=temp)
            loss = F.cross_entropy(out, y)
            loss = loss + 1.0 * symbolic_loss(p)
            loss = loss + 0.5 * usage_loss(p)
            loss = loss + conf_w * confidence_loss(p)
            loss = loss + 0.1 * model.ortho_penalty()
            if gm.any():
                loss = loss + 1.0 * grounding_loss(p[gm], a[gm])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
            correct += (out.argmax(-1) == y).sum().item()
        if verbose and (ep % 4 == 0 or ep == epochs - 1):
            print(f"  ep {ep:2d}  temp {temp:.2f}  loss {tot/n:.3f}  train-acc {correct/n:.3f}")
    return model

def train_mlp(seed, hidden=96, hardened=False, epochs=20, bs=128, lr=2e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    X_tr, y_tr, _ = make_split(10000, sample_seed=1000 + seed, hardened=hardened)
    model = MLPBaseline(hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X_tr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            out = model(X_tr[idx])
            loss = F.cross_entropy(out, y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model

# -----------------------------------------------------------------------------
# 4. Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_dcn(dcn, X_te, y_te, A_te):
    res = {}
    out, p, att = dcn(X_te, temp=0.5)
    states = p.argmax(-1)                                    # (B, N) collapsed states

    res["dcn_params"] = n_params(dcn)
    res["dcn_test_acc"] = (out.argmax(-1) == y_te).float().mean().item()

    # concept purity: NMI between each node's collapsed state and each true attribute
    nmi = np.zeros((dcn.n_nodes, len(ATTR_NAMES)))
    for n_i in range(dcn.n_nodes):
        for a_i in range(len(ATTR_NAMES)):
            nmi[n_i, a_i] = normalized_mutual_info_score(
                states[:, n_i].numpy(), A_te[:, a_i].numpy())
    res["purity_matrix"] = nmi.round(3).tolist()
    res["purity_best_match"] = {f"node_{i}": ATTR_NAMES[int(nmi[i].argmax())]
                                for i in range(dcn.n_nodes)}
    res["purity_diag_mean"] = float(np.mean([nmi[i].max() for i in range(dcn.n_nodes)]))
    res["purity_offdiag_max"] = float(max(
        nmi[i, j] for i in range(dcn.n_nodes) for j in range(len(ATTR_NAMES))
        if j != int(nmi[i].argmax())))

    # state accuracy: collapsed anchor == true value (anchors grounded by 5% supervision)
    state_acc = [(states[:, n_i] == A_te[:, n_i]).float().mean().item()
                 for n_i in range(dcn.n_nodes)]
    res["state_accuracy_per_node"] = dict(zip(ATTR_NAMES, [round(s, 3) for s in state_acc]))

    # codebook usage: thresholded "alive" metric (wins >1% of samples; the old
    # argmax-unique count is deprecated as of Phase 0) + raw count for reference
    usage = []
    for n_i in range(dcn.n_nodes):
        shares = [(states[:, n_i] == a).float().mean().item() for a in range(dcn.m)]
        usage.append({"node": ATTR_NAMES[n_i],
                      "anchors_alive": int(sum(s > 0.01 for s in shares)),
                      "anchors_used_argmax": int(sum(s > 0 for s in shares)),
                      "true_values": N_VALUES[n_i], "anchors_available": dcn.m})
    res["codebook_usage"] = usage

    # rule check: hard violations (sphere & yellow jointly collapsed) + soft mass
    viol = ((states[:, 0] == SPHERE_IDX) & (states[:, 1] == YELLOW_IDX)).float().mean().item()
    res["rule_violation_rate_hard"] = viol
    res["rule_violation_soft_mass"] = (p[:, 0, SPHERE_IDX] * p[:, 1, YELLOW_IDX]).mean().item()

    res["mean_routing_graph"] = np.round(att.mean(0).numpy(), 2).tolist()
    return res

# -----------------------------------------------------------------------------
# 5. Phase runners (5 seeds, medians + IQR, per-run JSON log)
# -----------------------------------------------------------------------------
def _median_iqr(vals):
    v = np.asarray(vals, dtype=float)
    return {"median": float(np.median(v)),
            "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))]}

def _write(outdir, phase, desc, seed, payload):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"results_{phase}_{desc}_{seed}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def _run_job(job):
    kind, seed, k, hidden, hardened = job
    X_te, y_te, A_te = make_split(2500, sample_seed=20000 + seed, hardened=hardened)
    if kind == "dcn":
        model = train_dcn(seed, k=k, hidden=hidden, hardened=hardened)
        res = evaluate_dcn(model, X_te, y_te, A_te)
    else:
        model = train_mlp(seed, hidden=hidden, hardened=hardened)
        with torch.no_grad():
            res = {"mlp_test_acc": (model(X_te).argmax(-1) == y_te).float().mean().item(),
                   "mlp_params": n_params(model)}
    return kind, seed, k, hidden, res

def run_phase0(seeds, outdir):
    mlp_h = match_mlp_hidden()
    jobs = [("dcn", s, 8, 64, False) for s in seeds] + \
           [("mlp", s, None, mlp_h, False) for s in seeds]
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_run_job, jobs))
    dcn_r = {s: r for kind, s, _, _, r in results if kind == "dcn"}
    mlp_r = {s: r for kind, s, _, _, r in results if kind == "mlp"}
    for kind, s, k, hidden, r in results:
        _write(outdir, "phase0", "mvp_" + kind, s,
               {"config": {"kind": kind, "k": k, "hidden": hidden, "hardened": False},
                "seed": s, **r})
    summary = {
        "phase": 0, "seeds": list(seeds),
        "dcn_params": next(iter(dcn_r.values()))["dcn_params"],
        "mlp_params": next(iter(mlp_r.values()))["mlp_params"],
        "dcn_test_acc": _median_iqr([r["dcn_test_acc"] for r in dcn_r.values()]),
        "mlp_test_acc": _median_iqr([r["mlp_test_acc"] for r in mlp_r.values()]),
        "purity_diag_mean": _median_iqr([r["purity_diag_mean"] for r in dcn_r.values()]),
        "purity_offdiag_max": _median_iqr([r["purity_offdiag_max"] for r in dcn_r.values()]),
        "state_accuracy": {a: _median_iqr([r["state_accuracy_per_node"][a]
                                           for r in dcn_r.values()]) for a in ATTR_NAMES},
        "anchors_alive": {a: [r["codebook_usage"][i]["anchors_alive"]
                              for r in dcn_r.values()]
                          for i, a in enumerate(ATTR_NAMES)},
        "rule_violation_rate_hard": _median_iqr(
            [r["rule_violation_rate_hard"] for r in dcn_r.values()]),
        "rule_violation_soft_mass": _median_iqr(
            [r["rule_violation_soft_mass"] for r in dcn_r.values()]),
        "all_nodes_matched_distinct_attrs": all(
            len(set(r["purity_best_match"].values())) == 4 for r in dcn_r.values()),
    }
    _write(outdir, "phase0", "summary", "5seeds", summary)
    print(json.dumps(summary, indent=2))
    return summary

def run_phase1(seeds, outdir, hardened=False, ks=(1, 2, 4, 8, 16)):
    desc_suffix = "hard" if hardened else "base"
    jobs = [("dcn", s, k, match_dcn_hidden(k), hardened) for k in ks for s in seeds]
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_run_job, jobs))
    by_k = {k: {} for k in ks}
    for kind, s, k, hidden, r in results:
        by_k[k][s] = r
        _write(outdir, "phase1", f"k{k}_{desc_suffix}", s,
               {"config": {"k": k, "hidden": hidden, "hardened": hardened},
                "seed": s, **r})
    summary = {"phase": 1, "task": desc_suffix, "seeds": list(seeds),
               "param_target": PARAM_TARGET, "per_k": {}}
    for k in ks:
        runs = by_k[k]
        summary["per_k"][k] = {
            "hidden": match_dcn_hidden(k),
            "n_params": next(iter(runs.values()))["dcn_params"],
            "acc": _median_iqr([r["dcn_test_acc"] for r in runs.values()]),
            "purity": _median_iqr([r["purity_diag_mean"] for r in runs.values()]),
            "state_acc_mean": _median_iqr(
                [float(np.mean(list(r["state_accuracy_per_node"].values())))
                 for r in runs.values()]),
            "rule_soft_mass": _median_iqr(
                [r["rule_violation_soft_mass"] for r in runs.values()]),
            "rule_hard_rate": _median_iqr(
                [r["rule_violation_rate_hard"] for r in runs.values()]),
        }
    # pre-committed Phase 1 criteria (ROADMAP.md): k=1 measurably worse means
    # purity gap k=4 vs k=1 >= 5 NMI pts OR >= 10x soft rule mass
    gap = (summary["per_k"][4]["purity"]["median"]
           - summary["per_k"][1]["purity"]["median"]) * 100
    m1 = summary["per_k"][1]["rule_soft_mass"]["median"]
    m4 = summary["per_k"][4]["rule_soft_mass"]["median"]
    summary["purity_gap_k4_vs_k1_nmi_pts"] = round(gap, 2)
    summary["soft_mass_ratio_k1_over_k4"] = float(m1 / m4) if m4 > 0 else float("inf")
    summary["k1_measurably_worse"] = bool(
        gap >= 5 or summary["soft_mass_ratio_k1_over_k4"] >= 10)
    _write(outdir, "phase1", f"summary_{desc_suffix}", "5seeds", summary)
    print(json.dumps(summary, indent=2))
    return summary

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[0, 1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--hardened", action="store_true",
                    help="Phase 1 fallback task; use only if k=1 ties on base")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    if args.phase == 0:
        run_phase0(args.seeds, args.outdir)
    else:
        run_phase1(args.seeds, args.outdir, hardened=args.hardened)
