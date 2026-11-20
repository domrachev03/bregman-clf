"""Numbers quoted in the text for the metric study, as LaTeX macros: reads metric_study_demand.json,
metric_study_table.json and metric_study_table.parquet in results/manipulator/ and writes
metric_study_numbers.tex there.

    python experiments/manipulator/metric_study_numbers.py
"""

import json

import pandas as pd
from _paths import RESULTS

dem = json.loads((RESULTS / "metric_study_demand.json").read_text())
tab = json.loads((RESULTS / "metric_study_table.json").read_text())
tp = pd.read_parquet(RESULTS / "metric_study_table.parquet")
hm = tp[tp.metric == "hessian"]


def fmt(v):
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


near_out = []
for key, v in tab.items():
    if key.startswith(("constant_", "target_")):
        near_out.append(int(v["cells"][5]))
h = dem["states_hessian"]
c_, t_ = dem["states_constant"], dem["states_target"]
mac = {
    "NEoneOutMaxPct": f"{100 * max(near_out) / 80:.0f}",
    "NEoneOutMax": str(max(near_out)),
    "NEoneVtwoErr": f"{dem['V2_rel_err_p99']:.1e}".replace("e-", "\\cdot10^{-") + "}",
    "NEoneNearN": str(h["n_near"]),
    "NEoneNearTrials": str(h["n_trials"]),
    "NEoneHonHess": fmt(h["near_p99_hessian"]),
    "NEoneHonConst": fmt(h["near_p99_constant"]),
    "NEoneHonTarget": fmt(h["near_p99_target"]),
    "NEoneConstOnHessMin": fmt(min(c_["near_p99_hessian"], t_["near_p99_hessian"])),
    "NEoneConstOnHessMax": fmt(max(c_["near_p99_hessian"], t_["near_p99_hessian"])),
    "NEoneDemFarMin": fmt(min(v for d in (h, c_, t_) for k, v in d.items() if k.startswith("far"))),
    "NEoneDemFarMax": fmt(max(v for d in (h, c_, t_) for k, v in d.items() if k.startswith("far"))),
    "NEoneEntOut": str(tab["entropy_constant"]["out"]),
    # smallest signed keep-in margin of the Hessian metric over all logged steps and trials
    "NEoneHessMinMargin": f"{1000 * hm[hm.psi != 'entropy'].min_dist.min():.2f}",
    "NEoneHessMinMarginEnt": f"{1e6 * hm[hm.psi == 'entropy'].min_dist.min():.2f}",
    "NEoneHessTrials": str(len(hm[hm.psi != "entropy"])),
}
(RESULTS / "metric_study_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print(mac)
