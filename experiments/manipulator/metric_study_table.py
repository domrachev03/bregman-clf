"""Manipulator metric-study table (Table II): per metric, constraint form and condition, with the
Euclidean keep-in distance recomputed from the logged end-effector positions. Needs the episode
logs of the sweep `configs/metric_study.yaml` in `runs/metric_study`.

    python experiments/manipulator/metric_study_table.py   (one process, about 1 minute)

Writes results/manipulator/metric_study_table.tex (Table II), .json and .parquet (per-trial records)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml
from _paths import RESULTS
from metric_study_demand import CFG, RUNS, Barrier, make_scenario

METRICS = [("constant", r"$\Gamma=I$"), ("target", r"$\Gamma=\nabla^2\psi(\xd)$"), ("hessian", r"$\Gamma=\nabla^2\psi(\x)$")]
FORMS = [("hard", "rate"), ("level", "level")]
CONDS = ["nom", "sat", "near"]


def cond_of(s):
    return "near" if s.endswith("near") else ("sat" if s.endswith("sat") else "nom")


def main():
    cfg = yaml.safe_load(CFG.read_text())
    opts = {s["name"]: s["options"] for s in cfg["scenarios"]}
    summ = pd.read_parquet(RUNS / "summary.parquet")
    recs = []
    for _, r in summ.iterrows():
        scen = r.scenario
        psi = scen.split("_")[2]
        sc = make_scenario(scen, **dict(opts[scen]))
        sc.reset(np.random.default_rng(int(r.seed)))
        bar = Barrier(sc.sets()["keep_in"]) if psi != "entropy" else None
        d = np.load(RUNS / scen / r.method / f"{int(r.seed)}.npz")
        X = np.asarray(d["ground_truth__ee_pos"], dtype=float)
        if bar is not None:
            dist = np.array([bar.dist(x) for x in X])
        else:  # entropic box: distance to the box
            ki = sc.sets()["keep_in"]
            dist = np.min(np.minimum(X - ki["lo"], ki["hi"] - X), axis=1)
        metric, form = r.method.split("_")[1], r.method.split("_")[2]
        recs.append(dict(psi=psi, cond=cond_of(scen), metric=metric, form=form, seed=int(r.seed),
                         fb=float(r.fallback_rate), out=bool(dist.min() < -1e-3), depth=float(max(0.0, -dist.min())),
                         min_dist=float(dist.min()),
                         jnt=bool(r.qpos_violation_rate_tol > 0)))
    df = pd.DataFrame(recs)
    df.to_parquet(RESULTS / "metric_study_table.parquet")
    sc_df = df[df.psi != "entropy"]
    out = {}
    lines = []
    for metric, mlabel in METRICS:
        for form, flabel in FORMS:
            sub = sc_df[(sc_df.metric == metric) & (sc_df.form == form)]
            cells = []
            for c in CONDS:
                s = sub[sub.cond == c]
                cells += [f"{100 * s.fb.mean():.2f}", f"{int(s.out.sum())}"]
            depth = 1000 * sub.depth.max()
            jnt = int(sub.jnt.sum())
            out[f"{metric}_{form}"] = dict(cells=cells, depth_mm=depth, jnt=jnt, n=len(sub))
            first = mlabel if form == "hard" else ""
            lines.append(f"{first} & {flabel} & " + " & ".join(cells) + f" & {depth:.0f} & {jnt} \\\\")
    ent = df[df.psi == "entropy"]
    for metric, _ in METRICS:
        s = ent[ent.metric == metric]
        out[f"entropy_{metric}"] = dict(out=int(s.out.sum()), fb=float(100 * s.fb.mean()), n=len(s),
                                        depth_mm=float(1000 * s.depth.max()))
    (RESULTS / "metric_study_table.json").write_text(json.dumps(out, indent=1))
    n = len(sc_df[(sc_df.metric == "hessian") & (sc_df.form == "hard")]) // 3
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Manipulator, backstepping-metric study (polytope and ball $\psi$, " + str(n)
           + r" trials per condition and row; $T=1.5$\,s). Per condition (nominal, $30\%$ torque bounds, near-boundary start): QP-infeasible steps [\%] and trials leaving $\dom$ by more than 1\,mm; deepest excursion [mm] and trials exceeding a joint limit by more than 1\,mrad, over all conditions.}",
           r"\label{tab:e1}", r"\footnotesize", r"\setlength{\tabcolsep}{2.2pt}",
           r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
           r"& & \multicolumn{2}{c}{nominal} & \multicolumn{2}{c}{$0.3\tau_{\max}$} & \multicolumn{2}{c}{near} & & \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
           r"metric & row & inf. & out & inf. & out & inf. & out & depth & jnt \\", r"\midrule"]
    tex += lines
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (RESULTS / "metric_study_table.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(tex))
    print(json.dumps({k: v for k, v in out.items() if k.startswith("entropy")}, indent=1))


if __name__ == "__main__":
    main()
