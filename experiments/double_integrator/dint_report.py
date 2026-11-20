"""Table I (dint_loop.tex), the macros of the text (dint_numbers.tex) and the demand figure
(dint_demand.pdf) of the double-integrator study, from dint_demand*.json and
dint_closed_loop_mu0.01*.json (see dint_study.py). Everything is in results/double_integrator/."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dint import RESULTS, A, B, Model, slacks
from dint_study import k_bound

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["Nimbus Roman", "STIX Two Text", "Times"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6})
LBL = {"hess": r"$\Gamma=\nabla^2\psi(\mathbf{x})$", "I": r"$\Gamma=I$", "target": r"$\Gamma=\nabla^2\psi(\mathbf{x}_d)$"}
COL = {"hess": "#1f5fa8", "I": "#c0392b", "target": "#e08a1e"}


def diameter():
    from itertools import combinations
    V = []
    for i, j in combinations(range(len(A)), 2):
        M = np.stack([A[i], A[j]])
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        v = np.linalg.solve(M, np.array([B[i], B[j]]))
        if np.all(slacks(v) > -1e-9):
            V.append(v)
    return max(np.linalg.norm(a - b) for a in V for b in V)


def figure(dem):
    c = np.array(dem["c"])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for metric in ["hess", "I", "target"]:
        ax.loglog(c, np.maximum.accumulate(dem[f"{metric}_mu0.01"]), "-o", ms=2.5, lw=1.0, color=COL[metric], label=LBL[metric])
        ax.loglog(c, np.maximum.accumulate(dem[f"{metric}_mu0.001"]), "--", lw=0.8, color=COL[metric])
    mb = Model(mu=1e-2, metric="hess", xd=np.array([0.3, -0.2]))
    bound = [k_bound(mb, np.sqrt(2 * ci), ci) for ci in c]
    sc = json.loads((RESULTS / "dint_demand_scaled.json").read_text())
    for lam, ls in [(1e2, (0, (1, 1.2))), (1e4, (0, (4, 1.5)))]:
        ax.loglog(c, np.maximum.accumulate(sc[f"I{lam:g}_mu0.01"]), linestyle=ls, lw=0.9, color="0.45",
                  label=r"$\Gamma=10^{%d}I$" % round(np.log10(lam)))
    ax.loglog(c, bound, ":", lw=1.0, color=COL["hess"], label=r"bound $K(\sqrt{2c})$")
    ax.set_xlabel(r"level $c$ of $\{V_2\leq c\}$")
    ax.set_ylabel(r"worst demand")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.grid(True, which="major", lw=0.3, alpha=0.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(RESULTS / "dint_demand.pdf")


def slope(c, y):
    m = c >= 1.0
    return np.polyfit(np.log(c[m]), np.log(np.asarray(y)[m]), 1)[0]


def main():
    dem = json.loads((RESULTS / "dint_demand.json").read_text())
    for mu in (1e-2, 1e-3):  # refresh archived bound arrays with the current K(c)
        mb = Model(mu=mu, metric="hess", xd=np.array([0.3, -0.2]))
        dem[f"bound_mu{mu:g}"] = [k_bound(mb, np.sqrt(2 * ci), ci) for ci in dem["c"]]
    (RESULTS / "dint_demand.json").write_text(json.dumps(dem, indent=1))
    figure(dem)
    c = np.array(dem["c"])
    mac = {
        "NDintDiam": f"{diameter():.2f}",
        "NDintRatio": f"{dem['I_mu0.01'][-1] / dem['hess_mu0.01'][-1]:.0f}",
        "NDintSlopeHess": f"{slope(c, dem['hess_mu0.01']):.2f}",
        "NDintSlopeI": f"{slope(c, dem['I_mu0.01']):.2f}",
        "NDintSlopeTarget": f"{slope(c, dem['target_mu0.01']):.2f}",
        "NDintMuHess": f"{dem['hess_mu0.001'][-1] / dem['hess_mu0.01'][-1]:.1f}",
        "NDintMuI": f"{dem['I_mu0.001'][-1] / dem['I_mu0.01'][-1]:.1f}",
        "NDintMuTarget": f"{dem['target_mu0.001'][-1] / dem['target_mu0.01'][-1]:.1f}",
    }
    loop = json.loads((RESULTS / "dint_closed_loop_mu0.01.json").read_text())
    for extra in ("_high", "_scaled", "_scaled_high"):
        f = RESULTS / f"dint_closed_loop_mu0.01{extra}.json"
        if f.exists():
            loop.update(json.loads(f.read_text()))
    amaxes = sorted({float(k.split("_a")[1]) for k in loop})
    n = len(loop[next(iter(loop))])
    mac["NDintTrials"] = str(n)
    certrow = []
    for amax in amaxes:
        hess = loop[f"hess_a{amax:g}"]
        cert, cert_inf = 0, 0
        for o in hess:
            m = Model(mu=1e-2, metric="hess", xd=np.array(o["xd"]))
            if k_bound(m, np.sqrt(2 * o["V20"]), o["V20"]) < amax:
                cert += 1
                cert_inf += o["infeas"] > 0
        certrow.append(f"{cert} ({cert_inf})")
    # Estimated certificates (dint_estcert.py): sampled worst demand on {V2 <= V2(0)} below a_max.
    estrow = []
    est = json.loads((RESULTS / "dint_estcert.json").read_text())["trials"]
    for amax in amaxes:
        hess = loop[f"hess_a{amax:g}"]
        assert all(abs(o["V20"] - e["V20"]) < 1e-9 for o, e in zip(hess, est))
        ids = [i for i, e in enumerate(est) if e["est"] < amax]
        estrow.append(f"{len(ids)} ({sum(hess[i]['infeas'] > 0 for i in ids)})")
        tag = {10.0: "Ten", 30.0: "Thirty", 100.0: "Hundred"}.get(amax)
        if tag:
            mac[f"NDintEst{tag}"] = str(len(ids))
            mac[f"NDintEstExitI{tag}"] = str(sum(loop[f"I_a{amax:g}"][i]["exit"] for i in ids))
            mac[f"NDintEstExitTarget{tag}"] = str(sum(loop[f"target_a{amax:g}"][i]["exit"] for i in ids))
            mac[f"NDintEstExitHess{tag}"] = str(sum(hess[i]["exit"] for i in ids))
    ratio = [e["K"] / e["est"] for e in est]
    mac["NDintKRatioMed"] = f"{np.median(ratio):.0f}"
    mac["NDintKRatioMax"] = f"{np.max(ratio):.0f}"
    rows = [(r"$\nabla^2\psi(\x)$", "hess"), (r"$I$", "I"), (r"$\nabla^2\psi(\xd)$", "target"),
            (r"$10^2I$", "I1e+02"), (r"$10^4I$", "I1e+04")]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Double integrator under $\|\ac\|\le\bar a$ (" + str(n) + r" trials per cell): trials with an infeasible step / trials leaving $\dom$, per metric $\Gamma$. Last rows: Hessian-metric trials certified by Cor.~\ref{cor:feas} via $K(V_2(0))<\bar a$, and via the sampled worst demand on $\{V_2\le V_2(0)\}$ (a lower estimate, not a proof); in parentheses, those that met an infeasible step.}",
           r"\label{tab:dint}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}l" + "r" * len(amaxes) + r"@{}}", r"\toprule",
           r"$\Gamma$ \textbackslash\ $\bar a$ & " + " & ".join(f"{a:g}" for a in amaxes) + r" \\", r"\midrule"]
    for label, key in rows:
        cells = []
        for amax in amaxes:
            out = loop.get(f"{key}_a{amax:g}")
            cells.append("--" if out is None else f"{sum(o['infeas'] > 0 for o in out)}/{sum(o['exit'] for o in out)}")
        tex.append(label + " & " + " & ".join(cells) + r" \\")
    tex += [r"\midrule", r"certified, $K$ & " + " & ".join(certrow) + r" \\",
            r"certified, est. & " + " & ".join(estrow) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (RESULTS / "dint_loop.tex").write_text("\n".join(tex) + "\n")
    (RESULTS / "dint_numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    print(mac)
    print("\n".join(tex))


if __name__ == "__main__":
    main()
