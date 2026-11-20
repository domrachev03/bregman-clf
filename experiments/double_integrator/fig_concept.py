"""Fig. 1 (four panels) from the double-integrator model of dint.py -> fig_concept.pdf, fig_concept.png.

(a) sublevel sets of D_psi(x_d, .) (inside C) vs a quadratic V (crossing dC);
(b) velocity-error tolerance {s : 1/2 s^T Gamma s <= 1/2} near a face, Hessian vs constant metric;
(c) demand d_Gamma at states approaching a face slowly (distance eta, velocity error kappa*eta*n) vs eta;
(d) worst demand on sublevel sets {V2 <= c} (dint_demand*.json, running maximum).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon

from dint import RESULTS, A, B, Model, slacks
from dint_report import COL, LBL
from dint_study import k_bound

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Nimbus Roman", "STIX Two Text", "Times"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.linewidth": 0.6, "axes.titlesize": 8.5})
XD = np.array([0.3, -0.2])


def vertices():
    from itertools import combinations
    V = []
    for i, j in combinations(range(len(A)), 2):
        M = np.stack([A[i], A[j]])
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        v = np.linalg.solve(M, np.array([B[i], B[j]]))
        if np.all(slacks(v) > -1e-9):
            V.append(v)
    V = np.array(V)
    c = V.mean(axis=0)
    return V[np.argsort(np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]))]


def panel_a(ax):
    m = Model(mu=0.15, p=1.0, xd=XD)  # larger barrier weight than in the study, for visibility
    g = np.linspace(-1.35, 1.35, 400)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X, Y], -1)
    inside = np.all(slacks(P) > 0, axis=-1)
    Dv = np.full(X.shape, np.nan)
    for i, j in zip(*np.nonzero(inside)):
        Dv[i, j] = m.D(P[i, j])
    Q = 0.5 * ((X - XD[0]) ** 2 + (Y - XD[1]) ** 2)
    ax.add_patch(Polygon(vertices(), closed=True, fill=False, lw=1.0, color="k"))
    ax.contour(X, Y, Dv, levels=[0.1, 0.35, 0.9, 2.5], colors=COL["hess"], linewidths=0.9)
    ax.contour(X, Y, Q, levels=[0.1, 0.35, 0.9, 2.5], colors="0.5", linewidths=0.7, linestyles="--")
    ax.plot(*XD, "k*", ms=5)
    ax.text(XD[0] + 0.06, XD[1] + 0.06, r"$\mathbf{x}_d$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"(a) level sets", pad=3)


def panel_b(ax):
    """Unit balls of the velocity-error metric at states approaching a face (face on top)."""
    mu = 1e-2
    ax.fill_between([-0.5, 3.2], 0, 0.45, color="0.88", lw=0)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.text(2.55, 0.13, r"$\partial\mathcal{C}$")
    for i, eta in enumerate([0.9, 0.4, 0.15]):
        xc, y = 0.45 + 1.0 * i, -eta
        hn = 1.0 + 30 * mu / eta**2  # normal Hessian entry grows like mu/eta^2 (illustrative scale)
        r = 0.36
        ax.add_patch(Ellipse((xc, y), 2 * r, 2 * r, fill=False, color=COL["I"], lw=0.8, ls="--"))
        ax.add_patch(Ellipse((xc, y), 2 * r, 2 * r / np.sqrt(hn), fill=False, color=COL["hess"], lw=1.1))
        ax.plot(xc, y, "k.", ms=3)
    ax.text(1.0, -1.05, r"$\|\mathbf{s}\|_{\nabla^2\psi}\!\leq\!1$", color=COL["hess"], fontsize=8)
    ax.text(1.0, -1.38, r"$\|\mathbf{s}\|\!\leq\!1$", color=COL["I"], fontsize=8)
    ax.set_xlim(-0.05, 3.0)
    ax.set_ylim(-1.5, 0.45)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"(b) metric (schematic)", pad=3)


def panel_c(ax):
    m0 = Model(mu=1e-2, xd=XD)
    sl = slacks(XD)
    i = int(np.argmin(sl))
    delta, n = sl[i], A[i]
    kappa = 2 * m0.k * len(A)
    etas = np.logspace(-5, -2, 40)
    # Gamma=I and Gamma=H(x_d) nearly coincide (P=I, mu=1e-2): draw H(x_d) dashed on top of I
    for key, lam, label, color, ls, lw in [("hess", 1, LBL["hess"], COL["hess"], "-", 1.1),
                                           ("I", 1, LBL["I"], COL["I"], "-", 2.0),
                                           ("target", 1, LBL["target"], COL["target"], (0, (2.5, 1.5)), 1.1),
                                           ("I", 1e4, r"$\Gamma=10^4I$", "0.45", (0, (4, 1.5)), 1.0)]:
        m = Model(mu=1e-2, metric=key, xd=XD, lam=lam)
        d = []
        for eta in etas:
            x = XD + (delta - eta) * n
            d.append(m.demand(x, m.xdot_r(x) + kappa * eta * n))
        d = np.array(d)
        pos = d > 0  # zero demand where the rate constraint is inactive (10^4 I, large eta)
        ax.loglog(etas[pos], d[pos], color=color, ls=ls, lw=lw, label=label)
    ax.text(1.5e-5, 2e5, r"$\propto\eta^{-2}$", fontsize=8)
    ax.set_xlabel(r"distance $\eta$ to the face")
    ax.set_ylabel(r"demand $d_\Gamma$")
    ax.legend(frameon=False, fontsize=7, loc="upper right", handlelength=1.8, labelspacing=0.2)
    ax.set_title(r"(c) demand at slow approach", pad=3)


def panel_d(ax):
    dem = json.loads((RESULTS / "dint_demand.json").read_text())
    sc = json.loads((RESULTS / "dint_demand_scaled.json").read_text())
    c = np.array(dem["c"])
    for metric, ls, lw in [("hess", "-", 1.1), ("I", "-", 2.0), ("target", (0, (2.5, 1.5)), 1.1)]:
        ax.loglog(c, np.maximum.accumulate(dem[f"{metric}_mu0.01"]), ls=ls, lw=lw, color=COL[metric])
    ax.loglog(c, np.maximum.accumulate(sc["I10000_mu0.01"]), color="0.45", ls=(0, (4, 1.5)), lw=0.9)
    mb = Model(mu=1e-2, metric="hess", xd=XD)
    ax.loglog(c, [k_bound(mb, np.sqrt(2 * ci), ci) for ci in c], ":", color=COL["hess"], lw=1.0)
    ax.text(0.012, 250, r"bound $K(c)$", color=COL["hess"], fontsize=8)
    ax.set_xlabel(r"level $c$ of $\{V_2\leq c\}$")
    ax.set_ylabel(r"worst demand")
    ax.set_title(r"(d) worst case on sublevel sets", pad=3)


def main():
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.0), gridspec_kw={"width_ratios": [1, 1.05, 1.2, 1.2]})
    panel_a(axs[0])
    panel_b(axs[1])
    panel_c(axs[2])
    panel_d(axs[3])
    for ax in axs[2:]:
        ax.grid(True, which="major", lw=0.3, alpha=0.5)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(RESULTS / "fig_concept.pdf")
    fig.savefig(RESULTS / "fig_concept.png", dpi=200)


if __name__ == "__main__":
    main()
