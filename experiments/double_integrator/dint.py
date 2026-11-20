"""Double-integrator model of the numerical study.

Plant: x_ddot = a in R^2, keep-in polytope C = {A x < b} (unit-norm rows), generating function
psi = p/2 |x|^2 + mu F, F = -sum log(b - A x). Order-two Bregman row with backstepping metric
Gamma in {"hess" (Gamma = Hess psi(x)), "I", "target" (Gamma = Hess psi(x_d))}:

    V2 = D_psi(x_d, x) + 1/2 s^T Gamma s,  s = x_dot - x_dot_r,
    x_dot_r = -k H(x)^{-1} (grad psi(x) - grad psi(x_d)),

rate constraint  V2_dot + gamma V2 <= 0  (affine in a). `demand` = norm of the minimum-norm a satisfying
it. `metric="I"` with `lam` gives Gamma = lam I.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Result files (JSON, tables, figures) are read from and written to this directory.
RESULTS = Path(__file__).resolve().parents[2] / "results" / "double_integrator"

# Irregular pentagon around the origin (unit outward normals), diameter ~ 2.
_ANG = np.array([0.1, 1.4, 2.6, 3.7, 5.0])
A = np.stack([np.cos(_ANG), np.sin(_ANG)], axis=1)
B = np.array([1.0, 0.8, 1.1, 0.9, 1.0])


def slacks(x):
    return B - x @ A.T  # (..., 5)


def inside(x, margin=0.0):
    return np.all(slacks(x) > margin, axis=-1)


class Model:
    def __init__(self, mu=1e-2, p=1.0, k=2.0, ks=2.0, gamma=None, metric="hess", xd=None, lam=1.0):
        self.mu, self.p, self.k, self.ks, self.lam = mu, p, k, ks, lam
        self.gamma = min(k, 2 * ks) if gamma is None else gamma
        self.metric = metric
        self.xd = np.zeros(2) if xd is None else np.asarray(xd, float)
        self.Hd = self.hess(self.xd)
        self.gd = self.grad(self.xd)

    # psi and derivatives (single point x: shape (2,))
    def psi(self, x):
        return 0.5 * self.p * x @ x - self.mu * np.sum(np.log(slacks(x)))

    def grad(self, x):
        return self.p * x + self.mu * A.T @ (1.0 / slacks(x))

    def hess(self, x):
        s = slacks(x)
        return self.p * np.eye(2) + self.mu * (A.T * (1.0 / s**2)) @ A

    def hdot(self, x, xdot):
        s = slacks(x)
        w = 2.0 * (A @ xdot) / s**3
        return self.mu * (A.T * w) @ A

    def D(self, x):  # D_psi(x_d, x)
        return self.psi(self.xd) - self.psi(x) - self.grad(x) @ (self.xd - x)

    def Gamma(self, x):
        if self.metric == "hess":
            return self.hess(x)
        if self.metric == "I":
            return self.lam * np.eye(2)
        return self.Hd

    def xdot_r(self, x):
        return -self.k * np.linalg.solve(self.hess(x), self.grad(x) - self.gd)

    def xddot_r(self, x, xdot):
        H = self.hess(x)
        e = np.linalg.solve(H, self.grad(x) - self.gd)
        return -self.k * (xdot - np.linalg.solve(H, self.hdot(x, xdot) @ e))

    def V2(self, x, s):
        return self.D(x) + 0.5 * s @ self.Gamma(x) @ s

    def row(self, x, xdot):
        """Row  c^T a <= r  (V2_dot + gamma V2 <= 0)."""
        H = self.hess(x)
        s = xdot - self.xdot_r(x)
        G = self.Gamma(x)
        xr2 = self.xddot_r(x, xdot)
        r = -self.gamma * self.V2(x, s) - (x - self.xd) @ H @ xdot + s @ G @ xr2
        if self.metric == "hess":
            r -= 0.5 * s @ self.hdot(x, xdot) @ s
        return G @ s, r

    def demand(self, x, xdot):
        c, r = self.row(x, xdot)
        if r >= 0:
            return 0.0
        n = np.linalg.norm(c)
        return np.inf if n == 0 else -r / n

    def law(self, x, xdot):
        """Backstepping law of the metric; it satisfies the rate constraint."""
        H = self.hess(x)
        s = xdot - self.xdot_r(x)
        xr2 = self.xddot_r(x, xdot)
        if self.metric == "hess":
            sd = -(x - self.xd) - self.ks * s - 0.5 * np.linalg.solve(H, self.hdot(x, xdot) @ s)
        else:
            G = self.Gamma(x)
            sd = -np.linalg.solve(G, H @ (x - self.xd)) - self.ks * s
        return xr2 + sd
