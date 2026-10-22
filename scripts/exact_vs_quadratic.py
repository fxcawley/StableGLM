"""Exact Rashomon quantities against their quadratic approximation, and whether the cubic and
quartic terms of the loss predict the difference.

For every case (a fitted logistic regression and a tolerance) the script records

- per coefficient: the exact half-ranges from ``coef_extremes`` relative to the ellipsoid
  half-range ``r * sd`` (``r = sqrt(2 eps)``), i.e. the observed relative deviation of the
  ellipsoid; the deviation predicted by ``RashomonSet.quadratic_error``; and the residual;
- on a subsample of rows: flips by the exact test (``can_flip``), by the ellipsoid screen
  (ellipsoid margin range straddles the threshold) and by the quartic-corrected screen;
- the fraction of Laplace-posterior draws inside the exact set against its nominal mass
  ``F_chi2_d(2 n eps)``.

Cases are the real datasets of ``evaluate.py`` (plus the unpenalized UCLA admissions fit) and
a synthetic grid over n, d, signal strength, feature correlation, penalty and tolerance.

Run from the repository root:  python scripts/exact_vs_quadratic.py [--quick]
Writes docs/_static/exact_vs_quadratic.json and .png and prints the Markdown tables used in
docs/evaluation.md. About ten minutes.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from scipy.stats import chi2  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_scale import load_adult, load_german_credit  # noqa: E402
from evaluate import select_c  # noqa: E402
from rashomon import RashomonSet, audit  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "tests", "data")
STATIC = os.path.join(ROOT, "docs", "_static")
QUICK = "--quick" in sys.argv
N_FLIP_ROWS = 100 if QUICK else 300
N_POSTERIOR = 1000 if QUICK else 2000


def measure(name: str, kind: str, rs: RashomonSet, X: np.ndarray, tol_label: str, extra: dict) -> dict:
    """All comparisons for one fitted RashomonSet; X holds the raw features (no intercept column)."""
    t0 = time.perf_counter()
    n, d, eps = rs._n, rs._d, float(rs._epsilon_value)
    r = np.sqrt(2 * eps)
    theta = rs._theta_hat
    Xa = rs._prepare_X(X)
    p = 1 / (1 + np.exp(-(Xa @ theta)))
    Hinv = np.linalg.inv(rs._hessian_matrix())
    sd = np.sqrt(np.diag(Hinv))

    # coefficient ranges: observed deviation of the ellipsoid, the ray estimate and the quartic one
    t1 = time.perf_counter()
    ex = rs.coef_extremes()
    t_exact_ranges = time.perf_counter() - t1
    observed = np.c_[(theta - ex[:, 0]) / (r * sd) - 1, (ex[:, 1] - theta) / (r * sd) - 1]
    ray = rs.quadratic_error()
    quartic = rs.quadratic_error(method="quartic")
    t1 = time.perf_counter()
    newton = rs.quadratic_error(method="newton")
    t_newton = time.perf_counter() - t1
    residual, residual_q, residual_n = observed - ray, observed - quartic, observed - newton

    # flips on a subsample of rows, threshold 0 on the logit scale
    rng = np.random.default_rng(0)
    rows = rng.choice(n, size=min(N_FLIP_ROWS, n), replace=False)
    Xs = Xa[rows]
    margin = Xs @ theta
    norms = np.sqrt(np.sum(Xs.T * rs._hinv(Xs.T), axis=0))
    t1 = time.perf_counter()
    flip_exact = rs.can_flip(X[rows], 0.0, coef_box=ex)
    t_flip_exact = time.perf_counter() - t1
    flip_ell = np.abs(margin) <= r * norms
    dev_rows = rs.quadratic_error(Xs)
    half_lo, half_hi = r * norms * (1 + dev_rows[:, 0]), r * norms * (1 + dev_rows[:, 1])
    flip_pred = np.where(margin > 0, margin - half_lo <= 0, margin + half_hi >= 0)
    dev_rows_n = rs.quadratic_error(Xs, method="newton")
    half_lo_n, half_hi_n = r * norms * (1 + dev_rows_n[:, 0]), r * norms * (1 + dev_rows_n[:, 1])
    flip_newton = np.where(margin > 0, margin - half_lo_n <= 0, margin + half_hi_n >= 0)

    # posterior mass of the exact set
    L = np.linalg.cholesky(Hinv / n)
    draws = theta[None, :] + rng.normal(size=(N_POSTERIOR, d)) @ L.T
    inside = rs.contains_many(draws)

    rec = {
        "name": name, "kind": kind, "n": int(n), "d": int(d), "tolerance": tol_label,
        "eps": eps, "eps_over_loss": eps / float(rs._L_hat), "r": float(r),
        "mean_p1p": float(np.mean(p * (1 - p))), "frac_saturated": float(np.mean((p < 0.05) | (p > 0.95))),
        "hpd_nominal": float(chi2.cdf(2 * n * eps, df=d)), "hpd_observed": float(inside.mean()),
        "ellipsoid_dev_median": float(np.median(np.abs(observed))), "ellipsoid_dev_max": float(np.abs(observed).max()),
        "residual_median": float(np.median(np.abs(residual))), "residual_max": float(np.abs(residual).max()),
        "residual_min_signed": float(residual.min()),  # negative would contradict the inner-bound property
        "quartic_residual_median": float(np.median(np.abs(residual_q))), "quartic_residual_max": float(np.abs(residual_q).max()),
        "newton_residual_median": float(np.median(np.abs(residual_n))), "newton_residual_max": float(np.abs(residual_n).max()),
        "newton_residual_min_signed": float(residual_n.min()), "seconds_newton": t_newton, "seconds_flip_exact": t_flip_exact,
        "seconds_exact_ranges": t_exact_ranges,
        "flip_newton": float(flip_newton.mean()), "flip_disagree_newton": float(np.mean(flip_newton != flip_exact)),
        "n_flip_rows": int(len(rows)), "flip_exact": float(flip_exact.mean()), "flip_ellipsoid": float(flip_ell.mean()),
        "flip_predicted": float(flip_pred.mean()),
        "flip_disagree_ellipsoid": float(np.mean(flip_ell != flip_exact)),
        "flip_disagree_predicted": float(np.mean(flip_pred != flip_exact)),
        "observed": observed.round(5).tolist(), "predicted": ray.round(5).tolist(), "quartic": quartic.round(5).tolist(),
        "newton": newton.round(5).tolist(), "seconds": time.perf_counter() - t0, **extra,
    }
    print(f"  {name:28} {tol_label:>8} eps/loss={rec['eps_over_loss']:6.1%} r={r:.3f} | ellipsoid dev max {rec['ellipsoid_dev_max']:.3f}"
          f" -> ray {rec['residual_max']:.3f} newton {rec['newton_residual_max']:.3f} quartic {rec['quartic_residual_max']:.3f}"
          f" | flips exact {rec['flip_exact']:.1%} ell {rec['flip_ellipsoid']:.1%} ray {rec['flip_predicted']:.1%} newton {rec['flip_newton']:.1%}"
          f" (disagree {rec['flip_disagree_ellipsoid']:.1%} / {rec['flip_disagree_predicted']:.1%} / {rec['flip_disagree_newton']:.1%})"
          f" | mass {rec['hpd_nominal']:.2f} vs {rec['hpd_observed']:.2f} | {rec['seconds']:.0f}s", flush=True)
    return rec


def real_cases() -> list[dict]:
    out = []
    bc = load_breast_cancer()
    y_bc = bc.target.astype(float)
    sets = [
        ("Breast Cancer PCA-10", PCA(n_components=10, random_state=0).fit_transform(StandardScaler().fit_transform(bc.data)), y_bc),
        ("Breast Cancer full", bc.data, y_bc),
    ]
    X_gc, y_gc = load_german_credit()
    sets.append(("German Credit", X_gc, y_gc))
    adult_path = os.path.join(DATA_DIR, "adult.data")
    if os.path.exists(adult_path) and not QUICK:
        X_ad, y_ad = load_adult(adult_path)
        idx = np.random.default_rng(0).choice(X_ad.shape[0], size=5000, replace=False)
        X_ad, y_ad = X_ad[idx], y_ad[idx]
        sets.append(("Adult Census (5k rows)", X_ad[:, X_ad.std(axis=0) > 0], y_ad))
    for name, X, y in sets:
        X = StandardScaler().fit_transform(X)
        C = select_c(X, y)
        model = LogisticRegression(C=C, max_iter=5000, tol=1e-8).fit(X, y)
        for tol in ("cv", 0.01, 0.03):
            rep = audit(model, X, y, tolerance=tol, n_samples=50, exact_ranges=False, exact_flips=False, random_state=0)
            out.append(measure(name, "real", rep.rashomon_set, X, tol if isinstance(tol, str) else f"{tol:g}", {"C_sklearn": C}))
    # unpenalized fit with the profile tolerance: the ranges are 95% profile-likelihood intervals
    df = pd.read_csv(os.path.join(DATA_DIR, "ucla_admissions.csv"))
    X = np.column_stack([df["gre"], df["gpa"], df["rank"] == 2, df["rank"] == 3, df["rank"] == 4]).astype(float)
    y = df["admit"].to_numpy(float)
    n = len(y)
    for label, eps in (("profile", chi2.ppf(0.95, 1) / (2 * n)), ("lr", chi2.ppf(0.95, 6) / (2 * n))):
        rs = RashomonSet(estimator="logistic", C=np.inf, fit_intercept=True, epsilon=eps, epsilon_mode="absolute").fit(X, y)
        out.append(measure("UCLA admissions (unpenalized)", "real", rs, X, label, {"C_sklearn": float("inf")}))
    return out


def synthetic_cases() -> list[dict]:
    out = []
    grid = product((200, 1000), (5, 20), (0.5, 2.0, 5.0), (0.0, 0.95), (1.0, 100.0))
    for n, d, signal, rho, C in grid:
        rng = np.random.default_rng(hash((n, d, signal, rho)) % 2**32)
        cov = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
        X = rng.multivariate_normal(np.zeros(d), cov, size=n)
        beta = rng.normal(size=d)
        beta *= signal / np.sqrt(beta @ cov @ beta)  # std of the linear predictor = signal
        y = (rng.random(n) < 1 / (1 + np.exp(-(X @ beta)))).astype(float)
        if y.min() == y.max():
            continue
        X = StandardScaler().fit_transform(X)
        try:
            model = LogisticRegression(C=C, max_iter=10000, tol=1e-8).fit(X, y)
        except Exception as e:  # pragma: no cover
            print(f"  skipped n={n} d={d} signal={signal} rho={rho} C={C}: {e}")
            continue
        name = f"n={n} d={d} signal={signal:g} rho={rho:g} C={C:g}"
        tols = [("0.01", 0.01, "percent_loss"), ("0.03", 0.03, "percent_loss"), ("0.1", 0.1, "percent_loss"),
                ("lr", chi2.ppf(0.95, d + 1) / (2 * n), "absolute")]
        for label, eps, mode in tols:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rs = RashomonSet.from_sklearn(model, X, y, epsilon=eps, epsilon_mode=mode, random_state=0, safety_override=True)
                out.append(measure(name, "synthetic", rs, X, label,
                                   {"signal": signal, "rho": rho, "C_sklearn": C, "accuracy": float(model.score(X, y))}))
            except Exception as e:
                print(f"  failed {name} {label}: {e}")
    return out


def markdown(results: list[dict]) -> str:
    real = [r for r in results if r["kind"] == "real"]
    lines = ["| Dataset | d | tolerance | ε / loss | r = √(2ε) | posterior mass: nominal | observed | remaining error, max over coefficients: ellipsoid | ray | ray + 1 Newton step | quartic | flips (exact / ellipsoid / ray / Newton) | rows misclassified: ellipsoid | ray | Newton | seconds: exact ranges | Newton |",
             "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|--:|"]
    for r in real:
        q = f"{r['quartic_residual_max']:.1%}" if np.isfinite(r["quartic_residual_max"]) else "no boundary"
        lines.append(f"| {r['name']} | {r['d']} | {r['tolerance']} | {r['eps_over_loss']:.1%} | {r['r']:.3f} | {r['hpd_nominal']:.3f} | {r['hpd_observed']:.3f} | "
                     f"{r['ellipsoid_dev_max']:.1%} | {r['residual_max']:.1%} | {r['newton_residual_max']:.1%} | {q} | "
                     f"{r['flip_exact']:.1%} / {r['flip_ellipsoid']:.1%} / {r['flip_predicted']:.1%} / {r['flip_newton']:.1%} | "
                     f"{r['flip_disagree_ellipsoid']:.1%} | {r['flip_disagree_predicted']:.1%} | {r['flip_disagree_newton']:.1%} | "
                     f"{r['seconds_exact_ranges']:.1f} | {r['seconds_newton']:.1f} |")
    syn = pd.DataFrame([r for r in results if r["kind"] == "synthetic"])
    lines += ["", "Synthetic grid, binned by r = √(2ε); each entry is the max over coefficients, then the median or worst case in the bin:", "",
              "| r | cases | ellipsoid: median | worst | ray: median | worst | ray + 1 Newton step: median | worst | quartic: median | worst | rows misclassified, worst case: ellipsoid | ray | Newton | posterior mass, observed − nominal: worst |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    edges = [0, 0.05, 0.1, 0.2, 0.4, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = syn[(syn["r"] >= lo) & (syn["r"] < hi)]
        if b.empty:
            continue
        lines.append(f"| {lo:g}–{hi:g} | {len(b)} | {b['ellipsoid_dev_max'].median():.1%} | {b['ellipsoid_dev_max'].max():.1%} | "
                     f"{b['residual_max'].median():.1%} | {b['residual_max'].max():.1%} | {b['newton_residual_max'].median():.1%} | {b['newton_residual_max'].max():.1%} | "
                     f"{b['quartic_residual_max'].median():.1%} | {b['quartic_residual_max'].max():.1%} | "
                     f"{b['flip_disagree_ellipsoid'].max():.1%} | {b['flip_disagree_predicted'].max():.1%} | {b['flip_disagree_newton'].max():.1%} | "
                     f"{(b['hpd_observed'] - b['hpd_nominal']).abs().max():.3f} |")
    worst = syn.sort_values("residual_max", ascending=False).head(5)
    lines += ["", "Synthetic cases where the ray estimate leaves the most unexplained:", "",
              "| case | tolerance | ε / loss | r | saturated rows | ellipsoid | ray | ray + 1 Newton step | quartic |", "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in worst.iterrows():
        lines.append(f"| {r['name']} | {r['tolerance']} | {r['eps_over_loss']:.1%} | {r['r']:.3f} | {r['frac_saturated']:.0%} | {r['ellipsoid_dev_max']:.1%} | "
                     f"{r['residual_max']:.1%} | {r['newton_residual_max']:.1%} | {r['quartic_residual_max']:.1%} |")
    df = pd.DataFrame(results)
    df["ray_dev_max"] = [np.abs(np.array(p)[np.isfinite(p)]).max() for p in df["predicted"]]
    lines += ["", "All cases, grouped by how far the ray estimate itself departs from the ellipsoid (max over coefficients):", "",
              "| ray estimate's deviation from the ellipsoid | cases | exact − ray, max over coefficients: median case | worst case | exact − Newton: worst case | rows misclassified by the ellipsoid screen: worst | by the ray screen: worst | by the Newton screen: worst |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for lo, hi in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, np.inf)):
        b = df[(df["ray_dev_max"] >= lo) & (df["ray_dev_max"] < hi)]
        if b.empty:
            continue
        label = f"{lo:.0%}–{hi:.0%}" if np.isfinite(hi) else f"over {lo:.0%}"
        lines.append(f"| {label} | {len(b)} | {b['residual_max'].median():.1%} | {b['residual_max'].max():.1%} | {b['newton_residual_max'].max():.1%} | "
                     f"{b['flip_disagree_ellipsoid'].max():.1%} | {b['flip_disagree_predicted'].max():.1%} | {b['flip_disagree_newton'].max():.1%} |")
    n_sides = int(sum(2 * r["d"] for r in results))
    lines.append(f"\nSmallest signed (exact − estimate) deviation over all {n_sides} coefficient sides in {len(results)} cases: "
                 f"ray {df['residual_min_signed'].min():.1e}, Newton {df['newton_residual_min_signed'].min():.1e} "
                 f"(neither estimate exceeded the exact range beyond the loss tolerance). "
                 f"Largest |observed − nominal| posterior mass: {(df['hpd_observed'] - df['hpd_nominal']).abs().max():.3f}.")
    return "\n".join(lines)


def figure(results: list[dict]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    obs = np.concatenate([np.ravel(r["observed"]) for r in results])
    pred = np.concatenate([np.ravel(r["predicted"]) for r in results])
    rr = np.concatenate([np.full(2 * r["d"], r["r"]) for r in results])
    finite = np.isfinite(pred)
    sc = ax0.scatter(pred[finite], obs[finite], c=rr[finite], s=6, cmap="viridis", norm=LogNorm())
    lim = max(np.abs(obs[finite]).max(), np.abs(pred[finite]).max()) * 1.05
    ax0.plot([-lim, lim], [-lim, lim], color="black", linewidth=0.8)
    ax0.set_xlabel("ray estimate of the deviation from the ellipsoid half-range (quadratic_error)")
    ax0.set_ylabel("exact deviation (exact / ellipsoid − 1)")
    ax0.set_title("Per coefficient and side, all cases; points lie on or above the line")
    fig.colorbar(sc, ax=ax0, label="r = √(2ε)")
    for kind, marker in (("synthetic", "o"), ("real", "s")):
        sub = [r for r in results if r["kind"] == kind]
        lab = kind == "synthetic"
        for key, color, name in (("ellipsoid_dev_max", "#ff7f0e", "ellipsoid"), ("quartic_residual_max", "#2ca02c", "quartic estimate"),
                                 ("residual_max", "#1f77b4", "ray estimate"), ("newton_residual_max", "#9467bd", "ray + 1 Newton step")):
            vals = np.array([max(r[key], 1e-6) for r in sub])  # floor so exact zeros show on the log axis
            ax1.scatter([r["r"] for r in sub], vals, marker=marker, s=14, color=color, label=name if lab else None)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("r = √(2ε)")
    ax1.set_ylabel("max over coefficients of the remaining relative error")
    ax1.set_title("Error of each approximation per case (squares: real data)")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(STATIC, "exact_vs_quadratic.png"), dpi=120)
    plt.close(fig)


def write_tables(text: str) -> None:
    """Replace the generated block in docs/evaluation.md, keeping the file's line endings."""
    path = os.path.join(ROOT, "docs", "evaluation.md")
    with open(path, "rb") as f:
        raw = f.read()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    body = raw.decode("utf-8").replace("\r\n", "\n")
    start, end = "<!-- generated: exact_vs_quadratic -->", "<!-- /generated -->"
    i = body.index(start) + len(start)
    j = body.index(end, i)
    body = body[:i] + "\n" + text + "\n" + body[j:]
    with open(path, "wb") as f:
        f.write(body.replace("\n", nl).encode("utf-8"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    path = os.path.join(STATIC, "exact_vs_quadratic.json")
    if "--render" in sys.argv:  # tables and figure from the saved results, without recomputing
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
    else:
        t0 = time.perf_counter()
        print("real datasets")
        results = real_cases()
        print("synthetic grid")
        results += synthetic_cases()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"\n{len(results)} cases in {time.perf_counter() - t0:.0f}s; wrote {path}")
    figure(results)
    write_tables(markdown(results))
    print(f"updated the tables in docs/evaluation.md and {os.path.join(STATIC, 'exact_vs_quadratic.png')}")
