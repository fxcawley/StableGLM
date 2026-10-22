"""Generate and check the numbers and figures quoted in docs/examples/tutorial.md.

The tutorial audits the README model (scikit-learn's default ``LogisticRegression`` on
the ten "mean" features of the Wisconsin breast-cancer data, standardized) and relates
the result to the Laplace posterior, to profile-likelihood intervals and to the
bootstrap. Every table in the tutorial sits between ``<!-- generated: NAME -->`` and
``<!-- /generated -->`` markers and is produced here; the prose quotes numbers from
those tables only.

Run from the repository root:
    python scripts/generate_tutorial_breast_cancer.py --write   # recompute, rewrite the blocks, save the figures
    python scripts/generate_tutorial_breast_cancer.py --check   # recompute and compare with the blocks (CI)

Exact quantities (flip rates, coefficient ranges, HPD levels) are compared tightly;
sampled ones (ESS, disagreement, bootstrap standard deviations) loosely.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rashomon import audit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TUTORIAL = ROOT / "docs" / "examples" / "tutorial.md"
STATIC = ROOT / "docs" / "_static"
SWEEP: list[Any] = [0.005, 0.01, 0.02, 0.05, "cv", ("profile", 0.05)]
# per block: (relative tolerance for the numbers, compare the surrounding text too)
CHECK = {
    "audit_summary": (0.3, False),
    "affected": (0.05, True),
    "coefficients": (0.02, True),
    "tolerance_sweep": (0.02, True),
    "intervals": (0.1, True),
}
# content may be empty and may not contain another marker
BLOCK = re.compile(r"<!-- generated: (\w+) -->\r?\n((?:(?!<!-- /?generated).)*?)(?:\r?\n)?<!-- /generated -->", re.DOTALL)
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


def label(tol: Any) -> str:
    if tol == "cv":
        return '`"cv"` (default)'
    if isinstance(tol, tuple):
        return f'`("{tol[0]}", {tol[1]:g})`'
    return f"`{tol:g}`"


def compute() -> dict:
    data = load_breast_cancer(as_frame=True)
    X, y = data.data.iloc[:, :10], data.target
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    report = audit(model, X, y, n_samples=2000, random_state=0)
    rs = report.rashomon_set
    n, d, eps = rs._n, rs._d, report.tolerance
    nH_inv = np.linalg.inv(n * rs._hessian_matrix())
    sd_laplace = np.sqrt(np.diag(nH_inv))[1:]
    out: dict[str, Any] = {"report": report, "n": n, "d": d, "eps": eps, "hpd": float(chi2.cdf(2 * n * eps, df=d))}

    # who is affected: flips near and far from the threshold
    p_hat = report.prediction_ranges["estimate"].to_numpy()
    near = (p_hat >= 0.2) & (p_hat <= 0.8)
    flipped = report.flipped
    out["affected"] = pd.DataFrame(
        {"rows": [int(near.sum()), int((~near).sum())],
         "can_flip": [int(flipped[near].sum()), int(flipped[~near].sum())],
         "p_min": [float(p_hat[near & flipped].min()), float(p_hat[~near & flipped].min())],
         "p_max": [float(p_hat[near & flipped].max()), float(p_hat[~near & flipped].max())]},
        index=["predicted probability between 0.2 and 0.8", "below 0.2 or above 0.8"],
    )

    # exact coefficient ranges against the Laplace prediction sqrt(2 n eps) * sd
    coef = report.coefficients.copy()
    coef["exact_half"] = (coef["high"] - coef["low"]) / 2
    coef["laplace_half"] = np.sqrt(2 * n * eps) * sd_laplace
    out["coefficients"] = coef

    # tolerance sweep: exact flip rate and sign stability, with the posterior mass of each set
    rows, stable = [], {}
    for tol in SWEEP:
        r = report if tol == "cv" else audit(model, X, y, tolerance=tol, n_samples=500, random_state=0)
        rows.append({
            "tolerance": label(tol), "eps": r.tolerance, "eps_over_loss": r.tolerance / r.loss_optimum,
            "hpd": float(chi2.cdf(2 * n * r.tolerance, df=d)), "flip_rate": r.flip_rate,
            "sign_stable": len(r.stable_features),
        })
        stable[label(tol)] = r.coefficients["sign_stable"]
    out["sweep"] = pd.DataFrame(rows)
    # the smallest tolerance in the sweep at which each coefficient's sign is not stable
    by_eps = list(out["sweep"].sort_values("eps")["tolerance"]) + ["no tolerance in the sweep"]
    coef["sign_fails_at"] = [next(t for t in by_eps if t == by_eps[-1] or not stable[t][name]) for name in coef.index]
    out["fail_order"] = by_eps

    # bootstrap of the penalized estimator vs the sandwich formula and the Laplace sd
    Xs = model[:-1].transform(X)
    comp = rs.compare_to_bootstrap(
        Xs, y.to_numpy(float), n_bootstrap=500, n_rashomon=50, feature_names=list(X.columns), random_state=0
    )
    Xa = rs._prepare_X(Xs)
    p = 1.0 / (1.0 + np.exp(-(Xa @ rs._theta_hat)))
    J = Xa.T @ (Xa * ((y.to_numpy(float) - p) ** 2)[:, None])  # empirical score covariance
    sd_sandwich = np.sqrt(np.diag(nH_inv @ J @ nH_inv))[1:]
    out["intervals"] = pd.DataFrame(
        {"bootstrap_sd": comp["bootstrap_std"], "sandwich_sd": sd_sandwich, "laplace_sd": sd_laplace,
         "rashomon_half": coef["exact_half"].to_numpy()},
        index=X.columns,
    )
    return out


def render(out: dict) -> dict[str, str]:
    blocks = {"audit_summary": "```text\n" + out["report"].summary() + "\n```"}

    lines = ["| rows with ... | rows | can flip | predicted probabilities of those that can flip |", "|---|--:|--:|---|"]
    for name, r in out["affected"].iterrows():
        lines.append(f"| {name} | {int(r['rows'])} | {int(r['can_flip'])} ({r['can_flip'] / r['rows']:.0%}) | "
                     f"{r['p_min']:.3f} to {r['p_max']:.3f} |")
    blocks["affected"] = "\n".join(lines)

    coef = out["coefficients"]
    lines = ["| feature | estimate | exact low | exact high | sign | Laplace half-width √(2nε)·sd | exact / Laplace |",
             "|---|--:|--:|--:|---|--:|--:|"]
    for name, r in coef.iterrows():
        flag = "stable" if r["sign_stable"] else "**unstable**"
        lines.append(f"| {name} | {r['estimate']:.3f} | {r['low']:.3f} | {r['high']:.3f} | {flag} | "
                     f"{r['laplace_half']:.3f} | {r['exact_half'] / r['laplace_half']:.3f} |")
    lines.append(f"\nTolerance ε = {out['eps']:.4f}; the set is a {out['hpd']:.1%} HPD region of the Laplace posterior "
                 f"(d = {out['d']} including the intercept, √(2nε) = {np.sqrt(2 * out['n'] * out['eps']):.2f}).")
    blocks["coefficients"] = "\n".join(lines)

    sw = out["sweep"]
    lines = ["| `tolerance=` | ε | ε / loss | posterior mass of the set | flip rate (exact) | sign-stable coefficients |",
             "|---|--:|--:|--:|--:|--:|"]
    for _, r in sw.iterrows():
        lines.append(f"| {r['tolerance']} | {r['eps']:.4f} | {r['eps_over_loss']:.1%} | {r['hpd']:.1%} | "
                     f"{r['flip_rate']:.1%} | {r['sign_stable']} of 10 |")
    fails = coef["sign_fails_at"]
    parts = [f"{', '.join(n.replace('mean ', '') for n in fails.index[fails == t])}: {t}"
             for t in out["fail_order"] if (fails == t).any()]
    lines.append("\nSmallest tolerance in the sweep at which each coefficient can change sign. " + "; ".join(parts) + ".")
    blocks["tolerance_sweep"] = "\n".join(lines)

    iv = out["intervals"]
    lines = ["| feature | bootstrap sd | sandwich prediction | Laplace sd | Rashomon half-range |",
             "|---|--:|--:|--:|--:|"]
    for name, r in iv.iterrows():
        lines.append(f"| {name} | {r['bootstrap_sd']:.3f} | {r['sandwich_sd']:.3f} | {r['laplace_sd']:.3f} | "
                     f"{r['rashomon_half']:.3f} |")
    blocks["intervals"] = "\n".join(lines)
    return blocks


def figures(out: dict) -> None:
    sw = out["sweep"].sort_values("eps_over_loss")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = sw["eps_over_loss"].to_numpy()
    flips = 100 * sw["flip_rate"].to_numpy()
    ax.plot(x, flips, "o-", color="#d62728")
    ax.set_xscale("log")
    ax.set_xlabel("tolerance as a fraction of the optimal training loss")
    ax.set_ylabel("rows that can flip (%)", color="#d62728")
    ax.set_ylim(0, 1.25 * flips.max())
    ax2 = ax.twinx()
    ax2.plot(x, sw["sign_stable"], "s--", color="#2ca02c")
    ax2.set_ylabel("sign-stable coefficients (of 10)", color="#2ca02c")
    ax2.set_ylim(0, 10.5)
    for xi, fi, h in zip(x, flips, sw["hpd"]):
        ax.annotate(f"{h:.1%}" if h < 0.1 else f"{h:.0%}", (xi, fi), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color="gray")
    ax.set_title("Tolerance sweep (grey: posterior mass of the set)")
    fig.tight_layout()
    fig.savefig(STATIC / "tutorial_tolerance_sweep.png", dpi=120)
    plt.close(fig)

    coef, iv = out["coefficients"], out["intervals"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    yy = np.arange(len(coef))[::-1]
    est = coef["estimate"].to_numpy()
    ax0.hlines(yy + 0.15, coef["low"], coef["high"], color="#1f77b4", linewidth=3, label="exact Rashomon range")
    ax0.hlines(yy - 0.15, est - coef["laplace_half"], est + coef["laplace_half"], color="#ff7f0e", linewidth=3,
               label="Laplace: θ̂ ± √(2nε)·sd")
    ax0.scatter(est, yy, color="black", s=16, zorder=3)
    ax0.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax0.set_yticks(yy)
    ax0.set_yticklabels(coef.index, fontsize=8)
    ax0.set_title("Rashomon range vs its quadratic approximation")
    ax0.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    z = 1.6449
    ax1.hlines(yy + 0.15, est - z * iv["bootstrap_sd"], est + z * iv["bootstrap_sd"], color="#9467bd", linewidth=3,
               label="bootstrap 90% (penalized refits)")
    ax1.hlines(yy - 0.15, est - z * iv["laplace_sd"], est + z * iv["laplace_sd"], color="#ff7f0e", linewidth=3,
               label="Laplace 90% credible")
    ax1.scatter(est, yy, color="black", s=16, zorder=3)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("Sampling intervals for the same coefficients")
    ax1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    for ax in (ax0, ax1):
        ax.set_xlabel("coefficient (standardized features)")
    fig.tight_layout()
    fig.savefig(STATIC / "tutorial_intervals.png", dpi=120)
    plt.close(fig)


def read_blocks(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in BLOCK.finditer(text)}


def write(blocks: dict[str, str]) -> None:
    raw = TUTORIAL.read_bytes()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8").replace("\r\n", "\n")

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in blocks:
            raise KeyError(f"tutorial block {name!r} has no generator")
        return f"<!-- generated: {name} -->\n{blocks[name]}\n<!-- /generated -->"

    new = BLOCK.sub(sub, text)
    missing = set(blocks) - set(read_blocks(text))
    if missing:
        raise KeyError(f"tutorial.md lacks blocks for {sorted(missing)}")
    TUTORIAL.write_bytes(new.replace("\n", nl).encode("utf-8"))


def check(blocks: dict[str, str]) -> list[str]:
    stored = read_blocks(TUTORIAL.read_text(encoding="utf-8"))
    failures = []
    for name, fresh in blocks.items():
        if name not in stored:
            failures.append(f"{name}: block missing from tutorial.md")
            continue
        old = stored[name].replace("\r\n", "\n")
        rel, compare_text = CHECK[name]
        if compare_text and NUMBER.sub("#", old) != NUMBER.sub("#", fresh):
            failures.append(f"{name}: text differs from the regenerated block (run --write)")
            continue
        a = [float(v) for v in NUMBER.findall(old)]
        b = [float(v) for v in NUMBER.findall(fresh)]
        if len(a) != len(b):
            failures.append(f"{name}: {len(a)} numbers in tutorial.md, {len(b)} regenerated")
            continue
        for u, v in zip(a, b):
            if abs(u - v) > max(rel * abs(v), 0.002):
                failures.append(f"{name}: tutorial says {u:g}, regenerated {v:g}")
    return failures


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode not in ("--write", "--check"):
        sys.exit(__doc__)
    out = compute()
    blocks = render(out)
    if mode == "--write":
        write(blocks)
        figures(out)
        print("\n\n".join(f"[{k}]\n{v}" for k, v in blocks.items()))
        print(f"\nrewrote the generated blocks in {TUTORIAL} and the figures in {STATIC}")
    else:
        failures = check(blocks)
        if failures:
            print("FAILED: tutorial.md does not match the regenerated numbers:")
            for f in failures:
                print("  - " + f)
            sys.exit(1)
        print(f"PASSED: {len(blocks)} generated blocks in tutorial.md match the recomputed values.")
