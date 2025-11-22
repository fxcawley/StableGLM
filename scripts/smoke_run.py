import json
from pathlib import Path

import numpy as np

from rashomon import RashomonSet


def main() -> None:
    out_dir = Path("build/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    # Logistic synthetic
    X_log = rng.normal(size=(64, 6))
    w_log = rng.normal(size=6)
    p = 1.0 / (1.0 + np.exp(-(X_log @ w_log)))
    y_log = (rng.random(64) < p).astype(float)
    rs_log = RashomonSet(estimator="logistic", random_state=0).fit(X_log, y_log)
    diag_log = rs_log.diagnostics()
    bands = rs_log.probability_bands(X_log[:5]).tolist()

    # Linear synthetic
    X_lin = rng.normal(size=(64, 6))
    w_lin = rng.normal(size=6)
    y_lin = X_lin @ w_lin + 0.1 * rng.normal(size=64)
    rs_lin = RashomonSet(estimator="linear", random_state=0).fit(X_lin, y_lin)
    diag_lin = rs_lin.diagnostics()
    s = rng.normal(size=6)
    hack = rs_lin.hacking_interval(s)

    payload = {
        "logistic": {
            "diagnostics": diag_log,
            "bands_first5": bands,
            "coef_norm": float(np.linalg.norm(rs_log.coef_)),
        },
        "linear": {
            "diagnostics": diag_lin,
            "hacking_interval_random_s": hack,
            "coef_norm": float(np.linalg.norm(rs_lin.coef_)),
            "r2_on_train": rs_lin.score(X_lin, y_lin),
        },
    }

    out_path = out_dir / "rashomon_smoke.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


