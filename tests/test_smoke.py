import numpy as np
from rashomon import RashomonSet


def test_fit_and_diagnostics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = (rng.random(20) > 0.5).astype(float)
    rs = RashomonSet(random_state=0).fit(X, y)
    diag = rs.diagnostics()
    assert diag["n"] == 20 and diag["d"] == 5
    assert diag["epsilon"] > 0
    assert diag["measure"] in {"param", "lr"}
