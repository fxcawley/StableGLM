"""CI entry point: recompute the numbers quoted in docs/examples/tutorial.md and compare.

Run from the repository root:
    python scripts/verify_tutorial_claims.py

Exit code 0 = every generated block in the tutorial matches the recomputed values;
1 = a mismatch (regenerate with ``python scripts/generate_tutorial_breast_cancer.py --write``).
"""

import os
import runpy
import sys

sys.argv = [os.path.join(os.path.dirname(__file__), "generate_tutorial_breast_cancer.py"), "--check"]
runpy.run_path(sys.argv[0], run_name="__main__")
