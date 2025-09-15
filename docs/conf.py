import os
import sys
from datetime import datetime

project = "StableGLM"
author = "StableGLM Team"
year = datetime.now().year
copyright = f"{year}, {author}"

extensions = [
    "myst_parser",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

html_theme = "sphinx_rtd_theme"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_static_path = ["_static"]
