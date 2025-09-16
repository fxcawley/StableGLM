import os
import sys
from datetime import datetime

# Add project root for autodoc
sys.path.insert(0, os.path.abspath(".."))

project = "StableGLM"
author = "StableGLM Team"
year = datetime.now().year
copyright = f"{year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
}

html_theme = "sphinx_rtd_theme"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_static_path = ["_static"]
