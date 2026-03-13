from importlib.metadata import PackageNotFoundError, version

from .rashomon_set import RashomonSet

try:
    __version__ = version("rashomon-py")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["RashomonSet", "__version__"]
