"""Pytest bootstrap so `import tot_360m.*` works from any checkout folder name."""
import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

if "tot_360m" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "tot_360m",
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["tot_360m"] = module
        spec.loader.exec_module(module)
