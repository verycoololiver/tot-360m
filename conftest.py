"""Pytest bootstrap so `import tot_360m.*` works from any checkout folder name (including child subprocesses)."""
import importlib.util
import os
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

# Also expose a lightweight sitecustomize bridge so `subprocess.run([sys.executable, ...])`
# in tests can `import tot_360m` without requiring an editable install when cloned as `tot-360m/`.
_BRIDGE_DIR = REPO_ROOT / ".pytest_cache" / "_tot_pkg_bridge"
_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
(_BRIDGE_DIR / "sitecustomize.py").write_text(
    f"import importlib.util, sys\n"
    f"if 'tot_360m' not in sys.modules:\n"
    f"    _r = {str(REPO_ROOT)!r}\n"
    f"    _s = importlib.util.spec_from_file_location('tot_360m', _r + '/__init__.py', submodule_search_locations=[_r])\n"
    f"    if _s and _s.loader:\n"
    f"        _m = importlib.util.module_from_spec(_s)\n"
    f"        sys.modules['tot_360m'] = _m\n"
    f"        _s.loader.exec_module(_m)\n",
    encoding="utf-8",
)
_existing_pp = os.environ.get("PYTHONPATH", "")
if str(_BRIDGE_DIR) not in _existing_pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        f"{_BRIDGE_DIR}{os.pathsep}{_existing_pp}" if _existing_pp else str(_BRIDGE_DIR)
    )
