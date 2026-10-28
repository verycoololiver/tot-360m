"""Tot-360M package root."""
from pathlib import Path
import sys

# Ensure `import tot_360m.*` resolves when running directly from a cloned repo root
# (even if the cloned folder is named `tot-360m` with a hyphen).
sys.modules.setdefault("tot_360m", sys.modules[__name__])
