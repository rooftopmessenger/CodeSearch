"""Root-level shim — preserves backward compatibility for `uv run main.py`.

All logic lives in backend/main.py.  This file just re-exports the entry point
so existing CLI invocations from the project root continue to work unchanged.
"""
import sys
from pathlib import Path
# Add backend/ to sys.path so bare imports inside backend/main.py resolve
# correctly when this shim is the __main__ module.
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from main import main  # noqa: E402

if __name__ == "__main__":
    main()
