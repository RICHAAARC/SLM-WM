"""转发到 experiments 层的正式依赖 profile 准备 CLI."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.dependency_preparation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
