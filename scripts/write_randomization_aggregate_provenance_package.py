"""转发层内精确9重复聚合来源包 CLI."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.runners.randomization_aggregate_provenance import main


if __name__ == "__main__":
    main()
