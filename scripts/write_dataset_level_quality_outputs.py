"""数据集级正式 Inception FID / KID 产物命令行入口。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.dataset_level_quality_outputs import main  # noqa: E402


if __name__ == "__main__":
    main()
