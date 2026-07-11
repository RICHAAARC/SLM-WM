"""调用论文实验层的外部 baseline 命令计划构造入口."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_experiments.baselines.command_plan_builder import main


if __name__ == "__main__":
    main()
