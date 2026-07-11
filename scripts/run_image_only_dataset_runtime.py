"""运行当前论文规模的真实 SLM-WM 图像盲检数据集实验."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runners.image_only_dataset_workload import main


if __name__ == "__main__":
    main()
