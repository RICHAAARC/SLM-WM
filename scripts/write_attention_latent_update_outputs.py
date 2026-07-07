"""命令行兼容入口; 正式实现位于 experiments.artifacts.attention_latent_update_outputs。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.artifacts.attention_latent_update_outputs import *  # noqa: F401,F403,E402
from experiments.artifacts.attention_latent_update_outputs import main  # noqa: E402


if __name__ == "__main__":
    main()
