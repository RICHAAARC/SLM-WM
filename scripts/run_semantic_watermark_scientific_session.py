"""调用实验运行层的主方法科学会话入口."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runtime.semantic_watermark_scientific_session import main


if __name__ == "__main__":
    raise SystemExit(main())
