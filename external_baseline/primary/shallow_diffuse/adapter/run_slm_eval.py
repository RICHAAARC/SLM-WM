"""运行 shallow_diffuse 的 SLM 外部 baseline adapter。"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_baseline.primary.sd35_diffusion_baseline_common import run_contract_or_report_required_adapter

BASELINE_ID = "shallow_diffuse"
MODEL_ALIGNMENT_STATUS = "sd35_medium_shallow_latent_adapter_required"
REAL_RUN_UNSUPPORTED_REASON = "shallow_diffuse_sd35_real_shallow_update_path_required"


def main() -> None:
    """CLI 入口。"""

    run_contract_or_report_required_adapter(
        baseline_id=BASELINE_ID,
        model_alignment_status=MODEL_ALIGNMENT_STATUS,
        real_run_unsupported_reason=REAL_RUN_UNSUPPORTED_REASON,
    )


if __name__ == "__main__":
    main()
