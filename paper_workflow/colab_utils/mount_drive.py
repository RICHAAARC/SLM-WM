"""封装 Google Drive 挂载探测, 避免 Notebook 直接承载 workflow 逻辑。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DriveMountReport:
    """记录 Drive 挂载动作或本地降级原因。"""

    mount_decision: str
    mount_point: str
    mounted: bool
    unsupported_reason: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        """转为可写入 JSON 的挂载报告。"""
        return asdict(self)


def build_drive_mount_report(
    mount_point: str = "/content/drive",
    force_remount: bool = False,
    perform_mount: bool = True,
) -> DriveMountReport:
    """尝试挂载 Google Drive; 非 Colab 环境会返回可审计降级报告。"""
    generated_at = datetime.now(timezone.utc).isoformat()
    if not perform_mount:
        return DriveMountReport(
            mount_decision="skipped",
            mount_point=mount_point,
            mounted=Path(mount_point).exists(),
            unsupported_reason="mount_not_requested",
            generated_at=generated_at,
        )
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return DriveMountReport(
            mount_decision="unsupported",
            mount_point=mount_point,
            mounted=False,
            unsupported_reason="google_colab_drive_unavailable",
            generated_at=generated_at,
        )
    drive.mount(mount_point, force_remount=force_remount)
    return DriveMountReport(
        mount_decision="pass",
        mount_point=mount_point,
        mounted=True,
        unsupported_reason="",
        generated_at=generated_at,
    )
