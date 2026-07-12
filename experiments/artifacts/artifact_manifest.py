"""构造论文产物 manifest, 该模块属于产物生成层而非核心方法层。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from main.core.digest import build_stable_digest
from experiments.artifacts.manifest_schema import ArtifactManifest


def build_artifact_manifest(
    artifact_id: str,
    artifact_type: str,
    input_paths: tuple[str, ...],
    output_paths: tuple[str, ...],
    config: dict[str, Any],
    code_version: str,
    rebuild_command: str,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """根据输入、输出和配置构造产物 manifest。

    完整 ``config`` 是打包器从正式产物独立复验配置身份的事实来源,
    ``config_digest`` 必须从同一份不可变快照计算。深拷贝用于避免调用方在
    manifest 构造后继续修改嵌套配置, 造成正文与摘要分离。
    """

    config_snapshot = deepcopy(config)
    return ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        input_paths=input_paths,
        output_paths=output_paths,
        config=config_snapshot,
        config_digest=build_stable_digest(config_snapshot),
        code_version=code_version,
        rebuild_command=rebuild_command,
        metadata=metadata or {},
    )
