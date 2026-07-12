"""为隔离测试根目录复制当前提交内受治理的正式 Prompt。"""

from __future__ import annotations

from pathlib import Path

from experiments.protocol.prompts import PROMPT_FILES


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def copy_governed_prompt_file(root: Path, paper_run_name: str) -> Path:
    """复制指定论文层级的规范 Prompt 文件并返回目标路径。

    该辅助函数只服务测试夹具。它让产物构建测试使用与正式配置相同的
    Prompt 字节身份, 从而不会用任意同名文本绕过生产代码的 SHA-256 门禁。
    """

    try:
        relative_path = PROMPT_FILES[paper_run_name]
    except KeyError as exc:
        raise ValueError(f"未知论文运行层级: {paper_run_name}") from exc
    source_path = REPOSITORY_ROOT / relative_path
    target_path = root / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(source_path.read_bytes())
    return target_path
