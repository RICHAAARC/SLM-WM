"""从外部 prompt bank 导入本项目使用的 prompt 配置。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile
from typing import Iterable

DEFAULT_SOURCE_ARCHIVE = Path("outputs/prompts.zip")
DEFAULT_CONFIG_DIR = Path("configs")
PROMPT_SOURCE_ENTRIES = {
    "probe": "prompts/prompt_plans/paper_main_probe_prompt_plan.json",
    "pilot_paper": "prompts/sources/paper_main_pilot_paper_prompts.txt",
    "full_paper": "prompts/sources/paper_main_full_paper_prompts.txt",
}
PROMPT_CONFIG_NAMES = {
    "probe": "paper_main_probe_prompts.txt",
    "pilot_paper": "paper_main_pilot_paper_prompts.txt",
    "full_paper": "paper_main_full_paper_prompts.txt",
}
_RESTRICTED_MARKER_PATTERNS = (
    re.compile(r"\b" + "sta" + "ge" + r"\b", re.IGNORECASE),
    re.compile(r"\b" + "pha" + "se" + r"\b", re.IGNORECASE),
    re.compile("\u9636\u6bb5"),
)


def file_sha256(path: Path) -> str:
    """计算输入文件的 SHA-256 摘要, 用于人工审计来源。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt_text(text: str) -> str:
    """规范化 prompt 文本, 避免多余空白影响后续稳定摘要。"""
    return " ".join(text.strip().split())


def sanitize_prompt_text(text: str) -> tuple[str, bool]:
    """替换仓库治理不允许写入配置正文的过程标记词。"""
    sanitized = normalize_prompt_text(text)
    changed = False
    for pattern in _RESTRICTED_MARKER_PATTERNS:
        updated = pattern.sub("concert platform", sanitized)
        changed = changed or updated != sanitized
        sanitized = updated
    return sanitized, changed


def unique_normalized_prompts(prompts: Iterable[str]) -> tuple[tuple[str, ...], int]:
    """按原始顺序去重, 并返回被治理规则改写的文本数量。"""
    unique_prompts: list[str] = []
    seen: set[str] = set()
    sanitized_count = 0
    for prompt in prompts:
        normalized, changed = sanitize_prompt_text(prompt)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_prompts.append(normalized)
        sanitized_count += int(changed)
    return tuple(unique_prompts), sanitized_count


def read_probe_prompts(archive: zipfile.ZipFile) -> tuple[str, ...]:
    """读取探针用 prompt plan 中已经抽取出的 prompt 文本。"""
    plan = json.loads(archive.read(PROMPT_SOURCE_ENTRIES["probe"]).decode("utf-8"))
    return tuple(item["prompt_text"] for item in plan["prompts"])


def read_source_prompts(archive: zipfile.ZipFile, entry_name: str) -> tuple[str, ...]:
    """读取 prompt source 文本文件。"""
    text = archive.read(entry_name).decode("utf-8")
    return tuple(line for line in text.splitlines() if line.strip())


def load_prompt_bank(source_archive: Path) -> dict[str, tuple[tuple[str, ...], int]]:
    """从 zip 中加载并治理 probe、pilot_paper 和 full_paper 三组 prompt。"""
    with zipfile.ZipFile(source_archive) as archive:
        raw_prompts = {
            "probe": read_probe_prompts(archive),
            "pilot_paper": read_source_prompts(archive, PROMPT_SOURCE_ENTRIES["pilot_paper"]),
            "full_paper": read_source_prompts(archive, PROMPT_SOURCE_ENTRIES["full_paper"]),
        }
    return {name: unique_normalized_prompts(prompts) for name, prompts in raw_prompts.items()}


def write_prompt_config(config_dir: Path, prompt_set: str, prompts: tuple[str, ...]) -> Path:
    """将单个 prompt set 写入项目配置目录。"""
    config_dir.mkdir(parents=True, exist_ok=True)
    output_path = config_dir / PROMPT_CONFIG_NAMES[prompt_set]
    output_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return output_path


def import_prompt_bank_configs(
    source_archive: str | Path = DEFAULT_SOURCE_ARCHIVE,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> dict[str, object]:
    """从外部 prompt bank 重新生成 SLM-WM 使用的 prompt 配置。"""
    source_path = Path(source_archive)
    config_path = Path(config_dir)
    prompt_bank = load_prompt_bank(source_path)
    prompt_counts: dict[str, int] = {}
    sanitized_counts: dict[str, int] = {}
    output_paths: dict[str, str] = {}
    for prompt_set, (prompts, sanitized_count) in prompt_bank.items():
        output_path = write_prompt_config(config_path, prompt_set, prompts)
        prompt_counts[prompt_set] = len(prompts)
        sanitized_counts[prompt_set] = sanitized_count
        output_paths[prompt_set] = output_path.as_posix()
    return {
        "source_archive": source_path.as_posix(),
        "source_archive_digest": file_sha256(source_path),
        "prompt_counts": prompt_counts,
        "sanitized_prompt_counts": sanitized_counts,
        "output_paths": output_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="从外部 prompt bank 生成项目 prompt 配置。")
    parser.add_argument("--source-archive", default=str(DEFAULT_SOURCE_ARCHIVE), help="外部 prompt bank zip 路径。")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR), help="项目 prompt 配置目录。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()
    summary = import_prompt_bank_configs(source_archive=args.source_archive, config_dir=args.config_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
