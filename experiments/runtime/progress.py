"""Notebook 和 repository runner 的总体工作量进度显示工具。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from os import PathLike
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")
PROGRESS_EVENT_ENV_NAME = "SLM_WM_PROGRESS_EVENT_PATH"


@dataclass
class WorkloadProgress:
    """按实际任务总量输出单行总体进度。

    该类属于通用工程写法: 调用方只需要提供运行时已经解析出的任务总数,
    例如 `len(records) * len(attack_configs)` 或命令计划数量。显示层不参与
    业务逻辑, 只根据 update 次数计算 elapsed 与 eta, 因而可以复用于不同
    Notebook 和 repository runner。
    """

    total: int
    desc: str
    enabled: bool = True
    profile: str = ""
    min_interval_seconds: float = 2.0

    def __post_init__(self) -> None:
        """初始化计时状态。"""

        self.total = max(0, int(self.total))
        self.completed = 0
        self.started_at = time.monotonic()
        self.last_emit_at = 0.0
        self.closed = False

    def __enter__(self) -> "WorkloadProgress":
        """进入上下文时输出初始进度。"""

        if self.enabled:
            self.emit(force=True)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """退出上下文时输出最终进度并换行。"""

        self.close()

    def update(self, count: int = 1, *, profile: str | None = None) -> None:
        """更新已完成任务数并按时间间隔刷新显示。"""

        if profile is not None:
            self.profile = profile
        self.completed = min(self.total, self.completed + int(count)) if self.total else self.completed + int(count)
        self.emit(force=self.completed >= self.total)

    def emit(self, *, force: bool = False) -> None:
        """按最小时间间隔输出进度行。"""

        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_emit_at < self.min_interval_seconds:
            return
        self.last_emit_at = now
        sys.stdout.write("\r" + self.progress_text())
        sys.stdout.flush()

    def close(self) -> None:
        """关闭进度显示。"""

        if self.closed:
            return
        self.closed = True
        if self.enabled:
            self.emit(force=True)
            sys.stdout.write("\n")
            sys.stdout.flush()

    def progress_text(self) -> str:
        """生成用户可读进度文本。"""

        elapsed_seconds = max(0.0, time.monotonic() - self.started_at)
        completed = max(0, int(self.completed))
        total = max(0, int(self.total))
        percent = (completed / total * 100.0) if total else 100.0
        if completed > 0 and total > 0:
            eta_seconds = elapsed_seconds * max(total - completed, 0) / completed
        else:
            eta_seconds = 0.0
        profile_text = self.profile or "none"
        return (
            f"工作量进度 | {self.desc} | {completed}/{total} ({percent:.1f}%) | "
            f"elapsed={elapsed_seconds / 60.0:.1f} min | eta={eta_seconds / 60.0:.1f} min | "
            f"profile={profile_text}"
        )


def maybe_tqdm(
    values: Iterable[T],
    *,
    total: int | None,
    desc: str,
    enabled: bool = True,
) -> Iterable[T]:
    """按需返回带总体工作量进度的迭代器。

    该函数属于通用工程写法: Notebook helper 可以统一关闭底层 pipeline
    的逐步进度条, 只暴露一个按样本或按任务计数的总体进度条。这样可以
    避免 Colab 输出区被每条样本的 diffusion step 刷屏, 同时保留用户需要
    的总体完成度。
    """

    if not enabled:
        return values

    def iterator() -> Iterator[T]:
        resolved_total = total if total is not None else 0
        with WorkloadProgress(resolved_total, desc=desc, enabled=True) as progress:
            for item in values:
                yield item
                progress.update()

    return iterator()


def progress_task_items(items: Iterable[T], *, total: int, desc: str, enabled: bool = True) -> Iterator[T]:
    """生成带总体进度条的任务迭代器。

    此处只负责展示进度, 不改变任务执行顺序和异常传播方式。调用方仍然
    在业务函数中完成真实图像生成、检测和落盘。
    """

    yield from maybe_tqdm(items, total=total, desc=desc, enabled=enabled)


def progress_event_path_from_environment(env: dict[str, str] | None = None) -> Path | None:
    """从环境变量读取子进程进度事件文件路径。

    该函数属于通用工程写法: Notebook 输出层只读取一个 JSONL 事件文件,
    真实样本循环仍由各 runner 或 adapter 自己执行。这样可以在保持低噪声
    的同时把长耗时子进程内部的 prompt、attack 或命令进度回传给外层进度条。
    """

    source = env if env is not None else os.environ
    value = str(source.get(PROGRESS_EVENT_ENV_NAME, "")).strip()
    if not value:
        return None
    return Path(value)


def write_progress_event(
    path: str | PathLike[str] | None,
    *,
    desc: str,
    completed: int,
    total: int,
    profile: str = "",
    **metadata: Any,
) -> None:
    """向 JSONL 文件追加一条低噪声进度事件。

    写入失败不应中断真实实验流程, 因为进度事件只用于可观察性。正式结果仍由
    records、manifests 和结果包决定。
    """

    if path is None:
        return
    try:
        event_path = Path(path)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "desc": str(desc),
            "completed": max(0, int(completed)),
            "total": max(0, int(total)),
            "profile": str(profile or "none"),
        }
        payload.update(metadata)
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def read_latest_progress_event(path: str | PathLike[str] | None) -> dict[str, Any] | None:
    """读取 JSONL 进度事件文件中的最后一条有效事件。"""

    if path is None:
        return None
    event_path = Path(path)
    if not event_path.is_file():
        return None
    try:
        with event_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            text = handle.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    for line in reversed([item.strip() for item in text.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def format_progress_event(event: dict[str, Any] | None) -> str:
    """把子进程进度事件压缩为单行 profile 片段。"""

    if not event:
        return ""
    desc = str(event.get("desc") or "child")
    try:
        completed = max(0, int(event.get("completed", 0) or 0))
        total = max(0, int(event.get("total", 0) or 0))
    except (TypeError, ValueError):
        completed = 0
        total = 0
    percent = (completed / total * 100.0) if total else 100.0
    profile = str(event.get("profile") or "none")
    baseline_id = str(event.get("baseline_id") or "").strip()
    baseline_text = f" baseline={baseline_id}" if baseline_id else ""
    return f"child={desc}{baseline_text} {completed}/{total} ({percent:.1f}%) {profile}"


def merge_child_progress_profile(base_profile: str, child_progress_path: str | PathLike[str] | None) -> str:
    """把父进程状态和最新子进程事件合并为一个可读 profile。"""

    child_profile = format_progress_event(read_latest_progress_event(child_progress_path))
    if not child_profile:
        return base_profile
    return f"{base_profile} {child_profile}"


@contextmanager
def progress_bar(total: int, *, desc: str, enabled: bool = True) -> Iterator[object | None]:
    """创建可手动更新的总体进度条。

    该函数用于一个 workflow 内部存在多个资源环节的情况。例如先用
    SD3.5 img2img pipeline 处理全图攻击, 再按需调用 flow-matching
    inversion 与 inpainting 后端。多个资源环节共享同一个总体进度条,
    用户看到的是整体任务完成度, 而不是每个样本内部 diffusion step。
    """

    if not enabled:
        yield None
        return
    with WorkloadProgress(total=total, desc=desc, enabled=True) as progress:
        yield progress


def update_progress(bar: object | None, count: int = 1, *, profile: str | None = None) -> None:
    """更新可选进度条。"""

    if bar is not None and hasattr(bar, "update"):
        try:
            bar.update(count, profile=profile)
        except TypeError:
            bar.update(count)


def emit_progress_status(bar: object | None, *, profile: str) -> None:
    """强制刷新当前工作量状态而不增加完成数。

    该函数属于通用工程写法: 长耗时 Colab 命令通常需要保持 stdout / stderr
    捕获落盘, 不能直接把第三方下载或推理日志持续刷到 Notebook 输出区。调用方
    可以在长命令启动、心跳和完成时刷新同一行状态, 从而保留可观察性并减少噪声。
    """

    if bar is None:
        return
    if hasattr(bar, "profile"):
        try:
            setattr(bar, "profile", profile)
        except Exception:
            return
    if hasattr(bar, "emit"):
        try:
            bar.emit(force=True)
        except TypeError:
            bar.emit()


def run_quiet_subprocess_with_progress(
    command: list[str] | str,
    *,
    cwd: str | PathLike[str] | None = None,
    timeout_seconds: int | float | None = None,
    shell: bool = False,
    env: dict[str, str] | None = None,
    progress: object | None = None,
    progress_profile: str = "",
    heartbeat_seconds: float = 30.0,
    child_progress_path: str | PathLike[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """执行子进程并用单行心跳显示长耗时状态。

    此处保留 `subprocess.run(..., capture_output=True)` 的核心语义: 子命令的
    stdout / stderr 不实时刷屏, 而是在命令结束后由调用方写入诊断文件。区别在于
    父进程会按固定间隔刷新一行总体进度, 适合 Colab 中的官方复现命令和 adapter
    命令。
    """

    started_at = time.monotonic()
    heartbeat_seconds = max(1.0, float(heartbeat_seconds))
    base_profile = progress_profile or "operation=subprocess"
    emit_progress_status(progress, profile=merge_child_progress_profile(f"{base_profile} status=running", child_progress_path))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        env=env,
    )
    while True:
        elapsed_seconds = time.monotonic() - started_at
        remaining_timeout = None if timeout_seconds is None else max(0.0, float(timeout_seconds) - elapsed_seconds)
        wait_seconds = heartbeat_seconds if remaining_timeout is None else min(heartbeat_seconds, remaining_timeout)
        try:
            stdout, stderr = process.communicate(timeout=wait_seconds)
            break
        except subprocess.TimeoutExpired as timeout_error:
            elapsed_minutes = (time.monotonic() - started_at) / 60.0
            if timeout_seconds is not None and elapsed_seconds >= float(timeout_seconds):
                process.kill()
                stdout, stderr = process.communicate()
                emit_progress_status(
                    progress,
                    profile=merge_child_progress_profile(
                        f"{base_profile} status=timeout elapsed={elapsed_minutes:.1f}min",
                        child_progress_path,
                    ),
                )
                raise subprocess.TimeoutExpired(
                    cmd=command,
                    timeout=timeout_seconds,
                    output=stdout,
                    stderr=stderr,
                ) from timeout_error
            emit_progress_status(
                progress,
                profile=merge_child_progress_profile(
                    f"{base_profile} status=running elapsed={elapsed_minutes:.1f}min",
                    child_progress_path,
                ),
            )
    elapsed_minutes = (time.monotonic() - started_at) / 60.0
    emit_progress_status(
        progress,
        profile=merge_child_progress_profile(
            f"{base_profile} status=completed return_code={process.returncode} elapsed={elapsed_minutes:.1f}min",
            child_progress_path,
        ),
    )
    return subprocess.CompletedProcess(command, int(process.returncode or 0), stdout, stderr)


def call_runner_with_progress_status(
    runner: Callable[..., dict[str, Any]],
    command: Any,
    *,
    cwd: str | PathLike[str] | None,
    timeout_seconds: int,
    progress: object | None = None,
    progress_profile: str = "",
) -> dict[str, Any]:
    """调用支持可选进度参数的命令 runner。

    该函数把测试替身兼容逻辑集中在配置边界附近: 真实 Colab runner 可以接收
    `progress` 和 `progress_profile`, 轻量测试替身则继续只接收命令、cwd 和
    timeout。业务函数只表达当前要执行的命令职责, 不重复维护兼容分支。
    """

    if progress is None:
        return runner(command, cwd=cwd, timeout_seconds=timeout_seconds)
    try:
        return runner(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            progress=progress,
            progress_profile=progress_profile,
        )
    except TypeError as error:
        message = str(error)
        if "progress" not in message and "progress_profile" not in message:
            raise
        return runner(command, cwd=cwd, timeout_seconds=timeout_seconds)
