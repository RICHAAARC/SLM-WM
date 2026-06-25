"""Notebook 和 repository runner 的总体工作量进度显示工具。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
import subprocess
import sys
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


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


@contextmanager
def progress_bar(total: int, *, desc: str, enabled: bool = True) -> Iterator[object | None]:
    """创建可手动更新的总体进度条。

    该函数用于一个 workflow 内部存在多个资源环节的情况。例如先用
    SD3.5 img2img pipeline 处理一部分攻击, 再释放显存并加载 DDIM
    inversion 后端处理另一部分攻击。两个资源环节共享同一个总体进度条,
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
    heartbeat_seconds: float = 60.0,
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
    emit_progress_status(progress, profile=f"{base_profile} status=running")
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
                emit_progress_status(progress, profile=f"{base_profile} status=timeout elapsed={elapsed_minutes:.1f}min")
                raise subprocess.TimeoutExpired(
                    cmd=command,
                    timeout=timeout_seconds,
                    output=stdout,
                    stderr=stderr,
                ) from timeout_error
            emit_progress_status(progress, profile=f"{base_profile} status=running elapsed={elapsed_minutes:.1f}min")
    elapsed_minutes = (time.monotonic() - started_at) / 60.0
    emit_progress_status(progress, profile=f"{base_profile} status=completed return_code={process.returncode} elapsed={elapsed_minutes:.1f}min")
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
