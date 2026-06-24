"""Notebook 和 repository runner 的总体工作量进度显示工具。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import sys
import time
from typing import TypeVar

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
