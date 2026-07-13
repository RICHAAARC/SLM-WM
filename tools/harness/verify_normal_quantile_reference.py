"""使用 MPFR 中点夹逼独立复验 Q20 量化标准正态表的逐项正确舍入."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import struct
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.core.digest import build_stable_digest
from main.core.normal_quantile_table import (
    NORMAL_QUANTILE_COUNT,
    NORMAL_QUANTILE_FLOAT32_CDF_ROUNDING_ERROR_BOUND,
    NORMAL_QUANTILE_REFERENCE_MPFR_ROUNDING_MODE,
    NORMAL_QUANTILE_REFERENCE_NEWTON_ITERATIONS,
    NORMAL_QUANTILE_REFERENCE_PRECISION_BITS,
    NORMAL_QUANTILE_REFERENCE_VERIFICATION_DIGEST,
    NORMAL_QUANTILE_REFERENCE_VERIFICATION_PROTOCOL,
    NORMAL_QUANTILE_TABLE_SHA256,
    standard_normal_quantile_float32_table,
)


DEFAULT_REPORT_PATH = Path(
    "outputs/audit_reports/normal_quantile_reference_verification.json"
)


def _load_gmpy2() -> Any:
    """按需加载仅供离线证书生成使用的 MPFR Python 绑定."""

    try:
        return importlib.import_module("gmpy2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "离线分位数参考复验需要开发环境安装 gmpy2"
        ) from exc


def _float32_neighbors(value: float) -> tuple[float, float]:
    """通过相邻 binary32 位模式返回正有限值的上下邻点."""

    bit_pattern = int.from_bytes(struct.pack(">f", value), "big")
    if not 0 < bit_pattern < 0x7F7FFFFF:
        raise ValueError("参考复验只接受正有限 binary32 分位数")
    return (
        struct.unpack(">f", (bit_pattern - 1).to_bytes(4, "big"))[0],
        struct.unpack(">f", (bit_pattern + 1).to_bytes(4, "big"))[0],
    )


def verify_normal_quantile_reference() -> dict[str, Any]:
    """用192位 MPFR 中点 CDF 夹逼和 Newton 根复验正半轴舍入区间."""

    gmpy2 = _load_gmpy2()
    context = gmpy2.get_context()
    previous_precision = context.precision
    previous_rounding = context.round
    context.precision = NORMAL_QUANTILE_REFERENCE_PRECISION_BITS
    context.round = gmpy2.RoundToNearest
    try:
        sqrt_two = gmpy2.sqrt(gmpy2.mpfr(2))
        sqrt_two_pi = gmpy2.sqrt(2 * gmpy2.const_pi())
        table = standard_normal_quantile_float32_table()
        mismatch_indices: list[int] = []
        maximum_cdf_rounding_error = gmpy2.mpfr(0)
        minimum_midpoint_probability_margin = gmpy2.mpfr("inf")
        for index in range(NORMAL_QUANTILE_COUNT // 2, NORMAL_QUANTILE_COUNT):
            probability = gmpy2.mpfr(2 * index + 1) / (
                2 * NORMAL_QUANTILE_COUNT
            )
            candidate = table[index]
            candidate_mpfr = gmpy2.mpfr(candidate)
            candidate_cdf = (
                1 + gmpy2.erf(candidate_mpfr / sqrt_two)
            ) / 2
            maximum_cdf_rounding_error = max(
                maximum_cdf_rounding_error,
                abs(candidate_cdf - probability),
            )
            root = candidate_mpfr
            for _ in range(NORMAL_QUANTILE_REFERENCE_NEWTON_ITERATIONS):
                cdf = (1 + gmpy2.erf(root / sqrt_two)) / 2
                density = gmpy2.exp(-(root * root) / 2) / sqrt_two_pi
                root -= (cdf - probability) / density
            previous_value, next_value = _float32_neighbors(candidate)
            lower_midpoint = (
                gmpy2.mpfr(previous_value) + gmpy2.mpfr(candidate)
            ) / 2
            upper_midpoint = (
                gmpy2.mpfr(candidate) + gmpy2.mpfr(next_value)
            ) / 2
            lower_midpoint_cdf = (
                1 + gmpy2.erf(lower_midpoint / sqrt_two)
            ) / 2
            upper_midpoint_cdf = (
                1 + gmpy2.erf(upper_midpoint / sqrt_two)
            ) / 2
            lower_probability_margin = probability - lower_midpoint_cdf
            upper_probability_margin = upper_midpoint_cdf - probability
            minimum_midpoint_probability_margin = min(
                minimum_midpoint_probability_margin,
                lower_probability_margin,
                upper_probability_margin,
            )
            if not (
                lower_probability_margin > 0
                and upper_probability_margin > 0
                and lower_midpoint < root < upper_midpoint
            ):
                mismatch_indices.append(index)
    finally:
        context.precision = previous_precision
        context.round = previous_rounding

    reference_payload = {
        "normal_quantile_reference_mismatch_count": len(mismatch_indices),
        "normal_quantile_reference_precision_bits": (
            NORMAL_QUANTILE_REFERENCE_PRECISION_BITS
        ),
        "normal_quantile_reference_newton_iterations": (
            NORMAL_QUANTILE_REFERENCE_NEWTON_ITERATIONS
        ),
        "normal_quantile_reference_verification_protocol": (
            NORMAL_QUANTILE_REFERENCE_VERIFICATION_PROTOCOL
        ),
        "normal_quantile_table_sha256": NORMAL_QUANTILE_TABLE_SHA256,
        "normal_quantile_reference_verified_positive_entry_count": (
            NORMAL_QUANTILE_COUNT // 2
        ),
        "normal_quantile_reference_mpfr_rounding_mode": (
            NORMAL_QUANTILE_REFERENCE_MPFR_ROUNDING_MODE
        ),
    }
    verification_digest = build_stable_digest(reference_payload)
    observed_cdf_rounding_error = float(maximum_cdf_rounding_error)
    ready = (
        not mismatch_indices
        and observed_cdf_rounding_error
        <= NORMAL_QUANTILE_FLOAT32_CDF_ROUNDING_ERROR_BOUND
        and verification_digest
        == NORMAL_QUANTILE_REFERENCE_VERIFICATION_DIGEST
    )
    if not ready:
        raise RuntimeError("Q20 标准正态分位数参考复验失败")
    return {
        **reference_payload,
        "normal_quantile_reference_verification_digest": (
            verification_digest
        ),
        "normal_quantile_reference_verification_ready": True,
        "normal_quantile_observed_maximum_float32_cdf_rounding_error": (
            observed_cdf_rounding_error
        ),
        "normal_quantile_reference_minimum_midpoint_probability_margin": (
            float(minimum_midpoint_probability_margin)
        ),
        "normal_quantile_declared_float32_cdf_rounding_error_bound": (
            NORMAL_QUANTILE_FLOAT32_CDF_ROUNDING_ERROR_BOUND
        ),
        "normal_quantile_reference_gmpy2_version": str(gmpy2.version()),
        "normal_quantile_reference_mpfr_version": str(gmpy2.mpfr_version()),
        "supports_paper_claim": False,
    }


def write_reference_verification_report(path: Path) -> Path:
    """把离线参考复验报告限制写入仓库 outputs 目录."""

    root = ROOT
    output_root = (root / "outputs").resolve()
    resolved_path = (root / path).resolve()
    resolved_path.relative_to(output_root)
    report = verify_normal_quantile_reference()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return resolved_path


def main() -> int:
    """执行命令行复验并输出受治理报告路径."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    report_path = write_reference_verification_report(args.output)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
