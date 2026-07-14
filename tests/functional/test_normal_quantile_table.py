"""验证跨平台位精确标准正态分位数表与 Gaussian PRG."""

from __future__ import annotations

import hashlib
import json
import struct

import pytest

from main.core import keyed_prg as keyed_prg_module
from main.core.digest import build_stable_digest
from main.core.keyed_prg import (
    KEYED_PRG_VERSION,
    _normal_quantile_indices,
    build_keyed_gaussian_tensor,
    keyed_prg_protocol_record,
    require_supported_keyed_prg_version,
)
from main.core.normal_quantile_table import (
    NORMAL_QUANTILE_COUNT,
    NORMAL_QUANTILE_FLOAT32_KS_DISTANCE_BOUND,
    NORMAL_QUANTILE_IDEAL_MIDPOINT_KS_DISTANCE,
    NORMAL_QUANTILE_MAXIMUM_CDF_CELL_WIDTH,
    NORMAL_QUANTILE_REFERENCE_VERIFICATION_DIGEST,
    NORMAL_QUANTILE_TABLE_SHA256,
    standard_normal_quantile_float32_table,
    standard_normal_quantile_reference_record,
    standard_normal_quantile_table_record,
)


pytestmark = pytest.mark.quick


def _table_bytes(values: tuple[float, ...]) -> bytes:
    """按协议规定的大端 binary32 顺序重建完整表字节."""

    return b"".join(struct.pack(">f", value) for value in values)


def test_legacy_box_muller_protocol_is_not_accepted() -> None:
    """当前项目不保留旧 Box-Muller 结果或配置兼容入口."""

    with pytest.raises(ValueError, match="keyed_prg_version"):
        require_supported_keyed_prg_version(
            "sha256_counter_box_muller_float32_legacy"
        )


def test_frozen_normal_quantile_table_has_exact_distribution_contract() -> None:
    """1048576点量化标准正态律必须满足摘要,对称性和矩门禁."""

    values = standard_normal_quantile_float32_table()
    record = standard_normal_quantile_table_record()
    reference_record = standard_normal_quantile_reference_record()

    assert len(values) == NORMAL_QUANTILE_COUNT == 1048576
    assert hashlib.sha256(_table_bytes(values)).hexdigest() == (
        NORMAL_QUANTILE_TABLE_SHA256
    )
    assert all(left < right for left, right in zip(values, values[1:]))
    assert all(
        values[index] == -values[-1 - index]
        for index in range(NORMAL_QUANTILE_COUNT // 2)
    )
    mean = sum(values) / NORMAL_QUANTILE_COUNT
    variance = sum(value * value for value in values) / (
        NORMAL_QUANTILE_COUNT
    )
    fourth_moment = sum(value**4 for value in values) / (
        NORMAL_QUANTILE_COUNT
    )
    assert mean == 0.0
    assert variance == pytest.approx(0.9999987224694579, abs=1e-15)
    assert fourth_moment == pytest.approx(2.9999312709276453, abs=1e-14)
    assert values[0] == -4.900964260101318
    assert values[-1] == 4.900964260101318
    assert NORMAL_QUANTILE_MAXIMUM_CDF_CELL_WIDTH == 1.0 / 1048576
    assert NORMAL_QUANTILE_IDEAL_MIDPOINT_KS_DISTANCE == 0.5 / 1048576
    assert NORMAL_QUANTILE_FLOAT32_KS_DISTANCE_BOUND == (
        pytest.approx(4.912236096776823e-07, abs=1e-21)
    )
    assert "normal_quantile_reference_verification_digest" not in record
    assert reference_record["normal_quantile_reference_mismatch_count"] == 0
    assert reference_record["normal_quantile_reference_verification_digest"] == (
        NORMAL_QUANTILE_REFERENCE_VERIFICATION_DIGEST
    )
    expected_reference_digest = reference_record.pop(
        "normal_quantile_reference_verification_digest"
    )
    assert build_stable_digest(reference_record) == expected_reference_digest


def test_gaussian_prg_uses_no_platform_transcendental_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gaussian 运行路径不得重新依赖系统 log,sqrt,sin 或 cos."""

    pytest.importorskip("torch")

    def reject_platform_math(*_args, **_kwargs):
        raise AssertionError("Gaussian PRG 不得调用平台 transcendental 函数")

    for function_name in ("log", "sqrt", "sin", "cos"):
        monkeypatch.setattr(
            keyed_prg_module.math,
            function_name,
            reject_platform_math,
        )

    tensor = build_keyed_gaussian_tensor(
        (4,),
        "known-key",
        {"operator": "known_answer_gaussian", "role": "cpu_test"},
    )

    assert tensor.tolist() == [
        -0.09923840314149857,
        1.7385268211364746,
        0.6552790999412537,
        0.8304281830787659,
    ]
    assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == (
        "589a60c85b588cc14bf41151021cbac35e94edf8f9a7c5e0603f7733abcefb4c"
    )


def test_sampling_protocol_digest_excludes_reference_verifier_metadata() -> None:
    """离线 MPFR 复验参数不得改变实际采样算法和正式 latent 身份."""

    protocol = keyed_prg_protocol_record()
    expected_digest = protocol.pop("keyed_prg_protocol_digest")

    assert not any(
        field_name.startswith("normal_quantile_reference_")
        for field_name in protocol
    )
    assert build_stable_digest(protocol) == expected_digest


def test_normal_indices_match_independent_concatenated_bit_string() -> None:
    """独立拼接 SHA-256 比特串, 复验20位索引的块内和跨块边界."""

    shape = (65,)
    key_material = "known-key"
    domain_fields = {
        "operator": "known_answer_gaussian",
        "role": "cpu_test",
    }
    domain_payload = {
        "keyed_prg_version": KEYED_PRG_VERSION,
        "key_material": key_material,
        "domain_fields": domain_fields,
        "shape": shape,
    }
    serialized = json.dumps(
        domain_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    domain_digest = hashlib.sha256(serialized).digest()
    block_count = (shape[0] * 20 + 255) // 256
    bit_string = "".join(
        f"{byte:08b}"
        for counter in range(block_count)
        for byte in hashlib.sha256(
            domain_digest + counter.to_bytes(16, "big")
        ).digest()
    )
    expected_indices = [
        int(bit_string[offset : offset + 20], 2)
        for offset in range(0, shape[0] * 20, 20)
    ]

    assert _normal_quantile_indices(shape[0], domain_digest) == (
        expected_indices
    )


@pytest.mark.parametrize(
    ("element_count", "expected_sha256"),
    (
        (12, "7c4c762ef96ee2ac30eda67551403567f8655a70ec75a75a8e15ebbd5896b450"),
        (13, "6de15d177e58daf0e744cc26ecd315429fec21740531e47c6a034b760965f22c"),
        (16, "3506518515b1ec120d3703af212820cc653afb9a4df700d35e3369326e2cc79a"),
        (17, "d1c2ca640182755e7e5a0b3e1af738108e3f9d271cc350e98ca6395622d8ab37"),
        (64, "83a4488c8c96ba631032bd4bc450c485419ccfa3cff70f643953ae6e9fe5aa9c"),
        (65, "ddb1a19cadad158aa1c95e028b91a05296d092e32a457416ee7f2ef7c7b45de4"),
    ),
)
def test_gaussian_prg_known_answers_cross_sha256_block_boundaries(
    element_count: int,
    expected_sha256: str,
) -> None:
    """20位索引跨 SHA-256 块时仍须保持冻结逐字节答案."""

    tensor = build_keyed_gaussian_tensor(
        (element_count,),
        "known-key",
        {"operator": "known_answer_gaussian", "role": "cpu_test"},
    )

    assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == (
        expected_sha256
    )
