"""验证隔离依赖 profile、精确直接输入和完整哈希锁门禁."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from experiments.runtime import dependency_profiles as dependency_profile_runtime
from experiments.runtime.dependency_profiles import (
    DEPENDENCY_PROFILE_REGISTRY_PATH,
    REQUIRED_DEPENDENCY_PROFILE_NAMES,
    build_dependency_profile_summary,
    get_dependency_profile,
    inspect_dependency_profile_environment,
    load_dependency_profile_registry,
    parse_exact_requirement_spec,
    require_dependency_profile_ready,
)


EXPECTED_RUNTIME_IDENTITIES = {
    "workflow_orchestrator": (
        "3.12.13",
        None,
        None,
        None,
        None,
    ),
    "sd35_method_runtime_gpu": (
        "3.12.13",
        "12.8",
        "2.11.0+cu128",
        "0.26.0+cu128",
        "https://download.pytorch.org/whl/cu128",
    ),
    "t2smark_sd35_gpu": (
        "3.12.13",
        "12.4",
        "2.5.0+cu124",
        "0.20.0+cu124",
        "https://download.pytorch.org/whl/cu124",
    ),
    "tree_ring_official_py39_cu117": (
        "3.9.19",
        "11.7",
        "1.13.0+cu117",
        "0.14.0+cu117",
        "https://download.pytorch.org/whl/cu117",
    ),
    "gaussian_shading_official_py38_cu117": (
        "3.8.20",
        "11.7",
        "1.13.0+cu117",
        "0.14.0+cu117",
        "https://download.pytorch.org/whl/cu117",
    ),
    "shallow_diffuse_official_py39_cu117": (
        "3.9.19",
        "11.7",
        "1.13.0+cu117",
        "0.14.0+cu117",
        "https://download.pytorch.org/whl/cu117",
    ),
}


def _copy_dependency_config(tmp_path: Path) -> Path:
    """复制登记表和直接输入, 供负向测试安全修改."""

    source_config_dir = DEPENDENCY_PROFILE_REGISTRY_PATH.parent
    target_config_dir = tmp_path / "configs"
    target_config_dir.mkdir(parents=True)
    shutil.copy2(
        DEPENDENCY_PROFILE_REGISTRY_PATH,
        target_config_dir / DEPENDENCY_PROFILE_REGISTRY_PATH.name,
    )
    shutil.copytree(
        source_config_dir / "dependency_profiles",
        target_config_dir / "dependency_profiles",
    )
    return target_config_dir / DEPENDENCY_PROFILE_REGISTRY_PATH.name


def _lock_line(specification: str) -> str:
    """构造仅用于 parser 测试的确定性哈希条目."""

    digest = hashlib.sha256(f"test-fixture:{specification}".encode("utf-8")).hexdigest()
    return f"{specification} --hash=sha256:{digest}"


@pytest.mark.quick
def test_registry_defines_one_cpu_parent_and_five_cuda_runtime_identities() -> None:
    """父编排 profile 必须是 CPU, 五个科学 profile 必须精确登记 CUDA identity."""

    profiles = load_dependency_profile_registry()

    assert tuple(profiles) == REQUIRED_DEPENDENCY_PROFILE_NAMES
    assert set(profiles) == set(EXPECTED_RUNTIME_IDENTITIES)
    for profile_name, expected_identity in EXPECTED_RUNTIME_IDENTITIES.items():
        profile = profiles[profile_name]
        assert (
            profile.python_version,
            profile.cuda_version,
            profile.torch_version,
            profile.torchvision_version,
            profile.pytorch_index_url,
        ) == expected_identity
        assert profile.python_implementation == "CPython"
        assert profile.operating_system == "linux"
        assert profile.machine == "x86_64"
        if profile_name == "workflow_orchestrator":
            assert profile.execution_role == "workflow_orchestration"
            assert profile.accelerator_runtime == "cpu"
            assert all(
                value is None
                for value in (
                    profile.cuda_version,
                    profile.torch_version,
                    profile.torchvision_version,
                    profile.pytorch_index_url,
                )
            )
            assert all(
                not specification.lower().startswith(("torch==", "torchvision=="))
                for specification in profile.direct_requirements
            )
        else:
            assert profile.accelerator_runtime == "cuda"
            assert f"torch=={profile.torch_version}" in profile.direct_requirements
            assert f"torchvision=={profile.torchvision_version}" in profile.direct_requirements


@pytest.mark.quick
def test_committed_direct_inputs_only_contain_exact_double_equals_specs() -> None:
    """直接输入不得出现范围、可漂移标签、安装选项或未版本化包."""

    for profile in load_dependency_profile_registry().values():
        assert profile.direct_requirements
        for specification in profile.direct_requirements:
            dependency = parse_exact_requirement_spec(specification)
            assert dependency.specification == specification
            assert specification.count("==") == 1
            assert all(token not in specification.lower() for token in (">=", "<=", "~=", "latest", "upgrade"))


@pytest.mark.quick
def test_t2smark_uses_exact_sd35_compatible_diffusers_candidate() -> None:
    """T2SMark SD3.5 输入必须固定为包含 StableDiffusion3Pipeline 的版本."""

    profile = get_dependency_profile("t2smark_sd35_gpu")

    assert "diffusers==0.32.0" in profile.direct_requirements
    assert "diffusers==0.21.4" not in profile.direct_requirements
    assert profile.formal_ready is False


@pytest.mark.quick
def test_gaussian_shading_direct_input_is_inference_minimal_and_python38_compatible() -> None:
    """Gaussian Shading profile 只登记正式推理入口依赖并固定兼容的 SciPy."""

    profile = get_dependency_profile("gaussian_shading_official_py38_cu117")
    normalized_names = {
        parse_exact_requirement_spec(specification).normalized_name
        for specification in profile.direct_requirements
    }

    assert "scipy==1.10.1" in profile.direct_requirements
    assert "huggingface_hub==0.25.2" in profile.direct_requirements
    assert normalized_names.isdisjoint(
        {
            "academictorrents",
            "albumentations",
            "clip",
            "horovod",
            "pytorch-lightning",
            "skimage",
        }
    )


@pytest.mark.quick
def test_missing_complete_hash_locks_fail_closed_with_stable_summaries() -> None:
    """已提交精确直接输入不能替代完整哈希锁, 缺锁时必须阻断正式运行."""

    for profile_name in REQUIRED_DEPENDENCY_PROFILE_NAMES:
        first_summary = build_dependency_profile_summary(profile_name)
        second_summary = build_dependency_profile_summary(profile_name)

        assert first_summary == second_summary
        assert first_summary["formal_ready"] is False
        assert first_summary["complete_hash_lock_present"] is False
        assert first_summary["complete_hash_lock_digest"] is None
        assert first_summary["readiness_blockers"] == ["complete_hash_lock_missing"]
        assert len(first_summary["profile_digest"]) == 64
        assert len(first_summary["summary_digest"]) == 64
        with pytest.raises(RuntimeError, match="complete_hash_lock_missing"):
            require_dependency_profile_ready(profile_name)


@pytest.mark.quick
@pytest.mark.parametrize(
    "invalid_specification",
    (
        "numpy>=1.24.4",
        "numpy<2",
        "numpy",
        "numpy==latest",
        "pip install --upgrade numpy==1.24.4",
    ),
)
def test_exact_requirement_parser_rejects_drifting_specs(invalid_specification: str) -> None:
    """统一 parser 必须拒绝所有非精确直接依赖形式."""

    with pytest.raises(ValueError):
        parse_exact_requirement_spec(invalid_specification)


@pytest.mark.quick
def test_registry_rejects_direct_input_digest_drift(tmp_path: Path) -> None:
    """直接输入即使新增精确包也必须同步更新受治理摘要."""

    registry_path = _copy_dependency_config(tmp_path)
    direct_path = tmp_path / "configs/dependency_profiles/workflow_orchestrator_direct.txt"
    direct_path.write_text(
        direct_path.read_text(encoding="utf-8") + "requests==2.32.3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="direct_requirements_digest"):
        load_dependency_profile_registry(registry_path)


@pytest.mark.quick
def test_registry_rejects_cuda_pair_or_index_drift(tmp_path: Path) -> None:
    """torch pair local version 与 CUDA index 任一漂移都必须被阻断."""

    registry_path = _copy_dependency_config(tmp_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["profiles"]["t2smark_sd35_gpu"]["pytorch"]["index_url"] = (
        "https://download.pytorch.org/whl/cu121"
    )
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="index_url 与 CUDA 版本不一致"):
        load_dependency_profile_registry(registry_path)


@pytest.mark.quick
def test_materialized_hash_lock_must_cover_direct_inputs_and_enables_gate(tmp_path: Path) -> None:
    """格式有效且覆盖全部直接输入的锁文件才能打开对应 profile 门禁."""

    registry_path = _copy_dependency_config(tmp_path)
    profile_name = "workflow_orchestrator"
    source_profile = get_dependency_profile(profile_name, registry_path)
    lock_path = tmp_path / source_profile.complete_hash_lock_path
    lock_path.write_text(
        "\n".join(_lock_line(specification) for specification in source_profile.direct_requirements) + "\n",
        encoding="utf-8",
    )

    ready_profile = require_dependency_profile_ready(profile_name, registry_path)
    summary = build_dependency_profile_summary(profile_name, registry_path)

    assert ready_profile.formal_ready is True
    assert ready_profile.complete_hash_lock_present is True
    assert ready_profile.complete_hash_lock_dependency_count == len(ready_profile.direct_requirements)
    assert ready_profile.complete_hash_lock_digest is not None
    assert len(ready_profile.locked_requirements) == len(ready_profile.direct_requirements)
    assert summary["locked_requirements"] == list(ready_profile.locked_requirements)
    assert summary["readiness_blockers"] == []


@pytest.mark.quick
def test_hash_lock_without_all_direct_inputs_is_rejected(tmp_path: Path) -> None:
    """带哈希但缺少直接依赖的文件不能被解释为完整闭包."""

    registry_path = _copy_dependency_config(tmp_path)
    profile_name = "workflow_orchestrator"
    source_profile = get_dependency_profile(profile_name, registry_path)
    lock_path = tmp_path / source_profile.complete_hash_lock_path
    lock_path.write_text(
        "\n".join(_lock_line(specification) for specification in source_profile.direct_requirements[:-1]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="完整哈希锁缺少直接依赖"):
        load_dependency_profile_registry(registry_path)


@pytest.mark.quick
def test_environment_inspection_stays_blocked_without_complete_lock() -> None:
    """即使只执行只读环境检查, 缺少完整锁仍必须出现在稳定 blocker 中."""

    first_report = inspect_dependency_profile_environment("workflow_orchestrator")
    second_report = inspect_dependency_profile_environment("workflow_orchestrator")

    assert first_report == second_report
    assert first_report["decision"] == "blocked"
    assert first_report["profile_formal_ready"] is False
    assert "complete_hash_lock_missing" in first_report["readiness_blockers"]
    assert len(first_report["inspection_digest"]) == 64


@pytest.mark.quick
def test_cpu_orchestrator_inspection_skips_torch_and_cuda_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """父编排环境只核验 CPU 平台和完整锁包, 不得导入 torch 或要求 CUDA."""

    registry_path = _copy_dependency_config(tmp_path)
    profile_name = "workflow_orchestrator"
    source_profile = get_dependency_profile(profile_name, registry_path)
    lock_path = tmp_path / source_profile.complete_hash_lock_path
    lock_path.write_text(
        "\n".join(
            _lock_line(specification)
            for specification in source_profile.direct_requirements
        )
        + "\n",
        encoding="utf-8",
    )
    installed_versions = {
        parse_exact_requirement_spec(specification).normalized_name:
        parse_exact_requirement_spec(specification).version
        for specification in source_profile.direct_requirements
    }
    monkeypatch.setattr(
        dependency_profile_runtime,
        "_installed_distribution_version",
        lambda package_name: installed_versions.get(
            parse_exact_requirement_spec(f"{package_name}==0").normalized_name
        ),
    )
    monkeypatch.setattr(
        dependency_profile_runtime,
        "_import_torch_module",
        lambda: (_ for _ in ()).throw(
            AssertionError("CPU 编排 inspection 不得导入 torch")
        ),
    )
    monkeypatch.setattr(
        dependency_profile_runtime.platform,
        "python_implementation",
        lambda: "CPython",
    )
    monkeypatch.setattr(
        dependency_profile_runtime.platform,
        "python_version",
        lambda: source_profile.python_version,
    )
    monkeypatch.setattr(dependency_profile_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        dependency_profile_runtime.platform,
        "machine",
        lambda: "x86_64",
    )

    report = inspect_dependency_profile_environment(
        profile_name,
        path=registry_path,
    )

    assert report["decision"] == "pass"
    assert report["mismatches"] == []
    assert report["expected_environment"]["accelerator_runtime"] == "cpu"
    assert report["expected_environment"]["cuda_version"] is None
    assert report["expected_environment"]["torch_version"] is None
    assert report["observed_environment"]["torch_module_available"] is None
    assert report["observed_environment"]["cuda_available"] is None


@pytest.mark.quick
def test_environment_inspection_pass_requires_lock_and_every_exact_runtime_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整锁、解释器、CUDA 和全部 installed version 精确一致时才允许 pass."""

    registry_path = _copy_dependency_config(tmp_path)
    profile_name = "t2smark_sd35_gpu"
    source_profile = get_dependency_profile(profile_name, registry_path)
    lock_path = tmp_path / source_profile.complete_hash_lock_path
    lock_path.write_text(
        "\n".join(_lock_line(specification) for specification in source_profile.direct_requirements) + "\n",
        encoding="utf-8",
    )
    installed_versions: dict[str, str] = {}
    for specification in source_profile.direct_requirements:
        dependency = parse_exact_requirement_spec(specification)
        installed_versions[dependency.package_name] = dependency.version
        installed_versions[dependency.normalized_name] = dependency.version
    monkeypatch.setattr(
        dependency_profile_runtime,
        "_installed_distribution_version",
        lambda package_name: installed_versions.get(package_name),
    )
    monkeypatch.setattr(dependency_profile_runtime.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(
        dependency_profile_runtime.platform,
        "python_version",
        lambda: source_profile.python_version,
    )
    monkeypatch.setattr(dependency_profile_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dependency_profile_runtime.platform, "machine", lambda: "x86_64")
    fake_torch = SimpleNamespace(
        __version__=source_profile.torch_version,
        version=SimpleNamespace(cuda=source_profile.cuda_version),
        cuda=SimpleNamespace(is_available=lambda: True),
    )

    passing_report = inspect_dependency_profile_environment(
        profile_name,
        torch_module=fake_torch,
        path=registry_path,
    )

    assert passing_report["decision"] == "pass"
    assert passing_report["profile_formal_ready"] is True
    assert passing_report["environment_match"] is True
    assert passing_report["mismatches"] == []
    assert passing_report["readiness_blockers"] == []

    installed_versions["numpy"] = "0.0.1"
    blocked_report = inspect_dependency_profile_environment(
        profile_name,
        torch_module=fake_torch,
        path=registry_path,
    )
    assert blocked_report["decision"] == "blocked"
    assert "locked_dependency_version_mismatch:numpy" in blocked_report["mismatches"]
    assert "direct_dependency_version_mismatch:numpy" not in blocked_report["mismatches"]


@pytest.mark.quick
def test_environment_inspection_rejects_extra_transitive_lock_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """直接依赖匹配但额外 transitive 版本漂移时仍必须阻断正式环境."""

    registry_path = _copy_dependency_config(tmp_path)
    profile_name = "sd35_method_runtime_gpu"
    source_profile = get_dependency_profile(profile_name, registry_path)
    transitive_specification = "transitive-evidence-package==9.8.7"
    lock_path = tmp_path / source_profile.complete_hash_lock_path
    lock_path.write_text(
        "\n".join(
            [
                *(
                    _lock_line(specification)
                    for specification in source_profile.direct_requirements
                ),
                _lock_line(transitive_specification),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ready_profile = get_dependency_profile(profile_name, registry_path)
    installed_versions: dict[str, str] = {}
    for specification in ready_profile.locked_requirements:
        dependency = parse_exact_requirement_spec(specification)
        installed_versions[dependency.normalized_name] = dependency.version
    installed_versions["transitive-evidence-package"] = "9.8.6"

    queried_distribution_names: list[str] = []

    def installed_distribution_version(package_name: str) -> str | None:
        """记录查询次数, 证明 direct 与 locked 映射共享一次版本缓存."""

        normalized_name = parse_exact_requirement_spec(
            f"{package_name}==0"
        ).normalized_name
        queried_distribution_names.append(normalized_name)
        return installed_versions.get(normalized_name)

    monkeypatch.setattr(
        dependency_profile_runtime,
        "_installed_distribution_version",
        installed_distribution_version,
    )
    monkeypatch.setattr(
        dependency_profile_runtime.platform,
        "python_implementation",
        lambda: "CPython",
    )
    monkeypatch.setattr(
        dependency_profile_runtime.platform,
        "python_version",
        lambda: ready_profile.python_version,
    )
    monkeypatch.setattr(dependency_profile_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dependency_profile_runtime.platform, "machine", lambda: "x86_64")
    fake_torch = SimpleNamespace(
        __version__=ready_profile.torch_version,
        version=SimpleNamespace(cuda=ready_profile.cuda_version),
        cuda=SimpleNamespace(is_available=lambda: True),
    )

    report = inspect_dependency_profile_environment(
        profile_name,
        torch_module=fake_torch,
        path=registry_path,
    )

    assert report["decision"] == "blocked"
    assert report["expected_environment"]["direct_dependencies"] == report[
        "observed_environment"
    ]["direct_dependencies"]
    assert report["expected_environment"]["locked_dependencies"][
        "transitive-evidence-package"
    ] == "9.8.7"
    assert report["observed_environment"]["locked_dependencies"][
        "transitive-evidence-package"
    ] == "9.8.6"
    assert "locked_dependency_version_mismatch:transitive-evidence-package" in report[
        "readiness_blockers"
    ]
    assert sorted(queried_distribution_names) == sorted(installed_versions)
    assert len(queried_distribution_names) == len(set(queried_distribution_names))
