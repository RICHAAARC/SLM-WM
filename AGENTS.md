# Repository Agent Contract

1. Before any modification, read `.codex/project_contract.md`.
2. Before any modification, read the relevant skill file under `.codex/skills/`.
3. Do not bypass harness audits.
4. Do not place runtime-heavy tests in the default pytest path.
5. Do not commit generated `outputs/` content; persistent output files must stay under `outputs/`.
6. Placeholder fields must end with `_placeholder`.
7. Random trace fields must end with `_random` or `_digest_random`.
8. Supported claims must map to governed records, tables, figures, reports, or manifests.
9. Task completion requires running `pytest -q` and `python tools/harness/run_all_audits.py` when available.
10. Git commit messages must be written in Chinese; code identifiers, paths, commands, and model names may keep their original spelling.
11. All persistent output files produced by repository commands must be written under `outputs/`; harness audit reports must use `outputs/audit_reports/`.
12. 避免在业务路径中大量重复防御式校验和错误信息构造; 重复校验应收敛到配置解析、dataclass 构造、schema validator 或测试中, 业务函数内部只保留关键边界校验。
13. 除 `docs/` 下的人类可读规划文件外, 路径、代码、配置、测试、脚本、skill 和根目录说明不得使用 `docs/naming_governance.md` 中登记的过程标记词; 必须使用表达职责、机制或协议角色的语义名称。
