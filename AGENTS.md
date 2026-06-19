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
