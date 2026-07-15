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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **SLM-WM** (21284 symbols, 42147 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/SLM-WM/context` | Codebase overview, check index freshness |
| `gitnexus://repo/SLM-WM/clusters` | All functional areas |
| `gitnexus://repo/SLM-WM/processes` | All execution flows |
| `gitnexus://repo/SLM-WM/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
