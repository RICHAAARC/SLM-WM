"""Run the real SD3.5 content-routing reference producer in an isolated child."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runners.content_routing_reference_runtime import (
    write_content_routing_reference_runtime_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="materialize real content-routing reference candidate bytes",
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--output-root",
        default="outputs/content_routing_reference_runtime",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = write_content_routing_reference_runtime_outputs(
        root=arguments.root,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
