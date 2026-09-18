"""Export the registered audited study as static read-only demo results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hypercast4d.comparison_archives import list_archives


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "web" / "public" / "offline-results.json",
    )
    args = parser.parse_args()
    results = list_archives(PROJECT_ROOT / "results" / "playground", PROJECT_ROOT)
    payload = {
        "schema_version": 1,
        "source": "tslib15-20260915",
        "result_count": len(results),
        "fit_count": sum(len(result["runs"]) for result in results),
        "results": results,
    }
    if payload["result_count"] != 120 or payload["fit_count"] != 360:
        raise RuntimeError(
            f"Expected the audited 120-result/360-fit archive; found "
            f"{payload['result_count']} results and {payload['fit_count']} fits"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"Wrote {payload['result_count']} results ({payload['fit_count']} fits) to {args.output}")


if __name__ == "__main__":
    main()
