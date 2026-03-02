from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry hook for OpenFOAM case generation")
    parser.add_argument("--case-dir", required=True, help="OpenFOAM case directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir)
    params_path = case_dir / "case_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}")

    params = json.loads(params_path.read_text(encoding="utf-8"))

    obstacle_dict = case_dir / "system" / "obstacleDict"
    obstacle_dict.write_text(
        "\n".join(
            [
                "// Geometry hook output for downstream mesh generator",
                f"shape {params['shape']};",
                f"Re {params['Re']};",
                f"dy {params['dy']};",
                f"eps {params['eps']};",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
