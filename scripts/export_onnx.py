from __future__ import annotations

import argparse
import json
from pathlib import Path

from cslr.training.export import export_checkpoint_to_onnx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence-length", type=int, default=48)
    args = parser.parse_args()

    result = export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
