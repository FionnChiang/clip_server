#!/usr/bin/env python3
"""推理服务入口脚本。

Usage:
    python scripts/serve.py
    python scripts/serve.py --checkpoint output/best_model.pth --port 8080
"""

import argparse
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(current_dir))

from src.inference.api import serve


def main():
    parser = argparse.ArgumentParser(description="Layout Classifier Inference Server")
    parser.add_argument("--checkpoint", type=str, default="output/best_model.pth")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--confidence-threshold", type=float, default=None,
                        help="Override confidence threshold (default: from checkpoint calibration)")
    parser.add_argument("--margin-threshold", type=float, default=None,
                        help="Override top1-top2 margin threshold (default: from checkpoint calibration)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override temperature scaling (default: from checkpoint calibration)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).is_absolute():
        checkpoint_path = str((current_dir / checkpoint_path).resolve())

    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"API docs:   http://{args.host}:{args.port}/docs")

    serve(
        checkpoint_path=checkpoint_path,
        host=args.host,
        port=args.port,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
