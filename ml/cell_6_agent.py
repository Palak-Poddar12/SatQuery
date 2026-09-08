"""Colab/local entry point for the GeoAI Analyst agent pipeline.

Run from the repository root:
    python ml/cell_6_agent.py

Run with a real image:
    python ml/cell_6_agent.py --image path/to/image.jpg --question "What land cover is visible?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from .pipeline import run_agent
except ImportError:
    from pipeline import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GeoAI Analyst agent")
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to a JPEG, PNG, or TIFF image. Uses a dummy image when omitted.",
    )
    parser.add_argument(
        "--question",
        default="What land cover is visible?",
        help="Question to route through the agent.",
    )
    parser.add_argument(
        "--has-sar",
        action="store_true",
        help="Treat the second image as SAR input.",
    )
    parser.add_argument(
        "--sar-image",
        type=Path,
        help="Optional SAR image path; requires --image and --has-sar.",
    )
    args = parser.parse_args()

    if args.image is None:
        images = [Image.new("RGB", (224, 224), (60, 120, 40))]
    else:
        if not args.image.is_file():
            raise FileNotFoundError(f"Image not found: {args.image}")
        images = [args.image]

    if args.has_sar:
        if args.sar_image is None:
            raise ValueError("--has-sar requires --sar-image")
        if not args.sar_image.is_file():
            raise FileNotFoundError(f"SAR image not found: {args.sar_image}")
        images.append(args.sar_image)

    result = run_agent(
        images,
        question=args.question,
        has_sar=args.has_sar,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
