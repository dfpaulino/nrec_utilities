#!/usr/bin/env python3
"""Count image files per scenario/trip directory under an NREC root.

Usage:
    python count_images.py [ROOT] [--ext png] [--subdir Images]

ROOT defaults to the current directory. For each immediate subdirectory of
ROOT (a scenario/trip), counts files matching *.<ext> inside <scenario>/<subdir>.
"""

import argparse
from pathlib import Path


def count_images(root: Path, subdir: str, ext: str) -> dict[str, int]:
    counts = {}
    for scenario in sorted(root.iterdir()):
        if not scenario.is_dir():
            continue
        image_dir = scenario / subdir
        if image_dir.is_dir():
            counts[scenario.name] = sum(1 for _ in image_dir.glob(f"*.{ext}"))
        else:
            counts[scenario.name] = 0
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="directory containing scenario/trip subdirectories")
    parser.add_argument("--ext", default="png", help="image file extension to count (default: png)")
    parser.add_argument("--subdir", default="Images", help="name of the image subdirectory inside each scenario (default: Images)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    counts = count_images(root, args.subdir, args.ext)

    name_width = max((len(name) for name in counts), default=0)
    for name, count in counts.items():
        print(f"{name:<{name_width}}  {count}")

    print("-" * (name_width + 8))
    print(f"{'TOTAL':<{name_width}}  {sum(counts.values())}")


if __name__ == "__main__":
    main()
