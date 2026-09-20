"""
Generic helpers for a YOLO (Ultralytics) dataset tree.

A YOLO tree here means:

    {root}/images/{split}/<stem>.png
    {root}/labels/{split}/<stem>.txt

with one label file per image, matched by stem. An *empty* label file is
meaningful: it declares the frame as explicit background.

This module is deliberately free of numpy and opencv. `nrec_utils.nrec_utils`
is NREC/VOC specific and stays importable with nothing but the standard
library (see the README), so the pieces both it and the occlusion generator
need live here rather than there.
"""

import os
from pathlib import Path

SPLITS = ("train", "val", "test")

# how many offending stems an aggregated layout error quotes before it stops
_MAX_REPORTED = 10


class YoloLayoutError(ValueError):
    """The directory handed over is not a usable YOLO dataset tree."""


def even_stride_select(items, factor: float) -> list:
    """
    Pick round(len(items) * factor) entries, at least one, spread evenly over
    `items` and keeping their order.

    Spreading the indices with round(i * total / count) keeps them strictly
    increasing (since total >= count), so we get exactly that many distinct
    entries at an even spacing rather than the nearest 1/N stride. For the NREC
    frames - captured ~133ms apart, so neighbours are near-duplicates - even
    spacing keeps coverage of the whole sequence while dropping the redundancy.

    Deterministic: no RNG, so the same arguments always yield the same entries.
    Ordering is the caller's responsibility; this does not sort.
    """
    if not 0 < factor <= 1:
        raise ValueError(f"factor must be in (0, 1], got {factor}")

    total = len(items)
    if total == 0:
        return []

    count = max(1, round(total * factor))
    return [items[round(i * total / count)] for i in range(count)]


def _describe(dir_path: Path) -> str:
    """List what is actually in a directory, to make an error message useful."""
    try:
        found = sorted(entry.name for entry in os.scandir(dir_path))
    except OSError:
        return "unreadable"
    if not found:
        return "empty"
    shown = ", ".join(found[:_MAX_REPORTED])
    return shown + (", ..." if len(found) > _MAX_REPORTED else "")


def _check_labels_not_xml(label_dir: Path, split: str, strict_xml: bool):
    """
    Guard against the most likely mistake: pointing this at a tree whose
    annotations were never converted from Pascal VOC .xml to YOLO .txt.
    """
    xml_count = sum(1 for f in os.scandir(label_dir) if f.name.lower().endswith(".xml"))
    if not xml_count:
        return

    txt_count = sum(1 for f in os.scandir(label_dir) if f.name.lower().endswith(".txt"))

    if txt_count == 0:
        raise YoloLayoutError(
            f"{label_dir} holds {xml_count} .xml annotations and no .txt labels - this "
            f"dataset was never converted to YOLO format. Run nrec_xml_to_yolo("
            f"src_base_dir=...) (or the nrec-yolo script) over it first."
        )

    # A mix is a real state rather than corruption: nrec_xml_to_yolo deliberately
    # leaves an .xml it could not convert in place, so the failure stays visible.
    message = (
        f"{label_dir} still holds {xml_count} .xml alongside {txt_count} .txt - these "
        f"annotations failed conversion and their images are unlabelled. Fix or remove "
        f"them, or pass allow_stray_xml/--allow-stray-xml to continue anyway."
    )
    if strict_xml:
        raise YoloLayoutError(message)
    print(f"WARN - split [{split}]: {message}")


def _pairs_for_split(image_dir: Path, label_dir: Path, split: str) -> list:
    """Match every .png in a split to its .txt label, collecting all problems."""
    images_by_stem = {}
    duplicates = []
    for entry in sorted(os.scandir(image_dir), key=lambda e: e.name):
        if not entry.is_file():
            continue
        name = entry.name
        if not name.lower().endswith(".png"):
            print(f"WARN - split [{split}]: ignoring non-png file {name}")
            continue
        stem = Path(name).stem
        # a.png and a.PNG would both claim labels/a.txt - only one could win, and
        # which one is filesystem dependent, so refuse rather than pick
        if stem in images_by_stem:
            duplicates.append(stem)
            continue
        images_by_stem[stem] = image_dir / name

    label_stems = {
        Path(f.name).stem
        for f in os.scandir(label_dir)
        if f.name.lower().endswith(".txt")
    }

    # existence only, never size: an empty .txt is a valid background frame
    missing = sorted(stem for stem in images_by_stem if stem not in label_stems)
    orphans = label_stems - set(images_by_stem)

    problems = []
    if duplicates:
        problems.append(
            f"{len(duplicates)} image stems appear more than once (differing only by "
            f"extension case): {', '.join(sorted(duplicates)[:_MAX_REPORTED])}"
        )
    if missing:
        problems.append(
            f"{len(missing)} images have no matching label .txt: "
            f"{', '.join(missing[:_MAX_REPORTED])}"
        )
    if problems:
        raise YoloLayoutError(f"split [{split}] is inconsistent - " + "; ".join(problems))

    if orphans:
        print(f"WARN - split [{split}]: {len(orphans)} label .txt have no image, ignored")

    # the List of tuples- pair image and label Ordered by name: 
    return [(images_by_stem[stem], label_dir / f"{stem}.txt")
            for stem in sorted(images_by_stem)]


def collect_yolo_pairs(src_base_dir, splits=SPLITS, strict_xml: bool = True) -> dict:
    """
    Validate a YOLO tree and enumerate it in one pass.

    Returns {split: [(image_path, label_path), ...]} for the splits that are
    actually present, each list sorted by image filename so callers get a
    deterministic order without rescanning.

    Raises YoloLayoutError - naming the offending path and the fix - when the
    tree is unusable: no images/ or labels/, no split at all, a split present
    on one side only, unconverted .xml annotations, or images without labels.
    A split directory that exists but is empty is warned about and skipped.

    An image's label must merely EXIST; an empty one is a valid background
    frame and passes.
    """
    root = Path(src_base_dir)
    if not root.is_dir():
        raise YoloLayoutError(f"not a directory: {root}")

    image_root = root / "images"
    label_root = root / "labels"
    if not image_root.is_dir() or not label_root.is_dir():
        raise YoloLayoutError(
            f"{root} is not a YOLO dataset root: expected images/ and labels/ "
            f"subdirectories but found [{_describe(root)}]. If this is the raw NREC "
            f"dataset, run the nrec-yolo migration over it first."
        )

    present = [split for split in splits if (image_root / split).is_dir()]
    if not present:
        raise YoloLayoutError(
            f"none of the splits {list(splits)} exist under {image_root} "
            f"(found [{_describe(image_root)}])"
        )

    pairs_by_split = {}
    for split in present:
        image_dir = image_root / split
        label_dir = label_root / split
        if not label_dir.is_dir():
            raise YoloLayoutError(
                f"split [{split}] has {image_dir} but no matching {label_dir}"
            )

        _check_labels_not_xml(label_dir, split, strict_xml)

        pairs = _pairs_for_split(image_dir, label_dir, split)
        if not pairs:
            print(f"WARN - split [{split}] has no .png images, skipping")
            continue
        pairs_by_split[split] = pairs

    if not pairs_by_split:
        raise YoloLayoutError(f"no split under {image_root} holds any .png image")

    return pairs_by_split
