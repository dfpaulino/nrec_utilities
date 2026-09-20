"""
Generate occluded copies of a YOLO dataset, to be folded back into a baseline
training set.

Given an existing clean YOLO tree (the output of the nrec-yolo migration), this
picks a controlled fraction of each split's frames, applies every requested
occlusion to them, and writes the results - images plus their labels - into a
second YOLO tree. You then copy that tree's images/ and labels/ into your
working dataset so the model trains on the clean frames plus the degraded ones.

    {src}/images/{split}/Image_141514_2014_023.png
    {src}/labels/{split}/Image_141514_2014_023.txt
        ->
    {dst}/images/{split}/motionVibration_Image_141514_2014_023.png
    {dst}/labels/{split}/motionVibration_Image_141514_2014_023.txt

Two independent dials:
  - copy_factor: HOW MANY frames of each split get occluded (0-1).
  - occ_factor:  HOW HARD the occlusion hits each of them (0-1).

`dst` is used exactly as given - no occ_<factor> level is inserted. Note that
the output filename carries the occlusion name but NOT the occlusion factor, so
two runs at different occ_factors produce the SAME filenames. Copying an
occ-0.1 set and an occ-0.3 set into one flat directory would silently overwrite
one with the other; keep one occ_factor per working dataset. Different
occlusion types are safe, since each has its own prefix.

Splits are processed independently and an occluded frame only ever lands in the
split it came from: an occluded copy of a train frame appearing in val would be
train/val leakage that inflates the metrics invisibly.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from nrec_utils.occlusions import (
    OcclusionBase,
    available_occlusions,
    create_occlusion,
    create_occlusions_instances
)
from nrec_utils.yolo_dataset import SPLITS, collect_yolo_pairs, even_stride_select


def _imread(path: Path):
    """
    Read an image as 3-channel 8-bit BGR.

    Goes through np.fromfile/cv2.imdecode rather than cv2.imread because
    imread cannot open a non-ASCII path on Windows and signals every failure by
    returning None instead of raising - a silent skip we would rather not have.

    IMREAD_COLOR keeps the result predictable (always HxWx3, alpha dropped).
    IMREAD_UNCHANGED would preserve grayscale and 16-bit sources, but then
    every occlusion would have to cope with 2-D arrays too.
    """
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise OSError(f"empty or unreadable image file: {path}")
    # BGR is kept end to end: imdecode returns BGR and imencode expects BGR, so
    # converting to RGB here would write every output with red and blue swapped
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not decode image: {path}")
    return image


def _imwrite(path: Path, image) -> None:
    """Write a BGR image as PNG. cv2.imwrite returns False rather than raising."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"could not encode image: {path}")
    buffer.tofile(str(path))


def occlude_pair(src_img: Path, src_label: Path, dst_img: Path, dst_label: Path,
                 occlusion: OcclusionBase) -> None:
    """
    Occlude one image and place its label beside the result.

    The label is copied byte for byte. That is only correct for a pixel-only
    occlusion, so the output is checked to have kept the input's height and
    width - see OcclusionBase for what to do when that stops holding.

    An empty label file is copied like any other: it is what marks a negative
    frame as explicit YOLO background, and dropping it would leave the occluded
    image unlabelled instead.
    """
    image = _imread(src_img)
    occluded = occlusion.generate_occluded_image(image)

    if not isinstance(occluded, np.ndarray):
        raise ValueError(
            f"{occlusion.get_occ_name()} returned {type(occluded).__name__}, "
            f"expected a numpy array"
        )
    if occluded.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"{occlusion.get_occ_name()} changed the image geometry "
            f"({image.shape[:2]} -> {occluded.shape[:2]}); its labels can no longer "
            f"be copied verbatim"
        )

    _imwrite(dst_img, occluded)
    # copyfile, not read/split/join, so empty files and line endings survive intact
    shutil.copyfile(src_label, dst_label)


def occlude_split(pairs: list, dst_root: Path, split: str, occlusion: OcclusionBase,
                  copy_factor: float, dry_run: bool = False,
                  strict: bool = False) -> dict:
    """
    Apply one occlusion to a fraction of one split, writing into dst_root.

    Selection runs on the (image, label) pairs rather than on the images with a
    label lookup afterwards, so the two can never diverge. It is confined to
    this one split, which is what keeps the occluded copies on the correct side
    of the train/val/test boundary.
    """
    prefix = occlusion.get_occ_name()
    selected = even_stride_select(pairs, copy_factor)

    img_dst_dir = dst_root / "images" / split
    label_dst_dir = dst_root / "labels" / split
    if not dry_run:
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        label_dst_dir.mkdir(parents=True, exist_ok=True)

    counters = {'images_written': 0, 'images_failed': 0}

    for src_img, src_label in selected:
        stem = src_img.stem
        dst_img = img_dst_dir / f"{prefix}_{stem}.png"
        dst_label = label_dst_dir / f"{prefix}_{stem}.txt"

        if dry_run:
            counters['images_written'] += 1
            continue

        try:
            occlude_pair(src_img, src_label, dst_img, dst_label, occlusion)
        except (OSError, ValueError) as e:
            if strict:
                raise
            print(f'WARN - could not occlude {src_img} ({e}), skipping')
            # never leave a half-written pair behind: an image with no label
            # trains as background, a label with no image confuses the loader
            dst_img.unlink(missing_ok=True)
            dst_label.unlink(missing_ok=True)
            counters['images_failed'] += 1
            continue

        counters['images_written'] += 1

    print(f"INFO - split [{split}] / {prefix}: selected {len(selected)} of "
          f"{len(pairs)}, wrote {counters['images_written']}, "
          f"failed {counters['images_failed']}"
          + (" (dry run, nothing written)" if dry_run else ""))
    return counters


def create_occ_set(src_base_dir: str, dst_base_dir: str, copy_factor: float,
                   occ_factor: float, occlusion_names=None, splits=SPLITS,
                   strict_xml: bool = True, dry_run: bool = False,
                   strict: bool = False) -> dict:
    """
    Build a YOLO tree of occluded images under `dst_base_dir` from the clean
    YOLO tree at `src_base_dir`.

    PARAMETERS
    - src_base_dir: root of a YOLO tree, i.e. what nrec-yolo produced. Its
      labels must already be .txt; the layout is validated up front so a bad
      path fails before anything is written.
    - dst_base_dir: root of the output tree, used exactly as given. Must not be
      inside src_base_dir.
    - copy_factor: fraction (0-1] of each split's frames to occlude.
    - occ_factor: occlusion strength [0-1] handed to each occlusion. 0 is legal
      and copies the images unchanged.
    - occlusion_names: which occlusions to apply; every registered one by
      default. They all run over the same selected frames, so their outputs are
      directly comparable.
    - splits: which of train/val/test to consider. Only those that exist in the
      source are processed, each in isolation.
    - strict_xml: fail when the source labels still hold unconverted .xml.
    - dry_run: report what would be written without creating anything.
    - strict: abort on the first unreadable image instead of warning and
      carrying on.

    RETURNS
    A dict of counters: images_written, images_failed.
    """
    if not 0 < copy_factor <= 1:
        raise ValueError(f"copy_factor must be in (0, 1], got {copy_factor}")
    if not 0 <= occ_factor <= 1:
        raise ValueError(f"occ_factor must be in [0, 1], got {occ_factor}")
    if occ_factor == 0:
        print("WARN - occ_factor is 0, the output will be an unmodified copy")

    if occlusion_names is None:
        occlusion_names = available_occlusions()
    if not occlusion_names:
        raise ValueError("no occlusions requested")

    src_root = Path(src_base_dir).resolve()
    dst_root = Path(dst_base_dir).resolve()
    # writing occluded images into the source would corrupt the baseline set and
    # make a second run occlude its own output
    if dst_root == src_root or src_root in dst_root.parents:
        raise ValueError(
            f"dest {dst_root} is inside source {src_root}; occluded images must go "
            f"to a separate tree"
        )

    pairs_by_split = collect_yolo_pairs(src_root, splits, strict_xml=strict_xml)
    #occlusions = [create_occlusion(name, occ_factor) for name in occlusion_names]
    occlusions = create_occlusions_instances(occ_factor)
    totals = {'images_written': 0, 'images_failed': 0}

    for occlusion in occlusions: 
        for split, pairs in pairs_by_split.items():
            print(f'INFO - Running occlusion type {occlusion.get_occ_name()} in split {split}')
            counters = occlude_split(pairs, dst_root, split, occlusion,
                                     copy_factor, dry_run=dry_run, strict=strict)
            for key in totals:
                totals[key] += counters[key]

    print(f'Total {totals}')
    return totals


def main(source: str, dest: str, copy_factor: float, occ_factor: float,
         occlusions=None, splits=SPLITS, allow_stray_xml: bool = False,
         dry_run: bool = False, strict: bool = False) -> dict:
    return create_occ_set(src_base_dir=source,
                          dst_base_dir=dest,
                          copy_factor=copy_factor,
                          occ_factor=occ_factor,
                          occlusion_names=occlusions,
                          splits=splits,
                          strict_xml=not allow_stray_xml,
                          dry_run=dry_run,
                          strict=strict)


def cli():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--source", required=True,
                        help="src directory of base images in YOLO layout")
    parser.add_argument("-d", "--dest", required=True,
                        help="dest directory, used as given (no occ_<factor> is added)")
    parser.add_argument("-p", "--copy-factor", required=True, type=float,
                        help="fraction of each split to occlude, 0 to 1 where 1 is 100%%")
    parser.add_argument("-f", "--occ-factor", required=True, type=float,
                        help="occlusion strength 0 to 1, eg 0.1 0.2 0.3")
    parser.add_argument("-o", "--occlusion", nargs="+", default=available_occlusions(),
                        choices=available_occlusions(),
                        help="which occlusions to apply (default: all)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS,
                        help="which splits to process (default: all)")
    parser.add_argument("--allow-stray-xml", action="store_true",
                        help="continue when the source labels still hold unconverted .xml")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written without creating anything")
    parser.add_argument("--strict", action="store_true",
                        help="abort on the first unreadable image instead of skipping it")
    args = parser.parse_args()

    # report a bad factor as usage text rather than a traceback
    if not 0 < args.copy_factor <= 1:
        parser.error(f"--copy-factor must be in (0, 1], got {args.copy_factor}")
    if not 0 <= args.occ_factor <= 1:
        parser.error(f"--occ-factor must be in [0, 1], got {args.occ_factor}")

    main(source=args.source,
         dest=args.dest,
         copy_factor=args.copy_factor,
         occ_factor=args.occ_factor,
         occlusions=args.occlusion,
         splits=args.splits,
         allow_stray_xml=args.allow_stray_xml,
         dry_run=args.dry_run,
         strict=args.strict)


if __name__ == "__main__":
    cli()
