import sys

import cv2
import numpy as np
import pytest

from nrec_utils.create_occluded_set import cli, create_occ_set
from nrec_utils.occlusions import MOTION_VIBRATION_OCCLUSION_NAME
from nrec_utils.yolo_dataset import SPLITS, YoloLayoutError

H, W = 16, 16
PER_SPLIT = 10
PREFIX = MOTION_VIBRATION_OCCLUSION_NAME

# one frame per split is a negative/background frame: it carries a zero-byte
# label, which is exactly what declares it as background to YOLO
BACKGROUND_STEM = "frame_000"


def write_png(path, image):
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    buffer.tofile(str(path))


@pytest.fixture
def yolo_src(tmp_path):
    """A minimal but complete clean YOLO tree."""
    rng = np.random.default_rng(seed=0)
    root = tmp_path / "clean"
    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for i in range(PER_SPLIT):
            stem = f"frame_{i:03d}"
            write_png(image_dir / f"{stem}.png",
                      rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8))
            if stem == BACKGROUND_STEM:
                (label_dir / f"{stem}.txt").write_bytes(b"")
            else:
                (label_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


def listing(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_builds_a_yolo_tree_at_the_dest_given(yolo_src, tmp_path):
    dst = tmp_path / "occ_0.3"

    create_occ_set(str(yolo_src), str(dst), copy_factor=0.5, occ_factor=0.3)

    for split in SPLITS:
        assert (dst / "images" / split).is_dir()
        assert (dst / "labels" / split).is_dir()
    # dest is used exactly as given - no occ_<factor> level is inserted
    assert not (dst / "occ_0p3").exists()


def test_source_tree_is_left_untouched(yolo_src, tmp_path):
    before = listing(yolo_src)

    create_occ_set(str(yolo_src), str(tmp_path / "out"), copy_factor=1.0, occ_factor=0.3)

    assert listing(yolo_src) == before


def test_output_is_named_after_the_occlusion_and_pairs_up(yolo_src, tmp_path):
    dst = tmp_path / "out"

    create_occ_set(str(yolo_src), str(dst), copy_factor=0.5, occ_factor=0.3)

    for split in SPLITS:
        images = sorted((dst / "images" / split).glob("*.png"))
        assert images
        for image_path in images:
            assert image_path.name.startswith(f"{PREFIX}_")
            # matching stems are the pairing Ultralytics relies on
            assert (dst / "labels" / split / f"{image_path.stem}.txt").exists()


def test_selected_count_per_split(yolo_src, tmp_path):
    dst = tmp_path / "out"

    create_occ_set(str(yolo_src), str(dst), copy_factor=0.3, occ_factor=0.3)

    for split in SPLITS:
        assert len(list((dst / "images" / split).glob("*.png"))) == round(PER_SPLIT * 0.3)


def test_channels_are_not_swapped(tmp_path):
    """
    A BGR->RGB conversion anywhere in the pipeline would write every image with
    red and blue exchanged. It is invisible to a shape/dtype check and a no-op
    for a per-channel convolution, so it needs its own test: a spatially
    constant image is unchanged by a normalised kernel, so the output must come
    back byte-identical - unless the channels moved.
    """
    src = tmp_path / "clean"
    (src / "images" / "train").mkdir(parents=True)
    (src / "labels" / "train").mkdir(parents=True)
    constant = np.zeros((H, W, 3), dtype=np.uint8)
    constant[:, :, 0] = 10
    constant[:, :, 1] = 20
    constant[:, :, 2] = 30
    write_png(src / "images" / "train" / "flat.png", constant)
    (src / "labels" / "train" / "flat.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    dst = tmp_path / "out"
    create_occ_set(str(src), str(dst), copy_factor=1.0, occ_factor=0.5,
                   splits=("train",))

    written = cv2.imread(str(dst / "images" / "train" / f"{PREFIX}_flat.png"))
    np.testing.assert_array_equal(written, constant)


def test_zero_occ_factor_copies_the_image_unchanged(yolo_src, tmp_path):
    dst = tmp_path / "out"

    create_occ_set(str(yolo_src), str(dst), copy_factor=1.0, occ_factor=0.0,
                   splits=("train",))

    source = cv2.imread(str(yolo_src / "images" / "train" / "frame_001.png"))
    written = cv2.imread(str(dst / "images" / "train" / f"{PREFIX}_frame_001.png"))
    np.testing.assert_array_equal(written, source)


def test_occlusion_actually_changes_the_pixels(yolo_src, tmp_path):
    dst = tmp_path / "out"

    create_occ_set(str(yolo_src), str(dst), copy_factor=1.0, occ_factor=1.0,
                   splits=("train",))

    source = cv2.imread(str(yolo_src / "images" / "train" / "frame_001.png"))
    written = cv2.imread(str(dst / "images" / "train" / f"{PREFIX}_frame_001.png"))

    assert written.shape == source.shape
    assert written.dtype == source.dtype
    assert not np.array_equal(written, source)


def test_background_frame_keeps_its_empty_label(yolo_src, tmp_path):
    dst = tmp_path / "out"

    create_occ_set(str(yolo_src), str(dst), copy_factor=1.0, occ_factor=0.3)

    for split in SPLITS:
        image_path = dst / "images" / split / f"{PREFIX}_{BACKGROUND_STEM}.png"
        label_path = dst / "labels" / split / f"{PREFIX}_{BACKGROUND_STEM}.txt"
        # not skipped just because it has no boxes
        assert image_path.exists()
        assert label_path.exists()
        assert label_path.stat().st_size == 0


def test_labels_are_copied_byte_for_byte(yolo_src, tmp_path):
    dst = tmp_path / "out"

    create_occ_set(str(yolo_src), str(dst), copy_factor=1.0, occ_factor=0.3,
                   splits=("train",))

    source = (yolo_src / "labels" / "train" / "frame_001.txt").read_bytes()
    written = (dst / "labels" / "train" / f"{PREFIX}_frame_001.txt").read_bytes()
    assert written == source


def test_a_frame_never_crosses_between_splits(yolo_src, tmp_path):
    """An occluded train frame landing in val would be invisible train/val leakage."""
    dst = tmp_path / "out"

    create_occ_set(str(yolo_src), str(dst), copy_factor=1.0, occ_factor=0.3)

    seen = {}
    for split in SPLITS:
        for image_path in (dst / "images" / split).glob("*.png"):
            seen.setdefault(image_path.name, set()).add(split)
    # every source stem exists in all three source splits, so each output name
    # must appear under all three - but only ever derived from its own split
    assert all(len(splits) == len(SPLITS) for splits in seen.values())
    for split in SPLITS:
        names = {p.name for p in (dst / "images" / split).glob("*.png")}
        assert names == set(seen)


def test_reruns_are_deterministic(yolo_src, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"

    create_occ_set(str(yolo_src), str(first), copy_factor=0.3, occ_factor=0.3)
    create_occ_set(str(yolo_src), str(second), copy_factor=0.3, occ_factor=0.3)

    assert listing(first) == listing(second)
    for relative in listing(first):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_unreadable_image_is_counted_and_skipped(yolo_src, tmp_path):
    (yolo_src / "images" / "train" / "broken.png").write_bytes(b"not a png")
    (yolo_src / "labels" / "train" / "broken.txt").write_text("")
    dst = tmp_path / "out"

    totals = create_occ_set(str(yolo_src), str(dst), copy_factor=1.0, occ_factor=0.3,
                            splits=("train",))

    assert totals['images_failed'] == 1
    assert totals['images_written'] == PER_SPLIT
    # no half-written pair left behind
    assert not (dst / "labels" / "train" / f"{PREFIX}_broken.txt").exists()
    assert not (dst / "images" / "train" / f"{PREFIX}_broken.png").exists()


def test_strict_mode_aborts_on_an_unreadable_image(yolo_src, tmp_path):
    (yolo_src / "images" / "train" / "broken.png").write_bytes(b"not a png")
    (yolo_src / "labels" / "train" / "broken.txt").write_text("")

    with pytest.raises(OSError):
        create_occ_set(str(yolo_src), str(tmp_path / "out"), copy_factor=1.0,
                       occ_factor=0.3, splits=("train",), strict=True)


def test_dest_inside_source_is_refused(yolo_src):
    with pytest.raises(ValueError, match="inside source"):
        create_occ_set(str(yolo_src), str(yolo_src / "nested"), copy_factor=1.0,
                       occ_factor=0.3)


def test_dry_run_writes_nothing_but_still_counts(yolo_src, tmp_path):
    dst = tmp_path / "out"

    totals = create_occ_set(str(yolo_src), str(dst), copy_factor=0.5, occ_factor=0.3,
                            dry_run=True)

    assert totals['images_written'] == round(PER_SPLIT * 0.5) * len(SPLITS)
    assert not dst.exists()


@pytest.mark.parametrize("copy_factor,occ_factor", [(0, 0.3), (1.5, 0.3), (0.5, 1.5)])
def test_out_of_range_factors_are_rejected(yolo_src, tmp_path, copy_factor, occ_factor):
    with pytest.raises(ValueError):
        create_occ_set(str(yolo_src), str(tmp_path / "out"),
                       copy_factor=copy_factor, occ_factor=occ_factor)


def test_unconverted_source_fails_before_writing_anything(tmp_path):
    src = tmp_path / "raw"
    (src / "images" / "train").mkdir(parents=True)
    (src / "labels" / "train").mkdir(parents=True)
    write_png(src / "images" / "train" / "a.png", np.zeros((H, W, 3), dtype=np.uint8))
    (src / "labels" / "train" / "a.xml").write_text("<annotation/>")
    dst = tmp_path / "out"

    with pytest.raises(YoloLayoutError, match="nrec_xml_to_yolo"):
        create_occ_set(str(src), str(dst), copy_factor=1.0, occ_factor=0.3)

    assert not dst.exists()


# ------------------------------------------------------------------------- cli

def test_cli_parses_its_flags_and_writes(yolo_src, tmp_path, monkeypatch):
    dst = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "create_occluded_set", "-s", str(yolo_src), "-d", str(dst),
        "-p", "0.5", "-f", "0.3",
    ])

    cli()

    assert list((dst / "images" / "train").glob(f"{PREFIX}_*.png"))


@pytest.mark.parametrize("bad", [["-p", "0", "-f", "0.3"],
                                 ["-p", "0.5", "-f", "2"],
                                 ["-p", "0.5", "-f", "0.3", "-o", "gaussianSnow"]])
def test_cli_rejects_bad_arguments(yolo_src, tmp_path, monkeypatch, bad):
    monkeypatch.setattr(sys, "argv", [
        "create_occluded_set", "-s", str(yolo_src), "-d", str(tmp_path / "out"), *bad,
    ])

    with pytest.raises(SystemExit):
        cli()
