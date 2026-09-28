import pytest

from nrec_utils.yolo_dataset import (
    SPLITS,
    YoloLayoutError,
    collect_yolo_pairs,
    even_stride_select,
)


# ---------------------------------------------------------------- even_stride_select

def test_selects_evenly_spread_indices():
    assert even_stride_select(list(range(10)), 0.3) == [0, 3, 7]


def test_full_factor_returns_everything():
    items = list(range(10))

    assert even_stride_select(items, 1.0) == items


def test_tiny_factor_still_returns_one_item():
    # max(1, ...) floor: a small fraction must not drop a short sequence entirely
    assert even_stride_select(list(range(10)), 0.01) == [0]


def test_selection_is_strictly_increasing_and_distinct():
    result = even_stride_select(list(range(100)), 0.5)

    assert len(result) == 50
    assert result == sorted(result)
    assert len(set(result)) == 50


@pytest.mark.parametrize("total", [1, 2, 3, 7, 10, 97, 1000])
@pytest.mark.parametrize("factor", [0.01, 0.1, 0.3, 0.5, 0.99, 1.0])
def test_matches_the_expression_it_replaced_in_nrec_select_images(total, factor):
    # the occluded subset must be drawn from the same frames the original
    # sampler would have picked, so this is the guard on that refactor
    items = list(range(total))
    count = max(1, round(total * factor))
    expected = [items[round(i * total / count)] for i in range(count)]

    assert even_stride_select(items, factor) == expected


def test_empty_input_returns_empty_rather_than_raising():
    # the expression it replaced computed max(1, 0) == 1 and then hit IndexError
    assert even_stride_select([], 0.5) == []


@pytest.mark.parametrize("factor", [0, -0.1, 1.1, 2])
def test_out_of_range_factor_is_rejected(factor):
    with pytest.raises(ValueError):
        even_stride_select(list(range(10)), factor)


def test_selection_is_deterministic():
    items = list(range(57))

    assert even_stride_select(items, 0.37) == even_stride_select(items, 0.37)


# ---------------------------------------------------------------- collect_yolo_pairs

def build_tree(root, splits=SPLITS, count=4, label_suffix=".txt"):
    """Minimal YOLO tree: `count` empty .png per split, each with a label."""
    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for i in range(count):
            (image_dir / f"frame_{i:03d}.png").write_bytes(b"")
            (label_dir / f"frame_{i:03d}{label_suffix}").write_text("0 0.5 0.5 0.1 0.1\n")
    return root


def test_returns_sorted_pairs_per_split(tmp_path):
    build_tree(tmp_path)

    pairs_by_split = collect_yolo_pairs(tmp_path)

    assert set(pairs_by_split) == set(SPLITS)
    for split, pairs in pairs_by_split.items():
        assert [p[0].name for p in pairs] == sorted(p[0].name for p in pairs)
        for image_path, label_path in pairs:
            assert image_path.stem == label_path.stem


def test_only_the_splits_that_exist_are_returned(tmp_path):
    build_tree(tmp_path, splits=("train",))

    assert set(collect_yolo_pairs(tmp_path)) == {"train"}


def test_missing_images_dir_is_reported(tmp_path):
    (tmp_path / "labels" / "train").mkdir(parents=True)

    with pytest.raises(YoloLayoutError, match="images"):
        collect_yolo_pairs(tmp_path)


def test_missing_label_split_dir_is_reported(tmp_path):
    build_tree(tmp_path, splits=("train",))
    # a split present under images/ but with no labels/ counterpart at all
    (tmp_path / "images" / "val").mkdir()
    (tmp_path / "images" / "val" / "frame_000.png").write_bytes(b"")

    with pytest.raises(YoloLayoutError, match="val"):
        collect_yolo_pairs(tmp_path)


def test_unconverted_xml_labels_name_the_fix(tmp_path):
    build_tree(tmp_path, splits=("train",), label_suffix=".xml")

    with pytest.raises(YoloLayoutError, match="nrec_xml_to_yolo"):
        collect_yolo_pairs(tmp_path)


def test_stray_xml_alongside_txt_is_a_different_error(tmp_path):
    build_tree(tmp_path, splits=("train",))
    (tmp_path / "labels" / "train" / "leftover.xml").write_text("<annotation/>")

    with pytest.raises(YoloLayoutError, match="failed conversion"):
        collect_yolo_pairs(tmp_path)


def test_stray_xml_can_be_downgraded_to_a_warning(tmp_path):
    build_tree(tmp_path, splits=("train",))
    (tmp_path / "labels" / "train" / "leftover.xml").write_text("<annotation/>")

    pairs_by_split = collect_yolo_pairs(tmp_path, strict_xml=False)

    assert len(pairs_by_split["train"]) == 4


def test_image_without_a_label_names_the_stem(tmp_path):
    build_tree(tmp_path, splits=("train",))
    (tmp_path / "images" / "train" / "unlabelled.png").write_bytes(b"")

    with pytest.raises(YoloLayoutError, match="unlabelled"):
        collect_yolo_pairs(tmp_path)


def test_empty_label_file_is_accepted(tmp_path):
    # an empty .txt is a valid background frame, not a broken label
    build_tree(tmp_path, splits=("train",))
    (tmp_path / "images" / "train" / "background.png").write_bytes(b"")
    (tmp_path / "labels" / "train" / "background.txt").write_bytes(b"")

    pairs_by_split = collect_yolo_pairs(tmp_path)

    assert len(pairs_by_split["train"]) == 5


def test_orphan_label_is_tolerated(tmp_path):
    build_tree(tmp_path, splits=("train",))
    (tmp_path / "labels" / "train" / "ghost.txt").write_text("")

    assert len(collect_yolo_pairs(tmp_path)["train"]) == 4


def test_split_with_no_images_is_skipped(tmp_path):
    build_tree(tmp_path, splits=("train",))
    (tmp_path / "images" / "val").mkdir()
    (tmp_path / "labels" / "val").mkdir()

    assert set(collect_yolo_pairs(tmp_path)) == {"train"}
