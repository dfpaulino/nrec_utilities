# nrec_utilities

Utilities to manage the NREC apple images dataset. The main use case is
sampling a percentage of the labeled images (positive and negative
observations, per split) out of the NREC "labeled" dataset layout and
aggregating the selected images (and their annotations) into a flat
dataset layout suitable for training (e.g. an object-detection pipeline).

## Scripts

- `src/nrec_utils/nrec_utils.py` — `nrec_select_images(...)`: samples a
  percentage of images from a scenario-based source layout and copies them
  (plus matching XML annotations, for positive images) into an aggregated
  destination layout.
- `src/nrec_utils/count_images.py` — counts image files per scenario/trip
  directory, useful for sanity-checking the source dataset before sampling.
- `src/nrec_utils/main.py` — example entry point calling
  `nrec_select_images`.

## Source directory structure

The source dataset must already exist (read-only) with the following
layout, per split (`type_dir`, e.g. `train`, `test`, `val`) and class
(`positive`/`negative`):

```
{src_base_dir}/{type_dir}/positive/2015.11Soergels/<scenario>/Images/*.png
{src_base_dir}/{type_dir}/positive/2015.11Soergels/<scenario>/Annotations/*.xml
{src_base_dir}/{type_dir}/negative/2015.11Soergels/<scenario>/Images/*.png
{src_base_dir}/{type_dir}/negative/2015.11Soergels/<scenario>/Annotations/*.xml
```

There can be any number of `<scenario>` folders (e.g. a distinct
field/row/lighting condition). Each image file must have a same-named
`.xml` annotation file for it to be copied.

## Destination directory structure

The destination folders must be created **before** calling
`nrec_select_images` (it does not create missing directories):

```
{dst_base_dir}/images/{type_dir}/
{dst_base_dir}/labels/{type_dir}/
```

### Bash commands to create the destination structure

```bash
DST_BASE_DIR=/content/aggregated

for split in train val test; do
  mkdir -p "$DST_BASE_DIR/images/$split"
  mkdir -p "$DST_BASE_DIR/labels/$split"
done
```

## Usage

### Selecting images (`nrec_select_images`)

Call it once per split (`train`/`test`/`val`) and once per class
(positive/negative) you want included in the aggregated dataset:

```python
import nrec_utils

# 100% of positive training images (image + annotation)
nrec_utils.nrec_select_images(
    src_base_dir="/content/apples_left_labeled",
    dst_base_dir="/content/aggregated",
    type_dir="train",
    percentage_to_copy=1.0,
    positive=True,
)

# 20% of negative training images (image only, no annotation)
nrec_utils.nrec_select_images(
    src_base_dir="/content/apples_left_labeled",
    dst_base_dir="/content/aggregated",
    type_dir="train",
    percentage_to_copy=0.2,
    positive=False,
)

# repeat with type_dir="val" and/or "test" as needed
```

Parameters:

- `src_base_dir` — root of the labeled source dataset (see layout above).
- `dst_base_dir` — root of the aggregated output dataset (see layout above).
- `type_dir` — the dataset split to read/write, e.g. `train`, `val`, `test`.
  Must match a real subfolder name on the source side.
- `percentage_to_copy` — fraction (0.0-1.0) of each scenario's images to
  keep, sampled by taking every Nth file (sorted by filename). Must be
  > 0.
- `positive` — `True` to pull from the `positive` branch (copies image +
  annotation); `False` to pull from the `negative` branch (copies image
  only, since there's nothing to annotate).

### Counting images

```bash
python src/nrec_utils/count_images.py /content/apples_left_labeled/train/positive/2015.11Soergels
```

Options: `--ext` (default `png`) and `--subdir` (default `Images`).

## Requirements

- Python >= 3.13
- No third-party dependencies (standard library only)
