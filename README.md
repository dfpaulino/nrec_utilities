# nrec_utilities

Utilities to manage the NREC dataset. The main use case is converting the
NREC "labeled" dataset into a dataset that can be used to train a YOLO
(Ultralytics) object-detection model:

1. **Restructure** — sample a fraction of the labeled images (positive and
   negative, per split) out of the NREC scenario-based directory layout and
   copy them into a flat YOLO `images/` + `labels/` layout.
2. **Convert annotations** — rewrite the Pascal VOC `.xml` annotations as YOLO
   `.txt` label files.

Both steps are run in one go by the `migrate_nrec_to_yolo` script.

## Scripts

- `src/nrec_utils/migrate_nrec_to_yolo.py` — command-line entry point that
  migrates the NREC layout into a YOLO layout and converts the annotations
  (see [Migrating NREC to YOLO](#migrating-nrec-to-yolo)).
- `src/nrec_utils/nrec_utils.py` — the library functions used by the script:
  - `nrec_select_images(...)` — samples images from one split/class and copies
    them into the YOLO layout.
  - `nrec_xml_to_yolo(...)` — converts the copied VOC `.xml` annotations into
    YOLO `.txt` labels, in place.
- `src/nrec_utils/create_occluded_set.py` — command-line entry point that
  generates occluded copies of an existing YOLO dataset, to be folded back into
  a training set (see [Generating an occluded set](#generating-an-occluded-set)).
- `src/nrec_utils/yolo_dataset.py` — generic YOLO-tree helpers shared by the two
  scripts: `even_stride_select(...)` (the even sampling both use) and
  `collect_yolo_pairs(...)` (validates a YOLO tree and enumerates its
  image/label pairs).
- `src/nrec_utils/occlusions/` — the occlusion implementations behind a common
  `OcclusionBase` interface, plus the registry the CLI resolves names against.
- `src/nrec_utils/count_images.py` — counts image files per scenario/trip
  directory, useful for sanity-checking the source dataset before sampling.

## Migrating NREC to YOLO

### What the script does

`migrate_nrec_to_yolo` runs two stages:

**Stage 1 — build the YOLO tree.** For every split (`train`, `val`, `test`)
and for both classes of image (`positive` and `negative`) it calls
`nrec_select_images`, which:

- treats every folder under `.../2015.11Soergels/` as a separate scenario;
- sorts each scenario's `.png` images by filename (chronological, since the
  name embeds the capture timestamp) and keeps `round(n * copy-factor)` of
  them, evenly spaced across the scenario (at least one per scenario).
  Frames are ~133 ms apart, so neighbouring frames are near-duplicates and
  keeping all of them adds little value;
- copies each selected image to `images/{split}/`;
- for **positive** images, copies the matching `.xml` annotation to
  `labels/{split}/` (images without an annotation are skipped with an error
  message);
- for **negative** images, writes an empty `labels/{split}/{name}.txt`, which
  marks the image as an explicit YOLO background frame.

The destination directories are created automatically if they don't exist.

**Stage 2 — convert annotations.** `nrec_xml_to_yolo` goes over
`labels/{train,val,test}/` and, for every `.xml`:

- reads the image size from `<size>` and converts each `<object>` into one
  YOLO line: `<class_id> <centre_x> <centre_y> <width> <height>`, all
  normalised to 0-1. The NREC boxes use the Pascal VOC 1-based pixel-centre
  convention, so 0.5 px is subtracted from each coordinate before converting;
- maps class names with `{"person": 0, "person-part": 1}`; boxes with any
  other class name, or with an invalid box, are skipped with a warning;
- writes `labels/{split}/{name}.txt` (same file name as the image);
- moves the original `.xml` to `labels_xml/{split}/` so `labels/` only holds
  `.txt` files while the original annotations are kept. An `.xml` that fails
  to parse is left in `labels/` so the failure is visible and can be retried.

### Arguments

| Argument | Short | Required | Description |
| --- | --- | --- | --- |
| `--source` | `-s` | yes | Root of the NREC labeled dataset (the directory containing `train/`, `val/`, `test/`). See [Source directory structure](#source-directory-structure). |
| `--dest` | `-d` | yes | Root of the YOLO dataset to create. See [Destination directory structure](#destination-directory-structure). |
| `--copy-factor` | `-p` | yes | Fraction of each scenario's images to keep, a float in `(0, 1]`. `1` copies 100% of the images, `0.5` copies half of them. The same factor is applied to positive and negative images and to every split. |

### How to run it

The script is run as a module from the `src` directory (it imports the
`nrec_utils` package, so it must be run with `python -m`, not by file path).
The migration script itself only uses the Python standard library, so it runs
on any Python >= 3.13. The package as a whole depends on numpy and
opencv-python (used by the occlusion classes); install them with
`pip install -e .` from the project root, or
`pip install numpy opencv-python`.

1. Clone the repository:

   ```bash
   git clone <repository-url> nrec_utilities
   ```

2. Change into the `src` directory:

   ```bash
   cd nrec_utilities/src
   ```

3. Run the module with its arguments:

   ```bash
   python -m nrec_utils.migrate_nrec_to_yolo --source /path/to/nrec/labeled --dest /path/to/yolo_dataset --copy-factor 0.5
   ```

   or with the short options:

   ```bash
   python -m nrec_utils.migrate_nrec_to_yolo -s /path/to/nrec/labeled -d /path/to/yolo_dataset -p 0.5
   ```

On Windows (PowerShell), for example with a conda environment:

```powershell
cd D:\github\nrec_utilities\src
D:\anaconda\envs\project\python.exe -m nrec_utils.migrate_nrec_to_yolo --source 'caminho/para/dados' --dest 'destino/para/dados' --copy-factor 0.5
```

#### Running / debugging from VS Code

When launched from VS Code, set the working directory to `src` and run the
`nrec_utils.migrate_nrec_to_yolo` module. VS Code then runs a command like:

```powershell
(project) PS D:\github\nrec_utilities\src> D:; cd 'D:\github\nrec_utilities/src'; & 'D:\anaconda\envs\project\python.exe' 'c:\Users\<user>\.vscode\extensions\ms-python.debugpy-2026.6.0-win32-x64\bundled\libs\debugpy\launcher' '52843' '--' '-m' 'nrec_utils.migrate_nrec_to_yolo' '--source' 'caminho/para/dados' '--dest' 'destino/para/dados' '--copy-factor' '0.5'
```

An equivalent `.vscode/launch.json` configuration:

```json
{
  "name": "migrate_nrec_to_yolo",
  "type": "debugpy",
  "request": "launch",
  "module": "nrec_utils.migrate_nrec_to_yolo",
  "cwd": "${workspaceFolder}/src",
  "args": ["--source", "caminho/para/dados", "--dest", "destino/para/dados", "--copy-factor", "0.5"]
}
```

## Generating an occluded set

`create_occluded_set` takes a **clean YOLO dataset** — the output of the
migration above — and produces a second YOLO tree holding degraded copies of a
fraction of its frames. You then copy that tree's `images/` and `labels/` into
your working dataset, so the model trains on the clean frames *plus* the
degraded ones.

**Prerequisite:** the source labels must already be `.txt`. Pointing this at a
tree whose annotations are still `.xml` fails immediately with a message telling
you to run the migration first — it does not silently produce an unlabelled set.

### What the script does

1. Validates that the source really is a YOLO tree (`images/{split}` +
   `labels/{split}`, every image paired with a label) and fails before writing
   anything if it is not.
2. For each split it finds, selects `round(n * copy-factor)` frames, evenly
   spaced across the sorted frame list — the same sampling the migration uses.
3. Applies every requested occlusion to each selected frame, writing the result
   to `{dest}/images/{split}/` and copying the frame's label verbatim to
   `{dest}/labels/{split}/`, both renamed with the occlusion's name as a prefix:

   ```
   {source}/images/train/Image_141514_2014_023.png
       -> {dest}/images/train/motionVibration_Image_141514_2014_023.png
   {source}/labels/train/Image_141514_2014_023.txt
       -> {dest}/labels/train/motionVibration_Image_141514_2014_023.txt
   ```

Splits are processed **independently**: an occluded frame only ever lands in the
split it came from, so occluded copies never cross the train/val/test boundary.
Selection is deterministic — no randomness — so a re-run with the same arguments
reproduces the same files.

Labels are copied unchanged, which is correct because the occlusions available
today only alter pixel values and move no geometry. The script verifies this at
runtime and refuses to write if an occlusion changes the image dimensions.

Background frames (the negatives, which carry an empty `.txt`) are occluded like
any other frame and keep their empty label — that empty file is what marks them
as background.

### Two factors, two jobs

| Factor | Controls |
| --- | --- |
| `--copy-factor` | **How many** frames of each split get occluded. `0.1` occludes every tenth frame. |
| `--occ-factor` | **How hard** the occlusion hits each of them. For `motionVibration` it scales the blur kernel; `0` is an identity copy. |

### Arguments

| Argument | Short | Required | Description |
| --- | --- | --- | --- |
| `--source` | `-s` | yes | Root of an existing **YOLO** dataset (the `--dest` of the migration). |
| `--dest` | `-d` | yes | Root of the occluded dataset to create, used **exactly as given** — no `occ_<factor>` level is added. Must not be inside `--source`. |
| `--copy-factor` | `-p` | yes | Fraction of each split's frames to occlude, a float in `(0, 1]`. |
| `--occ-factor` | `-f` | yes | Occlusion strength, a float in `[0, 1]`. |
| `--occlusion` | `-o` | no | Which occlusions to apply (default: all registered). All of them run over the *same* selected frames, so their outputs are directly comparable. |
| `--splits` | | no | Which of `train`/`val`/`test` to process (default: all that exist). |
| `--allow-stray-xml` | | no | Continue when the source labels still hold `.xml` that failed conversion. |
| `--dry-run` | | no | Report what would be written without creating anything. |
| `--strict` | | no | Abort on the first unreadable image instead of warning and skipping it. |

### ⚠️ Filenames do not encode the occlusion factor

The output name carries the occlusion name but **not** `--occ-factor`, so two
runs at different factors produce *identical* filenames. Copying an occ-0.1 set
and an occ-0.3 set into the same flat directory will silently overwrite one with
the other. Keep one `--occ-factor` per working dataset, and name the destination
directory after it (`occ_0.3/`) so you can tell them apart. Different occlusion
*types* are safe — each has its own prefix.

### How to run it

```bash
cd nrec_utilities/src
python -m nrec_utils.create_occluded_set --source /path/to/yolo_dataset --dest /path/to/occ_0.3 --copy-factor 0.1 --occ-factor 0.3
```

or with the short options:

```bash
python -m nrec_utils.create_occluded_set -s /path/to/yolo_dataset -d /path/to/occ_0.3 -p 0.1 -f 0.3
```

Check the selection before generating anything:

```bash
python -m nrec_utils.create_occluded_set -s /path/to/yolo_dataset -d /path/to/occ_0.3 -p 0.1 -f 0.3 --dry-run
```

Then fold the result into your baseline set:

```bash
cp -r /path/to/occ_0.3/images/* /path/to/yolo_dataset/images/
cp -r /path/to/occ_0.3/labels/* /path/to/yolo_dataset/labels/
```

### Adding a new occlusion

1. Add a class in `src/nrec_utils/occlusions/` subclassing `OcclusionBase`,
   taking `occ_factor` and passing its canonical name up to `super().__init__`.
2. Add one line to `OCCLUSION_REGISTRY` in
   `src/nrec_utils/occlusions/__init__.py`, keyed by that same name.

The CLI's `--occlusion` choices and `--help` pick it up automatically. A test
asserts every registry key matches its instance's `get_occ_name()`, so the name
used on the command line and the name used as a filename prefix cannot drift
apart. If the new occlusion moves geometry rather than only changing pixels,
read the note in `OcclusionBase` first — its labels cannot just be copied.

## Source directory structure

The source dataset must already exist (it is only read) with the following
layout, per split (`train`, `val`, `test`) and class (`positive`/`negative`):

```
{source}/{split}/positive/2015.11Soergels/<scenario>/Images/*.png
{source}/{split}/positive/2015.11Soergels/<scenario>/Annotations/*.xml
{source}/{split}/negative/2015.11Soergels/<scenario>/Images/*.png
{source}/{split}/negative/2015.11Soergels/<scenario>/Annotations/*.xml
```

There can be any number of `<scenario>` folders (e.g. a distinct
field/row/lighting condition). Each positive image must have a same-named
`.xml` annotation file for it to be copied.

## Destination directory structure

After running the script, the destination looks like this (directories are
created automatically):

```
{dest}/images/{split}/<name>.png       # selected images
{dest}/labels/{split}/<name>.txt       # YOLO labels (empty for negative/background images)
{dest}/labels_xml/{split}/<name>.xml   # original VOC annotations, kept as a backup
```

`labels_xml/` sits next to `labels/` (not inside it), so Ultralytics never
picks it up.

## Using the library functions directly

The two stages can also be called from Python (e.g. in a notebook), with
`src` on the Python path:

```python
import nrec_utils

for split in ("train", "val", "test"):
    for positive in (True, False):
        nrec_utils.nrec_select_images(
            src_base_dir="/content/apples_left_labeled",
            dst_base_dir="/content/aggregated",
            type_dir=split,
            percentage_to_copy=0.5,
            positive=positive,
        )

nrec_utils.nrec_xml_to_yolo(src_base_dir="/content/aggregated")
```

Calling `nrec_select_images` directly also lets you use a different fraction
per split or per class (e.g. 100% of positives and 20% of negatives).

`nrec_select_images` parameters:

- `src_base_dir` — root of the labeled source dataset.
- `dst_base_dir` — root of the YOLO output dataset.
- `type_dir` — the split to read/write, e.g. `train`, `val`, `test`.
- `percentage_to_copy` — fraction `(0, 1]` of each scenario's images to keep.
- `positive` — `True` for the `positive` branch (image + annotation);
  `False` for the `negative` branch (image + empty label).

`nrec_xml_to_yolo` parameters:

- `src_base_dir` — root of the YOLO tree (the `dst_base_dir` used above).
- `backup_dir` — where converted `.xml` files are moved; defaults to
  `{src_base_dir}/labels_xml`.
- `class_map` — annotation name to YOLO class id; defaults to
  `{"person": 0, "person-part": 1}`.

## Counting images

```bash
python src/nrec_utils/count_images.py /content/apples_left_labeled/train/positive/2015.11Soergels
```

Options: `--ext` (default `png`) and `--subdir` (default `Images`).

## Running tests

The tests live in `tests/` and use [pytest](https://docs.pytest.org/), which
is declared as a development dependency (the `dev` group in
`pyproject.toml`).

### Setting up the environment

Use a Python >= 3.13 environment, for example a conda environment. From the
project root (the directory containing `pyproject.toml`):

```powershell
conda activate project
pip install -e . --group dev
```

This installs the package in editable mode (changes under `src/` take effect
without reinstalling) plus the `dev` group (pytest). `--group` needs
pip >= 25.1. Re-run it whenever the dependencies in `pyproject.toml` change.

If pip fails with `[WinError 1260] This program is blocked by group policy`,
the build backend (`uv_build`), which pip downloads into a temporary folder,
is being blocked. Use uv to install into the same conda environment instead:

```powershell
uv pip install --python D:\anaconda\envs\project\python.exe -e . --group dev
```

### Running the tests

Run pytest from the project root with the environment active.
`testpaths = ["tests"]` in `pyproject.toml` only applies there, so running from
`src/` collects no tests.

```bash
python -m pytest                                                    # all tests
python -m pytest -v                                                 # one line per test
python -m pytest tests/test_occlusion_base.py                       # one file
python -m pytest tests/test_occlusion_base.py::test_get_occ_factor  # one test
python -m pytest -k occ_name                                        # tests whose name matches
```

### Adding a test dependency

Add the package to the `dev` group in `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "pytest>=9.1.1",
    "<package>",
]
```

then re-run `pip install -e . --group dev`.

## Requirements

- Python >= 3.13
- numpy >= 2.5 (runtime, used by `nrec_utils.occlusions`)
- opencv-python >= 5.0 (runtime, used by `nrec_utils.occlusions`)
- pytest for development (installed by `pip install -e . --group dev`)
