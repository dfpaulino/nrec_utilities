import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# YOLO class indices for the NREC person annotations. Both classes are kept:
# "person-part" marks a partially visible person and is a separate class here.
CLASS_MAP = {"person": 0, "person-part": 1}

# The NREC XML boxes use the Pascal VOC 1-based pixel-centre convention, so the
# valid coordinate range is [0.5, W+0.5] rather than [0, W]. Subtracting this
# offset moves them to 0-based edge coordinates; verified across all 31426
# boxes of the 2015.11Soergels set, it maps the observed range exactly onto
# [0, 720] x [0, 480] and brings the out-of-frame box count from 624 to 0.
VOC_PIXEL_OFFSET = 0.5

SPLITS = ("train", "val", "test")


def nrec_select_images(src_base_dir:str='/content/apples_left_labeled',
                       dst_base_dir:str='/content/aggregated',
                       type_dir:str='train',
                       percentage_to_copy:float=1.0,
                       positive:bool=True):
    """
    Sample a subset of images (and their matching XML annotations) out of the
    NREC "labeled" dataset layout, and copy them into a flat aggregated
    dataset layout suitable for training (e.g. an object-detection pipeline).

    WHAT IT DOES (step by step)
    1. Picks the "positive" or "negative" branch under src_base_dir/type_dir,
       inside the fixed scenario collection "2015.11Soergels".
    2. Treats every subdirectory found there as a separate "scenario" (e.g. a
       distinct field/row/lighting condition) and processes each in turn.
    3. Within a scenario, sorts the images by filename (which for this dataset
       is chronological, since the name embeds the capture timestamp) and keeps
       exactly round(n * percentage_to_copy) of them, spread evenly across the
       scenario. Frames are captured ~133ms apart, so neighbouring frames are
       near-duplicates: sampling at an even temporal spacing keeps coverage of
       the whole scenario while dropping that redundancy.
    4. Copies the selected images (always) and their matching .xml
       annotation (only when positive=True — negative/background images have
       no bounding boxes, and instead get an empty .txt label marking them as
       explicit YOLO background) into dst_base_dir.

    HOW TO CALL IT
    You typically call it once per split (train/test/val) and once per class
    (positive/negative) you want included in the aggregated dataset, e.g.:

        nrec_select_images(src_base_dir='/content/apples_left_labeled',
                            dst_base_dir='/content/aggregated',
                            type_dir='train', percentage_to_copy=1.0,
                            positive=True)
        nrec_select_images(src_base_dir='/content/apples_left_labeled',
                            dst_base_dir='/content/aggregated',
                            type_dir='train', percentage_to_copy=0.2,
                            positive=False)
        # repeat with type_dir='test' (and/or 'val') as needed

    PARAMETERS
    - src_base_dir: root of the labeled source dataset (see layout below).
    - dst_base_dir: root of the aggregated output dataset (see layout below).
    - type_dir: the dataset split to read/write, e.g. 'train' or 'test'.
      Must match a real subfolder name on the source side, and is also used
      to name the destination subfolder.
    - percentage_to_copy: fraction (0.0-1.0] of each scenario's images to
      keep. The count per scenario is exact — round(n * percentage_to_copy) —
      except that every scenario contributes at least one image, so very small
      fractions do not silently drop the short scenarios entirely.
    - positive: True to pull from the "positive" branch (copies image +
      annotation); False to pull from the "negative" branch (copies image
      plus an empty label, since there's nothing to annotate).

    RETURNS
    The total number of images copied across all scenarios.

    PRE-REQUISITES / REQUIRED DIRECTORY STRUCTURE

    Source (must already exist, read-only):
        {src_base_dir}/{type_dir}/positive/2015.11Soergels/<scenario>/Images/*.png
        {src_base_dir}/{type_dir}/positive/2015.11Soergels/<scenario>/Annotations/*.xml
        {src_base_dir}/{type_dir}/negative/2015.11Soergels/<scenario>/Images/*.png
        {src_base_dir}/{type_dir}/negative/2015.11Soergels/<scenario>/Annotations/*.xml
      (there can be any number of <scenario> folders; for positive scenarios
      each image file must have a same-named .xml annotation file for it to be
      copied)

    Destination (created automatically if missing):
        {dst_base_dir}/images/{type_dir}/
        {dst_base_dir}/labels/{type_dir}/
    """

    if not 0 < percentage_to_copy <= 1:
        raise ValueError(f"percentage_to_copy must be in (0, 1], got {percentage_to_copy}")

    if(positive):
        src_base_dir_full=src_base_dir+"/"+type_dir+"/positive/2015.11Soergels"
    else:
        src_base_dir_full=src_base_dir+"/"+type_dir+"/negative/2015.11Soergels"

    img_dst_dir = dst_base_dir+'/images/'+type_dir
    label_dst_dir = dst_base_dir+'/labels/'+type_dir
    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(label_dst_dir, exist_ok=True)

    scenario_dirs = [ f.name for f in os.scandir(src_base_dir_full) if f.is_dir()]


    total_img_copied=0

    for d in scenario_dirs:
        cur_scen_dir = src_base_dir_full+ "/" +d
        print(" INFO - Current Scenario: "+cur_scen_dir)

        (root,_,img_files) =next(os.walk(cur_scen_dir+"/Images"))
        (root,_,annotation_files) =next(os.walk(cur_scen_dir+"/Annotations"))

        img_files = sorted(f for f in img_files if f.lower().endswith(".png"))
        total_images = len(img_files)
        total_annotations = len(annotation_files)
        # Exact count, at least one frame per scenario. Spreading the indices
        # with round(i * total / count) keeps them strictly increasing (since
        # total >= count), so we get exactly img_to_copy_cnt distinct frames
        # at an even spacing rather than the nearest 1/N stride.
        img_to_copy_cnt = max(1, round(total_images*percentage_to_copy))
        selected = [img_files[round(i * total_images / img_to_copy_cnt)]
                    for i in range(img_to_copy_cnt)]


        if total_annotations == total_images :
            print (f'INFO - total_images [{total_images}] same as total_files [{total_images}] only {img_to_copy_cnt} will be copied.')
        else:
            print (f'WARN - total_images {total_images} not same as total_files {total_annotations}')


        scenario_img_copied=0
        for file_name in selected:

            cmn_file_name = Path(file_name).stem

            img_full_src_path = cur_scen_dir+'/Images/'+cmn_file_name+".png"
            img_full_dst_path = img_dst_dir+"/"+cmn_file_name+".png"

            annotation_full_src_path = cur_scen_dir+'/Annotations/'+cmn_file_name+".xml"
            annotation_full_dst_path = label_dst_dir+"/"+cmn_file_name+".xml"

            # annotations are only meaningful for positive (labeled) images, so
            # a negative image is copied on its own merit
            if positive and not Path(annotation_full_src_path).exists():
                print(f"Error - missing files {img_full_src_path} or {annotation_full_src_path}")
                continue
            if not Path(img_full_src_path).exists():
                print(f"Error - missing files {img_full_src_path}")
                continue

            shutil.copy(img_full_src_path,img_full_dst_path)
            if positive:
                shutil.copy(annotation_full_src_path,annotation_full_dst_path)
            else:
                # empty label = explicit YOLO background frame
                Path(label_dst_dir+"/"+cmn_file_name+".txt").touch()

            scenario_img_copied=scenario_img_copied+1

        total_img_copied=total_img_copied+scenario_img_copied
        print(f'Images copied for scenario [{scenario_img_copied}]')

    print(f'Total images copied [{total_img_copied}]')
    return total_img_copied


def _voc_box_to_yolo(bndbox, img_w:int, img_h:int):
    """
    Convert one Pascal VOC <bndbox> to YOLO (cx, cy, w, h) normalised to 0-1.

    Returns None if the box cannot be expressed as a valid YOLO box (zero/negative
    extent, or still out of range after clamping) so the caller can skip it.
    """
    x_min = float(bndbox.find('xmin').text) - VOC_PIXEL_OFFSET
    y_min = float(bndbox.find('ymin').text) - VOC_PIXEL_OFFSET
    x_max = float(bndbox.find('xmax').text) - VOC_PIXEL_OFFSET
    y_max = float(bndbox.find('ymax').text) - VOC_PIXEL_OFFSET

    # safety net - a no-op on the 2015.11Soergels set once the offset is applied,
    # but keeps a malformed annotation from producing an out-of-range label
    x_min = min(max(x_min, 0.0), img_w)
    x_max = min(max(x_max, 0.0), img_w)
    y_min = min(max(y_min, 0.0), img_h)
    y_max = min(max(y_max, 0.0), img_h)

    box_w = (x_max - x_min) / img_w
    box_h = (y_max - y_min) / img_h
    centre_x = (x_min + x_max) / 2 / img_w
    centre_y = (y_min + y_max) / 2 / img_h

    if box_w <= 0 or box_h <= 0:
        return None
    if not all(0.0 <= v <= 1.0 for v in (centre_x, centre_y, box_w, box_h)):
        return None

    return centre_x, centre_y, box_w, box_h


def nrec_xml_to_yolo(src_base_dir:str='/content/aggregated',
                     backup_dir:str=None,
                     class_map:dict=None):
    """
    Rewrite the Pascal VOC .xml annotations of an already-selected dataset into
    YOLO .txt label files, in place.

    This is the second stage of the pipeline. nrec_select_images() builds the
    YOLO directory tree but copies the annotations verbatim as .xml, which
    Ultralytics cannot read (it globs labels/**/*.txt, so an all-.xml labels
    folder trains as if every image were background). Run this once over the
    selection to make the dataset trainable:

        for type_dir in ('train','val','test'):
            for positive in (True, False):
                nrec_select_images(src_base_dir=..., dst_base_dir='/content/aggregated',
                                   type_dir=type_dir, percentage_to_copy=...,
                                   positive=positive)
        nrec_xml_to_yolo(src_base_dir='/content/aggregated')

    WHAT IT DOES
    1. Looks for labels/train, labels/val and labels/test under src_base_dir and
       processes whichever of them exist.
    2. For every .xml in a split, reads the image size from <size> (so no image
       decoding is needed) and converts each <object> to one YOLO line:
           <class_id> <centre_x> <centre_y> <width> <height>
       with all four values normalised to 0-1. See _voc_box_to_yolo for the
       coordinate handling - the source boxes are ABSOLUTE PIXELS in the VOC
       1-based pixel-centre convention, not fractions of the image.
    3. Writes labels/{split}/{stem}.txt, keeping the stem so it still matches
       its image in images/{split}/ - the pairing Ultralytics relies on.
    4. Moves the converted .xml into backup_dir/{split}/ so labels/ is left
       holding only .txt, while every original annotation is preserved and can
       be re-converted. An .xml that fails to convert is deliberately LEFT in
       labels/ so the failure stays visible and the run can be retried.

    Background (negative) frames already carry an empty .txt written by
    nrec_select_images and have no .xml, so they are never visited here - their
    empty labels survive untouched, which is what marks them as YOLO
    background.

    PARAMETERS
    - src_base_dir: root of the YOLO tree, i.e. the same path passed as
      dst_base_dir to nrec_select_images.
    - backup_dir: where converted .xml files are moved to. Defaults to
      {src_base_dir}/labels_xml, a sibling of labels/ so that Ultralytics'
      images/ -> labels/ path substitution never picks it up.
    - class_map: annotation <name> to YOLO class id. Defaults to CLASS_MAP,
      {'person': 0, 'person-part': 1}. Any name absent from the map is skipped
      with a warning rather than guessed at.

    RETURNS
    A dict of counters: files_converted, boxes_written, boxes_skipped,
    files_failed.
    """

    if class_map is None:
        class_map = CLASS_MAP
    if backup_dir is None:
        backup_dir = src_base_dir+"/labels_xml"

    totals = {'files_converted':0, 'boxes_written':0, 'boxes_skipped':0, 'files_failed':0}

    for split in SPLITS:
        label_dir = src_base_dir+"/labels/"+split
        if not Path(label_dir).is_dir():
            continue

        xml_files = sorted(f for f in os.listdir(label_dir) if f.lower().endswith(".xml"))
        if not xml_files:
            print(f'INFO - split [{split}] has no .xml to convert, skipping')
            continue

        split_backup_dir = backup_dir+"/"+split
        os.makedirs(split_backup_dir, exist_ok=True)

        converted=0; boxes=0; skipped=0; failed=0
        for xml_name in xml_files:
            xml_path = label_dir+"/"+xml_name
            cmn_file_name = Path(xml_name).stem

            try:
                root = ET.parse(xml_path).getroot()
                size = root.find('size')
                img_w = int(float(size.find('width').text))
                img_h = int(float(size.find('height').text))
                if img_w <= 0 or img_h <= 0:
                    raise ValueError(f"bad image size {img_w}x{img_h}")
            except (ET.ParseError, AttributeError, TypeError, ValueError) as e:
                print(f'WARN - could not read {xml_path} ({e}), leaving it in place')
                failed=failed+1
                continue

            lines=[]
            for obj in root.findall('object'):
                class_name = obj.find('name').text.strip()
                if class_name not in class_map:
                    print(f'WARN - unknown class [{class_name}] in {xml_path}, skipping box')
                    skipped=skipped+1
                    continue

                yolo_box = _voc_box_to_yolo(obj.find('bndbox'), img_w, img_h)
                if yolo_box is None:
                    print(f'WARN - unusable bndbox in {xml_path}, skipping box')
                    skipped=skipped+1
                    continue

                centre_x, centre_y, box_w, box_h = yolo_box
                lines.append(f'{class_map[class_name]} {centre_x:.6f} {centre_y:.6f} {box_w:.6f} {box_h:.6f}')

            # an .xml whose every box was skipped still becomes a (background)
            # .txt - dropping the file instead would leave the image unlabelled
            Path(label_dir+"/"+cmn_file_name+".txt").write_text("\n".join(lines)+("\n" if lines else ""))
            shutil.move(xml_path, split_backup_dir+"/"+xml_name)

            converted=converted+1
            boxes=boxes+len(lines)

        print(f'INFO - split [{split}]: converted {converted} files, wrote {boxes} boxes, '
              f'skipped {skipped} boxes, failed {failed} files')
        totals['files_converted'] += converted
        totals['boxes_written']   += boxes
        totals['boxes_skipped']   += skipped
        totals['files_failed']    += failed

    print(f'Total {totals}')
    return totals
