import os
import shutil
from pathlib import Path


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
    3. Within a scenario, looks at every file in its Images/ and
       Annotations/ subfolders and figures out how many images to keep based
       on percentage_to_copy (e.g. 0.25 keeps roughly 1 in every 4 images).
    4. Copies the selected images (always) and their matching .xml
       annotation (only when positive=True — negative/background images are
       assumed to have no bounding boxes to keep) into dst_base_dir.

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
    - percentage_to_copy: fraction (0.0-1.0) of each scenario's images to
      keep. 1.0 copies everything; 0.25 copies roughly every 4th image
      (sorted by filename). Must be > 0, otherwise a divide-by-zero occurs.
    - positive: True to pull from the "positive" branch (copies image +
      annotation); False to pull from the "negative" branch (copies image
      only, since there's nothing to annotate).

    PRE-REQUISITES / REQUIRED DIRECTORY STRUCTURE

    Source (must already exist, read-only):
        {src_base_dir}/{type_dir}/positive/2015.11Soergels/<scenario>/Images/*.png
        {src_base_dir}/{type_dir}/positive/2015.11Soergels/<scenario>/Annotations/*.xml
        {src_base_dir}/{type_dir}/negative/2015.11Soergels/<scenario>/Images/*.png
        {src_base_dir}/{type_dir}/negative/2015.11Soergels/<scenario>/Annotations/*.xml
      (there can be any number of <scenario> folders; each image file must
      have a same-named .xml annotation file for it to be copied)

    Destination (YOU must create these folders BEFORE calling this
    function — shutil.copy does not create missing directories and will
    raise an error if they don't exist):
        {dst_base_dir}/images/{type_dir}/
        {dst_base_dir}/labels/{type_dir}/

      e.g. before calling with type_dir='train':
        os.makedirs('/content/aggregated/images/train', exist_ok=True)
        os.makedirs('/content/aggregated/labels/train', exist_ok=True)
    """

    if(positive):
        src_base_dir_full=src_base_dir+"/"+type_dir+"/positive/2015.11Soergels"
    else:
        src_base_dir_full=src_base_dir+"/"+type_dir+"/negative/2015.11Soergels"

    dirs = [ f.name for f in os.scandir(src_base_dir_full) if f.is_dir()]


    total_img_copied=0

    for d in dirs:
        cur_scen_dir = src_base_dir_full+ "/" +d
        print(" INFO - Current Scenario: "+cur_scen_dir)

        (root,dirs,img_files) =next(os.walk(cur_scen_dir+"/Images"))
        (root,dirs,annotation_files) =next(os.walk(cur_scen_dir+"/Annotations"))

        total_images = len(img_files)
        total_annotations = len(annotation_files)
        img_to_copy_cnt = int(total_images*percentage_to_copy)
        # pace = "keep every Nth image" stride needed to hit img_to_copy_cnt.
        # NOTE: if percentage_to_copy is too small (img_to_copy_cnt == 0),
        # this divides by zero.
        pace = total_images//img_to_copy_cnt



        if total_annotations == total_images :
            print (f'INFO - total_images [{total_images}] same as total_files [{total_images}] only {img_to_copy_cnt} will be copied. pace {pace}')
        else:
            print (f'WARN - total_images {total_images} not same as total_files {total_annotations}')


        count=0
        for file_name in sorted(img_files):
            if count%pace==0:

                # remove the prefix "".png"
                cmn_file_name =file_name[:-4]

                img_full_src_path = cur_scen_dir+'/Images/'+cmn_file_name+".png"
                img_full_dst_path = dst_base_dir+'/images/'+type_dir+"/"+cmn_file_name+".png"


                annotation_full_src_path = cur_scen_dir+'/Annotations/'+cmn_file_name+".xml"
                annotation_full_dst_path = dst_base_dir+'/labels/'+type_dir+"/"+cmn_file_name+".xml"

                if(Path(img_full_src_path).exists() & Path(annotation_full_src_path).exists()) :
                    shutil.copy(img_full_src_path,img_full_dst_path)
                    # annotations are only meaningful for positive (labeled)
                    # images, so negative images are copied without one
                    if positive==True:
                        shutil.copy(annotation_full_src_path,annotation_full_dst_path)

                    total_img_copied=total_img_copied+1
                else:
                    print(f"Error - missing files {img_full_src_path} or {annotation_full_src_path}")

            count=count+1

        print(f'Total images copied [{total_img_copied}]')

