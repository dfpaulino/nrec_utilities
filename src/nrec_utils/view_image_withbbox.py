import argparse
import cv2
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import os

def read_annotation(annotation_file:str)->list:
    """
    return the BBOX as list of file is present and has content, otherwise, return empty array.
    Assumes a simple annotation per NREC images
    """
    bboxes = []
    if os.path.getsize(annotation_file)!=0:
        with open(annotation_file) as file:
            for line in file:
                contents = line.split(sep=' ') 
                x_c,y_c,w,h =float(contents[1]),float(contents[2]),float(contents[3]),float(contents[4])
                bboxes.append([x_c,y_c,w,h])

    return bboxes



def cli():
    parser = argparse.ArgumentParser(prog="view_image_withbbox",
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("-b", "--base", required=True,
                            help="base")
    parser.add_argument("-o", "--original", required=True,
                                help="original")
    parser.add_argument("-c", "--occ-dir", required=True,
                                    help="occ dir")                

    parser.add_argument("-t", "--split", required=True,
                                help="split test|train|val")
    parser.add_argument("-i", "--image", required=True,
                                    help="image")

    args = parser.parse_args()

    BASE_DIR = args.base
    ORIG_DIR = args.original
    OCC_DIR = args.occ_dir
    IMAGE = args.image
    SPLIT = args.split
    occ_prefix='fogEffect_'

    orig_image_path = BASE_DIR+ORIG_DIR+'/images/'+SPLIT+'/'+IMAGE
    
    occ_image_path = BASE_DIR+OCC_DIR+'/images/'+SPLIT+'/'+occ_prefix+IMAGE
    stem = Path(orig_image_path).stem
    label_path = BASE_DIR+ORIG_DIR+'/labels/'+SPLIT+'/'+stem+'.txt'

    image1 = cv2.imread(orig_image_path, cv2.IMREAD_COLOR)
    # converting BGR to RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.imread(occ_image_path, cv2.IMREAD_COLOR)
    # converting BGR to RGB
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    H,W = image1.shape[0],image1.shape[1]

    bboxs = read_annotation(label_path)
    for bbox in bboxs:
        # x_c,y_c,w,h]
        x0,x1 = int(bbox[0]*W-bbox[2]*W/2),int(bbox[0]*W+bbox[2]*W/2)
        y0,y1 = int(bbox[1]*H-bbox[3]*H/2),int(bbox[1]*H+bbox[3]*H/2)
        cv2.rectangle(image2, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.rectangle(image1, (x0, y0), (x1, y1), (0, 255, 0), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(image1)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(image2)
    ax2.set_title('occluded')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()



    

if __name__ =="__main__":
    cli()