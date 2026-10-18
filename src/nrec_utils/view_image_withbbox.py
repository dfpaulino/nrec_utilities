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
    
    parser.add_argument("-s", "--source", required=True,
                            help="src directory of base images in YOLO layout")

    parser.add_argument("-t", "--split", required=True,
                                help="split test|train|val")
    parser.add_argument("-i", "--image", required=True,
                                    help="image")

    args = parser.parse_args()

    DIR = args.source
    IMAGE = args.image
    SPLIT = args.split

    image_path = DIR+'/images/'+SPLIT+'/'+IMAGE
    stem = Path(image_path).stem
    label_path = DIR+'/labels/'+SPLIT+'/'+stem+'.txt'

    image1 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # converting BGR to RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    H,W = image1.shape[0],image1.shape[1]

    bboxs = read_annotation(label_path)
    for bbox in bboxs:
        # x_c,y_c,w,h]
        x0,x1 = int(bbox[0]*W-bbox[2]*W/2),int(bbox[0]*W+bbox[2]*W/2)
        y0,y1 = int(bbox[1]*H-bbox[3]*H/2),int(bbox[1]*H+bbox[3]*H/2)
        cv2.rectangle(image1, (x0, y0), (x1, y1), (0, 255, 0), 1)


    fig = plt.figure(figsize=(10, 10))
    plt.imshow(image1)
    plt.show()



    

if __name__ =="__main__":
    cli()