import pytest
import os

ANNOTATION_CONTENT=['0 0.454715 0.241813 0.062281 0.198538']
ANNOTATION_CONTENT_MULTI=['0 0.454715 0.241813 0.062281 0.198538\n','0 0.554715 0.241813 0.062281 0.198538']

def write_yolo_annotation(content:list,dest):
    with open(str(dest),"w", encoding='utf-8') as file:
        file.writelines(content)

def remove_yolo_annotation(dest):
    if os.path.exists(str(dest)):
        os.remove(str(dest))

def test_read_annotation_with_content():
    dest = '/tmp/Fog_Image_0_123456.txt'
    write_yolo_annotation(ANNOTATION_CONTENT,dest=dest)

    # file must not have size 0
    if os.path.exists(dest) and os.path.getsize(dest)==0:
            assert False
    
    file=open(dest)
    contents = file.readline().split(sep=' ') 
    print(contents)
    x_c,y_c,w,h =contents[1],contents[2],contents[3],contents[4]
    print(f'x {float(x_c)}, y {y_c}, w {w}, h {h}')

    remove_yolo_annotation(dest=dest)

def test_read_annotation_with_content_multi():
    dest = '/tmp/Fog_Image_0_123456.txt'
    write_yolo_annotation(ANNOTATION_CONTENT_MULTI,dest=dest)

    # file must not have size 0
    if os.path.getsize(dest)==0:
            assert False

    bboxes=[]
    
    with open(dest) as file:
        for line in file:
            contents = line.split(sep=' ') 
            print(contents)
            x_c,y_c,w,h =float(contents[1]),float(contents[2]),float(contents[3]),float(contents[4])
            print(f'x {x_c}, y {y_c}, w {w}, h {h}')
            bboxes.append([x_c,y_c,w,h])

    print(f'bboxs {bboxes}')
    remove_yolo_annotation(dest=dest)


def test_read_annotation_with_no_content():
    dest = '/tmp/Fog_Image_0_123456.txt'
    write_yolo_annotation([],dest=dest)
    
    if os.path.getsize(dest)==0:
        assert True

    remove_yolo_annotation(dest=dest)