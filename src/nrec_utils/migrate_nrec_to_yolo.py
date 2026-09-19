
import argparse
from nrec_utils import nrec_utils as nrec_util


#NREC_SRC_DIR = '/media/dpaulino/d9278585-8446-4967-a41e-50568bba88e0/AgriImages/NREC'
#YOLO_DST_DIR = '/media/dpaulino/d9278585-8446-4967-a41e-50568bba88e0/AgriImages/NREC/selected'

# fraction of each scenario to keep, per split. Frames are ~133ms apart, so
# neighbouring frames are near-duplicates and there is little value in keeping
# all of them.
#PERCENTAGE_TO_COPY = 1.0


def main(source_dir:str , dst_dir:str, copy_pc:float):
    # stage 1 - build the YOLO tree: one pass per split, per positive/negative
    for type_dir in ('train', 'val', 'test'):
        for positive in (True, False):
            nrec_util.nrec_select_images(src_base_dir=source_dir,
                                         dst_base_dir=dst_dir,
                                         type_dir=type_dir,
                                         percentage_to_copy=copy_pc,
                                         positive=positive)

    # stage 2 - rewrite the copied VOC .xml annotations as YOLO .txt labels
    nrec_util.nrec_xml_to_yolo(src_base_dir=dst_dir)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source",required=True, help="src directory")
    parser.add_argument("-d", "--dest", required=True,help="des directory")
    parser.add_argument("-p", "--copy-factor",required=True,type=float,help="copy factor float 0 to 1 where 1 is 100%%")
    args = parser.parse_args()
    main(args.source,args.dest,args.copy_factor)


if __name__ == "__main__":
    cli()
