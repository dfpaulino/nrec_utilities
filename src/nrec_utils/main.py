
import nrec_utils as nrec_utils

def main():
    nrec_utils.nrec_select_images( src_base_dir='/media/dpaulino/d9278585-8446-4967-a41e-50568bba88e0/AgriImages/NREC',
                                  dst_base_dir='/media/dpaulino/d9278585-8446-4967-a41e-50568bba88e0/AgriImages/NREC/selected',
                                  type_dir='val',
                                  percentage_to_copy=1.0,
                                  positive=True)

if __name__ == "__main__":
    main()
