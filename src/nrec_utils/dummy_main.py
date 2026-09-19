
from nrec_utils import nrec_utils as nrec_util
import argparse

def main(src:str, dst:str):
    print(f'source dir {src} : dst dir {dst}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="src directory")
    parser.add_argument("-d", "--dest", help="des directoru")
    args = parser.parse_args()
    
    main(args.source,args.dest)
