import argparse
import os
from readMat import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='0', help='matlab file name')

    args = parser.parse_args()
    mat = args.m

    read_mat(mat)

if __name__ == "__main__":
    main()