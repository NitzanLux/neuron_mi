import os
import argparse
# def create_dir_keys()

parser = argparse.ArgumentParser(description='Create evaluation object from file names')
parser.add_argument('-f', dest="folders", type=str,nargs='+',
                    help='folders_to_sample_from')
args = parser.parse_args()
