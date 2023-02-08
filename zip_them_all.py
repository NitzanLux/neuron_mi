import argparse
import gzip
import multiprocessing
import ntpath
import re
import traceback
from enum import Enum
from multiprocessing import Process, Queue
import EntropyHub as EH
from EntropyHub import SampEn
import numpy as np
import pickle as pickle
import entropy.DSampEN as DSEN
import entropy as ent
from typing import List, Dict
from utils.parse_file import parse_sim_experiment_file
from tqdm import tqdm
import os
import subprocess

parser = argparse.ArgumentParser(description='Regex expression for zipping different files')
parser.add_argument('-r', dest="regex_expression", type=str,
                    help='Regex expression for zipping files.',required=True)
parser.add_argument('-d', dest="dir", type=str,
                    help='Destination path', default='entropy_data')
args = parser.parse_args()

m = re.compile(args.regex_expression)
files=[]
for i in os.listdir(args.dir):
    if m.match(i):
        files.append(i)

cwd = os.getcwd()
os.chdir(os.path.join(cwd,args.dir))
for i in files:
    p = subprocess.Popen(f"zip -r {i}.zip {i}", stdout=subprocess.PIPE, shell=True)
    print(p.communicate())
