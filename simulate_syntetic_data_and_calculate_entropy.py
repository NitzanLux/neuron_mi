import argparse
import gzip
import multiprocessing
import ntpath
import traceback
from enum import Enum
from multiprocessing import Process, Queue
import EntropyHub as EH
from EntropyHub import SampEn
import numpy as np
import pickle as pickle
import entropy.DSampEN as DSEN
import entropy as ent
from entropy.CTW.CTW import UnboundProbabilityException
from typing import List, Dict
from utils.parse_file import parse_sim_experiment_file
from tqdm import tqdm
import os
import cProfile




import time
from entropy.CTW.efficient_inf_ctw import Node
ENTROPY_DATA_BASE_FOLDER = os.path.join(os.getcwd(), 'entropy_data')
number_of_cpus = multiprocessing.cpu_count()
print("start job")
from utils.utils import *
DEBUG_MODE=False
number_of_jobs = number_of_cpus//5
# number_of_jobs=1
def simulate_data(size,jitter):
    f