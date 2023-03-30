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
from typing import List, Dict
from utils.parse_file import parse_sim_experiment_file
from tqdm import tqdm
import os
import cProfile



import time
from entropy.CTW.efficient_inf_ctw import Node
ENTROPY_DATA_BASE_FOLDER = os.path.join(os.getcwd(), 'entropy_data')
number_of_cpus = multiprocessing.cpu_count()
MAX_INTERVAL = 300
print("start job")
from utils.utils import *
DEBUG_MODE=False
number_of_jobs = number_of_cpus//5
# number_of_jobs=1


SIMULATIONS_PATH = '/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron/simulations/data/'







class EntropyEstimation():
    def __init__(self, tag, file_name, file_index, sim_index, s, v,entropy=None,_tree=None):
        self.s = s
        self.v = v
        self.tag = tag
        self.file_index = file_index
        self.sim_index = sim_index
        self.file_name = file_name
        self.entropy=entropy
        self.__tree=_tree


    def build_tree(self):
        b = ent.CTW()
        b.insert_pattern(self.s.astype(int).tolist())
        self.entropy = b.get_entropy(len(self.s.astype(int).tolist()))
        # self.__tree=b.to_dict()

    @property
    def tree(self):
        if self.__tree is not None:
            return self.__tree
        else:
            pass
        # return ent.CTW.from_dict(self.__tree)

    def get_number_of_spikes(self):
        spike_number = np.sum(self.s)
        return np.sum(spike_number)

    def get_key(self):
        return (self.file_name, self.sim_index)

    def get_processed_data(self):
        s = None
        v = None
        if self.s is not None: s = self.s.copy()
        if self.v is not None: v = self.v.copy()
        keys = {}
        return v, s, keys

    def generate_file_name(self,remove_suffix_from_file=None):
        return self.generate_file_name_f(
            self.file_name[:-len(remove_suffix_from_file)] if remove_suffix_from_file is not None else self.file_name,
            self.sim_index)

    def to_dict(self,ignore_tree=True):
        data_dict = dict(s=self.s,
                         v=self.v,
                         tag=self.tag,
                         entropy = self.entropy,
                         file_index=self.file_index,
                         _tree=None if ignore_tree else self.__tree,
                         sim_index=self.sim_index,
                         file_name=self.file_name)
        return data_dict

    @staticmethod
    def generate_file_name_f(file_index, sim_index):
        return f'{file_index}_{sim_index}.pkl'

    def save(self):
        current_path = ENTROPY_DATA_BASE_FOLDER
        if not os.path.exists(current_path):
            os.mkdir(current_path)
        current_path = os.path.join(current_path, self.tag)
        if not os.path.exists(current_path):
            os.mkdir(current_path)
        current_path = os.path.join(current_path, self.generate_file_name())
        with open(current_path, 'wb') as pfile:
            pickle.dump(self.to_dict(), pfile)

    @staticmethod
    def load(path=None, tag=None, file_index=None, sim_index=None):
        if path is None:
            assert tag is not None and sim_index is not None and file_index is not None, 'If path is unfilled you nee ' \
                                                                                         'to fill tag sim index and ' \
                                                                                         'file index '
            path = os.path.join(ENTROPY_DATA_BASE_FOLDER, tag,
                                EntropyEstimation.generate_file_name_f(file_index, sim_index))
        try:
            with open(path, 'rb') as pfile:
                obj = pickle.load(pfile)
        except Exception as e:
            raise type(e)(str(e)+'\n '+path)

        return EntropyEstimation(**obj)

    @staticmethod
    def load_all_by_tag(tag):
        cur_path = os.path.join(ENTROPY_DATA_BASE_FOLDER, tag)
        data_list = []
        for i in tqdm(os.listdir(cur_path)):
            data_list.append(EntropyEstimation.load(os.path.join(cur_path, i)))
        return data_list


def load_file_path(base_dir):
    return os.listdir(base_dir)


def create_sample_entropy_file_multiprocessing(q, tag, use_v=True, use_s=True):
    while True:
        if q.empty():
            return
        data = q.get(block=120)
        if data is None:
            return
        f_path, f_index = data
        _, y_spike, y_soma = parse_sim_experiment_file(f_path)
        path, f = ntpath.split(f_path)
        if y_spike.ndim == 1:
            v = y_soma.astype(np.float64)
            s = y_spike.astype(np.float64)
            if DEBUG_MODE:
                v=v[:500]
                s=s[:500]
            eo = EntropyEstimation(tag, f, f_index, sim_index=0, s=s, v=v)
            eo.build_tree()
            eo.save()
            t = time.time()
            print(f"current sample number {f} {0}  total: {time.time() - t} seconds", flush=True)
        else:
            for index in range(y_spike.shape[0]):
                v = y_soma[index].astype(np.float64) if use_v else None
                s = y_spike[index].astype(np.float64) if use_s else None
                eo = EntropyEstimation(tag, f, f_index, sim_index=index, s=s, v=v)
                eo.build_tree()
                eo.save()
                t = time.time()
                print(f"current sample number {f} {index}  total: {time.time() - t} seconds", flush=True)
def create_sample_entropy_file(data, tag, use_v=True, use_s=True):
    f_path, f_index = data
    _, y_spike, y_soma = parse_sim_experiment_file(f_path)
    path, f = ntpath.split(f_path)
    if y_spike.ndim == 1:
        v = y_soma.astype(np.float64)
        s = y_spike.astype(np.float64)
        if DEBUG_MODE:
            v=v[:500]
            s=s[:500]
        eo = EntropyEstimation(tag, f, f_index, sim_index=0, s=s, v=v)
        eo.build_tree()
        eo.save()
        t = time.time()
        print(f"current sample number {f} {0}  total: {time.time() - t} seconds", flush=True)
    else:
        for index in range(y_spike.shape[0]):
            v = y_soma[index].astype(np.float64) if use_v else None
            s = y_spike[index].astype(np.float64) if use_s else None
            eo = EntropyEstimation(tag, f, f_index, sim_index=index, s=s, v=v)
            eo.build_tree()
            eo.save()
            t = time.time()
            print(f"current sample number {f} {index}  total: {time.time() - t} seconds", flush=True)

def get_entropy(tag, pathes, file_index_start, use_v, use_s):
    number_of_jobs = min(number_of_cpus - 1, len(pathes))
    print('Number of jobs :',number_of_jobs)
    if number_of_jobs ==1:
        create_sample_entropy_file((pathes[0],0+file_index_start), tag, use_v, use_s)
        return
    else:
        queue = Queue(maxsize=number_of_jobs)
        process = [
            Process(target=create_sample_entropy_file_multiprocessing, args=(
            queue, tag, use_v, use_s)) for i
            in range(number_of_jobs)]
        print('starting')
        for j, fp in enumerate(pathes):
            queue.put((fp, j + file_index_start))
            if j < len(process):
                process[j].start()

        # if number_of_jobs > 1:
        for p in process:
            p.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add ground_truth name')
    parser.add_argument('-f', dest="parent_dir_path", type=str,
                        help='parant directory path')
    parser.add_argument('-t', dest="tag", type=str,
                        help='tag for saving')
    parser.add_argument('-j', dest="cluster_num", type=int,
                        help="number of jobs on cluster")
    parser.add_argument('-mem', dest="memory", type=int,
                        help='set memory', default=-1)
    parser.add_argument('-v', dest="use_v", help='use voltage', type=str2bool, default=True)
    parser.add_argument('-s', dest="use_s", help='use spike data', type=str2bool, default=True)
    parser.add_argument('-ex', dest="files_that_do_not_exist",  type=str2bool,
                        help='simulate only files that do not exist', default=False)
    args = parser.parse_args()


    print(args)
    print("continue?y/n")
    response = input()
    while response not in {'y', 'n'}:
        print("continue?y/n")
        response = input()
    if response == 'n':
        exit(0)

    from utils.slurm_job import *

    number_of_clusters = args.cluster_num
    job_factory = SlurmJobFactory("cluster_logs")

    parent_dir_path = args.parent_dir_path
    if not os.path.exists(parent_dir_path):
        parent_dir_path = os.path.join(SIMULATIONS_PATH, parent_dir_path)
    list_dir_parent = os.listdir(parent_dir_path)
    list_dir_parent = [os.path.join(parent_dir_path, i) for i in list_dir_parent]

    keys = {}
    if args.memory > 0:
        keys['mem'] = args.memory
        print("Mem:", args.memory)
    if args.files_that_do_not_exist and os.path.exists(os.path.join(ENTROPY_DATA_BASE_FOLDER, args.tag)):
        files_that_exists = []
        for i,f in enumerate(os.listdir(os.path.join(ENTROPY_DATA_BASE_FOLDER, args.tag))):
            files_that_exists.append(f)
        new_list=[]
        for i in list_dir_parent:
            for j in files_that_exists:
                temp_i = os.path.basename(i)
                if temp_i in j:
                    break
            else:
                new_list.append(i)
        list_dir_parent = new_list
        if len(list_dir_parent)==0:
            print('no missing files, exiting...')
            exit(0)
    if number_of_clusters<=0: number_of_clusters=len(list_dir_parent)
    number_of_clusters=min(len(list_dir_parent),number_of_clusters)
    jumps = len(list_dir_parent) // (number_of_clusters)
    modulu_res = len(list_dir_parent) % (number_of_clusters)
    print(f'jumps {jumps} c_num = {number_of_clusters}')

    cur_start = 0
    for i in range(number_of_clusters):
        end_point = cur_start + jumps + (i < modulu_res)
        pathes = list_dir_parent[cur_start:min(end_point, len(list_dir_parent))]
        # if DEBUG_MODE: pathes=[pathes[0]]

        print(len(pathes))
        # use_voltage = args.sv == 'v'
        print(range(cur_start, min(end_point, len(list_dir_parent))))

        job_factory.send_job(f"entropy_{args.tag}_{i}",
                             f'python -c "from create_entropy_estimation import get_entropy; get_entropy(' + "'" + args.tag + "'" + f',{pathes},{i * jumps}, {args.use_v},{args.use_s})"',
                             **keys,mem='64G')
        print('job sent')
        cur_start = end_point

        if DEBUG_MODE: break
