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
from typing import List,Dict
from utils.parse_file import parse_sim_experiment_file
from tqdm import tqdm
import os
import time
ENTROPY_DATA_BASE_FOLDER = os.path.join(os.getcwd(), 'entropy_data')
number_of_cpus = multiprocessing.cpu_count()
MAX_INTERVAL = 1000
print("start job")
from utils.utils import *
number_of_jobs = number_of_cpus - 1
# number_of_jobs=1


SIMULATIONS_PATH = '/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron/simulations/data/'


class EntropyTypes(Enum):
    SAMPLE_ENTROPY = 'SampEn'
    APPROXIMATE_ENTROPY = 'ApEn'
    FUZZY_ENTROPY = 'FuzzEn'
    PERMUTATION_ENTROPY = 'PermEn'
    DISPERSION_ENTROPY = 'DispEn'
    SPECTRAL_ENTROPY = 'SpecEn'
    COSINE_ENTROPY = 'CoSiEn'
    # K2_ENTROPY = 'K2En'
    DISCRETE_SAMPLE_ENTROPY='DSampEn'
    def get_return_params(self):
        if self == EntropyTypes.SAMPLE_ENTROPY:
            return ['MSx','A', 'B']
        elif self == EntropyTypes.DISCRETE_SAMPLE_ENTROPY:
            return ['MSx','A']
    def get_function(self):
        if self.value in {'DSampEn'}:
            return getattr(DSEN,'DSampEn')
        else:
            return getattr(EH, str(self.value))

    @staticmethod
    def get_by_name(name):
        for i in EntropyTypes:
            if i.name == name:
                return i
        return None

class MultiScaleObj(Enum):
    MULTISCALE_ENTROPY = 'MSEn'
    R_MULTISCALE_ENTROPY = 'rMSEn'
    C_MULTISCALE_ENTROPY = 'cMSEn'

    def get_return_params(self):
        if self.name == MultiScaleObj.MULTISCALE_ENTROPY.name:
            return ['MSx', 'CI']



class EntropyObject():
    def __init__(self, tag, file_name, file_index, sim_index, s, v, use_derivative=False, smoothing_kernel=None,
                 max_scale=MAX_INTERVAL, multiscale_object=None, entropy_dict=None, multiscale_object_params=None,entropies_params=None):
        self.s = s
        self.v = v
        self.tag = tag
        self.multiscale_object = multiscale_object
        self.max_scale = max_scale
        self.file_index = file_index
        self.use_derivative = use_derivative
        self.smoothing_kernel = smoothing_kernel
        self.entropy_dict = entropy_dict if entropy_dict is not None else dict()
        self.sim_index = sim_index
        self.file_name = file_name
        self.entropies_params={} if entropy_dict is None else entropies_params
        self.multiscale_object_params = multiscale_object_params if multiscale_object_params is not None else {}

    def add_entropies(self, entropy_types: [EntropyTypes, List[EntropyTypes]],entropies_params:[None,Dict]=None):
        if not isinstance(entropy_types, list):
            entropy_types = [entropy_types]
        if entropies_params is None:
            entropies_params={}
        v, s, keys = self.get_processed_data()
        keys.update(entropies_params)
        for i in entropy_types:
            self.entropies_params[i]=keys
            self.add_entropy_measure(i, v, s, keys)

    def creat_entropy_function(self, entropy_type: EntropyTypes, keys=None):
        keys = {} if keys is None else keys
        if self.multiscale_object is not None:
            Mobj = EH.MSobject(entropy_type.value, **keys)
            return lambda x: getattr(EH, self.multiscale_object.value)(x, Mobj, Scales=self.max_scale,
                                                                       **self.multiscale_object_params)
        else:
            return lambda x: entropy_type.get_function()(x, m=self.max_scale, **keys)

    def add_entropy_measure(self, entropy_type: EntropyTypes, v=None, s=None, keys=None):
        # keys = {} if keys is None else keys
        if s is None and v is None:
            v, s, keys = self.get_processed_data()
        print(f"Current Entropy Measure {entropy_type.name} fidx{self.file_index} sidx{self.sim_index}", flush=True)
        start_time = time.time()
        # Mobj = EH.MSobject(entropy_type.value, **keys)\
        mobj = self.creat_entropy_function(entropy_type, keys)
        e_output_s, e_output_v = None, None
        if s is not None:
            e_output_s = mobj(s)
            # e_output_s = getattr(EH, self.multiscale_object.value)(s, Mobj, Scales=MAX_INTERVAL,**self.multiscale_object_params)
        if v is not None:
            e_output_v = mobj(v)
            # e_output_v = getattr(EH, self.multiscale_object.value)(v, Mobj, Scales=MAX_INTERVAL,**self.multiscale_object_params)
        print(
            f"Current Entropy Measure {entropy_type.name} fidx{self.file_index} sidx{self.sim_index}\n \t\t\t time:{time.time() - start_time}",
            flush=True)

        self.entropy_dict[entropy_type.name] = {'s': e_output_s, 'v': e_output_v}

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
        if self.use_derivative and v is not None:
            v = v[1:] - v[:-1]
        if self.smoothing_kernel is not None:  # for the future
            assert False, "smoothing is invalid right now."
        return v, s, keys

    def generate_file_name(self, remove_suffix_from_file=None):
        return self.generate_file_name_f(
            self.file_name[:-len(remove_suffix_from_file)] if remove_suffix_from_file is not None else self.file_name,
            self.sim_index)

    def get_entropy_dict(self):
        data_dict = dict()
        if self.multiscale_object is not None:
            return_params = self.multiscale_object.get_return_params()
            for k, v in self.entropy_dict.items():
                for tsv, vsv in v.items():
                    data_dict.update({f'{k}_{tsv}_{kk}': vv for kk, vv in zip(return_params, vsv)})
        else:
            for k, v in self.entropy_dict.items():
                return_params =EntropyTypes.get_by_name(k).get_return_params()
                for tsv, vsv in v.items():
                    data_dict.update({f'{k}_{tsv}_{kk}': vv for kk, vv in  zip(return_params,vsv)})
                    data_dict[f'{k}_{tsv}_CI']=data_dict[f'{k}_{tsv}_MSx'].sum()
        return data_dict

    def to_dict(self):
        data_dict = dict(s=self.s,
                         v=self.v,
                         tag=self.tag,
                         max_scale=self.max_scale,
                         file_index=self.file_index,
                         use_derivative=self.use_derivative,
                         smoothing_kernel=self.smoothing_kernel,
                         entropy_dict=self.entropy_dict,
                         sim_index=self.sim_index,
                         file_name=self.file_name,
                         multiscale_object_params=self.multiscale_object_params,
                         entropies_params=self.entropies_params,
                         multiscale_object=self.multiscale_object)
        return data_dict

    @staticmethod
    def generate_file_name_f(file_name, sim_index):
        return f'{file_name}_{sim_index}.pkl'

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
                                EntropyObject.generate_file_name_f(tag, file_index, sim_index))
        with open(path, 'rb') as pfile:
            obj = pickle.load(pfile)
        return EntropyObject(**obj)

    @staticmethod
    def load_all_by_tag(tag):
        cur_path = os.path.join(ENTROPY_DATA_BASE_FOLDER, tag)
        data_list = []
        for i in tqdm(os.listdir(cur_path)):
            data_list.append(EntropyObject.load(os.path.join(cur_path, i)))
        return data_list

    @staticmethod
    def load_and_save_in_single_file(tag):
        data_list = EntropyObject.load_all_by_tag(tag)
        data_dict = dict()
        file_set = set()
        for i in tqdm(data_list):
            file_set.add(i.file_index)
            data_dict[(i.file_index, i.sim_index)] = i
        file_path = os.path.join(ENTROPY_DATA_BASE_FOLDER, f'{tag}_fnum_{len(file_set)}_snum_{len(data_dict)}.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        # cur_path = os.path.join(ENTROPY_DATA_BASE_FOLDER, tag)
        # for i in tqdm(os.listdir(cur_path)):
        #     os.remove(os.path.join(cur_path, i))
        # os.rmdir(cur_path)

    @staticmethod
    def load_list(path):
        with open(path, 'rb') as file:
            object = pickle.load(file)

        return object


def load_file_path(base_dir):
    return os.listdir(base_dir)

def create_sample_entropy_file(q, tag, entropies_types,entropies_params=None,multiscale_object=None,multiscale_object_params=None, use_derivative=False,use_v=True,use_s=True):
    while True:
        if q.empty():
            return
        data = q.get(block=120)
        if data is None:
            return
        f_path, f_index = data
        _, y_spike, y_soma = parse_sim_experiment_file(f_path)
        path, f = ntpath.split(f_path)
        if y_spike.ndim==1:
            v = y_soma.astype(np.float64)
            s = y_spike.astype(np.float64)
            eo = EntropyObject(tag, f, f_index, sim_index=0, s=s, v=v, multiscale_object=multiscale_object,multiscale_object_params=multiscale_object_params,
                               use_derivative=use_derivative, max_scale=MAX_INTERVAL)
            eo.add_entropies(entropies_types,entropies_params)
            eo.save()
            t = time.time()
            print(f"current sample number {f} {0}  total: {time.time() - t} seconds", flush=True)
        else:
            for index in range(y_spike.shape[0]):
                v = y_soma[index].astype(np.float64) if use_v else None
                s = y_spike[index].astype(np.float64) if use_s else None
                eo = EntropyObject(tag, f, f_index, sim_index=index, s=s, v=v, multiscale_object=multiscale_object,multiscale_object_params=multiscale_object_params,
                                   use_derivative=use_derivative, max_scale=MAX_INTERVAL)
                eo.add_entropies(entropies_types,entropies_params)
                eo.save()
                t = time.time()
                print(f"current sample number {f} {index}  total: {time.time() - t} seconds", flush=True)


def get_entropy(tag, pathes, entropies, file_index_start, use_derivative, entropies_params, multiscale_object, multiscale_object_params,use_v,use_s):
    number_of_jobs = min(number_of_cpus - 1, len(pathes))
    entropies_list = []
    # entropy_params_dict={}
    for i in EntropyTypes:
        if i.name in entropies or i.value in entropies:
            entropies_list.append(i)
    assert len(entropies_list) > 0, f'No entropy measures, {entropies}'
    queue = Queue(maxsize=number_of_jobs)
    process = [
        Process(target=create_sample_entropy_file, args=(queue, tag, entropies_list,entropies_params,multiscale_object,multiscale_object_params, use_derivative,use_v,use_s)) for i
        in range(number_of_jobs)]
    print('starting')
    for j, fp in enumerate(pathes):
        queue.put((fp, j + file_index_start))
        if j < len(process):
            process[j].start()

    if number_of_jobs > 1:
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
    parser.add_argument('-der', dest="use_derivative", type=str,
                        help='add_derivative', default='False')
    parser.add_argument('-mem', dest="memory", type=int,
                        help='set memory', default=-1)
    parser.add_argument('-e', dest="entropies", nargs='+', help='<Required> entropies type', required=False)
    parser.add_argument('-v', dest="use_v", help='use voltage',type=str2bool, default=True)
    parser.add_argument('-s', dest="use_s", help='use spike data',type=str2bool, default=True)
    parser.add_argument('-edp', dest="entropies_params", nargs='+', help='<Required> entropies type params', required=False,default=None)
    parser.add_argument('-m', dest="multiscale_object", help='Multiscale object if needed', required=False,default=None)
    parser.add_argument('-mp', dest="multiscale_object_params", help='Multiscale object params if needed', required=False,default=None)
    parser.add_argument('-ex', dest="files_that_do_not_exist", type=bool,
                        help='simulate only files that do not exist', default=False)
    args = parser.parse_args()

    use_derivative = not args.use_derivative.lower() in {"false", '0', ''}

    # use_derivative = not args.use_derivative.lower() in {"false", '0', ''}
    if 'entropies' not in args or args.entropies is None:
        entropies = [i.name for i in EntropyTypes]
    else:
        entropies = args.entropies
    print(args, f'entropies = {entropies}')

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
    jumps = len(list_dir_parent) // (number_of_clusters)
    modulu_res = len(list_dir_parent) % (number_of_clusters)
    keys = {}
    print(f'jumps {jumps} c_num = {number_of_clusters}')
    if args.memory > 0:
        keys['mem'] = args.memory
        print("Mem:", args.memory)
    if args.files_that_do_not_exist:
        files_that_exists = []
        for i in enumerate(os.listdir(os.path.join(ENTROPY_DATA_BASE_FOLDER, args.tag))):
            assert False,'implement'
            pass  # todo implemnt

    cur_start=0
    for i in range(number_of_clusters):
        end_point= cur_start+jumps+(i<modulu_res)
        pathes = list_dir_parent[cur_start:min( end_point, len(list_dir_parent))]

        print(len(pathes))
        # use_voltage = args.sv == 'v'
        print(range(cur_start,min(end_point, len(list_dir_parent))))

        job_factory.send_job(f"entropy_{args.tag}_{i}_{MAX_INTERVAL}d",
                             f'python -c "from create_entropy_score import get_entropy; get_entropy(' + "'" + args.tag + "'" + f',{pathes},{entropies},{i * jumps},{use_derivative},{ args.entropies_params}, {args.multiscale_object}, {args.multiscale_object_params},{args.use_v},{args.use_s})"',
                             **keys)
        print('job sent')

        cur_start = end_point
