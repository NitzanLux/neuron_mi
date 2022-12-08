# %%
import copy

from scipy.stats import ttest_ind
import os
import dask.\
    array as da
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib
from matplotlib import colors
import pandas as pd
from scipy.stats import linregress
from create_entropy_score import EntropyObject
from itertools import combinations
import random
import warnings
import re
warnings.simplefilter(action='ignore', category=FutureWarning)

MSX_INDEX = 0
COMPLEXITY_INDEX = 1
FILE_INDEX = 2
SIM_INDEX = 3
SPIKE_NUMBER = 4
m = re.compile('.*randseed_[0-9]+')

def combination_sorting(orderd_models):
    comb_set = set()
    for i in combinations(np.arange(len(orderd_models)),2):
        sorted(i)
        if i[0]==i[1]:
            continue
        if i in comb_set:
            continue
        comb_set.add(i)
        yield i


def save_large_plot(fig, name, tags):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    cur_path = os.path.join('plots', str(tags))
    if not os.path.exists(cur_path):
        os.mkdir(cur_path)
    if '.' in name:
        name = f"{name[:name.find('.')]}_{tags}_{name[name.find('.'):]}"
    else:
        name = f"{name}_{tags}"
    fig.savefig(os.path.join(cur_path, name))


class ModelsSEData():
    def __init__(self, tags=None,data_dict=None):
        if data_dict is not None:
            self.__construct_from_data(data_dict)
        elif tags is not None:
            self.__construct_from_tags(tags)

    def __construct_from_data(self,data_dict):
        self.data_tags = set(data_dict.keys())
        self.data=dict()
        keys=[]
        for dn,d in data_dict.items():
            self.data[dn]=dict()
            temp_set=set()
            for k,v in d.items():
                eo = EntropyObject(**v)
                # print(k)
                # for j, v in enumerate(entropy_list):
                pos = m.match(eo.file_name).regs[0][1]
                eo.file_name = eo.file_name[:pos]
                # print(eo.file_name)
                temp_set.add(eo.get_key())

                self.data[dn][eo.get_key()]=eo
            keys.append(temp_set)
        i_keys = set.intersection(*keys)
        print(len(i_keys))

        d_keys = set.union(*keys)
        print(len(d_keys))
        # d_keys = d_keys.symmetric_difference(d_keys)
        d_keys = d_keys - i_keys
        print(len(d_keys))
        self.keys = set()
        for i in i_keys:
            if i in d_keys:  # if theres a sim from file that do not exists
                continue
            self.keys.add(i)

        self.entropy_keys = set()

        for i in self.data.values():
            for j in i.values():
                self.entropy_keys.update(set(j.get_entropy_dict().keys()))
    def __construct_from_tags(self,tags):
        self.data_tags = [(i[:-len('.pkl')] if i.endswith('.pkl') else i) for i in tags]
        self.data = dict()
        self.entropy_keys = set()
        keys=[]
        for i in tqdm(self.data_tags):
            # files=[]
            entropy_list = list(EntropyObject.load_list(os.path.join('entropy_data', i + '.pkl')).values())
            # for j in entropy_list:
            #     files.append(j.file_name)
            # suff = self.find_suffix_shared(files)
            for j,v in enumerate(entropy_list):
                pos= m.match(v.file_name).regs[1]
                entropy_list[j].file_name = v.file_name[:pos]
            keys.append({(v.file_name,v.sim_index) for v in entropy_list})
        i_keys = set.intersection(*keys)
        d_keys = set.union(*keys)
        d_keys = d_keys - i_keys
        d_keys_f = set([i[0] for i in d_keys])
        self.keys = set()
        for i in i_keys:
            if i[0] in d_keys_f or i in d_keys:  # if theres a sim from file that do not exists
                continue
            self.keys.add(i)
    def load_data(self,ratio):
        self.sample_from_set(ratio)
        for i in tqdm(self.data_tags):
            self.data[i]=dict()
            files=[]
            entropy_list = EntropyObject.load_list(os.path.join('entropy_data', i + '.pkl'))
            entropy_list = list(entropy_list.values())
            # for j in entropy_list:
            #     files.append(j.file_name)
            # suff = self.find_suffix_shared(files)
            for j,v in enumerate(entropy_list):
                entropy_list[j].file_name = v.file_name[:]
            for v in entropy_list:
                if (v.file_name,v.sim_index) in self.keys:
                    self.data[i][(v.file_name,v.sim_index)]=v
        for i in self.data.values():
            for j in i.values():
                self.entropy_keys.update(set(j.get_entropy_dict().keys()))

    def __len__(self):
        return len(self.keys)
    def sample_from_set(self,key_ratio=0.1):
        self.keys = random.sample(tuple(self.keys),int(len(self.keys)*key_ratio))
    @staticmethod
    def find_suffix_shared(files):
        base_str = ''
        pointer = -1
        cur_letter = None
        while True:
            for i in files:
                if cur_letter is None:
                    cur_letter = i[pointer]
                if i[pointer] != cur_letter:
                    break
            else:
                pointer -= 1
                base_str = cur_letter + base_str
                cur_letter = None
                continue
            break
        return base_str

    def get_by_shard_keys(self, key):
        assert key in self.keys, f'key is not shard amoung all , {key}'
        return {k: v[key] for k, v in self.data.items()}

    def iter_by_keys(self):
        for i in self.keys:
            yield self.get_by_shard_keys(i)

    def __iter__(self):
        """
        :return: modeltype ,file index , sim index , v
        """
        for dk, dv in self.data.items():
            for k, v in dv.items():
                yield [dk] + list(k) + list(v)

    def __iter_only_by_shard_keys(self):
        for i in self.keys:
            shard_keys = self.get_by_shard_keys(i)
            for k, v in shard_keys.items():
                yield k,v

    def get_as_dataframe(self, is_shared_keys=True):
        model_list = []
        file_list = []
        sim_list = []
        complexity_list = []
        msx_list = []
        spike_list = []
        voltage_list = []
        entropy_dict={}
        if is_shared_keys:
            generator = self.__iter_only_by_shard_keys()
        else:
            generator = self
        for k,v in tqdm(generator):
            model_list.append(v.tag)
            file_list.append(v.file_name)
            sim_list.append(v.sim_index)
            cur_entropy_dict = v.get_entropy_dict()
            if len(entropy_dict)==0:
                for e_k in cur_entropy_dict.keys():
                    entropy_dict[e_k]=[]
            for ent_k,ent_v in cur_entropy_dict.items():
                if ent_k not in entropy_dict:
                    entropy_dict[ent_k]=[None]*(len(file_list)-1)
                entropy_dict[ent_k].append(ent_v)

            spike_list.append(v.s)
            voltage_list.append(v.v)
        input_data={'model': model_list, 'file': file_list, 'sim_ind': sim_list,
                  'spikes': spike_list,'voltage':voltage_list}
        input_data.update(entropy_dict)
        df = pd.DataFrame(data=input_data)
        model_names = df['model'].unique()
        # print(df['file'])
        # print(df['sim_ind'].astype('str'))
        df['key'] = df['file'] + '#$#' + df['sim_ind'].astype('str')
        # df = pd.get_dummies(df, columns=['model'])
        return df, model_names.tolist()


def get_df_with_condition_balanced(df, condition, negate_condition=False):
    condition_files = df[condition]['key']
    if negate_condition:
        df = df[~df['key'].isin(condition_files)]
    else:
        df = df[df['key'].isin(condition_files)]
    return df
    # fi, c = np.unique(df['key'], return_counts=True)
