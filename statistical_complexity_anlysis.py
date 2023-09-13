import re
import os
import os
import re
import dash
from dash import dash_table

import matplotlib.pyplot as plt
import pandas as pd
# import mpmath
from utils.evaluations_utils import *
import numpy as np
# % matplotlib inline
import matplotlib
import seaborn as sns
from create_entropy_estimation import EntropyEstimation
import re
from typing import List, Iterable, Dict
from numpy import log
from scipy.special import betaln

AVARAGE_RATE_ROUND_FACTOR = 5
sns.set(font_scale=0.8)
import mpmath
from mpmath import mp, mpf,log

mp.dps = 50
# plt.rcParams['figure.figsize'] = [7, 7]

def get_ln_number_of_possible_states(k,n):
    n=int(n)
    k=int(k)
    # Assumes binom(n, k) >= 0
    # return -(betaln(1 + n - k, 1 + k) - log(n + 1))
    choose_n = mpf(0.)
    for i in range(n-k,n+1):
        choose_n+=mp.log(i,2)
    choose_k=1
    for i in range(1,k+1):
        choose_k+=mp.log(i,2)
    return (choose_n-choose_k)

def max_ent(l,n):
    return get_ln_number_of_possible_states(l,n)

def binary_ent(l, n):
    l, n = float(l), float(n)
    p = mpf(l) / mpf(n)
    # p = l / n
    # if p>=1. or  p<=0.:
    #     return 0.
    return -((1 - p) * log(1 - p,b=2) + (p) * log(p,b=2))


def binomial_ent(l, n):
    print(l, n)
    p = l / n
    return (1. / 2.) * np.log(2 * np.pi * np.exp(1) * n * p * (1 - p)) / n


file_name_regex = re.compile('(ID_[0-9]+_[0-9]+)')


class CTWHandler():

    def __init__(self):
        self.__data = dict()

    def add_data(self, path: Iterable[str]):
        # data:List[cee.EntropyEstimation]=[]
        for i in path:
            data = EntropyEstimation.load(i)
            if data.tag not in self.__data:
                self.__data[data.tag] = dict()
            if data.file_name not in self.__data[data.tag]:
                self.__data[data.tag][data.file_name] = dict()
            self.__data[data.tag][data.file_name][data.file_index] = data

    def get_item_by_file_name_and_index(self, f_name, index):
        return {k: v[f_name][index] for k, v in self.__data.items()}

    def __len__(self):
        conter = 0
        for kt, vt in self.__data.items():
            for kf, vf in vt.items():
                conter += len(vf)
        return conter

    def get_next_data(self):
        data_list = []
        for kt, vt in self.__data.items():
            for kf, vf in vt.items():
                for ki, vi in vf.items():
                    data_list.append(vi)  #,tree=vi.tree)
        return data_list

    def __iter__(self):
        for kt, vt in self.__data.items():
            for kf, vf in vt.items():
                for ki, vi in vf.items():
                    yield dict(tag=kt, file=kf, entropy=vi.entropy, rate=1000. * np.sum(vi.s) / vi.s.size,
                               n_spikes=np.sum(vi.s), l_segemnt=vi.s.size)  #,,tree=vi.tree)

    def to_df(self, tag_regex_to_remove: str | List[str] = '', normelize_entropy=True,
              group_dict: Dict[str, str] | None = None):
        key_data_arr = dict(full_tag=[], file=[], entropy=[], rate=[],
                            n_spikes=[], group=[], l_segemnt=[])  #tag=[], normalized_entropy=[])
        if isinstance(tag_regex_to_remove, str):
            tag_regex_to_remove = [tag_regex_to_remove]
        if group_dict is not None:
            group_dict = {k: re.compile(f'{v}') for k, v in group_dict.items()}
        for kv in self:
            for k, v in kv.items():
                if k == 'tag':
                    # disp_v = v
                    # for i in tag_regex_to_remove:
                    #     disp_v = re.sub(i, '', disp_v)
                    # disp_v = re.sub('([0-9]+-[0-9]+)',
                    #                 lambda x: x.group(1).replace('-', '.') if x.group(1) is not None else None,
                    #                 disp_v)  # replacing to decimal point
                    # key_data_arr['tag'].append(disp_v)
                    if group_dict is not None:
                        unique_flag = None
                        empty_flag = False
                        v_groups_data_for_warning = []
                        for k_group, v_group in group_dict.items():
                            if v_group.match(v):
                                v_groups_data_for_warning.append((v, v_group))
                                key_data_arr['group'].append(k_group)
                                if unique_flag is None:
                                    unique_flag = True
                                else:
                                    unique_flag = False
                                empty_flag = True
                        assert unique_flag is None or unique_flag, 'Keys of group are not unique ' + ' and '.join(
                            v_groups_data_for_warning)
                        assert empty_flag, 'one key has no group ' + v
                    else:
                        key_data_arr['group'].append('')
                    key_data_arr['full_tag'].append(v)
                elif k == 'file':
                    v = file_name_regex.match(v).group(1)
                    key_data_arr[k].append(v)
                elif k == 'entropy':

                    key_data_arr[k].append(v)
                else:
                    key_data_arr[k].append(v)

        df = pd.DataFrame(data=key_data_arr)
        # df['sim_id']=df['file']+'@'+df['sim_index']
        # df.drop(columns=['file','sim_index'],inplace=True)
        df['entropy']= df['entropy']*df['l_segemnt']
        return df

    def add_min_isi(self):

        data_arr = self.get_next_data()
        min_isi = {k: None for k in self.__data.keys()}
        for i in data_arr:
            indexes = np.where(i.s == 1)[0]
            cur_min_isi = np.min(indexes[1:] - indexes[:-1] - 1)
            if min_isi[i.tag] is None:
                min_isi[i.tag] = cur_min_isi
            elif min_isi[i.tag] > cur_min_isi:
                min_isi[i.tag] = cur_min_isi
        return min_isi

    def add_normalized(self, df, compress_to_minimal=False):
        tags = df['full_tag']

        if compress_to_minimal:
            min_isi_dict = self.add_min_isi()
            min_isi = np.array([min_isi_dict[t] for t in tags])
            print(f'min isi = {min_isi_dict}')
            df[''] = df[['n_spikes', 'l_segemnt']].apply(
                lambda x: max_ent(x[0], x[1] - (x[0] * min_isi)),axis=1)
                # lambda x: binary_ent(x[0], x[1] - (x[0] * min_isi)),axis=1)
        else:
            df['normalizing_factor'] = df[['n_spikes', 'l_segemnt']].apply(
                lambda x: max_ent(x[0], x[1]),axis=1)
                # lambda x: binary_ent(x[0]+1, x[1]),axis=1)
        # df['normalized_entropy'] = df['entropy'] / df['normalizing_factor']

        df['normalized_entropy'] = df['entropy']*df['l_segemnt'] / df['normalizing_factor']
        df['normalized_entropy'] = df['normalized_entropy']



    def add_log_prob(self,df):
        df['log_prob'] = -df['entropy'] * df['l_segemnt']

    def get_Q0(self,df):
        N=df['normalizing_factor'].apply(lambda x:mp.exp(x))

        denomenator = (1+mpf(1.)/N)*N.apply(lambda n :mp.log(n+1,2))-2*(mpf(1.)+df['normalizing_factor'])+df['normalizing_factor']
        df['Q0']=-mpf(2.)/denomenator
    def get_statistical_complexity(self,df):
        ans = -df['Q0']/df['l_segemnt']
        ans*=df['normalized_entropy']
        p_e=df['normalizing_factor']
        p=df['log_prob'].apply(lambda x:mp.exp(x))
        print("p",p)
        print("pe",p_e)
        # a=(2*(p*p_e).apply(lambda x:mp.sqrt(x)))
        # a= a[a==0]
        # print(a)
        ans*=((p+p_e).apply(lambda x:mp.log(x,2))-1.-0.5*(-df['normalizing_factor']+df['log_prob']))
        df['complexity']=ans
#set params for handler
models = {}
regex_match =re.compile(r".*_CTW")
gmax_value_regex = re.compile('([0-9](?:\.[0-9])?)[a-z,A-Z]*')
models_pathes = []
counter = 0
for i in os.listdir('entropy_data'):
    counter+=1
    cur_dir_path = os.path.join('entropy_data', i)
    if regex_match.match(i) and os.path.isdir(cur_dir_path) and 'current' not in i:
        for j in os.listdir(cur_dir_path):
            models_pathes.append(os.path.join(cur_dir_path, j))
    # if counter ==3:
    #     break
remove_v = True
#%% md

#%%

c = CTWHandler()
c.add_data(models_pathes)
df = c.to_df(
    tag_regex_to_remove=['Rat_L5b_PC_2_Hay_', '(?:_noNMDA_CTW)', 'noNMDA_', '_CTW', 'current_injection_synapses_',
                         ' Human_L23_PC_0603_11_937_Eyal_', 'morph_from_Rat_L5b_PC_2_Hay_', 'factor_'],
    group_dict={'Rat L5PC': 'Rat_L5b_PC_2_Hay_(?:(?:[0-9]-)?[0-9]|(?:DC_[0-9]{2}))_CTW',
                'Rat L5PC w/o NMDA': 'Rat_L5b_PC_2_Hay_(?:noNMDA_(?:[0-9]-)?[0-9]|(?:[0-9]-)?[0-9]_noNMDA)_CTW',
                'Human L23': 'Human_L23_PC_0603_11_937_Eyal_(?:[0-9]-)?[0-9]_CTW',
                'Rat L5PC DC': 'Rat_L5b_PC_2_Hay_factor_(?:(?:[0-9]-)?[0-9]|(?:DC_[0-9]{1,3}))_DC_-?[0-9]{1,3}_CTW',
                'Rat L5PC w/o NMDA DC': 'Rat_L5b_PC_2_Hay_noNMDA_factor_(?:(?:[0-9]-)?[0-9]|(?:DC_[0-9]{1,3}))_DC_-?[0-9]{1,3}_CTW',
                'Human L23 rat morph': 'Human_L23_PC_0603_11_937_Eyal_morph_from_Rat_L5b_PC_2_Hay_(?:[0-9]-)?[0-9]_CTW',
                'Current Synapse Rat L5PC': 'Rat_L5b_PC_2_Hay_current_injection_synapses_(?:[0-9]-)?[0-9]_CTW',
                'Current Synapse Rat L5PC w/o NMDA': 'Rat_L5b_PC_2_Hay_current_injection_synapses_(?:noNMDA_(?:[0-9]-)?[0-9]|(?:[0-9]-)?[0-9]_noNMDA)_CTW'}, )
for i in df['full_tag'].unique():
    df.loc[df['full_tag'] == i, 'Average Rate Per Parameters'] = np.round(df.loc[df['full_tag'] == i, 'rate'].mean(),
                                                                          AVARAGE_RATE_ROUND_FACTOR)

get_factor_from_tag_match = re.compile('^.*(?!DC)_([0-9](?:(?:\-|\.)[0-9]+)?)[^0-9\s]{3}.*$')
get_dc_from_tag_match = re.compile('^.*DC_(-?[0-9]{1,3})[^0-9\s]*$')


def convert_tag_to_weight_factor(x):
    a = get_factor_from_tag_match.match(x)
    # print(x)

    if a is None:
        return 1.
    a = a.group(1)
    a = a.replace('-', '.')
    # if a != x: print(x, a)

    return float(a)


def convert_tag_to_dc_factor(x):
    a = get_dc_from_tag_match.match(x)
    if a is None:
        return -70
    a = a.group(1)
    sign = int('-' not in a) * 2 - 1
    a = a.replace('-', '')
    # if a != x: print(x, a)
    return sign * float(a)


df['factor'] = df['full_tag'].apply(convert_tag_to_weight_factor)
df['DC_factor'] = df['full_tag'].apply(convert_tag_to_dc_factor)
df.drop_duplicates()
print("drop duplicates")
# factor_order = {i: convert_tag_to_weight_factor(i) for i in df['tag'].unique()}
df['DC_test'] = df['group'].apply(lambda x: True if 'DC' in x else False)
#%%
c.add_normalized(df, False)
c.get_Q0(df)
print('Q0')
c.add_log_prob(df)
