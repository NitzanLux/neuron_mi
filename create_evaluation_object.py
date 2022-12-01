import os
import argparse
import random
import sys
from create_entropy_score import EntropyObject
from tqdm import tqdm
import pickle
# def create_dir_keys()
seed = random.randrange(sys.maxsize)
rng = random.Random(seed)
parser = argparse.ArgumentParser(description='Create evaluation object from file names')
parser.add_argument('-f', dest="folders", type=str,nargs='+',
                    help='folders_to_sample_from')
parser.add_argument('-s', dest="sim_num", type=int,
                    help='number of simulations')
parser.add_argument('-n', dest="name_to_save", type=str,
                    help='name to save')
args = parser.parse_args()
b_path='entropy_data'
keys = []
for i in args.folders:
    cur_path=os.path.join(b_path,i)
    keys.append(set(os.listdir(cur_path)))
i_keys = set.intersection(*keys)
d_keys = set.union(*keys)
d_keys = d_keys.symmetric_difference(d_keys)
d_keys_f = set([i[0] for i in d_keys])
keys = set()
for i in i_keys:
    if i[0] in d_keys_f or i in d_keys:  # if theres a sim from file that do not exists
        continue
    keys.add(i)
print(len(keys),args.sim_num)
sim_num=min(len(keys),args.sim_num)
sim_names = rng.sample(keys,sim_num)
data_dict=dict()
total_files=0
for i in tqdm(args.folders):
    cur_path=os.path.join(b_path,i)
    data_dict[i]=dict()
    for sn in sim_names:
        total_files+=1
        eo = EntropyObject.load(os.path.join(cur_path,sn))
        data_dict[i][eo.get_key()]=eo.to_dict()
with open(os.path.join(b_path,f'{args.name_to_save}_fnum{total_files}_seed{seed}.pkl'),'wb')as f:
    pickle.dump(data_dict,f)
