import os
import re
from scipy import sparse
m = re.compile('[a-z]{3}_weighted_spikes.npz')
base_path= 'data'
for d_dir in os.listdir(base_path):
    d_dir_path=os.path.join(base_path,d_dir)
    for f_dir in os.listdir(d_dir_path):
        f_dir_path = os.path.join(d_dir_path,f_dir)
        to_com = list(filter(m.match,os.listdir(f_dir_path)))
        # to_com = map()
        if len(to_com)>0:
            for i in to_com:
                a = sparse.load_npz(os.path.join(f_dir_path,i))
                b = sparse.load_npz(os.path.join('data','input',i))
                if a==b:
                    continue
                print(f_dir,d_dir)
                break

