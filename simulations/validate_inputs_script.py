import os
import re
import numpy as np
from scipy import sparse
m = re.compile('[a-z]{3}_weighted_spikes.npz')
m_id = re.compile('ID_[0-9]+_[0-9]+')

base_path= 'data'
for d_dir in os.listdir(base_path):
    d_dir_path=os.path.join(base_path,d_dir)
    for f_dir in os.listdir(d_dir_path):
        f_dir_path = os.path.join(d_dir_path,f_dir)
        to_com = list(filter(m.match,os.listdir(f_dir_path)))
        # to_com = map()
        if len(to_com)>0:
            for i in to_com:
                ID= m_id.match(f_dir).group(0)
                a = sparse.load_npz(os.path.join(f_dir_path,i))
                a[a>0]=1
                b = sparse.load_npz(os.path.join('data','inputs',ID,i))
                b[b>0]=1
                if np.all(a==b.data):
                    continue
                print(f_dir,d_dir)
                break

