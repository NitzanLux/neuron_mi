import numpy as np
import pandas as pd
import os
import pickle
# from .. import utils as u
from utils.parse_file import parse_sim_experiment_file
data = dict()
# for r,d,_ in os.walk(os.path.join("simulation","data")):
# print([(i,p,j) for i,p,j in os.walk(os.path.join("simulations","data"))])
base_path = os.path.join("simulations","data")
for _,d,_ in os.walk(base_path):
    print(d)
    for d_m in d:
        data[d_m] = dict()

        for r_d,d_f,_ in os.walk(os.path.join(base_path,d_m)):
            for d_ff in d_f:
                # for r,d_a,f in os.walk(os.path.join(base_path,d_f)):
                _, s, v = parse_sim_experiment_file(os.path.join(r_d,d_ff))
                # print(s)
                data[d_m][d_ff] = s
with open('spikes_datar.pkl','wb') as f:
    pickle.dump(data,f)