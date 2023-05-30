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
for r,d,_ in os.walk(base_path):
    data[d]=dict()
    print(d)
    for f in d:
        _, s, v = parse_sim_experiment_file(os.path.join(base_path,f))
        # print(s)
        data[d][f] = s
with open('spikes_datar.pkl','wb') as f:
    pickle.dump(data,f)