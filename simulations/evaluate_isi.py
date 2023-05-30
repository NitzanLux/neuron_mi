import numpy as np
import pandas as pd
import os
import pickle
# from .. import utils as u
from utils.parse_file import parse_sim_experiment_file
data = dict()
# for r,d,_ in os.walk(os.path.join("simulation","data")):
for r,_,_ in os.walk(os.path.join("simulation","data")):
    data[r]=dict()
    for f in r:
        _, s, v = parse_sim_experiment_file(f)
        data[r][f] = s
with open('spikes_datar.pkl','wb') as f:
    pickle.dump(data,f)