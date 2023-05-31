import numpy as np
import pandas as pd
import os
import pickle
# from .. import utils as u
from tqdm import tqdm
from utils.parse_file import get_spikes
from utils.slurm_job import SlurmJobFactory
def pack():
    data = dict()
    # for r,d,_ in os.walk(os.path.join("simulation","data")):
    # print([(i,p,j) for i,p,j in os.walk(os.path.join("simulations","data"))])
    base_path = os.path.join("simulations","data")
    for _,d,_ in os.walk(base_path):
        print(d)
        for d_m in tqdm(d):
            data[d_m] = dict()

            for r_d,d_f,_ in os.walk(os.path.join(base_path,d_m)):
                for d_ff in tqdm(d_f):

                    s = get_spikes(os.path.join(r_d,d_ff))
                    if s is not None:
                        data[d_m][d_ff] = s

    with open('spikes_datar.pkl','wb') as f:
        pickle.dump(data,f)

if __name__ == '__main__':
    s= SlurmJobFactory("cluster_logs")
    s.send_job_for_function("isi_stat",'simulations.evaluate_isi','pack',[])