import argparse
import os
import sys

from utils.parse_file import parse_sim_experiment_file
from utils.slurm_job import SlurmJobFactory
import pandas as pd
import numpy as np
import argparse
def create_summery():
    parser = argparse.ArgumentParser(description='Add folder name for summery simulation data data')
    parser.add_argument('-f', dest="data_folder_name", type=str,
                        help='directory name for evaluation')
    args = parser.parse_args()
    data_folder_name = args.data_folder_name
    cur_path = os.path.join('simulations',data_folder_name)
    rate = []
    avarage_somatic_voltage = []
    model_name=[]
    sim_name=[]
    for m in os.listdir(cur_path):
        print(m)
        if 'input' in m:
            continue
        for i in os.listdir(os.path.join(cur_path,m)) :
            try:
                _, y_spike, y_soma  = parse_sim_experiment_file(os.path.join(cur_path,m,i))
                print(y_spike.sum(),y_spike.size)
                rate.append(1000.* y_spike.sum() / y_spike.size)
                avarage_somatic_voltage.append(y_soma.mean())
                model_name.append(m)
                sim_name.append(i)
            except:
                print(f"Could not load {i} from model {m}")
    df = pd.DataFrame({'model':model_name,'simulation':sim_name,'soma_average_voltage':avarage_somatic_voltage,'rate':rate})
    os.makedirs(os.path.join('simulations','dataframe_data'),exist_ok=True)
    with open(os.path.join('simulations','dataframe_data',f'{data_folder_name}_{np.random.randint(100000)}.pkl'),'wb') as f:
        df.to_pickle(f)

s = SlurmJobFactory('cluster_logs')

s.send_job_for_function('summery_creator','summery_creator_data_from_simulations','create_summery',sys.argv[1:])
