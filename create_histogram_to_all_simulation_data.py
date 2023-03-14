import os
from utils.parse_file import parse_sim_experiment_file
import numpy as np
import matplotlib.pyplot as plt
base_path=os.path.join('simulations', 'data')
dest_path=os.path.join('simulations','rate_hists')
os.makedirs(dest_path,exist_ok=True)
for model_name in os.listdir(base_path):
    if os.path.isfile(model_name):
        continue
    model_path = os.path.join(base_path, model_name)
    data=[]
    for i in os.listdir(model_path):
        cur_path=os.path.join(model_path,i)
        if os.path.isfile(cur_path):
            continue
        _, y_spike, _ = parse_sim_experiment_file(cur_path)
        # data['file_name'].append(i)
        # data['model_name'].append(model_simulations_dir)
        data.append(1000.*np.sum(y_spike)/y_spike.size)
    if len(data)==0:
        continue
    plt.hist(data, density=True)
    plt.xlabel('rate')
    plt.title(f'{model_name} (n = {len(data)})')
    plt.savefig(os.path.join(dest_path,f'{model_name}_rate_histogram.png'))