import os
from utils.parse_file import parse_sim_experiment_file
from utils.slurm_job import SlurmJobFactory
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
def create_histograms():
    base_path=os.path.join('simulations', 'data')
    dest_path=os.path.join('simulations','rate_hists')
    os.makedirs(dest_path,exist_ok=True)
    for model_name in os.listdir(base_path):
        print(f'Start {model_name}')
        if os.path.isfile(model_name):
            continue
        model_path = os.path.join(base_path, model_name)
        if model_name=='inputs':
            continue
        data=[]
        for i in tqdm(os.listdir(model_path)):
            cur_path=os.path.join(model_path,i)
            if os.path.isfile(cur_path):
                continue
            _, y_spike, _ = parse_sim_experiment_file(cur_path)
            # data['file_name'].append(i)
            # data['model_name'].append(model_simulations_dir)
            data.append(1000.*np.sum(y_spike)/y_spike.size)
        if len(data)==0:
            continue
        data = np.array(data)
        ax=plt.subplot()
        ax.hist(data, density=True)
        ax.set_xlabel('rate(Hrz)')
        ax.set_xlabel('Probability')
        ax.set_title(f'{model_name} \n std = {np.round(np.std(data),3)} mean = {np.round(np.mean(data),2)} med = {np.round(np.median(data),2)} (n = {len(data)}) ')
        plt.savefig(os.path.join(dest_path,f'{model_name}_rate_histogram.png'))
        plt.show()
if __name__ == '__main__':
    a=SlurmJobFactory('cluster_logs')
    a.send_job('hist_maker',f'python -c "from create_histogram_to_all_simulation_data import create_histograms; create_histograms()"')