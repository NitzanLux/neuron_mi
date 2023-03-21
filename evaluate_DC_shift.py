import os
from scipy import sparse
from subprocess import Popen
import sys
import numpy as np

def create_short_inputs(length=6000):
    src=os.path.join('simulations','data','inputs')
    dest=  os.path.join('simulations','data','short_inputs')
    os.makedirs(os.path.join('simulations','data','short_inputs'),exist_ok=True)
    for i in os.listdir(src):
        os.makedirs(os.path.join(dest,i),exist_ok=True)
        exc_weighted_spikes = sparse.load_npz(f'{os.path.join(src,i)}/exc_weighted_spikes.npz').A
        sparse.save_npz(f'{os.path.join(dest,i)}/exc_weighted_spikes.npz', sparse.csr_matrix(exc_weighted_spikes[:,:length]))
        inh_weighted_spikes = sparse.load_npz(f'{os.path.join(src,i)}/inh_weighted_spikes.npz').A
        sparse.save_npz(f'{os.path.join(dest,i)}/inh_weighted_spikes.npz', sparse.csr_matrix(inh_weighted_spikes[:,:length]))
def create_short_simulations(models,factors, dc_range,data_folder='data',input_dir='inputs'):
    assert len(factors)==len(models)
    for dc in dc_range:
        for mc,f in zip(models,factors):
            # dc=abs(dc)
            f_str=str(f).replace('.','-')
            commend = f"python3 -m simulations.simulate_neuron --neuron_model_folder simulations/neuron_models/{mc} --simulation_folder /ems/elsc-labs/segev-i/nitzan.luxembourg/projects/neuron_mi/neuron_mi/simulations/{data_folder}/{mc}_factor_{f_str}_DC_{dc}  --DC_shift {dc}  --weight_scale_factor {f} --save_plots True --simulation_duration_in_seconds 60 --input_dir '/ems/elsc-labs/segev-i/nitzan.luxembourg/projects/neuron_mi/neuron_mi/simulations/{data_folder}/{input_dir}/'"
            print(commend)
            # s.send_job(f'simulation_{mc}_{dc}_{f}',commend)
            process = Popen(commend, shell=True)
            stdout, stderr = process.communicate()
            print(stdout, file=sys.stdout)
            print(stderr, file=sys.stderr)
def create_entropy_approximation(models,factors, dc_range):
    assert len(factors)==len(models)
    for dc in dc_range:
        for mc,f in zip(models,factors):
            # dc=abs(dc)
            f_str=str(f).replace('.','-')
            command=f"python -m create_entropy_estimation -f simulations/data/{mc}_factor_{f_str}_DC_{dc}  -t {mc}_factor_{f_str}_DC_{dc}_CTW -j 0 -e True"
            print(command)
            # s.send_job(f'simulation_{mc}_{dc}_{f}',commend)
            process = Popen(command, shell=True)
            stdout, stderr = process.communicate()
            print(stdout, file=sys.stdout)
            print(stderr, file=sys.stderr)
create_short_simulations(['Rat_L5b_PC_2_Hay']*2,[0.25,0.5,0.1],list(range(-90,-30,10)),data_folder='short_data')