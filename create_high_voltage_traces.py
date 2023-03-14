import os
import numpy as np
from threading import Thread
import shlex, subprocess
from simulations.simulate_neuron import run_within_python_without_slurm
import re
namespace_vars=re.compile('(Namespace\([^\n]+\))')
def high_res_for_model_creator(model_name, input_file_name, destination_path=''):
    if not os.path.exists(os.path.join('simulations','data',model_name,f'{input_file_name}_{model_name}',f'{input_file_name}_{model_name}.out')):
        return
    with open(os.path.join('simulations','data',model_name,f'{input_file_name}_{model_name}',f'{input_file_name}_{model_name}.out'),'r') as f:
        out_data = f.readlines()
    out_data = '\n'.join(out_data)
    m = namespace_vars.search(out_data)
    args=None
    d=None
    if m:
        args = m.group(1)
        args = args.replace('Namespace','dict')
        print(args)
        command=f"d = {args}"
        eval(command,globals(),locals())
    else:
        return
    # args = shlex.split(f'python -m dummy_scripy.py --simulation_folder {model_name}')
    print('wattt',type(d))
def high_res_maneger(input_file_name):
    base_path=os.path.join('simulations', 'data')
    # dest_path=os.path.join('simulations','high_res_input',input_file_name)
    # os.makedirs(dest_path,exist_ok=True)
    models=[]
    for model_name in os.listdir(base_path):
        if os.path.isfile(model_name):
            continue
        model_path = os.path.join(base_path, model_name)
        dir_flag=False
        if model_name=='inputs':
            continue
        for i in os.listdir(model_path):

            cur_path=os.path.join(model_path,i)
            if os.path.isfile(cur_path):
                continue
            dir_flag=True
        if dir_flag:
            models.append(model_name)
    for i in models:
        high_res_for_model_creator(i,input_file_name)

if __name__ == '__main__':
    high_res_maneger("ID_0_512971")