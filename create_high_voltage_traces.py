import os
import numpy as np
from threading import Thread

def high_res_for_model_creator(model_name, input_file_path='', destination_path=''):
    exec(f'python dummy_scripy.py --simulation_folder {model_name}')
    print('wattt')
def high_res_maneger(input_file_path=''):
    # input_file_name=os.path.basename(input_file_path)
    base_path=os.path.join('simulations', 'data')
    # dest_path=os.path.join('simulations','high_res_input',input_file_name)
    # os.makedirs(dest_path,exist_ok=True)
    models=[]
    for model_name in os.listdir(base_path):
        print(f'Start {model_name}')
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
        high_res_for_model_creator(i)

if __name__ == '__main__':
    high_res_maneger()