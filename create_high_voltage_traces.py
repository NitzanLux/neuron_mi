import os
import numpy as np
def create_histograms(input_file_path):
    input_file_name=os.path.basename(input_file_path)
    base_path=os.path.join('simulations', 'data')
    dest_path=os.path.join('simulations','high_res_input',input_file_name)
    os.makedirs(dest_path,exist_ok=True)
    models=[]
    for model_name in os.listdir(base_path):
        print(f'Start {model_name}')
        if os.path.isfile(model_name):
            continue
        model_path = os.path.join(base_path, model_name)
        dir_flag=False
        for i in os.listdir(model_path):
            cur_path=os.path.join(model_path,i)
            if os.path.isfile(cur_path):
                continue
            dir_flag=True
        if dir_flag:
            models.append(model_name)
    print('\n'.join(models))