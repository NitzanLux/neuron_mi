import argparse
import os
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from utils.parse_file import parse_sim_experiment_file
parser = argparse.ArgumentParser(description='Regex expression for zipping different files')
parser.add_argument('-r', dest="regex_expression", type=str,
                    help='Regex expression for zipping files.',required=True)
parser.add_argument('-d', dest="dir", type=str,
                    help='Destination path', default=os.path.join('simulations','data'))
parser.add_argument('-f', dest="index", type=int,
                    help='file_index',default=0)
parser.add_argument('-i', dest="interval_length", type=int,
                    help='interval length',default=1000)
parser.add_argument('-s', dest="start_time", type=int,
                    help='starting time',default=1000)
args = parser.parse_args()

m = re.compile(args.regex_expression)
files=[]
for i in os.listdir(args.dir):
    if m.match(i) and os.path.isdir(os.path.join(args.dir,i)) and i!='inputs':
        files.append(i)

data_set=None
# f_dict=dict()
for f in files:
    print(f)
    simulations_data=[i.replace(f,'') for i in os.path.join(args.dir,f)]
    if data_set is None:
        data_set=set( simulations_data)
    else:
        data_set = data_set.intersection(set( simulations_data))
assert len(data_set)>0,"joint files were not found"
print(data_set)
cur_sim=list(data_set)[args.index]
print('current simulation: ',cur_sim)
cwd = os.getcwd()
cur_working_dir=os.path.join(cwd,args.dir)
# os.chdir(os.path.join(cwd,args.dir))
for f in files:
    print(f)
    f_path = os.path.join(cur_working_dir,f,cur_sim+f)
    _, y_spike, y_soma = parse_sim_experiment_file(f_path)
    x = np.where(y_spike==1)[0]
    plt.plot(y_soma[args.start_time:args.start_time+args.interval_length])
    plt.scatter(x,y_soma[args.start_time:args.start_time+args.interval_length][x],colort = 'red')
    cur_dir=f'{cur_sim}_{args.start_time}_{args.interval_length}'
    os.makedirs(cur_dir,exist_ok=True)
    plt.savefig(os.path.join(cur_dir,f'{f}.png'))
    plt.close()

    # p = subprocess.Popen(f"zip -r {i}.zip {i}", stdout=subprocess.PIPE, shell=True)
    # print(p.communicate())
