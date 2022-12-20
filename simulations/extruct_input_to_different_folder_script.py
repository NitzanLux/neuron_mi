import os
import argparse
from pathlib import Path
import re
import shutil
import time
parser = argparse.ArgumentParser(description='Simulate a neuron')
parser.add_argument('--path', type=str, default=1)  # for general runinng
args = parser.parse_args()
dest_dir = str(Path(args.path).parent.absolute())
# os.makedirs('')
m = re.compile('ID_[0-9]+_[0-9]+')

#clean faulted files
for i in os.listdir(args.path):
    if os.path.exists(os.path.join(args.path,i,'voltage.h5')):
        if os.path.exists(os.path.join(args.path,i,'inh_weighted_spikes.npz')) and\
            os.path.exists(os.path.join(args.path,i,'exc_weighted_spikes.npz')):
            continue
        else:
            ID = m.match(i).group(0)
            if os.path.exists(os.path.join(dest_dir,'inputs',ID,'inh_weighted_spikes.npz')) and \
                    os.path.exists(os.path.join(dest_dir,'inputs',ID,'inh_weighted_spikes.npz')):
                continue

    else:
        print(f'Should I delete {os.path.join(args.path,i)}?...',flush=True)
        time.sleep(0.5)
        list_dir = os.listdir(os.path.join(args.path,i))
        out_str=[]
        for j,f in enumerate(list_dir):
            file_stats = os.stat(os.path.join(args.path,i,f))
            out_str.append(f"({j})\t{f} with size: {file_stats.st_size} KB")
        out_str = "\n".join(out_str)
        print(f'The files inside are: {out_str}')
        time.sleep(5)
        print('Delete y/n')
        res=input()
        while(res not in {'y','n'}):
            res = input()
        if res=='y':
            shutil.rmtree(os.path.join(args.path,i))

#move input files
for i in os.listdir(args.path):
    ID = m.match(i).group(0)
    os.makedirs(os.path.join(dest_dir,'inputs',ID),exist_ok=True)
    shutil.move(os.path.join(args.path,i,'inh_weighted_spikes.npz'),os.path.join(dest_dir,'inputs',ID,'inh_weighted_spikes.npz'))
    shutil.move(os.path.join(args.path,i,'exc_weighted_spikes.npz'),os.path.join(dest_dir,'inputs',ID,'exc_weighted_spikes.npz'))
