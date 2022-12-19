import os
import argparse
from pathlib import Path
import re
import shutil

parser = argparse.ArgumentParser(description='Simulate a neuron')
parser.add_argument('--path', type=str, default=1)  # for general runinng
args = parser.parse_args()
dest_dir = str(Path(args.path).parent.absolute())
# os.makedirs('')
m = re.compile('ID_[0-9]+_[0-9]+')
for i in os.listdir(args.path):
    ID = m.match(i).group(0)
    os.makedirs(os.path.join(dest_dir,'inputs',ID),exist_ok=True)
    shutil.move(os.path.join(args.path,i,'inh_weighted_spikes.npz'),os.path.join(dest_dir,'inputs',ID,'inh_weighted_spikes.npz'))
    shutil.move(os.path.join(args.path,i,'exc_weighted_spikes.npz'),os.path.join(dest_dir,'inputs',ID,'exc_weighted_spikes.npz'))
