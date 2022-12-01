import os
from tqdm import tqdm
from create_entropy_score import EntropyObject
import re
from utils.evaluations_utils import ModelsSEData
m = re.compile('.*randseed_[0-9]+')
for folder in tqdm(os.listdir('entropy_data')):
    if not os.path.isdir(folder):
        continue
    cur_path = os.path.join('entropy_data',folder)
    files=[]
    for f in os.listdir(cur_path):
        eo = EntropyObject.load(os.path.join(cur_path,f))
        files.append(eo.file_name)

    for f in os.listdir(cur_path):
        eo = EntropyObject.load(os.path.join(cur_path,f))
        pos= m.match(eo.file_name).regs[1]
        m = eo.file_name[:pos]
        os.rename(os.path.join(cur_path,f),os.path.join(cur_path,eo.generate_file_name()+'.pkl'))