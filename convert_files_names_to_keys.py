import os
from tqdm import tqdm
from create_entropy_score import EntropyObject
from utils.evaluations_utils import ModelsSEData
for folder in tqdm(os.listdir('entropy_data')):
    if not os.path.isdir(folder):
        continue
    cur_path = os.path.join('entropy_data',folder)
    files=[]
    for f in os.listdir(cur_path):
        eo = EntropyObject.load(os.path.join(cur_path,f))
        files.append(eo.file_name)

    suffix=  ModelsSEData.find_suffix_shared(files)
    print(suffix)
    if len(suffix)==0:
        suffix=None
    for f in os.listdir(cur_path):
        eo = EntropyObject.load(os.path.join(cur_path,f))
        os.rename(os.path.join(cur_path,f),os.path.join(cur_path,eo.generate_file_name(suffix)+'.pkl'))