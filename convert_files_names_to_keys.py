import os
from tqdm import tqdm
from create_entropy_score import EntropyObject
for folder in tqdm(os.listdir('entropy_data')):
    if not os.path.isdir(folder):
        continue
    cur_path = os.path.join('entropy_data',folder)
    for f in os.listdir(cur_path):
        eo = EntropyObject.load(os.path.join(cur_path,f))
        os.rename(os.path.join(cur_path,f),os.path.join(cur_path,eo.generate_file_name()+'.pkl'))