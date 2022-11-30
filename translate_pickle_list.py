import os
import pickle
from utils.evaluations_utils import ModelsSEData as MSED
from create_entropy_score import EntropyObject
base_path = 'entropy_data'
from tqdm import tqdm
for i in tqdm(os.listdir(base_path)):
    if i.endswith('.pkl'):
        print(i,flush=True)
        dir_path=os.path.join(base_path,i[:-len('.pkl')])
        os.mkdir(dir_path)
        l = EntropyObject.load_list(os.path.join(base_path,i))
        files=[]
        for j in l.values():
            files.append(j.file_name)
        suff = MSED.find_suffix_shared(files)
        for eo in l.values():
            with open(os.path.join(dir_path,f'{eo.file_name[:-len(suff)]}_{eo.sim_index}.pkl'),'wb') as feo:
                pickle.dump(eo.to_dict(),feo)

