import os
from tqdm import tqdm
from create_entropy_score import EntropyObject
import re
from utils.evaluations_utils import ModelsSEData
def convert():
    m = re.compile('.*randseed_[0-9]+')
    for folder in tqdm(os.listdir('entropy_data')):
        print(folder)
        cur_path = os.path.join('entropy_data',folder)
        if os.path.isfile(cur_path):
            continue

        print(cur_path)
        # files=[]
        # for f in os.listdir(cur_path):
        #     eo = EntropyObject.load(os.path.join(cur_path,f))
        #     files.append(eo.file_name)

        for f in reversed(os.listdir(cur_path)):
            eo = EntropyObject.load(os.path.join(cur_path,f))
            pos= m.match(eo.file_name).regs[0][1]
            file_name = eo.file_name[:pos]
            os.rename(os.path.join(cur_path,f),os.path.join(cur_path,eo.generate_file_name_f(file_name,eo.sim_index)))
if __name__ == '__main__':
    # from utils import slurm_job
    # sl = slurm_job.SlurmJobFactory('cluster_logs')
    # sl.send_job("convert_names",'python -c "from convert_files_names_to_key import convert ; convert()"')
    convert()