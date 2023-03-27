from zipfile import ZipFile
import os
import shutil
from utils.slurm_job import SlurmJobFactory
def pack_data():
    src_path =os.path.join('simulations', 'data')
    dest_path='data'
    os.makedirs(dest_path)
    sims = ["ID_66_97136","ID_84_148396","ID_22_66248","ID_74_556493"]
    for m in os.listdir(src_path):
        temp_path = os.path.join(src_path, m)
        for o_sim in os.listdir(os.path.join(temp_path)):
            sim=o_sim.replace('_'+m,'')
            # os.makedirs(os.path.join(dest_path,sim),exist_ok=True)
            for f in os.listdir(os.path.join(temp_path,o_sim)):
                if '.h5' in f:
                    os.makedirs(os.path.join(dest_path, sim, m), exist_ok=True)
                    shutil.copyfile(os.path.join(src_path,m,o_sim,f),os.path.join(dest_path,sim,m,f))
                    print(os.path.join(src_path,m,o_sim,f),os.path.join(dest_path,sim,m,f))
    shutil.make_archive(f'{dest_path}', 'zip', dest_path)
    shutil.rmtree(dest_path)

s = SlurmJobFactory('cluster_logs')
s.send_job_for_function('packing_data..','pack_data_to_my_pc','pack_data',[])