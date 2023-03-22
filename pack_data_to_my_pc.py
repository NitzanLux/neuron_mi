from zipfile import ZipFile
import os
import shutil
src_path =os.path.join('simulations', 'slim_data')
dest_path='slim_data'
os.makedirs(dest_path)
for m in os.listdir(src_path):
    temp_path = os.path.join(src_path, m)
    for sim in os.path.join(temp_path):
        os.makedirs(os.path.join(dest_path,sim),exist_ok=True)
        os.makedirs(os.path.join(dest_path, sim,m), exist_ok=True)
        for f in os.listdir(os.path.join(temp_path,sim)):
            if '.h5' in f:
                shutil.move(os.path.join(src_path,m,sim,f),os.path.join(dest_path,sim,m,f))
shutil.make_archive(f'{dest_path}.zip', 'zip', dest_path)
shutil.rmtree(dest_path)
