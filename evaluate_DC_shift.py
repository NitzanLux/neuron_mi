import os
from scipy import sparse

length=6000
src=os.path.join('simulations','data','inputs')
dest=  os.path.join('simulations','data','short_inputs')
os.makedirs(os.path.join('simulations','data','short_inputs'))
for i in os.listdir(src):
    os.makedirs(os.path.join(dest,i))
    exc_weighted_spikes = sparse.load_npz(f'{os.path.join(src,i)}/exc_weighted_spikes.npz').A
    sparse.save_npz(f'{os.path.join(dest,i)}/exc_weighted_spikes.npz', sparse.csr_matrix(exc_weighted_spikes[:,:length]))
    inh_weighted_spikes = sparse.load_npz(f'{os.path.join(src,i)}/inh_weighted_spikes.npz').A
    sparse.save_npz(f'{os.path.join(dest,i)}/inh_weighted_spikes.npz', sparse.csr_matrix(inh_weighted_spikes[:,:length]))
