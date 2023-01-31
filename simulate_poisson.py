import numpy as np
import matplotlib.pyplot as plt
from entropy import *
from scipy.stats import poisson,binom
import os
poisson_ent = lambda x: poisson(x).entropy()
binomial_ent = lambda p,n: (1./2.)*np.log(2*np.pi*np.exp(1)*n*p*(1-p))
# binary_maximum_ent = lambda p: -((1-p)*np.log2(1-p)+(p)*np.log2(p))*n
# n= 60
def create_spike_trains(l,size):
    d = np.zeros((size,))
    e = np.random.poisson(l,size)
    e = np.cumsum(e)
    e=e[e<d.shape[0]]
    d[e]=1
    return d
def simulate_poisson(number):
    N = [2500,5000,10000]#,20000,40000]
    l = [1,5,10,2000]#, 100)
    anlytical=[]
    for i in l:
        y = []
        for j,n in enumerate(N):
            if j==0:
                anlytical.append(poisson_ent(i))
            s=create_spike_trains(i,n)
            ctw=CTW()
            ctw.insert_pattern(s)
            y.append(ctw.get_entropy(n))
        plt.plot(l,y,label=f'{str(i)}')
    plt.legend()
    plt.savefig(os.path.join('plots',f'results_{number}.png'))

if __name__ == '__main__':
    from utils.slurm_job import SlurmJobFactory
    number=np.random.randint(0,1e+7)
    job_factory = SlurmJobFactory("cluster_logs")
    job_factory.send_job(f"entropy_poiss_{number}",
                         f'python -c "from simulate_poisson import simulate_poisson; simulate_poisson({number})"')
