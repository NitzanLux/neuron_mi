import numpy as np
import matplotlib.pyplot as plt
from entropy import *
from scipy.stats import poisson,binom
import os
poisson_ent = lambda x: poisson(x).entropy()/x
binomial_ent = lambda p,n: (1./2.)*np.log(2*np.pi*np.exp(1)*n*p*(1-p))
# binary_maximum_ent = lambda p: -((1-p)*np.log2(1-p)+(p)*np.log2(p))*n
# n= 60
# def create_spike_trains(l,size):
#     d = np.zeros((size,))
#     e = np.random.poisson(l,size)
#     e = np.cumsum(e)
#     e=e[e<d.shape[0]]
#     d[e]=1
#     return d
def binary_ent(l, n):
    p=l/n
    return -((1-p)*np.log2(1-p)+(p)*np.log2(p))
def create_spike_trains_r(l,size,time_interval=1000):
    # l = size/l
    if time_interval is not None:
        s=time_interval
    else:
        if isinstance(size,tuple):
            s=size[0]
        else:
            s=size
    return np.random.binomial(1,float(l)/s,size)

def simulate_poisson(number):
    N = [2500,5000,10000,40000]
    l = np.arange(0,1000,100)#, 100)
    # l[0]=1
    # l=[1,5,10,20,50,100]
    anlytical=[]

    for j, n in enumerate(N):
        y = []
        for i in l:
            if j==0:
                anlytical.append(binary_ent(i, 1000))
            s=create_spike_trains_r(i,n)
            ctw=CTW()
            ctw.insert_pattern(s)
            y.append(ctw.get_entropy(n))
        plt.plot(l,y,label=f'{str(n)}',alpha=0.5)
    fig, ax = plt.subplots(1)
    fig.set_figwidth(16)
    fig.set_figheight(16)
    ax.plot(l,anlytical,label='analytical')
    ax.legend(title='CTW Entropy Approximation as Function of Sequence length')
    ax.set_xlabel('$lambda$')
    ax.set_ylabel('Entropy')
    plt.savefig(os.path.join('plots',f'results_{number}.png'))

if __name__ == '__main__':
    from utils.slurm_job import SlurmJobFactory
    number=np.random.randint(0,1e+7)
    job_factory = SlurmJobFactory("cluster_logs")
    job_factory.send_job(f"entropy_poiss_{number}",
                         f'python -c "from simulate_poisson import simulate_poisson; simulate_poisson({number})"')
    # simulate_poisson(100)