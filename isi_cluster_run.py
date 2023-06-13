import os

import matplotlib.pyplot as plt

import entropy as ent
from entropy.CTW.CTW import UnboundProbabilityException
from tqdm import tqdm
import numpy as np
from utils.slurm_job import SlurmJobFactory
def create_graphs():
    size=1000
    jumps=100
    cur_path=os.path.join("plots","cv_vs_ent_plots",f"plots_{size}_{jumps}_{np.random.randint(0,10000)}")
    os.makedirs(cur_path,exist_ok=True)
    a_regular = np.zeros((size,))+jumps
    print(a_regular.std() / a_regular.mean())

    genereate_less_regular = lambda x: a_regular+np.random.randint(0,x,size=size) if x>0 else a_regular
    c_cv = lambda x:x.std()/x.mean()
    cv_arr = []
    max_length=0
    r_arr=[]
    x=list(range(jumps*2))
    min_x=min(x)
    max_x=max(x)
    middel_x_f=lambda p:(max_x-min_x)*p+min_x
    traces_to_plot=[min_x,middel_x_f(0.25),middel_x_f(0.5),middel_x_f(0.75),max_x]
    plt.clf()
    for i in x:
        r= genereate_less_regular(i)
        r_a=np.array(r).cumsum()
        # r_a=r_a-r_a[0]
        # mask=size-r_a
        # mask[mask<0]=None
        # r = r[:np.nanargmin(np.abs(mask))]
        # max_length=max(max_length,r_a[r.shape[0]])
        max_length=max(max_length,r_a[-1])
        r_arr.append(r)
        # print(r.sum())
        cv_arr.append(c_cv(r))
    print(max_length)
    plt.scatter(x,cv_arr,s=0.1)
    plt.title("CV as function of jitter")
    plt.xlabel("Jitter")
    plt.ylabel("CV")

    plt.savefig(os.path.join(cur_path,"isi_eval_cv_graph.png"))
    plt.show()
    plt.clf()

    spike_arr = []
    spike_np = [np.cumsum(i) for i in r_arr]
    plt.eventplot(spike_np)  # , color=colorCodes, linelengths=lineSize)
    plt.title('Spike raster plot')
    plt.xlabel('Neuron')
    plt.ylabel('Spike')
    plt.savefig(os.path.join(cur_path, "spikes_raster.png"))
    plt.show()
    plt.clf()

    print(np.argmin(np.abs(np.array(cv_arr)-1)))
    r_ent=[]
    for x_v,r in zip(x,r_arr):
        z = np.zeros((int(max_length+1),))
        r = np.cumsum(r).astype(int)
        z[r]=1
        b = ent.CTW()
        tqdm(b.insert_pattern(z.astype(int).tolist()), disable=True)
        r_ent.append(b.get_entropy(max_length))

    plt.plot(x,r_ent)
    plt.title("Entropy as function of jitter")
    plt.xlabel("Jitter")
    plt.ylabel("Entropy")

    plt.savefig(os.path.join(cur_path,"isi_eval_ent_graph.png"))
    plt.show()
    plt.clf()

if __name__ == '__main__':
    s= SlurmJobFactory("cluster_logs")
    s.send_job_for_function(f"cv_vs_en_{np.random.randint(0,10000)}","isi_cluster_run","create_graphs",[])

# if __name__ == '__main__':
#
#     create_graphs()