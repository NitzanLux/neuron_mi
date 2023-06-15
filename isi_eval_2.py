import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

import os

import matplotlib.pyplot as plt

import entropy as ent
from entropy.CTW.CTW import UnboundProbabilityException
from tqdm import tqdm
import numpy as np
from utils.slurm_job import SlurmJobFactory


def create_graphs():
    # Set parameters
    mean = 100
    num_samples = 100
    max_param_value = 300
    num_steps = 20
    cur_path = os.path.join("plots", "cv_vs_ent_plots", f"plots_s_{num_samples}_m_{mean}_mv_{max_param_value}_ns_{num_steps}_{np.random.randint(0, 10000)}")
    os.makedirs(cur_path, exist_ok=True)

    # Generate parameters and empty list for CVs
    parameters = np.arange(0, max_param_value, max_param_value // num_steps)

    cvs = []
    s_data = []
    # Generate data and calculate CVs
    for param in parameters:
        data = np.random.normal(loc=mean, scale=param, size=num_samples).astype(int)
        # data_b = np.zeros_like(data)
        # data = np.cumsum(data)
        # data = np.sort(data)
        # data_b[0] = mean
        # data_b = data[1:] - data[:-1]
        # data= data_b
        cv = np.std(data) / np.mean(data)
        cvs.append(cv)
        s_data.append(data)

    # Calculate R^2 score
    slope, intercept = np.polyfit(parameters, cvs, 1)
    predicted_cvs = slope * parameters + intercept
    r2 = r2_score(cvs, predicted_cvs)

    s_data = np.array(s_data)
    # Print R^2 score
    print(f'R^2 score: {r2}')

    # Plot CVs
    plt.ylim([0.95 * min(predicted_cvs), 1.05 * max(predicted_cvs)])
    plt.scatter(parameters, cvs, s=10, label='Data')

    plt.plot(parameters, predicted_cvs, color='red', linewidth=2, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    plt.title('Coefficient of Variation as a Function of Parameter')
    plt.xlabel('Parameter')
    plt.ylabel('CV')
    plt.savefig(os.path.join(cur_path, "isi_eval_cv_graph.png"))
    plt.show()
    plt.clf()

    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()


    print(s_data.shape)
    x_scatter = np.repeat(np.arange(s_data.shape[0])[np.newaxis,:],s_data.shape[1],axis=0)
    # np.cumsum(s_data,axis=1)
    for i in range(s_data.shape[0]):
        plt.scatter(np.cumsum(s_data[i, :]), i + np.zeros_like(s_data[i, :]), linewidths=0.4, marker='|')

    plt.yticks(np.arange(0, s_data.shape[0], s_data.shape[0] // 10), parameters[::s_data.shape[0] // 10].astype(int))
    # plt.title('Scatter Eventplot')
    # plt.show()
    # plt.scatter(x_scatter,np.cumsum(s_data, axis=1),s=0.3)
    plt.yticks(np.arange(0, s_data.shape[0], s_data.shape[0] // 10), parameters[::s_data.shape[0] // 10].astype(int))
    plt.title('Spike raster plot')
    plt.xlabel('Neuron')
    plt.ylabel('Spike')
    plt.savefig(os.path.join(cur_path, "spikes_raster.png"))
    plt.show()
    # print(data.shape)
    plt.clf()


    s_data_c=np.cumsum(s_data,axis=1)
    min_val=np.min(s_data_c)
    max_val=np.max(s_data_c)

    r_ent = []
    # i=0
    for i,p in enumerate(parameters):
        template = np.zeros((max_val - min_val+1,))
        template_pos = s_data_c[i, :]
        template[template_pos - min_val] = 1
        # template_pos[r] = 1
        b = ent.CTW()
        tqdm(b.insert_pattern(template.astype(int).tolist()), disable=True)
        r_ent.append(b.get_entropy(max_val-min_val))
        # i+=1

    plt.plot(parameters, r_ent)
    plt.title("Entropy as function of jitter")
    plt.xlabel("Jitter")
    plt.ylabel("Entropy")

    plt.savefig(os.path.join(cur_path, "isi_eval_ent_graph.png"))
    plt.show()
    plt.clf()

if __name__ == '__main__':
    s= SlurmJobFactory("cluster_logs")
    s.send_job_for_function(f"cv_vs_en_{np.random.randint(0,10000)}","isi_eval_2","create_graphs",[])


