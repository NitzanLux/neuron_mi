import numpy as np
import time
import pickle
import os
from scipy import sparse
import h5py

def bin2dict(bin_spikes_matrix):
    spike_row_inds, spike_times = np.nonzero(bin_spikes_matrix)
    row_inds_spike_times_map = {}
    for row_ind, syn_time in zip(spike_row_inds, spike_times):
        if row_ind in row_inds_spike_times_map.keys():
            row_inds_spike_times_map[row_ind].append(syn_time)
        else:
            row_inds_spike_times_map[row_ind] = [syn_time]

    return row_inds_spike_times_map


def dict2bin(row_inds_spike_times_map, num_segments, sim_duration_ms):
    bin_spikes_matrix = np.zeros((num_segments, sim_duration_ms), dtype='bool')
    for row_ind in row_inds_spike_times_map.keys():
        for spike_time in row_inds_spike_times_map[row_ind]:
            bin_spikes_matrix[row_ind, spike_time] = 1.0

    return bin_spikes_matrix
def parse_sim_experiment_file_ido(sim_experiment_folder, print_logs=False):
    # ido_base_path="/ems/elsc-labs/segev-i/Sandbox Shared/Rat_L5b_PC_2_Hay_simple_pipeline_1/simulation_dataset/"
    exc_weighted_spikes = sparse.load_npz(f'{sim_experiment_folder}/exc_weighted_spikes.npz').A
    inh_weighted_spikes = sparse.load_npz(f'{sim_experiment_folder}/inh_weighted_spikes.npz').A

    exc_weighted_spikes_for_window = exc_weighted_spikes
    inh_weighted_spikes_for_window = inh_weighted_spikes

    all_weighted_spikes_for_window = np.vstack((exc_weighted_spikes_for_window, inh_weighted_spikes_for_window))

    somatic_voltage = h5py.File(f'{sim_experiment_folder}/voltage.h5', 'r')['somatic_voltage']
    somatic_voltage = np.array(somatic_voltage)
    summary = pickle.load(open(f'{sim_experiment_folder}/summary.pkl', 'rb'))

    output_spikes_for_window = np.zeros(somatic_voltage.shape[0])
    spike_times = summary['output_spike_times']
    output_spikes_for_window[spike_times.astype(int)] = 1
    return all_weighted_spikes_for_window, output_spikes_for_window, somatic_voltage



def parse_sim_experiment_file(sim_experiment_file):
    if not os.path.isfile(sim_experiment_file):
        return parse_sim_experiment_file_ido(sim_experiment_file)
    print('-----------------------------------------------------------------')
    print("loading file: '" + sim_experiment_file.__split("\\")[-1] + "'")
    loading_start_time = time.time()
    experiment_dict = pickle.load(open(str(sim_experiment_file).encode('utf-8'), "rb"))

    # gather params
    num_simulations = len(experiment_dict['Results']['listOfSingleSimulationDicts'])
    num_segments = len(experiment_dict['Params']['allSegmentsType'])
    sim_duration_ms = experiment_dict['Params']['totalSimDurationInSec'] * 1000
    num_ex_synapses = num_segments
    num_inh_synapses = num_segments
    num_synapses = num_ex_synapses + num_inh_synapses

    # collect X, y_spike, y_soma
    X = np.zeros((num_synapses, sim_duration_ms, num_simulations), dtype='bool')
    y_spike = np.zeros((sim_duration_ms, num_simulations))
    y_soma = np.zeros((sim_duration_ms, num_simulations))
    for k, sim_dict in enumerate(experiment_dict['Results']['listOfSingleSimulationDicts']):
        X_ex = dict2bin(sim_dict['exInputSpikeTimes'], num_segments, sim_duration_ms)
        X_inh = dict2bin(sim_dict['inhInputSpikeTimes'], num_segments, sim_duration_ms)
        X[:, :, k] = np.vstack((X_ex, X_inh))
        spike_times = (sim_dict['outputSpikeTimes'].astype(float) - 0.5).astype(int)
        y_spike[spike_times, k] = 1.0
        y_soma[:, k] = sim_dict['somaVoltageLowRes']

    loading_duration_sec = time.time() - loading_start_time
    print('loading took %.3f seconds' % (loading_duration_sec))
    print('-----------------------------------------------------------------')

    return X, y_spike, y_soma