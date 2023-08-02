import numpy as np
from matplotlib import pyplot as plt


def create_transition_matrix(k):
    states = [format(i, f'0{k}b') for i in range(2 ** k)]
    transition_matrix = dict()
    for i, v in enumerate(states):
        if (i > 1 and v[-1] == '1') or (i > 2 and v[-2] == '1'):
            a, b = 1., 0.
        elif i > 7 and v[-7] == '0' and v[-3] == '1':
            a, b = 1.-1e-3, 1e-3
        # elif i > 3 and v[-3] == '1':
        #     a, b = 0.1, 0.9
        else:
            a, b = 0.99, 0.01
        # repeate_string = "010101"
        # if v.endswith(repeate_string) or v[:-1].endswith(repeate_string):
        #     a,b = 0.99,0.01
        transition_matrix[v] = [a, b]
    # transition_matrix = {state: [0.7, 0.3] for state in states} # Example uniform probabilities
    return transition_matrix


def generate_sequence(transition_matrix, initial_state, length):
    k = len(initial_state)
    sequence = list(initial_state)
    for i in range(length - k):
        current_state = "".join(sequence[-k:])  # Last k bits
        p = np.array(transition_matrix[current_state])
        p = p / p.sum()  # normalizing
        next_state = np.random.choice([0, 1], p=p)
        sequence.append(str(next_state))
    return "".join(sequence)


def row_entropy(row):
    return -np.sum([p * np.log2(p) for p in row if p > 0])


def entropy_rate(transitions):
    # Determine the order of the Markov chain
    order = len(next(iter(transitions.keys())))
    num_states = 2 ** order

    # Building the transition matrix
    P = np.zeros((num_states, num_states))
    for context_from, transition_probs in transitions.items():
        row_idx = int(context_from, 2)
        # Transition to the next states by shifting the bits and appending 0 or 1
        for i, prob in enumerate(transition_probs):
            context_to = (row_idx << 1 | i) % num_states
            P[row_idx, context_to] = prob

    # Calculate the stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    stationary_distribution = eigenvectors[:, np.isclose(eigenvalues, 1)]
    stationary_distribution = stationary_distribution / stationary_distribution.sum()

    # Calculate the entropy for each state
    state_entropies = np.apply_along_axis(row_entropy, axis=1, arr=P)

    # Multiply by the stationary distribution and sum
    H = np.dot(stationary_distribution.flatten(), state_entropies)

    return H.real


# Create corrected trimmed transition matrix for k=15 by grouping states based on the sum of the last m bits
def create_corrected_trimmed_transition_matrix(k, m):
    states = [format(i, f'0{k}b') for i in range(2 ** k)]
    transition_matrix = {}
    for state in states:
        sum_last_m_bits = sum(int(bit) for bit in state[-m:])
        if sum_last_m_bits == m // 2:
            transition_matrix[state] = [0.7, 0.3]  # Example probabilities for equivalent states
        else:
            transition_matrix[state] = [0.5, 0.5]  # Default probabilities
    return transition_matrix


# def plot_raster(sequences):
#     plt.figure(figsize=(15, 6))
#     matrix = np.zeros((len(sequences),max([len(i) for i in sequences])))
#     for neuron_idx, sequence in enumerate(sequences):
#         # spike_times = [i for i, bit in enumerate(sequence) if bit == "1"]
#         for i, bit in enumerate(sequence):
#             matrix[neuron_idx,i] = bit == "1"
#         # plt.scatter(spike_times, [neuron_idx] * len(spike_times), marker='|')
#     plt.matshow(matrix)
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Neuron Index')
#     plt.title(f'Raster Plot for k={k}')
#     plt.ylim(-1, len(sequences))
#     plt.show()


def plot_raster(sequences):
    fig, ax = plt.subplots(figsize=(15, 6))
    matrix = np.zeros((len(sequences), max([len(i) for i in sequences])))
    for neuron_idx, sequence in enumerate(sequences):
        for i, bit in enumerate(sequence):
            matrix[neuron_idx, i] = bit == "1"
    ax.imshow(matrix, cmap='gray_r', aspect='auto')  # Use 'gray_r' colormap for black (1) and white (0)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Raster Plot')
    plt.ylim(-1, len(sequences))
    ax.xaxis.tick_bottom()  # Move x-axis to the bottom
    plt.show()


k = 10
sequence_length_k10_new = 500
num_neurons = 200
# Revised trimmed transition matrix (k=15, m=5)
transition_matrix = create_transition_matrix(k)

for i in range(1, 4):  # Example testing for k=1, 2, 3
    initial_state = "0" * k
    sequence_length = 100
    generated_sequence = generate_sequence(transition_matrix, initial_state, sequence_length)
    print(f"For k={k}, sequence: {generated_sequence}")

# Generate binary sequences for 20 neurons using the corrected trimmed transition matrix (k=15, m=5)
generated_sequences_k15_corrected = [generate_sequence(transition_matrix, "0" * k, sequence_length_k10_new) for _ in
                                     range(num_neurons)]

# Plot the sequences as a raster for 20 neurons
plot_raster(generated_sequences_k15_corrected)
print(entropy_rate(transition_matrix))
