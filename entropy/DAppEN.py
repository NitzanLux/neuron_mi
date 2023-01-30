import numpy as np
from tqdm import tqdm

NATURAL = np.exp(1)
BASE_2 = 2

def DAppEn(Sig, m=2, Logx=BASE_2):


    Sig = np.squeeze(Sig)
    N = Sig.shape[0]

    assert N > 10 and Sig.ndim == 1, "Sig:   must be a numpy vector"
    assert isinstance(m, int) and (m > 0), "m:     must be an integer > 0"
    assert isinstance(Logx, (int, float)) and (Logx > 0), "Logx:     must be a positive value"
    A = np.zeros((m+1,))
    for k in tqdm(range(1, m + 2)):
        A[k-1] = count_occurence(Sig,k,Logx)
    with np.errstate(divide='ignore', invalid='ignore'):
        Samp = A[:-1] -A[1:]
    return Samp, A

def count_occurence(sig,length,Logx):
    repeat_dict={}
    for i in range(sig.shape[0]-length+1):
        cur_sig = tuple(sig[i:i+length])
        if cur_sig not in repeat_dict:
            repeat_dict[cur_sig]=1
        else:
            repeat_dict[cur_sig]+=1
    cumulative_sum=0
    for k,v in repeat_dict.items():
        assert v>0,f'pattern {k} appears zero times....'
        cumulative_sum+=np.log(v*v)/np.log(Logx)
    return cumulative_sum/(sig.shape[0]-length+1)

def count_occurence(sig,length):
    assert length>0,'length must be positive'
    if length ==1:
        cumulative_sum_B = (sig.shape[0] * (sig.shape[0]-1)/2)
        cumulative_sum_A = np.sum([i*(i-1) for i in np.unique(sig,return_counts=True)[1]])/2
        return cumulative_sum_A,cumulative_sum_B
    repeat_dict_A={}
    repeat_dict_B={}
    for i in range(sig.shape[0]-length+1):
        cur_sig_A = tuple(sig[i:i+length])
        cur_sig_B = tuple(sig[i:i+length-1])

        if cur_sig_B not in repeat_dict_B:
            repeat_dict_B[cur_sig_B]=1
        else:
            repeat_dict_B[cur_sig_B]+=1
        if cur_sig_A not in repeat_dict_A:
            repeat_dict_A[cur_sig_A]=1
        else:
            repeat_dict_A[cur_sig_A]+=1

    cumulative_sum_B = 0
    cumulative_sum_A = 0
    for k,v in repeat_dict_B.items():
        # assert v>0,f'pattern {k} appears zero times....'
        cumulative_sum_B += (v)*(v-1)/2
    for k,v in repeat_dict_A.items():
        # assert v>0,f'pattern {k} appears zero times....'
        cumulative_sum_A += (v)*(v-1)/2
    return cumulative_sum_A ,cumulative_sum_B