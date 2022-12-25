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