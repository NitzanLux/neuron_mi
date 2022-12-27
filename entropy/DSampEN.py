import numpy as np
from tqdm import tqdm
""" Base Sample Entropy function."""

NATURAL = np.exp(1)
BASE_2 = 2
def DSampEn(Sig, m=2, Logx=NATURAL):


    """SampEn  estimates the sample entropy of a univariate data sequence.

    .. code-block:: python

        Samp, A= SampEn(Sig)

    Returns the sample entropy estimates (``Samp``) and the number of matched state
    vectors (``m: B``, ``m+1: A``) for ``m`` = [0, 1, 2] estimated from the data sequence (``Sig``)
    logarithm = natural

    .. code-block:: python

        Samp, A= SampEn(Sig, keyword = value, ...)

    Returns the sample entropy estimates (``Samp``) for dimensions = [0, 1, ..., ``m``]
    estimated for the data sequence (``Sig``) using the specified keyword arguments:
        :m:     - Embedding Dimension, a positive integer
        :Logx:  - Logarithm base, a positive scalar


    :References:
        [1] Joshua S Richman and J. Randall Moorman.
            "Physiological time-series analysis using approximate entropy
            and sample entropy."
            American Journal of Physiology-Heart and Circulatory Physiology
            2000
    """

    Sig = np.squeeze(Sig)
    N = Sig.shape[0]

    assert N > 10 and Sig.ndim == 1, "Sig:   must be a numpy vector"
    assert isinstance(m, int) and (m > 0), "m:     must be an integer > 0"
    assert isinstance(Logx, (int, float)) and (Logx > 0), "Logx:     must be a positive value"
    A = np.zeros((m+1,))
    B = np.zeros((m+1,))
    for k in tqdm(range(1,m+2)):
        a,b = count_occurence(Sig,k)
        A[k-1]=a
        B[k-1]=b
    with np.errstate(divide='ignore', invalid='ignore'):
        Samp = -np.log(A/B)/np.log(Logx)
    return Samp, A,B

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
    # return cumulative_sum_B
