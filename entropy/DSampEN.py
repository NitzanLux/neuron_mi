import numpy as np

""" Base Sample Entropy function."""


def DSampEn(Sig, m=2, Logx=np.exp(1)):
    """SampEn  estimates the sample entropy of a univariate data sequence.

    .. code-block:: python

        Samp, A, B = SampEn(Sig)

    Returns the sample entropy estimates (``Samp``) and the number of matched state
    vectors (``m: B``, ``m+1: A``) for ``m`` = [0, 1, 2] estimated from the data sequence (``Sig``)
    using the default parameters: embedding dimension = 2, time delay = 1,
    radius threshold = 0.2*SD(``Sig``), logarithm = natural

    .. code-block:: python

        Samp, A, B = SampEn(Sig, keyword = value, ...)

    Returns the sample entropy estimates (``Samp``) for dimensions = [0, 1, ..., ``m``]
    estimated for the data sequence (``Sig``) using the specified keyword arguments:
        :m:     - Embedding Dimension, a positive integer
        :tau:   - Time Delay, a positive integer
        :r:     - Radius Distance Threshold, a positive scalar
        :Logx:  - Logarithm base, a positive scalar

    :See also:
        ``ApEn``, ``FuzzEn``, ``PermEn``, ``CondEn``, ``XSampEn``, ``SampEn2D``, ``MSEn``

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
    for k in range(1, m + 2):
        A[k-1] = count_occurence(Sig,k)
    with np.errstate(divide='ignore', invalid='ignore'):
        Samp = -np.log(A[2:] / A[:-1] ) / np.log(Logx)
    return Samp, A

def count_occurence(sig,length):
    repeat_dict={}
    for i in range(sig.shape[0]-length):
        cur_sig = tuple(sig[i:i+length])
        if cur_sig not in repeat_dict:
            repeat_dict[cur_sig]=1
        else:
            repeat_dict[cur_sig]+=1
    cumulative_sum=0
    for k,v in repeat_dict.items():
        assert v>0,f'pattern {k} appears zero times....'
        cumulative_sum+=(v*(v-1))/2
    return cumulative_sum