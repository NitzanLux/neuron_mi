import numpy as np
def __calculate_LZ76(sig):
    if isinstance(sig, np.ndarray):
        sig=sig.tolist()
    sig = ''.join([str(int(i)) for i in sig])
    sig_set=[]
    start_counter=0
    end_counter=1
    while end_counter<len(sig):
        cur_seq=sig[start_counter:end_counter]
        if cur_seq in sig[:start_counter]:
            end_counter+=1
        else:
            sig_set.append(cur_seq)
            start_counter,end_counter=end_counter,end_counter+1
    if end_counter-start_counter>0:
        sig_set.append(sig[start_counter:])
    return len(sig_set)

def __calculate_entropy(sig):
    p = __calculate_LZ76(sig)
    n=len(sig)
    return (p/n)*np.log(n)

# sig='01011010001101110010'
# out,s= calculate_LZ76(sig)
def LZ76(sig):
    return __calculate_entropy(sig)
#%%
