import os
import re
compile_pattern=re.compile('.*(_((?:[0-9]-[0-9])|([0-9]))_c_noNMDA)')
compile_pattern_finnished = re.compile('.*(synapses_noNMDA_((?:[0-9]-[0-9])|([0-9])))')
cur_path = os.path.join('simulations','data')
def rename(cur_path):
    for i in os.listdir(cur_path):
        m=compile_pattern.match(i)
        m_f=compile_pattern_finnished.match(i)
        # print(i)
        if m or m_f:
            print(m,m_f)
            if os.path.isdir(os.path.join(cur_path,i)):
                rename(os.path.join(cur_path,i))
            if m:
                os.rename(os.path.join(cur_path,i),os.path.join(cur_path,i.replace(m.group(1),'_noNMDA_'+m.group(2))))


rename(cur_path)