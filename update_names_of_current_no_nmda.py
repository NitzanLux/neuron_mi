import os
import re
compile_pattern=re.compile('.*(_((?:[0-9]-[0-9])|([0-9]))_c_noNMDA)')
cur_path = os.path.join('simulations','data')
def rename(cur_path):
    for i in os.listdir(cur_path):
        m=compile_pattern.match(i)
        if m:
            os.rename(os.path.join(cur_path,i),os.path.join(cur_path,i.replace(m.group(1),'_noNMDA_'+m.group(2))))
            if os.path.isdir(os.path.join(cur_path,i)):
                rename(os.path.join(cur_path,i))

