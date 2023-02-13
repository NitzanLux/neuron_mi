import logging
import os
import subprocess
import sys
base_path= os.getcwd()
cur_path = os.path.join(base_path,'neuron_models')
for i in os.listdir(cur_path):
    # try:
    cwd_path=os.path.join('~',cur_path,i)
    if os.path.exists(os.path.join(cwd_path,'mechanisms')):
        command = "nrnivmodl "+os.path.join('.','mechanisms')
    elif os.path.exists(os.path.join(cwd_path,'mods')):
        command = "nrnivmodl " + os.path.join('.', 'mods')
    else:
        logging.error(f'cannot find mod/mechanisms directory for {i}')
        continue
    print(''.join(command),flush=True)
    if os.path.isfile(cwd_path):
        continue
    res = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True,cwd=cwd_path)
    output,error = res.communicate()
    if output:
        print ("ret> ",res.returncode)
        print ("OK> output ",output)
    if error:
        print ("ret> ",res.returncode)
        print ("Error> error ",error.strip())
        # exit(0)
    # except OSError as e:
    #     print("OSError > ", e.errno)
    #     print("OSError > ", e.strerror)
    #     print("OSError > ", e.filename)
    #     raise e
    # except Exception as e:
    #     print("Error > ", sys.exc_info()[0])
    #     raise e
