
import subprocess
import re

def get_works_on_cluster(match_filter:str):
    result = subprocess.run(['squeue', '--me', '-o', '"%.1i %.1P %100j %1T %.1M  %.R"'], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = str(result)
    result = result.split('\n')
    for i, s in enumerate(result):
        result[i] = re.split('[\s]+', s)
    index = result[0].index("NAME")
    print(f"{match_filter}")
    m = re.compile(f"{match_filter}")
    filterd_names = []
    for i, arr in enumerate(result):
        if i == 0 or len(arr) < index + 1:
            continue
        if m.match(arr[index]):
            print(arr[index])
            filterd_names.append(arr[index])
    return filterd_names

