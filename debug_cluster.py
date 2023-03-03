import time
import datetime
from utils.slurm_job import SlurmJobFactory
value = datetime.datetime.fromtimestamp(time.time())
print(value)
def demo_run():
    prev_time = None
    while True:
        cur_time = datetime.datetime.fromtimestamp(time.time())
        if prev_time is None or (cur_time-prev_time).seconds>=600:
            prev_time=cur_time
            print(prev_time)

if __name__ == '__main__':
    job_factory= SlurmJobFactory("cluster_logs")
    job_factory.send_job(f"demo_debug",
                         f'python -c "from debug_cluster import demo_run; demo_run()"',timelimit=True)