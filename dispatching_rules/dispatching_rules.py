import random

# def dispatch_rule_1(t_cur, tard_job, UC_job):
#     if not tard_job:
#         j = min()
#     else:
#         j = max()
    
#     j = OP + 1
#     M = min()
#     # assign Oij on Mk

# def dispatch_rule_2(t_cur, tard_job, UC_job):
#     if not tard_job:
#         j = min()
#     else:
#         j = max()
    
#     j = OP + 1
#     M = min()
#     # assign Oij on Mk

# def dispatch_rule_3(t_cur, UC_job, ji, j):
#     if random.uniform(0, 1) < 0.5:
#         M = min()
#     else:
#         M = min()
#     # assign Oij on Mk

def SPT(available_jobs, machines):
  shortest_time = (99999999, -1, -1)
  for job in available_jobs:
    for machine_op in job.plan[job.current_op - 1].machine_ops:
      if machine_op.proc_time < shortest_time[0]:
        shortest_time = (machine_op.proc_time, job.id, machine_op.machine_no)
  return (shortest_time[1], shortest_time[2])