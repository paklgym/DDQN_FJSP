class Operation:
  def __init__(self, id, job_id, no_proc_machines):
    self.id = id
    self.job_id = job_id
    self.machine_ops = []
    self.no_proc_machines = no_proc_machines
    self.cur_machine = -1 # Para identificar a MachineOperation no array machine_ops