class Job:
  def __init__(self, id, no_operations):
    self.id = id
    self.no_operations = no_operations
    self.plan = []
    self.current_op = 1 # Para identificar a operação no array plan
    self.makespan = 0
    self.due_date = 0