class Machine:
  def __init__(self, id):
    self.id = id
    self.time_left_on_op = 0
    self.current_job = 0
    self.queue = []
    self.processing = False