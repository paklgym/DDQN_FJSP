import entities.job as Job
import entities.machine as Machine
import entities.operation as Operation
import entities.machine_operation as MachineOperation
import entities.finished_job as FinishedJob
import dispatching_rules.dispatching_rules as DR
import numpy as np
import re
from copy import deepcopy

class Environment:
    def __init__(self, file):
        self.machines = []
        self.jobs = []
        self.available_jobs = []
        self.completed_jobs = []
        self.events = []
        self.dataset = []
        self.n_jobs = 0
        self.n_machines = 0
        self.time = 0

        content = file.readlines()
        content = [i.replace('\n','').replace('\t', ' ') for i in content]
        self.content_formatted = []
        for lines in content:
            lines = " ".join(re.split(r"\s+", lines))
            self.content_formatted.append(lines)
    
        n_jobs = int(self.content_formatted[0].split(' ')[0])
        n_machines = int(self.content_formatted[0].split(' ')[1])

        ## Adiciona os valores das linhas do dataset em listas de jobs
        jobs_arr_np = [np.array([int(x) for x in line.split()]) for line in self.content_formatted[1:] if len(line) > 0]

        ## Monta a configuração das máquinas do dataset
        no_machines = int(self.content_formatted[0].split(' ')[1])
        for machine in range(1, no_machines + 1):
            self.machines.append(Machine.Machine(machine))

        ## Monta os Jobs
        for job_idx in range(len(jobs_arr_np)):
            job_id = job_idx + 1
            self.jobs.append(Job.Job(job_id, jobs_arr_np[job_idx][0]))
            current_job = jobs_arr_np[job_idx]
        # adiciona as informações do dataset nas maquinas e operações
            op_idx = 1
            op_id = 1
            steps = [op_idx]
            for step in steps:
                operation = Operation.Operation(op_id, job_idx+1, current_job[op_idx])
                m_idx_strt = op_idx + 1
                op_data_length = 1 + current_job[op_idx]*2

                for machine_idx in range(m_idx_strt, m_idx_strt + op_data_length - 1, 2):
                    m_op = MachineOperation.MachineOperation(current_job[machine_idx], current_job[machine_idx+1])
                    operation.machine_ops.append(m_op)
                
                op_idx += op_data_length

                self.jobs[job_idx].plan.append(operation)
                
                if op_idx < len(current_job):
                    steps.append(op_idx)
                
                op_id += 1

    def pass_time(self):
        self.time += 1
        #incrementa makespan dos jobs
        for job in self.jobs:
            job.makespan += 1

        #decrementa tempo das máquinas
        for machine in self.machines:
            if machine.processing and machine.time_left_on_op > 0:
                machine.time_left_on_op -= 1
    
    def queue_job(self, job_id, machine_id):
        machine = self.machines[machine_id - 1]
        machine.queue.append(self.jobs[job_id - 1])
        self.available_jobs = [available_job for available_job in self.available_jobs if available_job.id != job_id]

    def end_operations(self):
        for machine in self.machines:
            if machine.time_left_on_op == 0 and machine.processing:
                job = self.jobs[machine.current_job - 1]
                self.available_jobs.append(job)
                job.current_op += 1
                machine.current_job = 0
                machine.processing = False


    def start_jobs(self):
        for machine in self.machines:
        # Se a máquina não está processando e possui job na fila
            if machine.processing == False and len(machine.queue) > 0:
                # Alocar primeiro job da fila
                job = machine.queue.pop(0)
                operation = job.plan[job.current_op - 1]
                machine.current_job = job.id
                # machine.time_left_on_op = 0
                for machine_op in operation.machine_ops:
                    if machine_op.machine_no == machine.id:
                        machine.time_left_on_op = machine_op.proc_time # Definir o tempo da operação na máquina
                    machine.processing = True

    def finish_jobs(self):
        for job in self.available_jobs:
            if job.current_op > job.no_operations:
                self.completed_jobs.append(FinishedJob(job.id, deepcopy(job.makespan)))
                job.makespan = 0
                job.current_op = 1

    def run_environment(self):

        while len(self.available_jobs) > 0:
            decision = DR.SPT(self.available_jobs, self.machines)
            self.queue_job(*decision)

            self.start_jobs()
            while len(self.completed_jobs) < 1500:
                self.pass_time()
                self.end_operations()
                self.finish_jobs()

        # Chamada para decisão do agente (deve retornar job_id, machine_id) e usar o método:
        # env.queue_job(job_id, machine_id)
        while len(self.available_jobs) > 0:
            decision = DR.SPT(self.available_jobs, self.machines)
            self.queue_job(*decision)

        self.start_jobs()

        jobs = [np.array([]) for job in self.jobs]


        for job in self.completed_jobs:
            jobs[job.job_id - 1] = np.append(jobs[job.job_id - 1], job.makespan)

        completed_jobs = [len(job) for job in jobs]

        jobs = [np.average(job) for job in jobs]

    def env_reset(self):
        self.machines = []
        self.jobs = []
        self.available_jobs = []
        self.completed_jobs = []
        self.events = []
        self.dataset = []
        self.n_jobs = 0
        self.n_machines = 0
        self.time = 0

    def env_render(self):
        data = {'Job': [i for i in range(1, len(self.jobs)+1)],
                'Makespan médio': self.jobs,
                'Jobs completados': self.completed_jobs }
        return data

    def print_jobs(self):
        retorno = []
        for job in self.jobs:
            retorno.append('Job ' + str(job.id) + ': ' + str(job.no_operations) + ' operações')
            for operacao in job.plan:
                retorno.append('\tOperação ' + str(operacao.id) + ': ' + str(operacao.no_proc_machines) + ' máquinas podem processar essa operação')
                for maquina_op in operacao.machine_ops:
                    retorno.append('\t\tMáquina ' + str(maquina_op.machine_no) + ': ' + str(maquina_op.proc_time) + ' UT')
        return(retorno)

    def print_dataframe(self):
        aux_plan = []
        for job in self.jobs:
            aux_plan.append(job.plan)
        return len(aux_plan)


    def print_events(self):
        for event in self.events:
            print(str(event.time) + ' - ' + event.event)

    def dataset_content(self):
        return self.content_formatted