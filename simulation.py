import copy
import random
import numpy as np


# generate actual arrival times for all tasks in the simulation
def generate_arrival_times(lam, size):
    current_av = 0.0
    arrivals = []
    sz = int(size * lam * 2)
    durations = np.random.exponential(1 / lam, size=sz)
    ind = 0
    while current_av < size:
        av = durations[ind]
        ind += 1
        current_av = current_av + av
        arrivals.append(current_av)

    return arrivals


# generate all tasks for simulation. duration times will be determined later (on realtime).
def generate_tasks(job_type_distributions, total_duration):
    total_tasks = 0
    tasks = []
    cnt = 0

    for job_type, distribution in job_type_distributions.items():
        arrival_times = generate_arrival_times(distribution.job_arrival_lambda, int(total_duration))
        for arrival_time in arrival_times:
            task_name = f"{job_type} - Task {cnt}"
            task = Task(task_name, job_type, arrival_time)
            cnt += 1
            tasks.append(task)
            total_tasks += 1

    return tasks, total_tasks


# a utility function
def aggregate_list_items_by_type(lst):
    aggregated_items = {}
    for item in lst:
        item_type = item.job_type
        if item_type not in aggregated_items:
            aggregated_items[item_type] = []
        aggregated_items[item_type].append(item)

    return aggregated_items


# reports the maximum waiting time and the maximum worker utilization for any algorithm
def generate_report(completed_tasks, current_time, resources):
    total_time = current_time
    max_util = max([resource.usage / total_time for resource in resources])
    aggregated_completed_tasks = aggregate_list_items_by_type(completed_tasks)

    mx_waits = 0
    for key, val in aggregated_completed_tasks.items():
        dur_per_worker = {}
        wait_per_worker = {}
        task_per_worker = {}

        for task in val:
            if task.worker not in dur_per_worker:
                dur_per_worker[task.worker] = 0
                wait_per_worker[task.worker] = 0
                task_per_worker[task.worker] = 0

            dur_per_worker[task.worker] += task.dur
            wait_per_worker[task.worker] += task.wait_time
            task_per_worker[task.worker] += 1
        wait_for_task = sum(
            [(wait_per_worker[worker] / dur_per_worker[worker]) * task_per_worker[worker] / len(val) for worker in
             task_per_worker])

        mx_waits = max(mx_waits, wait_for_task)

    return max_util, mx_waits


# resource class for the workers
class Resource:
    def __init__(self, name):
        self.name = name
        self.capacity = 1
        self.queue = []
        self.eusage = 0
        self.usage = 0
        self.available = True
        self.update_time = 0
        self.find = 0


# a class for the tasks and their properties
class Task:
    def __init__(self, name, job_type, arrival_time):
        self.name = name
        self.resource = -1
        self.job_type = job_type
        self.duration = 0
        self.start_time = -1
        self.end_time = -1
        self.dur = 0
        self.arrival_time = arrival_time
        self.is_finished = False
        self.wait_time = 0
        self.worker = -1


# a class for task type
class JobType:
    def __init__(self, name, job_arrival_lambda):
        self.name = name
        self.job_arrival_lambda = job_arrival_lambda
        self.queue = []


# The simulation class. Initialize parameters
class ResourceAllocationSimulation:
    def __init__(self, total_duration, job_type_distributions, graph, x_values_PT, x_values_PS, tasks, total_tasks,
                 job_types,
                 mu):
        self.total_duration = total_duration
        self.job_type_distributions = job_type_distributions
        self.resources = []
        self.tasks = tasks
        self.total_tasks = total_tasks

        self.graph = graph
        self.completed_tasks = []
        self.x_values_PT = x_values_PT
        self.x_values_PS = x_values_PS
        self.job_types = job_types
        self.durations = {}
        self.duration_indeces = {}
        self.Mu = mu
        for i in self.Mu.keys():
            self.durations[i] = {}
            self.duration_indeces[i] = {}
            for j in self.Mu[i].keys():
                self.duration_indeces[i][j] = 0
                self.durations[i][j] = np.random.exponential(1 / self.Mu[i][j], total_tasks)

    # adding a worker to the simulation
    def add_resource(self, resource):
        self.resources.append(resource)

    # used probabilities from the PT minimax solution for deciding which worker will be assigned to which task
    def generate_job_workers_PT(self):
        task_workers = {}
        for job_type, distribution in self.job_type_distributions.items():
            neighbors = list(self.graph.neighbors(job_type))
            probabilities = [self.x_values_PT[job_type][x] for x in neighbors]
            ln = len([task for task in self.tasks if task.job_type == job_type])
            task_workers[job_type] = random.choices(neighbors, weights=probabilities, k=ln)
            if job_type == 2:
                workers = {}
                for worker in task_workers:
                    if worker not in workers:
                        workers[worker] = 0
                    workers[worker] += 1

        return task_workers

    # used probabilities from the PS minimax solution for deciding which worker will be assigned to which task
    def generate_job_workers_PS(self):
        task_workers = {}

        for job_type, distribution in self.job_type_distributions.items():
            neighbors = list(self.graph.neighbors(job_type))
            probabilities = [self.x_values_PS[job_type][x] for x in neighbors]
            ln = len([task for task in self.tasks if task.job_type == job_type])
            task_workers[job_type] = random.choices(neighbors, weights=probabilities, k=ln)

        return task_workers

    # choose a worker for a task, following the PT minimax solution out of a subset of the (free) workers
    def generate_job_worker_part_PT(self, job_type, neighbors):

        neighbors = [neighbor.name for neighbor in neighbors]
        probabilities = [self.x_values_PT[job_type][x] for x in neighbors]
        try:
            choice = random.choices(neighbors, weights=probabilities, k=1)
        except:
            choice = [neighbors[0]]

        return choice

    # choose a worker for a task, following the PS minimax solution out of a subset of the (free) workers
    def generate_job_worker_part_PS(self, job_type, neighbors):

        neighbors = [neighbor.name for neighbor in neighbors]
        probabilities = [self.x_values_PS[job_type][x] for x in neighbors]
        try:
            choice = random.choices(neighbors, weights=probabilities, k=1)
        except:
            choice = [neighbors[0]]

        return choice

    #run the Greedy Worker Utilization heuristic (See the end of Section 4.1)
    def run_simulation_GWU(self):
        duration_indeces = copy.deepcopy(self.duration_indeces)
        durations = copy.deepcopy(self.durations)
        tasks = copy.deepcopy(self.tasks)
        resources = copy.deepcopy(self.resources)
        tasks.sort(key=lambda x: x.arrival_time)
        graph = self.graph
        total_time = 0

        for cnt, task in enumerate(tasks):

            for resource in resources:  # update durations of all candidates before adding a task
                if resource.update_time < task.arrival_time:
                    update_time = task.arrival_time - resource.update_time
                    for item in resource.queue[resource.find:]:
                        usage_update_time = min(update_time, item.duration)
                        if update_time < item.duration:
                            item.duration = item.duration - update_time
                            resource.usage += usage_update_time
                            break
                        else:
                            update_time -= item.duration
                            item.duration = 0
                            resource.find += 1
                            resource.usage += usage_update_time

                    resource.update_time = task.arrival_time

            candidates = list(graph.neighbors(task.job_type))  # fetch candidates for assignments.
            min_usage = 10000000000
            min_resource = -1
            for candidate in candidates:
                resource = next((r for r in resources if r.name == candidate), None)
                if resource.usage < min_usage:  # choose candidate with the lower utilization
                    min_usage = resource.usage
                    min_resource = resource

            if min_resource:
                task.duration = durations[min_resource.name][task.job_type][
                    duration_indeces[min_resource.name][task.job_type]]
                duration_indeces[min_resource.name][task.job_type] += 1
                task.dur = task.duration
                min_resource.queue.append(task)
                task.wait_time = 0 if len(min_resource.queue) == 1 else sum(
                    ## wait time can be calculated after we are done.
                    task.duration for task in min_resource.queue[min_resource.find:-1])
                task.start_time = task.arrival_time + task.wait_time
                task.end_time = task.arrival_time + task.wait_time + task.duration
                total_time = max(total_time, task.arrival_time + task.wait_time + task.duration)
        for resource in resources:  # update durations of all candidates before adding a task
            if resource.update_time < total_time:
                update_time = total_time - resource.update_time
                for item in resource.queue[resource.find:]:
                    usage_update_time = min(update_time, item.duration)
                    if update_time < item.duration:
                        item.duration = item.duration - update_time
                        resource.usage += usage_update_time
                        break
                    else:
                        update_time -= item.duration
                        item.duration = 0
                        resource.find += 1
                        resource.usage += usage_update_time

                resource.update_time = total_time
        return generate_report(tasks, total_time, resources)

    # run the Greedy Task Waiting time heuristic (See the end of Section 4.1)
    def run_simulation_GTW(self):
        tasks = copy.deepcopy(self.tasks)
        resources = copy.deepcopy(self.resources)
        tasks.sort(key=lambda x: x.arrival_time)
        duration_indeces = copy.deepcopy(self.duration_indeces)
        durations = copy.deepcopy(self.durations)

        graph = self.graph

        total_time = 0
        for task in tasks:
            for resource in resources:  # update durations of all candidates before adding a task
                if resource.update_time < task.arrival_time:
                    update_time = task.arrival_time - resource.update_time
                    for item in resource.queue[resource.find:]:
                        if update_time < item.duration:
                            item.duration = item.duration - update_time
                            break
                        else:
                            update_time -= item.duration
                            item.duration = 0
                            resource.find += 1
                    resource.update_time = task.arrival_time
            candidates = graph.neighbors(task.job_type)  # fetch candidates for assignments.
            min_wait = 100000
            min_resource = -1

            for candidate in candidates:
                resource = next((r for r in resources if r.name == candidate), None)
                expected_wait = sum([0 if task.duration == 0 else self.Mu[resource.name][
                    task.job_type] if task.dur == task.duration else
                max(0, self.Mu[resource.name][task.job_type] - (task.dur - task.duration)) for task
                                     in resource.queue[resource.find:]])  # choose worker for which the expected
                # waiting time is minimal
                if expected_wait < min_wait:
                    min_wait = expected_wait
                    min_resource = resource

            if min_resource:
                min_resource.queue.append(task)
                task.duration = durations[min_resource.name][task.job_type][
                    duration_indeces[min_resource.name][task.job_type]]
                duration_indeces[min_resource.name][task.job_type] += 1

                task.dur = task.duration
                min_resource.usage += task.duration
                task.wait_time = 0 if len(min_resource.queue) == 1 else sum(
                    task.duration for task in min_resource.queue[min_resource.find:-1])
                task.start_time = task.arrival_time + task.wait_time
                task.end_time = task.arrival_time + task.wait_time + task.duration

                total_time = max(total_time, task.arrival_time + task.wait_time + task.duration)

        for resource in resources:  # update durations of all candidates before adding a task
            if resource.update_time < total_time:
                update_time = total_time - resource.update_time
                for item in resource.queue[resource.find:]:
                    if update_time < item.duration:
                        item.duration = item.duration - update_time
                        break
                    else:
                        update_time -= item.duration
                        item.duration = 0
                        resource.find += 1
                resource.update_time = total_time

        return generate_report(tasks, total_time, resources)

    # run Algorithm 1 with a solution for PT
    def run_simulation_SIM_PT(self):
        duration_indeces = copy.deepcopy(self.duration_indeces)
        durations = copy.deepcopy(self.durations)
        tasks = copy.deepcopy(self.tasks)
        resources = copy.deepcopy(self.resources)
        tasks.sort(key=lambda x: x.arrival_time)
        job_workers = self.generate_job_workers_PT()
        ind = {}
        total_time = 0
        for task in tasks:
            if task.job_type not in ind:
                ind[task.job_type] = 0
            for resource in resources:  # update durations of all candidates before adding a task
                if resource.update_time < task.arrival_time:
                    update_time = task.arrival_time - resource.update_time
                    for item in resource.queue[resource.find:]:
                        if update_time < item.duration:
                            item.duration = item.duration - update_time
                            break
                        else:
                            update_time -= item.duration
                            item.duration = 0
                            resource.find += 1
                    resource.update_time = task.arrival_time

            resource_name = job_workers[task.job_type][ind[task.job_type]]  # job_workers contains probabilities for PT.
            ind[task.job_type] += 1

            resource = next((r for r in resources if r.name == resource_name), None)
            min_resource = resource

            if min_resource:
                min_resource.queue.append(task)
                task.duration = durations[min_resource.name][task.job_type][
                    duration_indeces[min_resource.name][task.job_type]]
                duration_indeces[min_resource.name][task.job_type] += 1

                task.dur = task.duration
                min_resource.usage += task.duration
                task.wait_time = 0 if len(min_resource.queue) == 1 else sum(
                    task.duration for task in min_resource.queue[min_resource.find:-1])
                task.start_time = task.arrival_time + task.wait_time
                task.worker = min_resource.name
                task.end_time = task.arrival_time + task.wait_time + task.duration
                total_time = max(total_time, task.arrival_time + task.wait_time + task.duration)

        for resource in resources:  # update durations of all candidates before adding a task
            if resource.update_time < total_time:
                update_time = total_time - resource.update_time
                for item in resource.queue[resource.find:]:
                    if update_time < item.duration:
                        item.duration = item.duration - update_time
                        break
                    else:
                        update_time -= item.duration
                        item.duration = 0
                        resource.find += 1
                resource.update_time = total_time
        return generate_report(tasks, total_time, resources)

    # run Algorithm 1 with a solution for PS
    def run_simulation_SIM_PS(self):

        duration_indeces = copy.deepcopy(self.duration_indeces)
        durations = copy.deepcopy(self.durations)
        tasks = copy.deepcopy(self.tasks)
        resources = copy.deepcopy(self.resources)
        tasks.sort(key=lambda x: x.arrival_time)
        job_workers = self.generate_job_workers_PS()
        ind = {}
        total_time = 0
        for cnt, task in enumerate(tasks):
            if task.job_type not in ind:
                ind[task.job_type] = 0
            for resource in resources:  # update durations of all candidates before adding a task
                if resource.update_time < task.arrival_time:
                    update_time = task.arrival_time - resource.update_time
                    for item in resource.queue[resource.find:]:
                        if update_time < item.duration:
                            item.duration = item.duration - update_time
                            break
                        else:
                            update_time -= item.duration
                            resource.find += 1
                            item.duration = 0
                    resource.update_time = task.arrival_time

            resource_name = job_workers[task.job_type][ind[task.job_type]]
            ind[task.job_type] += 1

            resource = next((r for r in resources if r.name == resource_name), None)
            min_resource = resource

            if min_resource:
                min_resource.queue.append(task)
                task.duration = durations[min_resource.name][task.job_type][
                    duration_indeces[min_resource.name][task.job_type]]
                duration_indeces[min_resource.name][task.job_type] += 1

                task.dur = task.duration
                min_resource.usage += task.duration
                task.wait_time = 0 if len(min_resource.queue) == 1 else sum(
                    task.duration for task in min_resource.queue[min_resource.find:-1])
                task.start_time = task.arrival_time + task.wait_time
                task.end_time = task.arrival_time + task.wait_time + task.duration
                total_time = max(total_time, task.arrival_time + task.wait_time + task.duration)

        for resource in resources:  # update durations of all candidates before adding a task
            if resource.update_time < total_time:
                update_time = total_time - resource.update_time
                for item in resource.queue[resource.find:]:
                    if update_time < item.duration:
                        item.duration = item.duration - update_time
                        break
                    else:
                        update_time -= item.duration
                        resource.find += 1
                        item.duration = 0
                resource.update_time = total_time

        return generate_report(tasks, total_time, resources)

    # run Algorithm 2 with a solution for PS
    def run_simulation_SIMF_PS(self):
        duration_indeces = copy.deepcopy(self.duration_indeces)
        durations = copy.deepcopy(self.durations)
        tasks = copy.deepcopy(self.tasks)
        resources = copy.deepcopy(self.resources)
        tasks.sort(key=lambda x: x.arrival_time)
        graph = copy.deepcopy(self.graph)
        job_workers = self.generate_job_workers_PS()
        ind = {}

        total_time = 0
        for task in tasks:
            if task.job_type not in ind:
                ind[task.job_type] = 0
            for resource in resources:  # update durations of all candidates before adding a task
                if resource.update_time < task.arrival_time:
                    update_time = task.arrival_time - resource.update_time
                    for item in resource.queue[resource.find:]:
                        if update_time < item.duration:
                            item.duration = item.duration - update_time
                            break
                        else:
                            update_time -= item.duration
                            item.duration = 0
                            resource.find += 1
                    resource.update_time = task.arrival_time

            candidates = list(graph.neighbors(task.job_type))  # fetch candidates for assignments.
            min_resources = []
            for candidate in candidates:
                resource = next((r for r in resources if r.name == candidate), None)
                expected_wait = sum([0 if task.duration == 0 else self.Mu[resource.name][
                    task.job_type] if task.dur == task.duration else
                max(0, self.Mu[resource.name][task.job_type] - (task.dur - task.duration)) for task
                                     in resource.queue[resource.find:]])
                if expected_wait == 0:
                    min_resources.append(resource)

            if len(min_resources) == 0:
                resource_name = job_workers[task.job_type][ind[task.job_type]]
                ind[task.job_type] += 1
                resource = next((r for r in resources if r.name == resource_name), None)
                min_resource = resource
            else:
                resource_name = self.generate_job_worker_part_PS(task.job_type, min_resources)
                resource = next((r for r in resources if r.name == resource_name[0]), None)
                min_resource = resource

            if min_resource:
                min_resource.queue.append(task)
                task.duration = durations[min_resource.name][task.job_type][
                    duration_indeces[min_resource.name][task.job_type]]
                duration_indeces[min_resource.name][task.job_type] += 1
                task.dur = task.duration
                min_resource.usage += task.duration
                task.wait_time = 0 if len(min_resource.queue) == 1 else sum(
                    task.duration for task in min_resource.queue[min_resource.find:-1])
                task.start_time = task.arrival_time + task.wait_time
                task.end_time = task.arrival_time + task.wait_time + task.duration
                total_time = max(total_time, task.arrival_time + task.wait_time + task.duration)

        for resource in resources:  # update durations of all candidates before adding a task
            if resource.update_time < total_time:
                update_time = total_time - resource.update_time
                for item in resource.queue[resource.find:]:
                    if update_time < item.duration:
                        item.duration = item.duration - update_time
                        break
                    else:
                        update_time -= item.duration
                        resource.find += 1
                        item.duration = 0
                resource.update_time = total_time

        return generate_report(tasks, total_time, resources)

    # run Algorithm 2 with a solution for PT
    def run_simulation_SIMF_PT(self):
        duration_indeces = copy.deepcopy(self.duration_indeces)
        durations = copy.deepcopy(self.durations)
        tasks = copy.deepcopy(self.tasks)
        resources = copy.deepcopy(self.resources)
        tasks.sort(key=lambda x: x.arrival_time)
        graph = copy.deepcopy(self.graph)
        job_workers = self.generate_job_workers_PT()
        ind = {}

        total_time = 0
        for task in tasks:
            if task.job_type not in ind:
                ind[task.job_type] = 0
            for resource in resources:  # update durations of all candidates before adding a task
                if resource.update_time < task.arrival_time:
                    update_time = task.arrival_time - resource.update_time
                    for item in resource.queue[resource.find:]:
                        if update_time < item.duration:
                            item.duration = item.duration - update_time
                            break
                        else:
                            update_time -= item.duration
                            item.duration = 0
                            resource.find += 1
                    resource.update_time = task.arrival_time

            candidates = graph.neighbors(task.job_type)  # fetch candidates for assignments.
            min_resources = []
            for candidate in candidates:
                resource = next((r for r in resources if r.name == candidate), None)
                expected_wait = sum([0 if task.duration == 0 else self.Mu[resource.name][
                    task.job_type] if task.dur == task.duration else
                max(0, self.Mu[resource.name][task.job_type] - (task.dur - task.duration)) for task
                                     in resource.queue[resource.find:]])
                if expected_wait == 0:
                    min_resources.append(resource)

            if len(min_resources) == 0:
                resource_name = job_workers[task.job_type][ind[task.job_type]]
                ind[task.job_type] += 1
                resource = next((r for r in resources if r.name == resource_name), None)
                min_resource = resource
            else:
                resource_name = self.generate_job_worker_part_PT(task.job_type, min_resources)
                resource = next((r for r in resources if r.name == resource_name[0]), None)
                min_resource = resource

            if min_resource:
                min_resource.queue.append(task)
                task.duration = durations[min_resource.name][task.job_type][
                    duration_indeces[min_resource.name][task.job_type]]
                duration_indeces[min_resource.name][task.job_type] += 1
                task.dur = task.duration
                min_resource.usage += task.duration
                task.wait_time = 0 if len(min_resource.queue) == 1 else sum(
                    task.duration for task in min_resource.queue[min_resource.find:-1])
                task.start_time = task.arrival_time + task.wait_time
                task.end_time = task.arrival_time + task.wait_time + task.duration
                total_time = max(total_time, task.arrival_time + task.wait_time + task.duration)

        for resource in resources:  # update durations of all candidates before adding a task
            if resource.update_time < total_time:
                update_time = total_time - resource.update_time
                for item in resource.queue[resource.find:]:
                    if update_time < item.duration:
                        item.duration = item.duration - update_time
                        break
                    else:
                        update_time -= item.duration
                        resource.find += 1
                        item.duration = 0
                resource.update_time = total_time

        return generate_report(tasks, total_time, resources)
