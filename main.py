import copy
import networkx as nx
import run_minmax_problems
import simulation

# time constant
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_MINUTE = 60
UNITS_PER_SECOND = 10

# Average (real) running times for the different workers.
AVERAGE_MU = {
    5: 5.33,
    6: 5,
    7: 5.5,
    8: 8,
    9: 4.5,
    10: 3.33,
    11: 4,
    12: 6,
    13: 6.5
}


# moving from average duration to Mu
def seconds_to_p(mean):
    return 1 / (mean)


# a utility function for generating Mu (duration distributions)
def generate_uniform_list(size, min_val, max_val):
    step_size = (max_val - min_val) / (size - 1)
    uniform_list = [min_val + i * step_size for i in range(size - 1)]
    uniform_list.append(max_val)
    return uniform_list


# generate Mu (duration times) according to the second approach. See the last paragraph of section E.1 for details
def generate_mu(workers, kappa, graph):
    mu = {}
    for worker in workers:
        a = (2 * kappa * AVERAGE_MU[worker]) / (1 + kappa)
        b = 2 * AVERAGE_MU[worker] - a
        mu[worker] = {}

        neighbors = list(graph.neighbors(worker))
        if len(neighbors) == 1:
            mu[worker][neighbors[0]] = seconds_to_p(AVERAGE_MU[worker])
        else:
            ls = generate_uniform_list(len(neighbors), min(a, b), max(a, b))
            for cnt, neighbor in enumerate(neighbors):
                mu[worker][neighbor] = seconds_to_p(ls[cnt])
    return mu

# Generate lambda (arrival distribution parameter)
def generate_lam(tasks, config):
    arrivals = [arrival * (config[
                               'requests_per_day'] / HOURS_PER_DAY / MINUTES_PER_HOUR / SECONDS_PER_MINUTE)
                for arrival in config['lam']]
    lam = {}
    for e2, j in enumerate(tasks):
        lam[j] = arrivals[e2 % len(arrivals)]
    return lam


# load the compatibility graph from the configuration file (which workers are allowed to perform each task)
def load_graph():
    with open("graph_conf", "r") as gcnf:

        for line in gcnf:
            if line.startswith("["):
                current = {}
                continue
            key, val = line.strip().split(":")
            if key == "tasks":
                current['bot_nodes'] = [int(task) for task in val.split(",")]  # bot are tasks
            if key == "workers":
                current['top_nodes'] = [int(worker) for worker in val.split(",")]
            if key == "edges":
                current['edges'] = [(int(spp.split(";")[1]), int(spp.split(";")[0])) for spp in val.split(",")]

        top_nodes = current['top_nodes']
        bot_nodes = current['bot_nodes']
        edges = current['edges']
        g = nx.Graph()
        g.add_nodes_from(top_nodes, bipartite=0)
        g.add_nodes_from(bot_nodes, bipartite=1)
        g.add_edges_from(edges)

        return copy.deepcopy(g), top_nodes, bot_nodes


# run the simulation itself and present its results.
# first two lines allow to adjust the parameters.
def run():
    kappa, load, lam = 2, 120000, [0.25, 0.25, 0.25, 0.25]
    config = {'total_duration': 604800, 'name': "\kappa-{},load-{},lam-{}".format(kappa, load, lam)}
    graph, workers, tasks = load_graph()
    config['requests_per_day'] = load
    config['mu'] = generate_mu(workers, kappa, graph)
    config['lam'] = lam  # initial values of lam will be used for determining the actual values
    lam = generate_lam(tasks, config)
    config['lam'] = lam

    #run the minimax problems
    xx1, pt1, ps1, xx2, pt2, ps2 = run_minmax_problems.get_probabilities_per_worker(config, graph)

    # generate objects required for running the simulation
    # Define the job type distributions
    job_type_distributions = {}
    for taskid in range(len(config['lam'])):  # for each task define a relevant object
        job_type_distributions[taskid + 1] = simulation.JobType(taskid + 1, config['lam'][taskid + 1])

    job_types = {}
    for job_type, distribution in job_type_distributions.items():
        job_type_instance = simulation.JobType(job_type, distribution.job_arrival_lambda)
        job_types[job_type] = job_type_instance

    # create the tasks
    tasks, total_tasks = simulation.generate_tasks(job_type_distributions,
                                                   config[
                                                       'total_duration'])  # there is no change for for different amounts of workers.

    # Create the simulation object
    rasimulation = simulation.ResourceAllocationSimulation(config['total_duration'], job_type_distributions, graph,
                                                           xx1, xx2,
                                                           copy.deepcopy(tasks), copy.deepcopy(total_tasks),
                                                           job_types, config['mu'])

    # Create resources and add them to simulation
    top_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}
    for i in top_nodes:
        resource = simulation.Resource(i)
        rasimulation.add_resource(resource)

    # report the results
    print("Parameters: kappa: {} load: {} arrival dist: {} simulation duration: {} seconds".
          format(kappa, load, lam, config['total_duration']))

    print("OPT(PT) Maximum task waiting time: {} Maximum worker utilization: {}".format(pt1, ps1))

    print("OPT(PS) Maximum task waiting time: {} Maximum worker utilization: {}".format(pt2, ps2))

    u, w = rasimulation.run_simulation_SIM_PT()
    print("SIM(PT) Maximum task waiting time: {} Maximum worker utilization: {}".format(w, u))

    u, w = rasimulation.run_simulation_SIM_PS()
    print("SIM(PS) Maximum task waiting time: {} Maximum worker utilization: {}".format(w, u))

    u, w = rasimulation.run_simulation_SIMF_PT()
    print("SIM-F(PT) Maximum task waiting time: {} Maximum worker utilization: {}".format(w, u))

    u, w = rasimulation.run_simulation_SIMF_PS()
    print("SIM-F(PS) Maximum task waiting time: {} Maximum worker utilization: {}".format(w, u))

    u, w = rasimulation.run_simulation_GTW()
    print("GTW Maximum task waiting time: {} Maximum worker utilization: {}".format(w, u))

    u, w = rasimulation.run_simulation_GWU()
    print("GWU Maximum task waiting time: {} Maximum worker utilization: {}".format(w, u))

#run the simulation
run()
