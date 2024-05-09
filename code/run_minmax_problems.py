import run_PS
import run_PT


def get_probabilities_per_worker(config, graph):
    Mu = config['mu']
    Lam = config['lam']
    xx1, pt1, ps1 = run_PT.get_probabilities_per_worker(graph, Mu, Lam)
    xx2, pt2, ps2 = run_PS.get_probabilities_per_worker(graph, Mu, Lam)

    return xx1, pt1, ps1, xx2, pt2, ps2
