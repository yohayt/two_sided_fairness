import csv
import itertools
import os
import shutil
import matlab.engine
import numpy as np

# Python wrapper for running the minimax program of PT

def get_probabilities_per_worker( graph, Mu, Lam):
    graphs = [graph]
    for gid, graph in enumerate(graphs):
        mu = []
        g = graph
        top_nodes = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}  # workers
        bot_nodes = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}  # tasks
        N = {}
        dct = {}
        # define variables
        c = 0
        for bot in bot_nodes:
            for top in top_nodes:
                mu.append(0)  # add placeholders for mu
                dct[(top, bot)] = c
                c += 1
        x_vars = c

        for top in top_nodes:  # for each worker
            dct[top] = c
            c += 1

        for top in top_nodes:  # for each worker
            dct[(top, top, top)] = c
            c += 1

        for bot in bot_nodes:  # for each task
            dct[bot] = c
            c += 1

        dct['all'] = c

        for i in top_nodes:  # workers- add neighbor tasks
            if i not in N:
                N[i] = set()
            for j in g.neighbors(i):
                N[i].add(j)

        for j in bot_nodes:  # tasks add neighbor workers
            if j not in N:
                N[j] = set()
            for i in g.neighbors(j):
                N[j].add(i)

        fname = "mnmxfair1"
        # define mu
        for e2, j in enumerate(bot_nodes):
            for i in N[j]:
                mu[dct[i, j]] = Mu[i][j]
        I = []
        J = []
        V = []
        B = []
        Ie = []
        Je = []
        Ve = []
        Be = []
        # build constraint 7:
        row = 0  #
        for j in bot_nodes:
            for i in top_nodes:
                if i in N[j]:
                    Ie.append(row)
                    Je.append(dct[i, j])
                    Ve.append(1.0)

            Be.append(1.0)
            row += 1
        end_sums = row
        # constraint 8, left
        for i in top_nodes:  # for each worker, sum of params is rho_i
            for j in N[i]:
                Ie.append(row)
                Je.append(dct[i, j])
                Ve.append(Lam[j] / Mu[i][j])

            Ie.append(row)
            Je.append(dct[i])
            Ve.append(-1)
            Be.append(0)
            row += 1
        # additional parameter for simplifying the presentation of the objective.
        for i in top_nodes:  # for each worker, sum of params sq is rho_isq
            for j in N[i]:
                Ie.append(row)
                Je.append(dct[i, j])
                Ve.append(Lam[j] / Mu[i][j] ** 2)

            Ie.append(row)
            Je.append(dct[(i, i, i)])
            Ve.append(-1)
            Be.append(0)
            row += 1

        for j in bot_nodes: # constraint 7 (redundant but do not harm)
            for i in N[j]:  # for each job, sum of params is 1
                Ie.append(row)
                Je.append(dct[i, j])
                Ve.append(1)

            Be.append(1)
            row += 1

        erows = row

        row = 0
        for j in bot_nodes:  # (constraint 9)
            I.append(row)
            J.append(dct[j])
            V.append(1.0)

            I.append(row)
            J.append(dct['all'])
            V.append(-1.0)

            B.append(0.0)
            row += 1

        # constraint 8(right):
        for i in top_nodes:
            for j in N[i]:
                I.append(row)
                J.append(dct[i, j])
                V.append(Lam[j] / Mu[i][j])
            B.append(1.0)
            row += 1

        combinations = list(itertools.product(top_nodes, bot_nodes))

        for i, j in combinations: # upper bounds
            J.append(dct[i, j])
            I.append(row)
            V.append(1.0)
            B.append(1.0)
            row += 1

        for i, j in combinations: # lower bounds
            I.append(row)
            J.append(dct[i, j])
            V.append(-1.0)
            B.append(0.0)
            row += 1

        ierows = row

        num_f = len(top_nodes)
        num_v = len(dct.keys())
        folder = "temp"

        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)

        os.mkdir(folder)
        # write constraints to csv
        np.array(I).tofile("{}/I_{}.csv".format(folder, fname), sep=",")
        np.array(J).tofile("{}/J_{}.csv".format(folder, fname), sep=",")
        np.array(B).tofile("{}/b_{}.csv".format(folder, fname), sep=",")
        np.array(V).tofile("{}/vals_{}.csv".format(folder, fname), sep=",")
        np.array(Ie).tofile("{}/Ie_{}.csv".format(folder, fname), sep=",")
        np.array(Je).tofile("{}/Je_{}.csv".format(folder, fname), sep=",")
        np.array(Be).tofile("{}/be_{}.csv".format(folder, fname), sep=",")
        np.array(Ve).tofile("{}/valse_{}.csv".format(folder, fname), sep=",")
        np.array(mu).tofile("{}/Mu_{}.csv".format(folder, fname), sep=",")
        filename = '{}/fa_{}.csv'.format(folder, fname)

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x_vars', 'end_sums', 'erows', 'ierows', 'nf', 'nv'])
            writer.writerow([x_vars, end_sums, erows, ierows, num_f, num_v])

        try:
            # initial matlab engine and run the minimax problem
            eng = matlab.engine.start_matlab()
            x, v = eng.pt(nargout=2)
            x_p = np.array(x)
            ind = 0
            # process the results, very low probabilites are ignored (for cleaning).
            xx = {}
            for bot in bot_nodes:
                for top in top_nodes:
                    if bot not in xx:
                        xx[bot] = {}
                    if x_p[ind] > 0.000000001:
                        xx[bot][top] = x_p[ind]  # task and then worker.
                    else:
                        xx[bot][top] = 0  # task and then worker.
                    ind += 1
            # calculate the maximum expected utilization of a worker.
            xw = {}
            for top in top_nodes:
                if x_p[ind][0] > 0.000000001:
                    xw[top] = x_p[ind][0]
                else:
                    xw[top] = 0
                ind += 1
        except Exception as e:
            print(e)
            return "ERROR when running MINMAX for PT"
        pa = v
        pt = max(xw.values())

    return xx, pa, pt
