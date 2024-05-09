import numpy as np
import csv
import os
import shutil
import matlab.engine
# Python wrapper for running the linear program of PS


def get_probabilities_per_worker(graph, Mu, Lam):
    g = graph
    top_nodes = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 0}
    bot_nodes = {n for n, d in graph.nodes(data=True) if d['bipartite'] == 1}
    N = {}
    dct = {}
    #define variables
    c = 0
    for bot in bot_nodes:
        for top in top_nodes:
            dct[(top, bot)] = c
            c += 1

    for top in top_nodes:
        dct[top] = c
        c += 1

    dct['all'] = c
    # define neighbors sets
    for i in top_nodes:
        if i not in N:
            N[i] = set()
        for j in g.neighbors(i):
            N[i].add(j)

    for j in bot_nodes:
        if j not in N:
            N[j] = set()
        for i in g.neighbors(j):
            N[j].add(i)

    fname = "mnmxfair1"

    I = []
    J = []
    V = []
    B = []
    Ie = []
    Je = []
    Ve = []
    Be = []
    # build constraint 11:
    row = 0  #

    for j in bot_nodes:
        for i in N[j]:
            Ie.append(row)
            Je.append(dct[i, j])
            Ve.append(1.0)

        Be.append(1.0)
        row += 1
    # define constraint 12 (left)
    for i in top_nodes:
        for j in N[i]:
            Ie.append(row)
            Je.append(dct[i, j])
            Ve.append(Lam[j] / Mu[i][j])

        Ie.append(row)
        Je.append(dct[i])
        Ve.append(-1)
        Be.append(0)
        row += 1
    erows = row
    # define the maximum rho variable by letting it be ge to all worker rhos
    row = 0
    for i in top_nodes:
        I.append(row)
        J.append(dct[i])
        V.append(1.0)

        I.append(row)
        J.append(dct['all'])
        V.append(-1.0)

        B.append(0.0)
        row += 1

    # build constraint 12(right):
    for i in top_nodes:
        for j in N[i]:
            I.append(row)
            J.append(dct[i, j])
            V.append(Lam[j] / Mu[i][j])
        B.append(1.0)
        row += 1

    for i in top_nodes: #constraint 13
        for j in bot_nodes:
            J.append(dct[i, j])
            I.append(row)
            V.append(1.0)
            B.append(1.0)  # higher bound is 1
            row += 1
            I.append(row)
            J.append(dct[i, j])
            V.append(-1.0)
            B.append(0.0) # lower bound is 0
            row += 1

    ierows = row
    num_f = len(top_nodes)
    num_v = len(dct.keys())
    folder = "temp"

    if os.path.exists(folder) and os.path.isdir(folder):
        shutil.rmtree(folder)

    os.mkdir(folder)
    #store constraints in csv files
    np.array(I).tofile("{}/I_{}.csv".format(folder, fname), sep=",")
    np.array(J).tofile("{}/J_{}.csv".format(folder, fname), sep=",")
    np.array(B).tofile("{}/b_{}.csv".format(folder, fname), sep=",")
    np.array(V).tofile("{}/vals_{}.csv".format(folder, fname), sep=",")
    np.array(Ie).tofile("{}/Ie_{}.csv".format(folder, fname), sep=",")
    np.array(Je).tofile("{}/Je_{}.csv".format(folder, fname), sep=",")
    np.array(Be).tofile("{}/be_{}.csv".format(folder, fname), sep=",")
    np.array(Ve).tofile("{}/valse_{}.csv".format(folder, fname), sep=",")

    filename = '{}/fa_{}.csv'.format(folder, fname)

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['num_f', 'num_v', 'erows', 'ierows', 'all'])
        writer.writerow([num_f, num_v, erows, ierows, dct['all']])
    # initialize the matlab engine and run the linear program
    eng = matlab.engine.start_matlab()
    x, v = eng.ps(nargout=2)
    #process the results of the linear program
    x_p = np.array(x)
    ind = 0
    xx = {}
    for bot in bot_nodes:
        for top in top_nodes:
            if bot not in xx:
                xx[bot] = {}
            xx[bot][top] = x_p[dct[(top, bot)]]  # task and then worker.
    xw = {}
    for top in top_nodes:
        xw[top] = x_p[ind]
        ind += 1
    # compute expected waiting time (PT):
    mx_job = 0
    mx_vals = []
    for j in bot_nodes:
        sm = 0
        for i in N[j]:
            sm0 = 0
            sm1 = 0
            for jj in N[i]:
                sm0 += xx[jj][i][0] * Lam[jj] / (Mu[i][jj] ** 2)
                sm1 += xx[jj][i][0] * Lam[jj] / Mu[i][jj]
            sm += xx[j][i][0] * Mu[i][j] * sm0 / (1 - sm1)

        mx_vals.append(sm)
        mx_job = max(sm, mx_job)
    pt = mx_job
    ps = v

    return xx, pt, ps
