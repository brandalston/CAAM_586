import networkx as nx
import math, random, numpy, statistics


def x_given_y(y,B):
    u = numpy.random.uniform(0,1)
    return -(1/y)*math.log(1-u*(1-math.exp(-B*y)))


def y_given_x(x,B):
    u = numpy.random.uniform(0,1)
    return -(1/x)*math.log(1-u*(1-math.exp(-B*x)))


def problem_1():
    N = 100000
    B = [1,5,10,100]
    for b in B:
        x_0, y_0 = numpy.random.uniform(0, b), numpy.random.uniform(0, b)
        X, Y, XY = [x_0], [y_0], [x_0*y_0]
        for i in range(1, N):
            x = x_given_y(Y[-1], b)
            X.append(x)
            y = y_given_x(X[-1], b)
            Y.append(y)
            XY.append(X[-1]*Y[-1])
        avg_X = statistics.mean(X)
        avg_Y = statistics.mean(Y)
        avg_XY = statistics.mean(XY)
        print('B value:', b, 'E[X]', avg_X)
        print('B value:', b, 'E[Y]', avg_Y)
        print('B value:', b, 'E[XY]', avg_XY)
        print('\n')
    return


def x_g_yz(y,z,a,b):
    u = numpy.random.uniform(0, 1)
    return -math.log(1-u)/(1+a*y+b*z)


def y_g_xz(x,z,a,c):
    u = numpy.random.uniform(0, 1)
    return -math.log(1-u)/(1+a*x+c*z)


def z_g_xy(x,y,b,c):
    u = numpy.random.uniform(0, 1)
    return -math.log(1-u)/(1+b*x+c*y)


def problem_2():
    a, b, c = 1, 1, 1
    x_0, y_0, z_0 = 1/numpy.random.uniform(0, 1), 1/numpy.random.uniform(0, 1), 1/numpy.random.uniform(0, 1)
    N = 100000
    X, Y, Z, XYZ = [x_0], [y_0], [z_0], [x_0 * y_0*z_0]
    for i in range(1, N):
        x = x_g_yz(Y[-1], Z[-1], a, b)
        X.append(x)
        y = y_g_xz(X[-1], Z[-1], a, c)
        Y.append(y)
        z = z_g_xy(X[-1], Y[-1], b, c)
        Z.append(z)
        XYZ.append(x*y*z)
    avg_X = statistics.mean(X)
    avg_Y = statistics.mean(Y)
    avg_Z = statistics.mean(Z)
    avg_XYZ = statistics.mean(XYZ)
    print('Start: X', x_0, 'Y', y_0, 'Z', z_0)
    print('E[X]', avg_X)
    print('E[Y]', avg_Y)
    print('E[Z]', avg_Z)
    print('E[XYZ]', avg_XYZ)
    return


def problem_5():
    n = 4
    updates = 10000
    G = nx.grid_graph(dim=[n,n])
    zeta = numpy.random.uniform(0, 1)
    beta = 1/numpy.random.uniform(0, 1)
    for v in G.nodes:
        if numpy.random.uniform(0,1) < .5: G.nodes[v]['val'] = 1
        else: G.nodes[v]['val'] = -1
    X = {v: [G.nodes[v]['val']] for v in G.nodes}
    N = {v: None for v in G.nodes}
    M = {v: None for v in G.nodes}
    A = {v: None for v in G.nodes}
    B = {v: None for v in G.nodes}
    for i in range(1, updates):
        for v in G.nodes:
            N[v] = sum([1 if X[u][-1] == X[v][-1] else 0 for u in G.neighbors(v)])
            M[v] = len(list(G.neighbors(v)))-N[v]
            delta_v = 1 if X[v][-1] == X[v][0] else 0
            A[v] = zeta ** (1-delta_v) * (1-zeta) ** delta_v
            B[v] = zeta ** delta_v * (1 - zeta) ** (1 - delta_v)
            u = numpy.random.uniform(0, 1)
            pi_v = math.exp(-beta*M[v])*B[v] / (math.exp(-beta*N[v])*A[v]+math.exp(-beta*M[v])*B[v])
            if pi_v < u: X[v].append(-1)
            else: X[v].append(1)
    avg_v = {v: statistics.mean(X[v]) for v in G.nodes}
    for v in G.nodes:
        print(v, avg_v[v])
    return


# problem_1()
# problem_2()
problem_5()

