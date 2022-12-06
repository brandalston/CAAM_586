import math, pandas, numpy, statistics, networkx
desired_width = 320
pandas.set_option('display.width', desired_width)
numpy.set_printoptions(linewidth=desired_width)
pandas.set_option('display.max_columns', 10)


def a_given_b(B, x=None, y=None):
    u = numpy.random.uniform(0,1)
    if x is None:
        return -(1/y)*math.log(1-u*(1-math.exp(-B*y)))
    if y is None:
        return -(1/x)*math.log(1-u*(1-math.exp(-B*x)))


def problem_1():
    N = 10**6
    B = [1, 5, 10, 100]
    for b in B:
        x_0, y_0 = numpy.random.uniform(0, b), numpy.random.uniform(0, b)
        X, Y, XY = [x_0], [y_0], [x_0*y_0]
        for i in range(1, N):
            x = a_given_b(b, x=None, y=Y[-1])
            X.append(x)
            y = a_given_b(b, x=X[-1], y=None)
            Y.append(y)
            XY.append(X[-1]*Y[-1])
        avg_X = statistics.mean(X)
        avg_Y = statistics.mean(Y)
        avg_XY = statistics.mean(XY)
        print('\nB value:', b)
        print('E[X]:', round(avg_X, 4))
        print('E[Y]:', round(avg_Y, 4))
        print('E[XY]:', round(avg_XY, 4))
    return


def tri_var_cond(a, b, c, x=None, y=None, z=None):
    u = numpy.random.uniform(0,1)
    if x is None: return -math.log(1-u)/(1+a*y+b*z)
    if y is None: return -math.log(1-u)/(1+a*x+c*z)
    if z is None: return -math.log(1-u)/(1+b*x+c*y)


def problem_2():
    a, b, c = 1, 1, 1
    x_0, y_0, z_0 = 1/numpy.random.uniform(0, 1), 1/numpy.random.uniform(0, 1), 1/numpy.random.uniform(0, 1)
    N = 10**6
    X, Y, Z, XYZ = [x_0], [y_0], [z_0], [x_0 * y_0*z_0]
    for i in range(1, N):
        x = tri_var_cond(a, b, c, x=None, y=Y[-1], z=Z[-1])
        X.append(x)
        y = tri_var_cond(a, b, c, x=X[-1], y=None, z=Z[-1])
        Y.append(y)
        z = tri_var_cond(a, b, c, x=X[-1], y=Y[-1], z=None)
        Z.append(z)
        XYZ.append(x*y*z)
    avg_X = statistics.mean(X)
    avg_Y = statistics.mean(Y)
    avg_Z = statistics.mean(Z)
    avg_XYZ = statistics.mean(XYZ)
    print('Start X:', round(x_0, 4), 'Y:', round(y_0, 4), 'Z:', round(z_0, 4))
    print('E[X]:', round(avg_X, 4))
    print('E[Y]:', round(avg_Y, 4))
    print('E[Z]:', round(avg_Z, 4))
    print('E[XYZ]:', round(avg_XYZ, 4))
    return


def image_analysis():
    grid_size = [3, 5, 10]
    grid_size = [3]
    updates = 10**6
    zeta = numpy.random.uniform(0, 1)
    beta = 1 / numpy.random.uniform(0, 1)
    for n in grid_size:
        G = networkx.grid_graph(dim=[n, n])
        X = {v: [+1 if numpy.random.uniform(0, 1) < .5 else -1] for v in G.nodes}
        G_0 = pandas.DataFrame(numpy.nan, index=range(n), columns=range(n))
        for v in G.nodes:
            G_0.at[v[0], v[1]] = X[v][-1]
        N, M = {v: None for v in G.nodes}, {v: None for v in G.nodes}
        A, B = {v: None for v in G.nodes}, {v: None for v in G.nodes}
        for i in range(1, updates):
            for v in G.nodes:
                N[v] = sum([1 if X[u][-1] != X[v][-1] else 0 for u in G.neighbors(v)])
                M[v] = len(list(G.neighbors(v)))-N[v]
                delta_v = 1 if X[v][-1] == X[v][0] else 0
                A[v] = zeta ** (1-delta_v) * (1-zeta) ** delta_v
                B[v] = zeta ** delta_v * (1 - zeta) ** (1 - delta_v)
                pi_v = (math.exp(-beta*M[v])*B[v]) / (math.exp(-beta*N[v])*A[v] + math.exp(-beta*M[v])*B[v])
                X[v].append(X[v][-1]) if pi_v <= numpy.random.uniform(0, 1) else X[v].append(-X[v][-1])
        avg_v = pandas.DataFrame(numpy.nan, index=range(n), columns=range(n))
        for v in G.nodes:
            avg_v.at[v[0], v[1]] = statistics.mean(X[v])
        print(f'\nImage size ({n}x{n})', 'Zeta:', round(zeta, 4), 'Beta:', round(beta, 4))
        print('Initial')
        print(G_0)
        print('updated')
        print(avg_v)
    return


# print('\nProblem 1 (Ross 12.6)')
# problem_1()
# print('\nProblem 2 (Ross 12.11)')
# problem_2()
print('\nProblem 5 (Asmussen and Glynn 5.7)')
image_analysis()
