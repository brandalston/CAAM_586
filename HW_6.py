import numpy as np
import statistics, math, random
import scipy.stats as st


def problem_2():
    N, p, a = 40, 0.2, 16
    E_raw, E_tilt = [], []
    repeats, num_ = 10000, 100
    for k in range(repeats):
        raw_X, tilt_X = [], []
        for i in range(num_):
            iid_raw = np.random.binomial(size=N, n=1, p=p)
            if sum(iid_raw) >= a: raw_X.append(1)
            else: raw_X.append(0)
            iid_tilt = np.random.binomial(size=N, n=1, p=.4)
            if sum(iid_tilt) >= a: tilt_X.append((4/3)**40*(3/8)**sum(iid_tilt))
            else: tilt_X.append(0)
        E_raw.append(statistics.mean(raw_X))
        E_tilt.append(statistics.mean(tilt_X))

    print(f'Raw E: {round(statistics.mean(E_raw), 6)}. Tilted E: {round(statistics.mean(E_tilt), 6)}')
    print(f'Raw Var: {round(statistics.variance(E_raw), 6)}. Tilted Var: {round(statistics.variance(E_tilt), 6)}')
    return


def problem_3():
    X, mu, sigma = 10, -.1, .5
    theta = -2*mu/sigma
    repeats = 10000
    E, B = [], {}
    for i in range(repeats):
        S_tau = np.random.normal(mu+theta*sigma, math.sqrt(sigma))
        while S_tau <= X:
            S_tau += np.random.normal(mu+theta*sigma,math.sqrt(sigma))
        B[i] = S_tau - X
        E.append(math.exp(-X*theta)/len(B)*sum([math.exp(-theta*B[i]) for i in B]))
    print(f'E: {statistics.mean(E)}, Var: {statistics.variance(E)}'
          f'\n95% CI: {st.t.interval(alpha=0.99, df=len(E)-1, loc=statistics.mean(E), scale=statistics.variance(E))}')
    return


def problem_4():
    d, lamda, mu = 6, 1, 3
    theta = .697224
    E, B = [], {}
    repeats = 10000
    for i in range(repeats):
        delta = 0
        while delta <= d:
            delta += (st.erlang.rvs(a=2, loc=mu-theta)-np.random.exponential(lamda-theta))
        B[i] = delta-d
        E.append(math.exp(-d*theta)/len(B)*sum([math.exp(-theta*B[i]) for i in B]))
    print(f'E: {statistics.mean(E)}')
    return



print('\nProblem 2')
problem_2()
print('\nProblem 3')
problem_3()
print('\nProblem 4')
problem_4()
