import numpy as np
import statistics, math, random
import scipy.stats


def sum_exp_raw_estimator():
    X = []
    for k in range(1000):
        if sum([-i*math.log(random.random()) for i in range(1,5)]) >= 21.6: X.append(1)
        else: X.append(0)
    return statistics.mean(X)


def sum_exp_anti_estimator():
    X = []
    for k in range(1000):
        val1, val2 = 0, 0
        for i in range(1, 5):
            u = random.random()
            x1, x2 = -math.log(u), -math.log(1-u)
            val1 += i*x1
            val2 += i*x2
        if val1 >= 21.6 or val2 >= 21.6: X.append(1)
        else: X.append(0)
    return 0.5*statistics.mean(X)


def problem_1():
    K = 1000
    E, E_anti = [], []
    for i in range(K):
        E.append(sum_exp_raw_estimator())
        E_anti.append(sum_exp_anti_estimator())
    print(f'Raw E: {statistics.mean(E)}, Var: {statistics.variance(E)}')
    print(f'Antithetic E: {statistics.mean(E_anti)}, Var: {statistics.variance(E_anti)}')
    print(f'Variance reduction factor using antithetic estimator: '
          f'{round(statistics.variance(E)/statistics.variance(E_anti),4)}')
    return


def related_normals_raw_estimator():
    X = []
    for k in range(1000):
        if np.random.normal(np.random.normal(1,1),4) < 1: X.append(1)
        else: X.append(0)
    return X


def related_normals_cond_estimator():
    X = []
    for k in range(1000):
        y = np.random.normal(1,1)
        A_y = 1-scipy.stats.norm.cdf(y, loc=1, scale=1.0)
        X.append(A_y)
    return X


def related_normals_cond_anti_estimator():
    X = []
    for k in range(1000):
        y1, y2 = np.random.normal(0,1/2), np.random.normal(1,math.sqrt(3)/2)
        a, b = y1+y2, y1-y2
        A = 1-scipy.stats.norm.cdf(a, loc=0, scale=1/2)
        B = 1-scipy.stats.norm.cdf(b, loc=1, scale=math.sqrt(3)/2)
        X.append(A+B)
    return X


def problem_3():
    raw_X = related_normals_raw_estimator()
    cond_X = related_normals_cond_estimator()
    cond_anti_X = related_normals_cond_anti_estimator()
    print(f'Raw E: {statistics.mean(raw_X)}, Var: {statistics.variance(raw_X)}\n'
          f'Conditional E: {statistics.mean(cond_X)}, Var: {statistics.variance(cond_X)}\n'
          f'Conditional + antithetic E: {statistics.mean(cond_anti_X)/2}, Var: {statistics.variance(cond_anti_X)}')
    return


def insurance_raw_estimator(N):
    X = []
    for k in range(N):
        u = random.random()
        c = np.random.poisson(15/(.5+u),1)
        if c >= 20: X.append(1)
        else: X.append(0)
    return X


def insurance_cond_control_estimator(N):
    X = []
    for k in range(N):
        u = random.random()
        lamd = 15/(0.5+u)
        X.append(1 - sum([math.exp(-lamd)*(lamd**k)/math.factorial(k) for k in range(20)]) + (1/lamd)*(u+0.5))
    return X


def insurance_cond_anti_estimator(N):
    X = []
    for k in range(N):
        u = random.random()
        lamd1, lamd2 = 15/(0.5+u), 15/(1.5-u)
        X.append(1/2*(1 - (sum([math.exp(-lamd1)*(lamd1**k)/math.factorial(k) for k in range(20)])) +
                      1 - (sum([math.exp(-lamd2)*(lamd2**k)/math.factorial(k) for k in range(20)]))
                      ))
    return X


def problem_4():
    N = 10000
    raw_X = insurance_raw_estimator(N)
    cond_control_X = insurance_cond_control_estimator(N)
    cond_anti_X = insurance_cond_anti_estimator(N)
    print(f'Raw E: {statistics.mean(raw_X)}, Var: {statistics.variance(raw_X)}')
    print(f'Conditional + control E: {statistics.mean(cond_control_X)}, Var: {statistics.variance(cond_control_X)}')
    print(f'Conditional + anti E: {statistics.mean(cond_anti_X)}, Var: {statistics.variance(cond_anti_X)}')
    return


def SSQ(T):
    # dicts storing system through time
    Arrivals, Departures, Num_System, U_dict_gen = {}, {}, {}, {}
    # variables of system
    t, num_arriv, num_depart, num_system, num_event = 0, 0, 0, 0, 0
    # first arrival time t_A, departure time t_D
    u = random.random()
    U_dict_gen[num_event] = u
    t_A, t_D = -(1/2)*math.log(u), math.inf
    running = True
    while running:
        # event is arrival
        if t_A < t_D and t_A < T:
            num_event += 1
            # update t, number arrivals, number people in system
            t = t_A
            num_arriv += 1
            num_system += 1
            # update arrival time
            u = random.random()
            U_dict_gen[num_event] = u
            t_A = t-(1/2)*math.log(u)
            # update departure time based on number of people in system
            if num_system == 1:
                t_D = t-math.log(u)
            # store event data
            Arrivals[num_arriv] = t
        # event is departure
        elif t_D < t_A and t_D < T:
            num_event += 1
            t = t_D
            num_system -= 1
            num_depart += 1
            u = random.random()
            U_dict_gen[num_event] = u
            # update/reset departure time
            if num_system == 0:
                t_D = math.inf
            elif num_system > 0:
                t_D = t - math.log(u)
            # store event data
            Departures[num_depart] = t
        # at close but people in queue
        elif min(t_A, t_D) > T and num_system > 0:
            num_event += 1
            # next event must be departure
            t = t_D
            num_system -= 1
            num_depart += 1
            u = random.random()
            U_dict_gen[num_event] = u
            # reset departure time
            t_D = t - math.log(u)
            Departures[num_depart] = t
        elif min(t_A, t_D) > T and num_system == 0:
            running = False
    return Arrivals, Departures, U_dict_gen


def anti_SSQ(T):
    A, D, U_dict = SSQ(T)
    # dicts storing system through time
    Arrivals, Departures, Num_System = {}, {}, {}
    # variables of system
    t, num_arriv, num_depart, num_system, num_event = 0, 0, 0, 0, 0
    # first arrival time t_A, departure time t_D
    u = 1-U_dict[num_event]
    t_A, t_D = -(1 / 2) * math.log(u), math.inf
    running = True
    while running:
        if len(Departures) > 11:
            running=False
        # event is arrival
        if t_A < t_D and t_A < T:
            num_event += 1
            # update t, number arrivals, number people in system
            t = t_A
            num_arriv += 1
            num_system += 1
            # update arrival time
            u = 1 - U_dict[num_event]
            t_A = t - (1 / 2) * math.log(u)
            # update departure time based on number of people in system
            if num_system == 1:
                t_D = t - math.log(u)
            # store event data
            Arrivals[num_arriv] = t
        # event is departure
        elif t_D < t_A and t_D < T:
            num_event += 1
            t = t_D
            num_system -= 1
            num_depart += 1
            u = 1 - U_dict[num_event]
            # update/reset departure time
            if num_system == 0:
                t_D = math.inf
            elif num_system > 0:
                t_D = t - math.log(u)
            # store event data
            Departures[num_depart] = t
        # at close but people in queue
        elif min(t_A, t_D) > T and num_system > 0:
            num_event += 1
            # next event must be departure
            t = t_D
            num_system -= 1
            num_depart += 1
            u = 1 - U_dict[num_event]
            # reset departure time
            t_D = t - math.log(u)
            Departures[num_depart] = t
        elif min(t_A, t_D) > T and num_system == 0:
            running = False
    Arrivals_combo = {i: 0.5*(A[i]+Arrivals[i]) for i in Arrivals}
    Departures_combo = {i: 0.5*(D[i]+Departures[i]) for i in Arrivals}

    return Arrivals_combo, Departures_combo


def problem_6():
    T, N = 10, 2
    X_raw, X_anti, X_cond_est = [], [], []
    for k in range(N):
        # print(k)
        A_raw, D_raw, U_dict = SSQ(T)
        times_raw = {i: D_raw[i]-A_raw[i] for i in A_raw}
        # print('raw:', times)
        A_anti, D_anti = anti_SSQ(T)
        times_anti = {i: D_anti[i]-A_anti[i] for i in A_anti}
        # print('anti:', times_anti)
        X_raw.append(statistics.mean(times_raw.values()))
        X_anti.append(statistics.mean(times_anti.values()))
    print('raw:', statistics.mean(X_raw), statistics.variance(X_raw))
    print('anti:', statistics.mean(X_anti), statistics.variance(X_anti))
    return


def twoD_gauss_MC(a, mean, cov):
    rvs = np.random.multivariate_normal(mean, cov, size=100000)
    X = [1 if (i[0] >= a and i[1] >= a) else 0 for i in rvs]
    if (mean != np.array([0,0])).all():
        L = [math.exp(-mean@np.linalg.inv(cov)@np.array(i)+
             0.5*mean@np.linalg.inv(cov)@mean)
             for i in rvs]
        X = np.array(X)*L
    print(f'E: {statistics.mean(X)}, Var: {statistics.variance(X)}')
    CI = scipy.stats.t.interval(alpha=0.95, df=len(X)-1, loc=np.mean(X), scale=scipy.stats.sem(X))
    print(f'95% CI: {CI}')
    return


def problem_7():
    a = [1,3,10]
    mean_raw = np.array([0, 0])
    cov = np.array([[4, -1], [-1, 4]])
    deltas = [0.001, 2, 10]
    for i in a:
        print(f'Crude MC simulation, a = {i}')
        twoD_gauss_MC(i, mean_raw, cov)
        print(f'Importance Sampling MC simulation, a = {i}')
        mean_best = np.array([i, i])
        twoD_gauss_MC(i, mean_best, cov)
        for d in deltas:
            print(f'Importance Sampling MC simulation, a = {i}, delta = {d}')
            twoD_gauss_MC(i, mean_best, d*cov)
        print('')
    return


def CMC(repeats, size):
    K = []
    for k in range(repeats):
        P = {i: np.random.beta(1,19) for i in range(size)}
        X = {i: np.random.normal(3,1) for i in range(size)}
        D = {i: np.random.binomial(1,P[i]) for i in P}
        L_n = {i: D[i]*X[i] for i in range(size)}
        if sum(list(L_n.values())) > 45: K.append(1)
        else: K.append(0)
    return K


def cond_MC(repeats, size):
    X = []
    for k in range(repeats):
        P = {i: np.random.beta(1,19) for i in range(size)}
        D = {i: np.random.binomial(1,P[i]) for i in P}
        theta = 3*sum(list(D.values()))
        if theta > 45: X.append(1)
        else: X.append(0)

    return X


def problem_9():
    size, repeats = 100, 10000
    raw_X = CMC(repeats, size)
    cond_X = cond_MC(repeats, size)
    print(f'Raw E: {statistics.mean(raw_X)}, Var: {statistics.variance(raw_X)}')
    print(f'Conditional E: {statistics.mean(cond_X)}, Var: {statistics.variance(cond_X)}')

    return

"""
# Executable code
print('\nProblem 1')
problem_1()
print('\nProblem 3')
problem_3()
print('\nProblem 4')
problem_4()
print('\nProblem 6')
problem_6()
print('\nProblem 7')
problem_7()
print('\nProblem 9')
problem_9()
"""
problem_7()
