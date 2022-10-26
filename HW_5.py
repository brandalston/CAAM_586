import numpy as np
import pandas as pd
import statistics, math, random
import scipy.stats
from matplotlib import pyplot as plt


def sum_exp_raw_estimator():
    X = []
    for k in range(1000):
        val = 0
        for i in range(1,5):
            val -= i*math.log(random.random())
        if val >= 21.6: X.append(1)
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
    print('Variance reduction using antithetic estimator',
          round(2*statistics.variance(E_anti)/statistics.variance(E),4))
    return


def related_normals_raw_estimator():
    X = []
    for k in range(1000):
        y = np.random.normal(1,1)
        x = np.random.normal(y,4)
        if x < 1: X.append(1)
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
    print(f'Raw expectation: {statistics.mean(raw_X)}. Variance: {statistics.variance(raw_X)}\n'
          f'Conditional: {statistics.mean(cond_X)}. Variance: {statistics.variance(cond_X)}\n'
          f'Conditional + antithetic: {statistics.mean(cond_anti_X)/2}. Variance: {statistics.variance(cond_anti_X)}')
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


def insurance_cond_anti(N):
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
    cond_anti_X = insurance_cond_control_estimator(N)
    print(statistics.mean(raw_X), statistics.variance(raw_X))
    print(statistics.mean(cond_control_X), statistics.variance(cond_control_X))
    print(statistics.mean(cond_anti_X), statistics.variance(cond_anti_X))
    return


def SSQ(T, N):
    X = []
    for k in range(N):
        # first arrival time t_A, departure time t_D
        t_A, t_D = -(1/2)*math.log(random.random()), math.inf
        # dicts storing system through time
        Arrivals, Departures = {}, {}
        # variables of system
        t, num_arriv, num_depart, num_system = 0, 0, 0, 0
        running = True
        while running:
            # event is arrival
            if t_A < t_D and t_A < T:
                # update t, number arrivals, number people in system
                t = t_A
                num_arriv += 1
                num_system += 1
                # update arrival time
                t_A = t-(1/2)*math.log(random.random())
                # update departure time based on number of people in system
                if num_system == 1:
                    t_D = t-math.log(random.random())
                # store event data
                Arrivals[num_arriv] = t
            # event is departure
            elif t_D < t_A and t_D < T:
                t = t_D
                num_system -= 1
                num_depart += 1
                Y = -math.log(random.random())
                # update/reset departure time
                if num_system == 0:
                    t_D = math.inf
                elif num_system > 0:
                    t_D = t + Y
                # store event data
                Departures[num_depart] = t
            # at close but people in queue
            elif min(t_A, t_D) > T and num_system > 0:
                # next event must be departure
                t = t_D
                num_system -= 1
                num_depart += 1
                Y = -math.log(random.random())
                # reset departure time
                t_D = t + Y
                Departures[num_depart] = t
            elif min(t_A, t_D) > T and num_system == 0:
                running = False
        Total_time = {i: Departures[i]-Arrivals[i] for i in range(1,11)}
        X.append(statistics.mean(list(Total_time.values())))
    return X


def anti_SSQ(T, N):
    # first arrival time t_A, departure time t_D
    u = random.random()
    t_A1, t_A2 = -(1/2)*math.log(u), -(1/2)*math.log(1-u)
    t_D1, t_D2 = math.inf, math.inf
    # dicts storing system through time
    Arrivals1, Departures1 = {}, {}
    Arrivals2, Departures2 = {}, {}
    # variables of system
    num_arriv, num_depart, num_system = 0, 0, 0
    t1, t2 = 0, 0
    running = True
    while running:
        # event is arrival
        if t_A1 < t_D1 and t_A1 < T:
            # update t, number arrivals, number people in system
            t1 = t_A1
            t2 = t_A2
            num_arriv += 1
            num_system += 1
            # update arrival time
            u = random.random()
            t_A1 = t1-(1/2)*math.log(u)
            t_A2 = t2-(1/2)*math.log(1-u)
            # update departure time based on number of people in system
            if num_system == 1:
                u_D = random.random()
                t_D1 = t1-math.log(u_D)
                t_D2 = t2-math.log(1-u_D)
            # store event data
            Arrivals1[num_arriv] = t1
            Arrivals2[num_arriv] = t2
        # event is departure
        elif t_D1 < t_A1 and t_D1 < T:
            t1 = t_D1
            t2 = t_D2
            num_system -= 1
            num_depart += 1
            u = random.random()
            Y1, Y2 = -math.log(u), -math.log(1-u)
            # update/reset departure time
            if num_system == 0:
                t_D1 = math.inf
                t_D2 = math.inf
            elif num_system > 0:
                t_D1 = t1 + Y1
                t_D2 = t2 + Y2
            # store event data
            Departures1[num_depart] = t1
            Departures2[num_depart] = t2
        # at close but people in queue
        elif min(t_A1, t_D1) > T and num_system > 0:
            # next event must be departure
            t1 = t_D1
            t2 = t_D2
            num_system -= 1
            num_depart += 1
            u = random.random()
            Y1, Y2 = -math.log(u), -math.log(1-u)
            # reset departure time
            t_D1 = t1 + Y1
            t_D2 = t2 + Y2
            Departures1[num_depart] = t1
            Departures2[num_depart] = t2
        elif min(t_A1, t_D1) > T and num_system == 0:
            running = False
    Total_time1 = {i: Departures1[i]-Arrivals1[i] for i in range(1,11)}
    Total_time2 = {i: Departures2[i]-Arrivals2[i] for i in range(1,11)}
    for i in range(1,11):
        print(Total_time1[i], Total_time2[i])
        # X.append(statistics.mean(list(Total_time1.values())))
    return


def problem_5():
    T, N = 20, 1000
    raw_X = SSQ(T, N)
    anti_X = anti_SSQ(T, N)
    # print(statistics.mean(raw_X))
    # print(statistics.variance(raw_X))
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
    return

# problem_1()
# problem_3()
# problem_4()
# problem_5()
problem_9()
