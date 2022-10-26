import numpy as np
import pandas as pd
import statistics, math, random
from matplotlib import pyplot as plt


def poisson(T, lamda):
    Ta = []
    t = -math.log(random.random()) / lamda
    while t < T:
        Ta.append(t)
        t = t -math.log(random.random()) / lamda
    return Ta


def CTMC(T):
    # using DTMC
    t, N, markov_path_1 = 0, 0, {}
    Q = [[-3,3,0,0,0,0],
         [2,-5,3,0,0,0],
         [0,4,-7,3,0,0],
         [0,0,4,-7,3,0],
         [0,0,0,4,-7,3],
         [0,0,0,0,4,-4]]
    P = [[0,1,0,0,0,0],
         [2/5,0,3/5,0,0,0],
         [0,4/7,0,3/7,0,0],
         [0,0,4/7,0,3/7,0],
         [0,0,0,4/7,0,3/7],
         [0,0,0,0,1,0]]
    P, Q = pd.DataFrame(P), pd.DataFrame(Q)
    while t < T:
        # update v_N according to state
        v_N = -Q[N][N]
        # generate poisson time in state
        T_state = -(1/v_N)*math.log(random.random())
        if N == 0: N = 1
        elif N == 5: N = 4
        else:
            # generate next state
            if random.random() < P[N][N+1]: N += 1
            else: N -= 1
        # store state at time t
        markov_path_1[t] = N
        t += T_state

    # uniformization method
    P_tilda = [[4/7,3/7,0,0,0,0],
               [2/7,2/7,3/7,0,0,0],
               [0,4/7,0,2/7,0,0],
               [0,0,4/7,0,2/7,0],
               [0,0,0,4/7,0,2/7],
               [0,0,0,0,4/7,3/7]]
    N = 0
    markov_path_2 = {}
    # get lamda >= v_n for all states n
    lamda_star = max([-Q[n][n] for n in range(6)])
    # generate all poisson state times up to T
    times = poisson(T, lamda_star)
    for t in times:
        if N == 0: N = 1
        elif N == 5: N = 4
        else:
            # generate next state
            if random.random() < P_tilda[N][N + 1]: N += 1
            else: N -= 1
        # store state at time t
        markov_path_2[t] = N

    return markov_path_1, markov_path_2


def problem_1():
    T, repeats = 10, 1000
    IDC_dict1, IDC_dict2 = {}, {}
    for i in range(repeats):
        path_1, path_2 = CTMC(T)
        IDC_dict1[i], IDC_dict2[i] = path_1, path_2
    E1 = np.mean([max(IDC_dict1[i]) for i in IDC_dict1])
    Var1 = np.var([max(IDC_dict1[i]) for i in IDC_dict1])
    IDC_1 = Var1 / E1
    E2 = np.mean([max(IDC_dict2[i]) for i in IDC_dict2])
    Var2 = np.var([max(IDC_dict2[i]) for i in IDC_dict2])
    IDC_2 = Var2 / E2
    print('DTMC Method IDC:', IDC_1,'\nUniformization Method IDC:',IDC_2)
    return


def MMPP(T):
    lamda = {0: 1, 1: 2}
    N = 0
    A = {}
    t = 0
    state = 0
    while t < T:
        # generate poisson time T_state in state
        T_state = -(1/lamda[state])*math.log(random.random())
        # generate all arrival times up to T_state
        arrival_times = poisson(T_state, lamda[state])
        # update interval to only contain generated arrival times after t
        arrival_times = [x + t for x in arrival_times]
        for t in arrival_times:
            # store state at time
            N += 1
            A[t] = N
        t += T_state
        # update state
        state = N%2
    return A


def problem_2():
    IDC_dict = {}
    repeats = 500
    T = 50
    for i in range(repeats):
        IDC_dict[i] = MMPP(T)
    E_t = np.mean([max(IDC_dict[i]) for i in IDC_dict])
    Var_t = np.var([max(IDC_dict[i]) for i in IDC_dict])
    IDC = Var_t/E_t
    print('IDC:',round(IDC,5))


def SPRW(K):
    Z = {n: random.uniform(-0.5,1.5) for n in range(1,K)}
    X = {n: 0 for n in range(0, K)}
    for n in range(0,K-1):
        X[n+1] = max(X[n]+Z[n+1],0)
    return X


def problem_3():
    repeats, P, K = 1000, {}, 50
    for i in range(repeats):
        P[i] = SPRW(K)
    X_40 = [P[i][40] for i in P]
    E_40 = np.mean(X_40)
    Var_40 = np.var(X_40)
    print('E[X_40]:',E_40,'\nVar[X_40]:',Var_40)
    return


def problem_4():
    over = {i: {'var': 0, 'mean': 0} for i in range(100)}
    for i in range(100):
        # generate first two normals
        X = [np.random.normal(), np.random.normal()]
        # while sample variance < .1 and N < 100
        while statistics.variance(X)/math.sqrt(len(X)) > .1 or len(X) < 100:
            # generate new normal
            X.append(np.random.normal())
        over[i]['var'] = statistics.variance(X)
        over[i]['mean'] = statistics.mean(X)
    var = statistics.mean([over[i]['var'] for i in over])
    mean = statistics.mean([over[i]['mean'] for i in over])
    print('Number generated: ',len(over),'\nSample mean:', mean,'\nSample var: ',var,'\nSample var/sqrt(n):',var/math.sqrt(len(over)))


def problem_5():
    e_est, e_dict = 0, {}
    for i in range(1000):
        n = 2
        u_past = random.random()
        u_present = random.random()
        while u_past <= u_present:
            u_past = u_present
            u_present = random.random()
            n += 1
        e_dict[i] = n
    e_est, e_var = statistics.mean(e_dict.values()), statistics.variance(e_dict.values())
    confidence_inter = e_var*1.96/math.sqrt(len(e_dict))
    print('95% CI: [',e_est-confidence_inter,',',e_est+confidence_inter,']')
    return


def CSSQ(T):
    N, t = 0, 0
    t_A, t_D = -(1/4)*math.log(random.random()), math.inf
    # dicts storing system through time
    A, D = [], []
    # variables of system
    t, num_arriv, num_depart, num_system, num_event = 0, 0, 0, 0, 0
    running = True
    while running:
        # print(t)
        # event is arrival
        if t_A < t_D and t_A < T and N <= 3:
            # less than 3 customers, customer enters
            N += 1
            # update total time t and gen next arrival time
            t = t_A
            t_A = t-(1/4)*math.log(random.random())
            # update departure time
            if N == 1:
                t_D = t - (1/4.2)*math.log(random.random())
            A.append(t)
        # event is departure
        elif t_D < t_A and t_D < T:
            N -= 1
            t = t_D
            # update departure time
            if N == 0:
                t_D = math.inf
            elif N > 0:
                t_D = t - (1/4.2)*math.log(random.random())
            D.append(t)
        elif min(t_A, t_D) > T and num_system == 0:
            running = False
    print(A)
    print(D)
    return A, D


def problem_6():
    T = 8
    CSSQ(T)
    return


print('\nProblem 1')
problem_1()
print('\nProblem 2')
problem_2()
print('\nProblem 3')
problem_3()
print('\nProblem 4')
problem_4()
print('\nProblem 5')
problem_5()
print('\nProblem 6')
print('couldnt get my code to work properly')
# problem_6()