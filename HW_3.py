import numpy as np
from matplotlib import pyplot as plt
import math
import random


def SSQ(T:float):
    # first arrival time t_A, departure time t_D
    t_A, t_D = -0.1*math.log(random.random()), math.inf
    # dicts storing system through time
    Arrivals, Departures, System, idle_time, overtime = {}, {}, {}, 0, 0
    # variables of system
    t, num_arriv, num_depart, num_system, num_event = 0, 0, 0, 0, 0
    running = True
    # event is arrival
    while running:
        if t_A < t_D and t_A < T:
            # update t, number arrivals, number people in system
            t_old = t
            t = t_A
            num_arriv += 1
            num_system += 1
            # update arrival time
            t_A = t-(1/10)*math.log(random.random())
            # update departure time based on number of people in system
            if num_system == 1:
                t_D = t-(1/40)*math.log(np.prod(np.random.rand(3)))
            # store event data
            Arrivals[num_arriv] = t
            num_event += 1
            System[t] = num_system
            # update idle time
            if num_system == 1:
                idle_time += (t-t_old)
        # event is departure
        elif t_D < t_A and t_D < T:
            t = t_D
            num_system -= 1
            num_depart += 1
            Y = -(1/40)*math.log(np.prod(np.random.rand(3)))
            # update/reset departure time
            if num_system == 0:
                t_D = math.inf
            elif num_system > 0:
                t_D = t + Y
            # store event data
            Departures[num_depart] = t
            num_event += 1
            System[t] = num_system
        # at close but people in queue
        elif min(t_A, t_D) > T and num_system > 0:
            # next event must be departure
            t = t_D
            num_system -= 1
            num_depart += 1
            Y = -(1/40)*math.log(np.prod(np.random.rand(3)))
            # reset departure time
            t_D = t + Y
            Departures[num_depart] = t
            num_event += 1
            System[t] = num_system
        elif min(t_A, t_D) > T and num_system == 0:
            running = False
            overtime = max(0, t-T)
    return Arrivals, Departures, System, idle_time, overtime


def problem_1():
    customer_time, overtime, idle_time = 0, 0, 0
    iterations = [100,1000]
    for m in iterations:
        for i in range(m):
            A, D, S, idle, over = SSQ(9)
            idle_time += idle
            overtime += over
            time_in_sys = {customer: D[customer]-A[customer] for customer in A}
            customer_time += sum(list(time_in_sys.values()))/len(time_in_sys)
        customer_time /= m
        overtime /= m
        idle_time /= m
        print(f'{m} iterations: Avg customer time in system: {round(customer_time,4)}. '
              f'Avg idle time: {round(idle_time,4)}. Avg overtime: {round(overtime,4)}')
    return


def PP_lambda(t):
    return 19 - 3*abs((t % 10) - 5)


def next_arrival(t):
    k = 2
    while k > PP_lambda(t)/19:
        k = random.random()
        t -= (1/19)*math.log(random.random())
    return t


def NH_SSQ(T):
    t_A = next_arrival(0)
    t_B, t_D = 0.3*random.random(), math.inf
    t, N_A, N_D = 0, 0, 0
    A, D, B_L, B_W, E, S, J = {}, {}, {}, {}, {}, {}, {}  # dicts to record event times/number of each event
    N_B, K = 1, 1  # assume start on break
    S[K], E[K], J[K] = 0, 0, 0
    B_L[N_B] = 0
    while min(t_A, t_D, t_B) <= T:
        K += 1
        # customer arrives as event
        if t_A < t_D and t_A < t_B:
            t = t_A
            N_A += 1
            A[N_A] = t
            E[K] = t
            S[K] = S[K-1]+1
            J[K] = J[K-1]
            # update next arrival/departure times
            t_A = next_arrival(t)
            # server working
            if S[K] > 1:
                t_D = t_D
            elif S[K] == 1 and J[K] == 1:
                t_D = t-(1/25)*math.log(random.random())
            # server not working
            elif J[K] == 0:
                t_D = math.inf

        # complete job as event
        elif t_D < t_A and t_D < t_B:
            t = t_D
            N_D += 1
            E[K] = t
            S[K] = S[K-1]-1
            # update server break position
            if S[K] > 0:
                J[K] = 1
            elif S[K] == 0:
                J[K] = 0
            # update t_B and t_D
            if S[K] == 0:
                t_D = math.inf
            elif S[K] > 0:
                t_D = t - (1/25)*math.log(random.random())
            if J[K] == 0:
                t_B = t+0.3*random.random()
            elif J[K] == 1:
                t_B = math.inf
            # store departure event data
            D[N_D] = t
            if S[K] == 0:
                N_B += 1
                B_L[N_B] = t
        # return from break as event
        elif t_B < t_A and t_B < t_D:
            t = t_B
            B_W[N_B] = t
            E[K] = t
            S[K] = S[K-1]
            if S[K] > 0:
                J[K] = 1
            elif S[K] == 0:
                J[K] = 0
            if S[K] == 0:
                N_B += 1
                B_L[N_B] = t
            # update t_B and t_D
            if S[K] == 0:
                t_D = math.inf
            elif S[K] > 0:
                t_D = t-(1/25)*math.log(random.random())
            if S[K] == 0:
                t_B = t + 0.3*random.random()
            elif J[K] == 1:
                t_B = math.inf
    if J[K] == 0:
        B_W[N_B] = T
    return A, D, B_L, B_W, S, E, J, K


def problem_2():
    m = 500
    break_time = 0
    for i in range(m):
        A, D, B_L, B_W, S, E, J, K = NH_SSQ(100)
        time_on_break = {number: B_W[number]-B_L[number] for number in B_L}
        break_time += sum(list(time_on_break.values()))
    break_time /= m
    print('Avg time on break:', round(break_time,4))
    return


def poisson(T, lamda):
    Ta = []
    t = -math.log(random.random()) / lamda
    while t < T:
        Ta.append(t)
        t = t -math.log(random.random()) / lamda
    return Ta


def CD_SSQ(T, lamda):
    arrival_times = poisson(T, lamda)  # arrival times
    n = len(arrival_times)  # num arrivals
    server_times = np.zeros(n)  # Server Start time
    departure_times = np.zeros(n)  # Depart time
    idle_times = np.zeros(n)  # Idle time
    wait_times = np.zeros(n)  # Wait time

    server_times[0] = arrival_times[0]  # enter system time of first customer
    G = -(1/4)*math.log(random.random())  # service time of first customer
    departure_times[0] = server_times[0] + G  # exit time of first customer
    idle_times[0] = arrival_times[0]  # idle time of server at beginning
    wait_times[0] = 0  # no wait time of first customer to enter system

    jobs_leaving = np.zeros(n)  # event number aka customer who left
    for iter in range(1,n):
        # next event is arrival or departure
        server_times[iter] = max(departure_times[iter-1], arrival_times[iter])
        # wait time of next event to enter service
        wait_times[iter] = server_times[iter] - arrival_times[iter]
        # generate leave time
        F = 5 * random.random()
        # leave if leave time < wait time
        if wait_times[iter] > F:
            departure_times[iter] = departure_times[iter - 1]
            jobs_leaving[iter] = 1
        else:
            # generate service time and update time to leave system
            G = -(1/4)*math.log(random.random())
            departure_times[iter] = server_times[iter] + G
    number_left = sum(jobs_leaving)
    return number_left


def problem_3():
    m = 500
    avgleft = 0

    for iter in range(m):
        avgleft = avgleft + CD_SSQ(100, 5)

    avgleft = avgleft / m
    print("Expected number of lost customers:", avgleft)
    return


def insurance_claim(T, lamda, capital):
    t = 0
    while t < T:
        # generate claim arrival time
        ta = -(1/lamda)*math.log(random.random())
        # update time
        t = t + ta
        # claim amount
        claim = -1000*math.log(random.random())
        # payment amount
        payment = ta*11000
        # update capital
        capital += (payment - claim)
        # if capital reach 0 at any point then option risk policy fails
        if capital < 0:
            return 0
    return 1


def problem_4():
    m = 1000
    prob_good = 0
    for iter in range(m):
        prob_good += insurance_claim(365, 10, 25000)
    prob_good /= m
    print("Probability always positive:", prob_good)


def shock_value():
    running = True
    while running:
        y = -2*math.log(2*random.random())
        u = random.random()
        if u <= (y*math.exp(-y))/((1/2)*math.exp(-y/2)):
            running = False
        return y


def shocks(C, alpha):
    ASV, T, A, shock_residuals = 0, 0, {}, {}
    iter = 0
    while ASV < C:
        # generate shock time
        t = -(1/10)*math.log(random.random())
        # update total time
        T += t
        # generate shock value
        shock_val = shock_value()
        A[t] = shock_val
        # shock residual
        shock_residuals[t] = shock_val*math.exp(-alpha*(T-list(A.values()).index(shock_val)))
        ASV = sum([shock_residuals[t] for t in shock_residuals])
    return T


def problem_5():
    C, alpha = 5, 0.5
    avg_time, K = 0, 1000
    for i in range(K):
        avg_time += shocks(C, alpha)
    avg_time /= K
    print('Average time to system fail:',avg_time)


def problem_6():
    N, K, S_0, mu, theta = 20, 100, 100, -0.05, .03
    n, R, V, avg_V = 0, S_0, np.zeros(N), {i: None for i in range(N)}
    while n < 20:
        x = np.random.normal(mu,theta)
        R *= math.exp(x)
        P = R
        V[n] = K - P
        n += 1
    for n in range(N):
        avg_V[n] = sum(V[1:n])/N
    print(f'Expected stock gain at beginning of day:', avg_V)
    return


def JN_arrivals(i):
    if i == 0:
        return -(1/5)*math.log(random.random())
    if i == 1:
        return -(1/20)*math.log(random.random())
    if i == 2:
        return -(3/91)*math.log(random.random())


def JN_service(i):
    if i == 0:
        return -(1/10)*math.log(random.random())
    if i == 1:
        return -(1/50)*math.log(random.random())
    if i == 2:
        return -(1/100)*math.log(random.random())


def jackson_network(T):
    X = [[], [], []]
    C = [[], [], []]
    # arrival times and departures
    t_A = [0,0,0]
    t_D = [math.inf, math.inf, math.inf]
    flag = False
    # generate first arrivals to each server
    for i in [0,1,2]:
        t_A[i] = JN_arrivals(i)
    m = min(min(t_A), min(t_D))
    for i in [0,1,2]:
        if T > m == t_A[i]:
            # store customer arrival time and server they went to
            X[i].append([0,i])
            # generate new arrival time
            t_A[i] = JN_arrivals(i)
            if len(X[i])==1:
                #  generate service time
                s = JN_arrivals(i)
                t_D[i] = t_A[i]+s
        elif T > m == t_D[i]:
            chosen = X[i][0]


    return


def NH_ISQ_arrival_time(t):
    k = 2
    while k > (5*math.sin(t/2)+5)/10:
        k = random.random()
        t -= (1/10)*math.log(random.random())
    return t


def ISQ(T, type_process):
    A, D, p, mu_1, mu_2, lamda = [], [], 0.6, 1, 0.5, 10
    t = 0
    while t < T:
        if type_process == 'NH':
            t_A = -(1/10)*math.log(random.random())
        else:
            t_A = NH_ISQ_arrival_time(t)
        t += t_A
        A.append(t)
    for i in range(len(A)):
        u = random.random()
        if u > p:
            t_D = -(1/mu_1)*math.log(u)
        else:
            t_D = -(1/mu_2)*math.log(u)
        D.append(t_D)
    depart_times = {i: A[i]+D[i] for i in range(len(A))}
    num_in_sys_at_t = {}
    for i in A:
        num_in_sys_at_t[i] = len([k for k in depart_times if (depart_times[k] > i >= A[k])])

    time_in_system_50 = {A[i]: A[i] + D[i] for i in range(len(A)) if ((A[i]+D[i]) > 50 and A[i] <= 50)}
    time_in_system_100 = {A[i]: A[i] + D[i] for i in range(len(A)) if (A[i]+D[i]) > T}
    return time_in_system_50, time_in_system_100, num_in_sys_at_t


def problem_9():
    T, K = 100, 1000
    iter_fifty, iter_hundred = [], []
    for i in range(K):
        num_left_at_fifty, num_left_at_hundred, num_in_sys_at_t = ISQ(T, 'H')
        iter_fifty.append(len(num_left_at_fifty))
        iter_hundred.append(len(num_left_at_hundred))
    print('Using homogeneous arrival process')
    print(f'1K simulations Mean of customers in system at time = 50: {np.mean(iter_fifty)}\n'
          f'1K simulations Variance of customers in system at time = 50: {np.var(iter_fifty)}\n'
          f'1K simulations Mean of customers in system at time = 100: {np.mean(iter_hundred)}\n'
          f'1K simulations Variance of customers in system at time = 100: {np.var(iter_hundred)}')

    num_left_at_fifty, num_left_at_hundred, num_in_sys_at_t = ISQ(T, 'NH')
    lists = sorted(num_in_sys_at_t.items())
    x, y = zip(*lists)
    plt.plot(x, y)
    plt.xlabel('t')
    plt.ylabel('# in system')
    plt.savefig('People in NH_ISQ single path.png', dpi=300)
    plt.show()
    plt.close()

    paths = {i: (None, None) for i in range(K)}
    fig = plt.figure()
    fig.suptitle('Average People in NH_ISQ 1K paths')
    for k in range(K):
        num_left_at_fifty, num_left_at_hundred, num_in_sys_at_t = ISQ(T, 'NH')
        paths[k] = (num_in_sys_at_t.keys(), num_in_sys_at_t.values())

    plt.ylim(0)
    plt.xlim(0)
    plt.xlabel('t')
    plt.ylabel('# in system')
    plt.savefig('Average People in NH_ISQ 1K paths.png', dpi=300)
    plt.close()

"""
# executable code
print('\n\nProblem 1')
problem_1()
print('\n\nProblem 2')
problem_2()
print('\n\nProblem 3')
problem_3()
print('\n\nProblem 4')
problem_4()
print('\n\nProblem 5')
problem_5()
print('\n\nProblem 6')
problem_6()
print('\n\nProblem 9')
problem_9()
"""

problem_2()
