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

    A, D, S, idle, over = SSQ(9)
    plt.plot(S.keys(), S.values())
    plt.xlabel("time")
    plt.ylabel("# in system")
    plt.title('Single Server Queue over Time T')
    plt.savefig('SSQ.png', dpi=300)

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
        print(f'{m} iterations. Avg customer time in system: {round(customer_time,4)}. '
              f'Avg idle time: {round(idle_time,4)}. Avg overtime: {round(overtime,4)}')
    return


def PP_lambda(t):
    val = 19 - 3*abs((t % 10) - 5)
    return val


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
    N_B, K = 1, 1  # assume start on break -> event at beginning
    A, D, B_L, B_R, S, E, J = {}, {}, {}, {}, {}, {}, {}  # dicts to record event times/number of each event
    S[K], E[K], J[K] = 0, 0, 0
    while min([t_A, t_B, t_D]) <= T:
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
            # server on break
            if S[K] > 1:
               t_D = t_D
            if J[K] == 0:
                t_D = math.inf
            elif J[K] == 1:
                if S[K] == 1:
                    t_D = t-(1/25)*math.log(random.random())
        # complete job as event
        elif t_D < t_A and t_D < t_B:
            t = t_D
            N_D += 1
            E[K] = t
            S[K] = S[K-1]-1
            # update server break position
            if S[K] == 0:
                J[K] = 0
            elif S[K] > 0:
                J[K] = 1
            # update t_B and t_D
            if S[K] == 0:
                t_D = math.inf
            elif S[K] > 0:
                t_D = t-(1/25)*math.log(random.random())
            if J[K] == 0:
                t_B += (t+0.3*random.random())
            else:
                t_B = math.inf
            # store departure event data
            D[N_D] = t
            if S[K] == 0:
                N_B += 1
                B_L[N_B] = t
        # return from break as event
        elif t_B < t_A and t_B < t_D:
            t = t_B
            B_R[N_B] = t
            E[K] = t
            S[K] = S[K-1]
            if S[K] > 0:
                J[K] = 1
            if S[K] == 0:
                J[K] = 0
            if S[K] == 0:
                N_B += 1
                B_L[N_B] = t
            # update t_B and t_D
            if S[K] == 0:
                t_D = math.inf
            if S[K] > 0:
                t_D = t-(1/25)*math.log(random.random())
            if S[K] == 0:
                t_B = t + 0.3*random.random()
            if J[K] == 0:
                t_B = math.inf
    if J[K] == 0:
        B_R[N_B] = T
    return A, D, B_L, B_R, S, E, J, K


def problem_2():
    A, D, B_L, B_R, S, E, J, K = NH_SSQ(10)
    print(A)
    print(D)
    return
# problem_1()
problem_2()