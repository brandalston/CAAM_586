import numpy as np
import statistics, math, random
import scipy.stats


def sum_exp_estimators(K):
    X_raw, X_anti = [], []
    for k in range(K):
        val_raw, val1, val2 = 0, 0, 0
        for i in range(1, 5):
            u = random.random()
            x, x1, x2 = -math.log(u), -math.log(u), -math.log(1 - u)
            val_raw += i * x
            val1 += i * x1
            val2 += i * x2
        if val1 >= 21.6 or val2 >= 21.6: X_anti.append(1)
        else: X_anti.append(0)
        if val_raw >= 21.6: X_raw.append(1)
        else: X_raw.append(0)
    return statistics.mean(X_raw), 0.5 * statistics.mean(X_anti)


def problem_1():
    K = 1000
    E_raw, E_anti = [], []
    for i in range(K):
        X_raw, X_anti = sum_exp_estimators(K)
        E_raw.append(X_raw)
        E_anti.append(X_anti)
    print(f'Raw E: {round(statistics.mean(E_raw),6)}, Antithetic E: {round(statistics.mean(E_anti),6)}')
    print(f'Raw Var: {round(statistics.variance(E_raw),6)}')
    print(f'Antithetic Var: {round(statistics.variance(E_anti),6)}, '
          f'Reduction over Raw: {round(statistics.variance(E_raw)/statistics.variance(E_anti),4)}')
    return


def related_normals_estimators(N, gamma):
    X_raw, X_cond, X_cond_anti, X_cond_control = [], [], [], []
    for k in range(N):
        y = np.random.normal(1, 1)
        if np.random.normal(y, 4) < 1: X_raw.append(1)
        else: X_raw.append(0)
        X_cond.append(1 - scipy.stats.norm.cdf(y, 1, 1))
        y1, y2 = np.random.normal(0, 0.5), np.random.normal(1, math.sqrt(3) / 2)
        A = 1 - scipy.stats.norm.cdf(y1, 0, 0.5)
        B = 1 - scipy.stats.norm.cdf(y2, 1, math.sqrt(3) / 2)
        X_cond_anti.append((A + B) / 2)
        X_cond_control.append(1 - scipy.stats.norm.cdf(y, 1, 1) + gamma * (y - 1))
    return X_raw, X_cond, X_cond_anti, X_cond_control


def problem_3():
    N = 100000
    Y, Z = [], []
    for k in range(10000):
        y_cov = np.random.normal(1, 1)
        Y.append(y_cov)
        z_cov = 1 - scipy.stats.norm.cdf(y_cov, 1, 1)
        Z.append(z_cov)
    gamma = -np.cov(Y, Z)[0, 1] / statistics.variance(Y)
    E_raw, E_cond, E_cond_anti, E_cond_control = related_normals_estimators(N, gamma)
    print(f'Raw E: {round(statistics.mean(E_raw),6)}, Conditional E: {round(statistics.mean(E_cond),6)}, '
          f'Conditional + Antithetic Var E: {round(statistics.mean(E_cond_anti),6)}, Conditional + Control Var E: {round(statistics.mean(E_cond_control),6)}')
    print(f'Raw Var: {round(statistics.variance(E_raw),6)}')
    print(f'Conditional Var: {round(statistics.variance(E_cond),6)}, '
          f'Reduction over Raw: {round(statistics.variance(E_raw)/statistics.variance(E_cond),4)}')
    print(f'Conditional + Antithetic Var: {round(statistics.variance(E_cond_anti),6)}, '
          f'Reduction over Conditional: {round(statistics.variance(E_raw)/statistics.variance(E_cond_anti),4)}')
    print(f'Conditional + Control Var: {round(statistics.variance(E_cond_control),6)}, '
          f'Reduction over Conditional: {round(statistics.variance(E_raw)/statistics.variance(E_cond_control),4)}')
    return


def insurance_estimators(N):
    E_raw, E_cond_control, E_cond_anti = [], [], []
    for k in range(N):
        u = random.random()
        c = np.random.poisson(15 / (.5 + u), 1)
        lamd = 15 / (0.5 + u)
        lamd1, lamd2 = 15 / (0.5 + u), 15 / (1.5 - u)
        if c >= 20: E_raw.append(1)
        else: E_raw.append(0)
        E_cond_control.append(
            1 - sum([math.exp(-lamd) * (lamd ** k) / math.factorial(k) for k in range(20)]) + (1 / lamd) * (u - 0.5))
        E_cond_anti.append(0.5 * (1 - (sum([math.exp(-lamd1) * (lamd1 ** k) / math.factorial(k) for k in range(20)])) +
                        1 - (sum([math.exp(-lamd2) * (lamd2 ** k) / math.factorial(k) for k in range(20)]))
                        ))
    return E_raw, E_cond_control, E_cond_anti


def problem_4():
    N = 100000
    E_raw, E_cond_control, E_cond_anti = insurance_estimators(N)
    print(f'Raw E: {round(statistics.mean(E_raw),6)}, '
          f'Conditional + Control Var E: {round(statistics.mean(E_cond_control),6)}, '
          f'Conditional + Antithetic Var E: {round(statistics.mean(E_cond_anti),6)}')
    print(f'Raw Var: {round(statistics.variance(E_raw),6)}')
    print(f'Conditional + Control Var: {round(statistics.variance(E_cond_control),6)}, Reduction over Raw: {round(statistics.variance(E_raw)/statistics.variance(E_cond_control),4)}')
    print(f'Conditional + Antithetic Var: {round(statistics.variance(E_cond_anti),6)}, Reduction over Raw: {round(statistics.variance(E_raw)/statistics.variance(E_cond_anti),4)}')
    return


def SSQ_estimators(N):
    """
    S_cov_est, T_cov_est = [], []
    for i in range(1000):
        U_arriv_cov = {k: random.random() for k in range(N)}
        U_service_cov = {k: random.random() for k in range(N)}
        inter_arrivals_cov = {k: -0.5 * math.log(U_arriv_cov[k]) for k in range(N)}
        Arrival_time_cov = {k: sum(list(inter_arrivals_cov.values())[:(k + 1)]) for k in range(N)}
        Service_time_cov = {k: -math.log(U_service_cov[k]) for k in range(N)}
        Departure_time_cov = {0: Arrival_time_cov[0] + Service_time_cov[0]}
        for k in range(1, N):
            Departure_time_cov[k] = max(Departure_time_cov[k - 1], Arrival_time_cov[k]) + Service_time_cov[k]
        Times_cov = {k: max(Arrival_time_cov[k], Departure_time_cov[k - 1]) +
                    Service_time_cov[k] - Arrival_time_cov[k] for k in range(1, N)}
        S_cov_est.append(sum(list(Service_time_cov.values())))
        T_cov_est.append(sum(list(Times_cov.values())))
    """

    U_arriv = {k: random.random() for k in range(N)}
    U_service = {k: random.random() for k in range(N)}

    inter_arrival_time = {k: -0.5*math.log(U_arriv[k]) for k in range(N)}
    anti_inter_arrival_time = {k: -0.5*math.log(1-U_arriv[k]) for k in range(N)}

    Arrival_time = {k: sum(list(inter_arrival_time.values())[:(k+1)]) for k in range(N)}
    anti_Arrival_time = {k: sum(list(anti_inter_arrival_time.values())[:(k+1)]) for k in range(N)}

    Service_time = {k: -math.log(U_service[k]) for k in range(N)}
    anti_Service_time = {k: -math.log(1-U_service[k]) for k in range(N)}

    Departure_time = {0: Arrival_time[0]+Service_time[0]}
    anti_Departure_time = {0: anti_Arrival_time[0]+anti_Service_time[0]}
    for k in range(1,N):
        Departure_time[k] = max(Departure_time[k-1], Arrival_time[k])+Service_time[k]
        anti_Departure_time[k] = max(anti_Departure_time[k-1],anti_Arrival_time[k])+anti_Service_time[k]

    Times = {k: max(Arrival_time[k], Departure_time[k - 1]) +
                Service_time[k] - Arrival_time[k] for k in range(1, N)}
    Times[0] = Service_time[0]
    anti_Times = {k: max(anti_Arrival_time[k], anti_Departure_time[k - 1]) +
                anti_Service_time[k] - anti_Arrival_time[k] for k in range(1, N)}
    anti_Times[0] = anti_Service_time[0]
    anti_Times = {k: anti_Times[k] + Times[k] for k in range(1, N)}
    # print(Times)
    # print(anti_Times)
    return Service_time, inter_arrival_time, Times, anti_Times


def problem_6():
    N, repeats = 10, 100000
    E_raw, E_anti, E_control_S, E_control_S_I = [], [], [], []
    S_cov_est, S_I_cov_est, T_cov_est = [], [], []
    for i in range(repeats):
        U_arriv_cov = {k: random.random() for k in range(N)}
        U_service_cov = {k: random.random() for k in range(N)}
        inter_arrivals_cov = {k: -0.5 * math.log(U_arriv_cov[k]) for k in range(N)}
        Arrival_time_cov = {k: sum(list(inter_arrivals_cov.values())[:(k + 1)]) for k in range(N)}
        Service_time_cov = {k: -math.log(U_service_cov[k]) for k in range(N)}
        Departure_time_cov = {0: Arrival_time_cov[0] + Service_time_cov[0]}
        for k in range(1, N):
            Departure_time_cov[k] = max(Departure_time_cov[k - 1], Arrival_time_cov[k]) + Service_time_cov[k]
        Times_cov = {k: max(Arrival_time_cov[k], Departure_time_cov[k - 1]) +
                        Service_time_cov[k] - Arrival_time_cov[k] for k in range(1, N)}
        Times_cov[0] = Service_time_cov[0]
        S_cov_est.append(sum(list(Service_time_cov.values())))
        S_I_cov_est.append(sum(list(Service_time_cov.values()))-sum(list(inter_arrivals_cov.values())[:-1]))
        T_cov_est.append(sum(list(Times_cov.values())))
    control_S = {'gamma': -np.cov(S_cov_est, T_cov_est)[0, 1]/statistics.variance(S_cov_est), 'mu': statistics.mean(S_cov_est)}
    control_S_I = {'gamma': -np.cov(S_I_cov_est, T_cov_est)[0, 1]/statistics.variance(S_I_cov_est), 'mu': statistics.mean(S_I_cov_est)}

    for k in range(repeats):
        Serv_times, inter_arrival_times, times_raw, times_anti = SSQ_estimators(N)

        E_raw.append(sum(times_raw.values()))
        E_anti.append(sum(times_anti.values())/2)
        E_control_S.append(sum(times_raw.values())+
                           control_S['gamma']*(sum(Serv_times.values())-control_S['mu']))
        E_control_S_I.append(sum(times_raw.values())+
                             control_S_I['gamma']*(
                                     sum(Serv_times.values())-sum(list(inter_arrival_times.values())[:-1])-control_S_I['mu']))
    print(f'Raw E: {round(statistics.mean(E_raw),6)}, Antithetic E: {round(statistics.mean(E_anti),6)}, Control of S E: {statistics.mean(E_control_S)}, Control of S and I E: {round(statistics.mean(E_control_S_I),6)}')
    print(f'Raw Var: {round(statistics.variance(E_raw),6)}')
    print(f'Antithetic Var: {round(statistics.variance(E_anti),6)}, Reduction over Raw: {round(statistics.variance(E_raw)/statistics.variance(E_anti),4)}')
    print(f'Control of S Var: {round(statistics.variance(E_anti),6)}, Reduction over Raw: {round(statistics.variance(E_raw)/statistics.variance(E_control_S),4)}')
    print(f'Control on S and I Var: {round(statistics.variance(E_anti),6)}, Reduction over Raw: {round(statistics.variance(E_raw)/statistics.variance(E_control_S_I),4)}')

    return


def twoD_gauss_MC_estimator(a, mean, cov):
    rvs = np.random.multivariate_normal(mean, cov, size=100000)
    E = [1 if (i[0] >= a and i[1] >= a) else 0 for i in rvs]
    if (mean != np.array([0, 0])).all():
        L = [math.exp(-mean @ np.linalg.inv(cov) @ np.array(i) +
                      0.5 * mean @ np.linalg.inv(cov) @ mean)
             for i in rvs]
        E = np.array(E) * L
    if (mean == np.array([0,0])).all(): print(f'E: {statistics.mean(E)}, Var: {statistics.variance(E)}')
    else: print(f'Var: {statistics.variance(E)}')
    CI = scipy.stats.t.interval(alpha=0.95, df=len(E) - 1, loc=np.mean(E), scale=scipy.stats.sem(E))
    print(f'95% CI: {CI}')
    return


def problem_7():
    a = [1, 3, 10]
    mean_raw = np.array([0, 0])
    cov = np.array([[4, -1], [-1, 4]])
    deltas = [0.001, 2, 10]
    for i in a:
        print(f'\nCrude MC simulation, a = {i}')
        twoD_gauss_MC_estimator(i, mean_raw, cov)
        print(f'Importance Sampling MC simulation, a = {i}')
        mean_best = np.array([i, i])
        twoD_gauss_MC_estimator(i, mean_best, cov)
    for i in a:
        mean_best = np.array([i, i])
        for d in deltas:
            print(f'Importance Sampling MC simulation, a = {i}, delta = {d}')
            twoD_gauss_MC_estimator(i, mean_best, d * cov)
        print('')
    return


def CMC_estimators(repeats, size):
    X_raw, X_cond = [], []
    for i in range(repeats):
        K_raw_i, K_cond_i = [], []
        for k in range(1000):
            D = scipy.stats.bernoulli.rvs(p=np.random.beta(1, 19), size=size)
            X = np.random.normal(3, 1, size=size)
            if sum(D*X) > 45: K_raw_i.append(1)
            else: K_raw_i.append(0)
            K_cond_i.append(1 - scipy.stats.norm.cdf((45-3*sum(D))/math.sqrt(sum(D)),0,1))
        X_raw.append(statistics.mean(K_raw_i))
        X_cond.append(statistics.mean(K_cond_i))
    return X_raw, X_cond


def problem_9():
    size, repeats = 100, 1000
    E_raw, E_cond = CMC_estimators(repeats, size)
    print(f'Raw E: {round(statistics.mean(E_raw),6)}, Conditional E: {round(statistics.mean(E_cond),6)}')
    print(f'Raw Var: {round(statistics.variance(E_raw),6)}')
    print(f'Conditional Var: {round(statistics.variance(E_cond),6)}, Reduction over Raw: {round(statistics.variance(E_raw)/statistics.variance(E_cond),4)}')
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