import random

iterations = [100, 1000, 10000]
expectation = {iter: None for iter in iterations}

for iter in iterations:
    e_iter = 0
    for i in range(iter+1):
        count, sum = 0, 0
        while sum < 1:
            sum += random.uniform(0,1)
            count += 1
        e_iter += count
    expectation[iter] = e_iter/iter

print(expectation)
