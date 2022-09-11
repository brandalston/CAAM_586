import numpy as np
from matplotlib import pyplot as plt


def problem_3():
    """ PROBLEM 3 (ROSS 5.32) """
    # disk info
    r = 5
    x_center, y_center = 0, 0
    area_disc = np.pi * (r ** 2)

    # Point process parameters (poisson intensity)
    lambda_poisson = 1

    # Simulate Poisson point process
    # Poisson number of points
    numbPoints = np.random.poisson(lambda_poisson * area_disc)
    # polar angular coordinates for points
    theta = 2 * np.pi * np.random.uniform(0, 1, numbPoints)
    # polar radial coordinates for points
    rho = r * np.sqrt(np.random.uniform(0, 1, numbPoints))

    # Convert from polar to Cartesian coordinates
    x_generated = rho * np.cos(theta)
    y_generated = rho * np.sin(theta)

    # Shift disk center
    x_generated = x_generated + x_center
    y_generated = y_generated + y_center

    # Plot
    fig = plt.figure()
    plt.scatter(x_generated, y_generated, edgecolor='k', facecolor='none', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    fig.suptitle('Poisson process on disc, r=5, lambda=1')
    plt.savefig('poisson disc.png',dpi=300)
    plt.close()


def problem_4():
    pass

problem_3()