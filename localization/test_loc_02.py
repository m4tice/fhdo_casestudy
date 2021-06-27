import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fhdo_casestudy.cs_utils.tools as to

from scipy.spatial.transform import Rotation as Rot


# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]


def load_data(file_name):
    data_df = pd.read_csv(file_name)
    data = data_df.values
    return data


def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def my_observation(xTrue, u, z):
    # add noise to gps x-y
    # z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
    z = z + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xTrue, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


map_csv = "../database/town02/town02-waypoints.csv"
csv_file = "loc_data_03.csv"

hz = []
d = load_data(csv_file)

xDRs = []

xz0, yz0, w0, wz0, v0, rad0 = d[0]
xTrue = np.array([[xz0], [yz0], [rad0], [v0]])
xDR = xTrue
xDRs.append(xDR)

for i, (xz, yz, w, wz, v, rad) in enumerate(d[0:]):
    xTrue = np.array([[xz], [yz], [rad], [v]])
    z = np.array([[xz], [yz]]) + GPS_NOISE @ np.random.randn(2, 1)
    ud = np.array([[v], [wz]]) + INPUT_NOISE @ np.random.randn(2, 1)
    xDR = motion_model(xDR, ud)
    xDRs.append(xDR)

xDRs = np.asarray(xDRs)

ax = to.map_plot(map_csv)
ax.plot(d[:, 0], d[:, 1])
ax.plot(xDRs[:, 0], xDRs[:, 1])

plt.show()
