import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fhdo_casestudy.cs_utils.tools as to

DT = 0.1


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


def load_data(file_name):
    data_df = pd.read_csv(file_name)
    data = data_df.values
    return data


map_csv = "../database/town02/town02-waypoints.csv"
csv_file = "loc_data_05.csv"

data = load_data(csv_file)

ox, oy = data[:, 0], data[:, 1]
d0 = data[0]
xTrue = np.array(([d0[0]], [d0[1]], [d0[5]], [d0[4]]))
hxTrue = xTrue
fig, ax = plt.subplots()

for x, y, w, wz, v, rad in data:
    u = np.array([[v], [wz]])
    xTrue = motion_model(xTrue, u)
    hxTrue = np.hstack((hxTrue, xTrue))

    ax.cla()
    offset = 40
    waypoints = to.load_map(map_csv)
    mxs = waypoints[:, 0]
    mys = waypoints[:, 1]

    ax.axis('equal')
    ax.grid(True)
    ax.axis([min(mxs) - offset, max(mxs) + offset, min(mys) - offset, max(mys) + offset])
    # ax.axis([min(mxs) - offset, max(mxs) + offset, 100, 160])
    ax.set_facecolor('gray')
    ax.scatter(mxs, mys, color='white', marker='.')

    ax.plot(ox, oy)
    ax.plot(hxTrue[0, :].flatten(),
            hxTrue[1, :].flatten(), "-o")
    plt.pause(0.001)

plt.show()


