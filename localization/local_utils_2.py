import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fhdo_casestudy.cs_utils.tools as to

from scipy.spatial.transform import Rotation as Rot


def load_data(file_name):
    data_df = pd.read_csv(file_name)
    data = data_df.values
    return data


map_csv = "loc_data.csv"
csv_file = "loc_data_04.csv"

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

show_animation = True


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def my_observation(xTrue, xd, u, z):
    # add noise to gps x-y
    # z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)
    # z = z + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)

    # xd = motion_model(xTrue, ud)

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


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst, ax):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    ax.plot(px, py, "--r")


def main():
    print(__file__ + " start!!")

    time = 0.0
    data = load_data(csv_file)

    # State Vector [x y yaw v]'
    # xEst = np.zeros((4, 1))
    # xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    fd = data[0]
    # xDR = np.zeros((4, 1))  # Dead reckoning
    xTrue = np.array((fd[0], fd[1], fd[5], fd[4]))
    xTrue = np.reshape(xTrue, (4, 1))
    xDR = xTrue
    xEst = xTrue

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    # hz = np.zeros((2, 1))
    hz = np.array(([fd[0]], [fd[1]]))

    fig, ax = plt.subplots()
    # while SIM_TIME >= time:
    for i, (x, y, xz, yz, w, wz, v, rad) in enumerate(data[1:]):
        time += DT

        xTrue = np.array([[xz], [yz], [rad], [v]])
        z = np.array([[x], [y]])
        xDR = np.array([[x], [y], [rad], [v]])
        ud = np.array([[v], [w]])

        if i == 0:
            xEst = xTrue

        # xTrue, z, xDR, ud = my_observation(xTrue, xDR, u, z)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        # hxEst.append(xEst)
        # hxDR.append(xDR)
        # hxTrue.append(xTrue)
        # hz.append(z)

        if show_animation:
            ax.cla()

            # MAP
            # offset = 0.1
            # waypoints = to.load_map(map_csv)
            # mxs = waypoints[:, 0]
            # mys = waypoints[:, 1]
            #
            # # fig, ax = plt.subplots()
            # ax.axis('equal')
            # ax.grid(True)
            # ax.axis([min(mxs) - offset, max(mxs) + offset, min(mys) - offset, max(mys) + offset])
            # # ax.axis([min(mxs) - offset, max(mxs) + offset, 100, 160])
            # ax.set_facecolor('gray')
            # ax.scatter(mxs, mys, color='white', marker='.')

            # for stopping simulation with the esc key.
            # plt.gcf().canvas.mpl_connect('key_release_event',
            #         lambda event: [exit(0) if event.key == 'escape' else None])

            ax.plot(hz[0, :], hz[1, :], ".g")
            ax.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            ax.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), ".k")
            ax.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst, PEst, ax)
            # ax.axis("equal")
            # ax.grid(True)
            plt.gca().invert_yaxis()
            plt.pause(0.001)


if __name__ == '__main__':
    main()

