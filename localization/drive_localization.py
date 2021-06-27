import glob
import os
import sys

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])

except IndexError:
    pass

import carla

import time
import math
import keyboard
import warnings
warnings.filterwarnings("ignore")
import cvxpy
import random

import numpy as np

sys.path.append(r"../../")
from fhdo_casestudy.cs_utils import dictionaries as dic
from fhdo_casestudy.cs_utils import sensor_util as su
from fhdo_casestudy.cs_utils import tools as to
from fhdo_casestudy.cs_utils import module_p2
from fhdo_casestudy.mpc import cubic_spline_planner as cubic_spline_planner

from queue import Queue
from queue import Empty

np.random.seed(2)

# -- VARIABLE INITIALIZATION --
# Specify the wanted map
town_dic = dic.town02

map_name, \
    weather_type, \
    current_town, \
    gnss_csv, \
    gps_intersection_csv, \
    waypoints_csv, \
    spawn_csv, \
    carla_intersection_csv, \
    road_segments_file = to.paths_initialization(town_dic)


def update_carla_state(vehicle):
    control = vehicle.get_control()
    throttle, brake, steer = control.throttle, control.brake, control.steer

    # get location
    location = vehicle.get_location()
    loc_x, loc_y = location.x, location.y

    # get rotation
    yaw = vehicle.get_transform().rotation.yaw
    yaw = np.deg2rad(yaw)

    # get velocity
    v = vehicle.get_velocity()
    ms = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    kmh = int(3.6 * ms)


def carla_world_plot(world, path, z=1.0, lt=0.0):
    debug = world.debug
    for x, y in path:
        location = carla.Location(x=x, y=y, z=z)
        debug.draw_point(
            location,
            size=0.1,
            life_time=lt,
            persistent_lines=False,
            color=carla.Color(0, 255, 0))


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_name))


def game_loop(reload=True, hp=False, cp=False):  # hp: Horizon plot - cp: Course plot
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # List initialization
    actor_list, sensor_list = [], []
    gnss_list, zgnss_list = [], []
    imu_list, zimu_list = [], []
    loc_data = []
    sensor_queue = Queue()

    try:
        if reload:
            print("Loading world...")
            world = client.load_world(town_dic['name'])
        else:
            world = client.get_world()

        settings = world.get_settings()
        settings.synchronous_mode = True
        world.apply_settings(settings)
        traffic_manager = client.get_trafficmanager()

        # Spectator
        spectator = world.get_spectator()
        transform = carla.Transform(carla.Location(x=town_dic['x'], y=town_dic['y'], z=town_dic['z']),
                                    carla.Rotation(pitch=town_dic['pitch'], yaw=town_dic['yaw']))
        spectator.set_transform(transform)

        # World setting
        world.set_weather(getattr(carla.WeatherParameters, town_dic['weather']))  # set weather
        blueprint_library = world.get_blueprint_library()  # get blueprint library

        # Vehicle
        vehicle_model = "mercedesccc"  # "model3"/"lincoln"/"mercedesccc"
        vehicle_color = dic.petronas_color
        vehicle = su.spawn_vehicle(world, blueprint_library, vehicle_model, vehicle_color)
        actor_list.append(vehicle)

        # == GNSS: zero noise
        zgnss = su.spawn_gnss(world, blueprint_library, vehicle, dic.gnss)
        zgnss.listen(lambda zgnss_event: sensor_callback(zgnss_event, sensor_queue, 0))
        sensor_list.append(zgnss)

        # == GNSS
        gnss = su.spawn_gnss(world, blueprint_library, vehicle, dic.gnss_noise)
        gnss.listen(lambda gnss_event: sensor_callback(gnss_event, sensor_queue, 1))
        sensor_list.append(gnss)

        # == IMU: zero noise
        zimu = su.spawn_imu(world, blueprint_library, vehicle, dic.imu)
        zimu.listen(lambda zimu_event: sensor_callback(zimu_event, sensor_queue, 2))
        sensor_list.append(zimu)

        # == IMU:
        imu = su.spawn_imu(world, blueprint_library, vehicle, dic.imu_noise)
        imu.listen(lambda imu_event: sensor_callback(imu_event, sensor_queue, 3))
        sensor_list.append(imu)

        # == Camera: Spectator
        cam_spectate = su.spawn_camera_rgb(world, blueprint_library, vehicle, dic.cam_spectate_1)
        cam_spectate.listen(lambda data: sensor_callback(data, sensor_queue, 4))
        sensor_list.append(cam_spectate)

        # Sensor listening
        print("Stage: listening to sensor")
        while True:
            world.tick()
            try:
                ms, kmh = su.speed_estimation(vehicle)
                traffic_manager.global_percentage_speed_difference(80)
                degree = vehicle.get_transform().rotation.yaw
                # rad = degree * np.pi / 180
                rad = np.deg2rad(degree)

                location = vehicle.get_location()
                loc_x = location.x
                loc_y = location.y

                vehicle.set_autopilot(True)
                su.passing_trafficlight(vehicle)

                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    if s_frame[1] == 0:
                        xz, yz = su.gnss_function(s_frame[0], gnss_list)
                    elif s_frame[1] == 1:
                        x, y = su.gnss_function(s_frame[0], gnss_list)
                    elif s_frame[1] == 2:
                        gyroscopez = su.imu_function(s_frame[0], imu_list)
                    elif s_frame[1] == 3:
                        gyroscope = su.imu_function(s_frame[0], imu_list)
                    else:
                        su.live_cam(s_frame[0], dic.cam_spectate_1)

                xTrue = [xz, yz, ms, rad]
                xDR = [x, y, ms, rad]
                zTrue = [xz, yz]
                z = [x, y]
                u = [ms, gyroscopez[2]]
                ud = [ms, gyroscope[2]]

                # data = [x, y, xz, yz, gyroscope[2], gyroscopez[2], ms, rad]
                data2 = [loc_x, loc_y, gyroscope[2], gyroscopez[2], ms, rad]
                # data3 = [xz, yz]
                loc_data.append(data2)

                # Stopping key
                if keyboard.is_pressed("q"):
                    print("Simulation stopped")
                    break

            except Empty:
                print("- Some of the sensor information is missed")

    finally:
        print("Finally...")
        # Switch back to synchronous mode
        settings.synchronous_mode = False
        world.apply_settings(settings)
        to.export_csv("loc_data.csv", loc_data)

        try:
            # Destroying actors
            print("Destroying {} actor(s)".format(len(actor_list)))
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

        except Exception as e:
            print("Final Exception: ", e)


def main():
    game_loop(reload=True, hp=True, cp=False)
    time.sleep(0.5)


if __name__ == '__main__':
    main()
