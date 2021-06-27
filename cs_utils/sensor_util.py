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

import math
import csv
import cv2
import time
import random
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(r"../../")
from caa.c_utils import training_util as tu

# -- DEFINE VARIABLES --
IM_WIDTH = 320
IM_HEIGHT = 160


# == SPAWN FUNCTIONS =========
# -- Create camera RGB -
def spawn_camera_rgb(world, blueprint_library, attached_object, dic_set):
    cam_rgb_bp = blueprint_library.find(dic_set['name'])

    if 'image_size_x' in dic_set:
        cam_rgb_bp.set_attribute('image_size_x', dic_set['image_size_x'])

    if 'image_size_y' in dic_set:
        cam_rgb_bp.set_attribute('image_size_y', dic_set['image_size_y'])

    if 'fov' in dic_set:
        cam_rgb_bp.set_attribute('fov', dic_set['fov'])

    if 'sensor_tick' in dic_set:
        cam_rgb_bp.set_attribute('sensor_tick', dic_set['sensor_tick'])

    cam_rgb_sp = carla.Transform(carla.Location(x=dic_set['loc_x'], z=dic_set['loc_z']),
                                 carla.Rotation(pitch=dic_set['rot_pitch']))
    cam_rgb = world.spawn_actor(cam_rgb_bp, cam_rgb_sp, attach_to=attached_object)

    return cam_rgb


# -- Create camera RGB -
def spawn_camera_ss(world, blueprint_library, attached_object, dic_set):
    cam_ss_bp = blueprint_library.find(dic_set['name'])
    cam_ss_bp.set_attribute('image_size_x', dic_set['image_size_x'])
    cam_ss_bp.set_attribute('image_size_y', dic_set['image_size_y'])
    cam_ss_bp.set_attribute('fov', dic_set['fov'])
    cam_ss_bp.set_attribute('sensor_tick', dic_set['sensor_tick'])
    cam_ss_sp = carla.Transform(carla.Location(x=dic_set['loc_x'], z=dic_set['loc_z']),
                                carla.Rotation(pitch=dic_set['rot_pitch']))
    cam_ss = world.spawn_actor(cam_ss_bp, cam_ss_sp, attach_to=attached_object)

    return cam_ss


def spawn_lidar(world, blueprint_library, attached_object, dic_set):
    lidar_bp = blueprint_library.find(dic_set['name'])
    lidar_bp.set_attribute('channels', dic_set['channels'])
    lidar_bp.set_attribute('points_per_second', dic_set['points_per_second'])
    lidar_bp.set_attribute('rotation_frequency', dic_set['rotation_frequency'])
    lidar_bp.set_attribute('range', dic_set['range'])
    lidar_bp.set_attribute('upper_fov', dic_set['upper_fov'])
    lidar_bp.set_attribute('lower_fov', dic_set['lower_fov'])
    lidar_location = carla.Location(0, 0, 2)
    lidar_rotation = carla.Rotation(0, 0, 0)
    lidar_transform = carla.Transform(lidar_location, lidar_rotation)
    lidar_sen = world.spawn_actor(lidar_bp, lidar_transform, attach_to=attached_object)

    return lidar_sen


def spawn_gnss(world, blueprint_library, attached_object, dic_set):
    gnss_bp = blueprint_library.find(dic_set['name'])
    gnss_bp.set_attribute('noise_alt_bias', dic_set['noise_alt_bias'])
    gnss_bp.set_attribute('noise_alt_stddev', dic_set['noise_alt_stddev'])
    gnss_bp.set_attribute('noise_lat_bias', dic_set['noise_lat_bias'])
    gnss_bp.set_attribute('noise_lat_stddev', dic_set['noise_lat_stddev'])
    gnss_bp.set_attribute('noise_lon_bias', dic_set['noise_lon_bias'])
    gnss_bp.set_attribute('noise_lon_stddev', dic_set['noise_lon_stddev'])
    gnss_bp.set_attribute('noise_seed', dic_set['noise_seed'])
    gnss_bp.set_attribute('sensor_tick', dic_set['sensor_tick'])
    gnss_transform = carla.Transform(carla.Location(x=dic_set['loc_x'], z=dic_set['loc_z']))
    gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=attached_object)

    return gnss


def spawn_imu(world, blueprint_library, attached_object, dic_set):
    imu_bp = blueprint_library.find(dic_set['name'])
    imu_bp.set_attribute('noise_accel_stddev_x', dic_set['noise_accel_stddev_x'])
    imu_bp.set_attribute('noise_accel_stddev_y', dic_set['noise_accel_stddev_y'])
    imu_bp.set_attribute('noise_accel_stddev_z', dic_set['noise_accel_stddev_z'])
    imu_bp.set_attribute('noise_gyro_bias_x', dic_set['noise_gyro_bias_x'])
    imu_bp.set_attribute('noise_gyro_bias_y', dic_set['noise_gyro_bias_y'])
    imu_bp.set_attribute('noise_gyro_bias_z', dic_set['noise_gyro_bias_z'])
    imu_bp.set_attribute('noise_gyro_stddev_x', dic_set['noise_gyro_stddev_x'])
    imu_bp.set_attribute('noise_gyro_stddev_y', dic_set['noise_gyro_stddev_y'])
    imu_bp.set_attribute('noise_gyro_stddev_z', dic_set['noise_gyro_stddev_z'])
    imu_bp.set_attribute('noise_seed', dic_set['noise_seed'])
    imu_bp.set_attribute('sensor_tick', dic_set['sensor_tick'])
    imu_transform = carla.Transform()

    imu = world.spawn_actor(imu_bp, imu_transform, attach_to=attached_object)

    return imu


def spawn_vehicle(world, blueprint_library, model, color=None, spawn_number=None):
    car_bp = blueprint_library.filter(model)[0]

    # set color
    if color is not None:
        car_bp.set_attribute('color', color)

    if spawn_number is None:
        car_sp = random.choice(world.get_map().get_spawn_points())
    else:
        car_sps = world.get_map().get_spawn_points()
        car_sp = car_sps[spawn_number]

    vehicle = world.spawn_actor(car_bp, car_sp)

    return vehicle


def spawn_mpc_vehicle(world, blueprint_library, initial_state, model, color=None,):
    car_bp = blueprint_library.filter(model)[0]

    # set color
    if color is not None:
        car_bp.set_attribute('color', color)

    car_sp = carla.Transform(carla.Location(x=initial_state.x, y=initial_state.y, z=0.500000),
                             carla.Rotation(pitch=0.000000, yaw=initial_state.yaw, roll=0.000000))

    vehicle = world.spawn_actor(car_bp, car_sp)

    return vehicle


# == CAMERA FUNCTIONS =====
# -- Display image -----
def live_cam(image, dic_set, pp1=False):
    cam_height, cam_width = int(dic_set['image_size_y']), int(dic_set['image_size_x'])
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (cam_height, cam_width, 4))
    image = array[:, :, :3]

    if pp1:
        image = image[:, :, ::-1]
        image = tu.preprocess1(image)

    cv2.imshow("back camera", image)
    cv2.waitKey(1)


# -- Export waypoints --
def export_map_waypoints(world, csv_file, distance=2.0):
    print("Calling: exporting map waypoints (writing into map text file!)")
    # initialization
    map = world.get_map()
    waypoints_list = []
    map_waypoints = map.generate_waypoints(distance)

    # add waypoints to list
    for item in map_waypoints:
        loc = item.transform.location
        waypoints_list.append([loc.x, loc.y])

    if os.path.isfile(csv_file):
        print("- Removing existed file")
        os.remove(csv_file)

    # write coordinates to csv file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for loc_x, loc_y in waypoints_list:
            writer.writerow([loc_x, loc_y])


# -- Export spawn points --
def export_spawn_pts(world, csv_file):
    print("Calling: exporting spawn coordinates (writing into map text file!)")
    sp = world.get_map().get_spawn_points()
    sp_list = []

    for p in sp:
        x, y, z = p.location.x, p.location.y, p.location.z
        pitch, yaw, roll = p.rotation.pitch, p.rotation.yaw, p.rotation.roll
        sp_list.append([x, y, z, pitch, yaw, roll])

    if os.path.isfile(csv_file):
        print("- Removing existed file")
        os.remove(csv_file)

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for x, y, z, pitch, yaw, roll in sp_list:
            writer.writerow([x, y, z, pitch, yaw, roll])


# -- Sensor based: Position tracking -
def position_tracking(vehicle, pos_tracking, display=False):
    location = vehicle.get_location()
    pos_tracking.append([location.x, location.y])

    if display:
        print(location.x, location.y, location.z)


# == Collision sensor =====
def detect_collision(event, vehicle, actor_list):
    actor_we_collide_against = event.other_actor
    impulse = event.normal_impulse
    intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)

    print("- Collision against: ", actor_we_collide_against)

    for actor in actor_list:
        actor.destroy()
    sys.exit()


# == GNSS functions =====
def gnss_function(event, gnss_data=None):
    lat, lon, alt = event.latitude, event.longitude, event.altitude
    tag = time.time()
    gnss_data.append([tag, lat, lon, alt])

    # print("<LAT: {}> - <LON: {}> - <ALT: {}>".format(lat, lon, alt))
    return lat, lon


def check_enter_gps_int(event, gps_intersection):
    # Initialize variable that indicates whether the car is at the intersection or not
    current_latitude, current_longitude, alt = event.latitude, event.longitude, event.altitude
    ent_int = False

    # check if vehicle is entering an intersection
    for tag, lat, lon, dis in gps_intersection:
        lat_dis = km2latdeg(dis)
        lon_dis = km2londeg(dis, lat)
        if lat - lat_dis <= current_latitude <= lat + lat_dis and lon + lon_dis >= current_longitude >= lon - lon_dis:
            ent_int = True
            break

    print(ent_int)


def imu_function(sensor_data, imu_data=None):
    limits = (-99.9, 99.9)
    accelerometer = (
        max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
        max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
        max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
    gyroscope = (
        max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
        max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
        max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
    compass = math.degrees(sensor_data.compass)

    # print("<ACCE: {}> - <GYRO: {}> - <COMP: {}>".format(accelerometer, gyroscope, compass))

    a0, a1, a2 = accelerometer[0], accelerometer[1], accelerometer[2]
    g0, g1, g2 = gyroscope[0], gyroscope[1], gyroscope[2]

    if imu_data is not None:
        imu_data.append([a0, a1, a2, g0, g1, g2, compass])

    return gyroscope


# -- Passing red lights --------
def passing_trafficlight(vehicle):
    if vehicle.is_at_traffic_light():
        traffic_light = vehicle.get_traffic_light()
        if traffic_light.get_state() == carla.TrafficLightState.Red:
            # world.hud.notification("Traffic light changed! Good to go!")
            traffic_light.set_state(carla.TrafficLightState.Green)


def latdeg2km(lat):
    km = lat * 110.574
    return km


def londeg2km(lat_coordinate, lon):
    factor = 111.320 * math.cos(lat_coordinate)
    km = lon * factor
    return km


def km2latdeg(km):
    lat = km / 110.574
    return lat


def km2londeg(km, lat_coordinate):
    factor = 111.320 * math.cos(lat_coordinate)
    lon = km / factor
    return lon


def gps_distance(lat1, lon1, lat2, lon2):
    # radius of the Earth
    R = 6373.0

    # coordinates
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    print("Result:", distance)


# -- Speed estimation
def speed_estimation(vehicle, display=False):
    v = vehicle.get_velocity()
    ms = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    kmh = int(3.6 * ms)
    if display:
        print('Speed: %.4f (m/s) - %.4f (km/h)' % (ms, kmh))

    return ms, kmh

