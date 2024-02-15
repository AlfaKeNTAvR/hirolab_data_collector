#!/usr/bin/env python

from mat4py import loadmat
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
from geometry_msgs.msg import TransformStamped, Point, Pose
from numpy.linalg import norm
from std_msgs.msg import String, Float64, Float64MultiArray
import rospy
import math
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import sys
import tf2_ros
import tf.transformations
from kortex_driver.srv import *
from kortex_driver.msg import *
from sensor_msgs.msg import Joy
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, quaternion_from_matrix, euler_from_matrix
import time
import actionlib
from scipy.spatial.distance import cdist

from data_collector.msg import (UpperBodyKeypoints)

# # ########################
# # # Initial parameters
rh_x, rh_y, rh_z = np.ones(3)
lh_x, lh_y, lh_z = np.ones(3)
chest_x, chest_y, chest_z = np.ones(3)
lua_x, lua_y, lua_z = np.ones(3)
rua_x, rua_y, rua_z = np.zeros(3)
lla_x, lla_y, lla_z = np.ones(3)
rla_x, rla_y, rla_z = np.ones(3)
waist_x, waist_y, waist_z = np.ones(3)
# - - - - - - -
# # # Initial arrays for filter
lua_x_array = None
lua_y_array = None
lua_z_array = None
rua_x_array = None
rua_y_array = None
rua_z_array = None
lla_x_array = None
lla_y_array = None
lla_z_array = None
rla_x_array = None
rla_y_array = None
rla_z_array = None
waist_x_array = None
waist_y_array = None
waist_z_array = None
chest_x_array = None
chest_y_array = None
chest_z_array = None
rh_x_array = None
rh_y_array = None
rh_z_array = None
lh_x_array = None
lh_y_array = None
lh_z_array = None
# - - - - - - -
accumulate_start_time = None
total_time_dwell = None
velocity_magnitude = 0
max_velocity = 0.5
realtime_physical = None
accumulated_workload = None
accumulated_workload_bar = None
per_fixation_total = None
old_velocity = [0, 0, 0]
current_velocity = [0, 0, 0]
first_time = 0
gripper_state_value = 0
reset_condn = 0
grasp_condn = 0
caught_workload = 0
# # ########################

rospy.init_node('physical_workload')
start_time = rospy.get_time()
vel_list = [0] * 10

# # ################################################################################################
# # Load workload mapping data (TC version)

# NOTE: What are those .mat files?
muscle_mapping_folder = '/home/fetch/catkin_workspaces/iona_devel_ws/src/hirolab_data_collector/muscle_mapping'

AD_mapping_raw = loadmat(f'{muscle_mapping_folder}/AD_Effort.mat')
AD_mapping_data = AD_mapping_raw['AD_E']
AD_mapping = np.zeros(150)  # joint limited range of motion
for i in range(len(AD_mapping_data)):
    AD_mapping[i] = float(str(AD_mapping_data[i])[1:-1])

MD_mapping_raw = loadmat(f'{muscle_mapping_folder}/MD_Effort.mat')
MD_mapping_data = MD_mapping_raw['MD_E']
MD_mapping = np.zeros(120)  # joint limited range of motion
for i in range(len(MD_mapping_data)):
    MD_mapping[i] = float(str(MD_mapping_data[i])[1:-1])

BI_mapping_raw = loadmat(f'{muscle_mapping_folder}/BI_Effort.mat')
BI_mapping_data = BI_mapping_raw['B_E']
BI_mapping = np.zeros(150)  # joint limited range of motion
for i in range(len(BI_mapping_data)):
    BI_mapping[i] = float(str(BI_mapping_data[i])[1:-1])
# ################################################################################################


def callback_gripper(data):

    global gripper_state_value
    # print(data)
    gripper_state_value = data.data


def callback_lh(data):
    global lh_x, lh_y, lh_z
    lh_x = -data.transform.translation.x
    lh_y = -data.transform.translation.y
    lh_z = -data.transform.translation.z


def callback_rh(data):
    global rh_x, rh_y, rh_z, first_time, old_velocity, current_velocity, velocity_magnitude, first_time
    rh_x = -data.transform.translation.x
    rh_y = -data.transform.translation.y
    rh_z = -data.transform.translation.z

    # NOTE: Why do we need velocity?

    if first_time == 0:
        old_velocity = np.asarray([rh_x, rh_y, rh_z])
        first_time = 1

    current_velocity = np.asarray([rh_x, rh_y, rh_z])

    velocity_magnitude = np.linalg.norm(current_velocity - old_velocity)

    old_velocity = current_velocity


def callback_chest(data):
    global chest_x, chest_y, chest_z
    chest_x = -data.transform.translation.x
    chest_y = -data.transform.translation.y
    chest_z = -data.transform.translation.z


def callback_lua(data):
    global lua_x, lua_y, lua_z
    lua_x = -data.transform.translation.x
    lua_y = -data.transform.translation.y
    lua_z = -data.transform.translation.z


def callback_rua(data):
    global rua_x, rua_y, rua_z
    rua_x = -data.transform.translation.x
    rua_y = -data.transform.translation.y
    rua_z = -data.transform.translation.z


def callback_lla(data):
    global lla_x, lla_y, lla_z
    lla_x = -data.transform.translation.x
    lla_y = -data.transform.translation.y
    lla_z = -data.transform.translation.z


def callback_rla(data):
    global rla_x, rla_y, rla_z
    rla_x = -data.transform.translation.x
    rla_y = -data.transform.translation.y
    rla_z = -data.transform.translation.z


def callback_waist(data):
    global waist_x, waist_y, waist_z
    waist_x = -data.transform.translation.x
    waist_y = -data.transform.translation.y
    waist_z = -data.transform.translation.z


def upperbody_keypoints_callback(message):
    """
    
    """

    global lua_x, lua_y, lua_z, rua_x, rua_y, rua_z
    global lla_x, lla_y, lla_z, rla_x, rla_y, rla_z
    global lh_x, lh_y, lh_z, rh_x, rh_y, rh_z
    global chest_x, chest_y, chest_z
    global waist_x, waist_y, waist_z

    global velocity_magnitude, current_velocity, old_velocity, first_time

    # NOTE: Why all coordinates are negated?

    try:
        # # Upper arm (Shoulder):
        lua_x = -message.keypoints[0].x
        lua_y = -message.keypoints[0].y
        lua_z = -message.keypoints[0].z

        rua_x = -message.keypoints[1].x
        rua_y = -message.keypoints[1].y
        rua_z = -message.keypoints[1].z

        # # Lower arm (Elbow):
        lla_x = -message.keypoints[2].x
        lla_y = -message.keypoints[2].y
        lla_z = -message.keypoints[2].z

        rla_x = -message.keypoints[3].x
        rla_y = -message.keypoints[3].y
        rla_z = -message.keypoints[3].z

        # # Hand (Wrist):
        lh_x = -message.keypoints[4].x
        lh_y = -message.keypoints[4].y
        lh_z = -message.keypoints[4].z

        rh_x = -message.keypoints[5].x
        rh_y = -message.keypoints[5].y
        rh_z = -message.keypoints[5].z

        # # Chest and Waist:
        chest_x = -message.keypoints[6].x
        chest_y = -message.keypoints[6].y
        chest_z = -message.keypoints[6].z

        waist_x = -message.keypoints[7].x
        waist_y = -message.keypoints[7].y
        waist_z = -message.keypoints[7].z

        if first_time == 0:
            old_velocity = np.asarray([rh_x, rh_y, rh_z])
            first_time = 1

        current_velocity = np.asarray([rh_x, rh_y, rh_z])

        velocity_magnitude = np.linalg.norm(current_velocity - old_velocity)

        old_velocity = current_velocity

    except:
        return


# def callback_velocity(data):

#     global velocity_magnitude, max_velocity, vel_list

#     x = data.twist.linear_x
#     y = data.twist.linear_y
#     z = data.twist.linear_z

#     velocity_magnitude = math.sqrt(x ** 2 + y ** 2 + z ** 2)
#     # print(velocity_magnitude)
#     if velocity_magnitude > 0.5:

#         velocity_magnitude = 0.5

#     vel_list.append(velocity_magnitude)
#     vel_list.pop(0)

#     velocity_magnitude = sum(vel_list)/len(vel_list)


def filter_creator(list_, message_value, window=1):

    # use this function for the filter
    if list_ is None:
        list_ = [message_value] * window
    else:
        list_.append(message_value)
        list_.pop(0)

    return list_


def average_calculator(list_):

    return sum(list_) / len(list_)


def run():

    global rh_x, rh_y, rh_z, lh_x, lh_y, lh_z
    global chest_x, chest_y, chest_z, lua_x, lua_y, lua_z, rua_x, rua_y, rua_z, lla_x, lla_y, lla_z, rla_x, rla_y, rla_z, waist_x, waist_y, waist_z
    global velocity_magnitude, max_velocity, vel_list, physical_accumulated, grasp_condn, caught_workload

    # rospy.init_node('human_tracking')
    # rospy.sleep(0.5)

    # for accumulated workload
    physical_accumulated = 0
    same_accumulated = 0
    rate = rospy.Rate(10)  # 10 Hz
    start_time = rospy.get_time()
    total_time = rospy.get_time() - start_time
    first_time_nf = 0
    grasp_trigger = 0
    # for accumulated workload

    fig = plt.figure(figsize=(11, 9))
    ax = plt.axes(projection='3d')
    ax.set_xlim3d([-1.5, 1.5])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1.3, 0.7])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(180, 0)

    # get the controllers data
    rospy.Subscriber('/Left_Hand', TransformStamped, callback_lh, queue_size=1)
    rospy.Subscriber('/Right_Hand', TransformStamped, callback_rh, queue_size=1)

    # get the trackers data
    rospy.Subscriber('/Chest', TransformStamped, callback_chest, queue_size=1)
    rospy.Subscriber(
        '/L_Upperarm', TransformStamped, callback_lua, queue_size=1
    )
    rospy.Subscriber(
        '/R_Upperarm', TransformStamped, callback_rua, queue_size=1
    )
    rospy.Subscriber(
        '/L_Lowerarm', TransformStamped, callback_lla, queue_size=1
    )
    rospy.Subscriber(
        '/R_Lowerarm', TransformStamped, callback_rla, queue_size=1
    )
    rospy.Subscriber('/Waist', TransformStamped, callback_waist, queue_size=1)
    rospy.Subscriber('/gripper_state', Float64, callback_gripper, queue_size=1)

    rospy.Subscriber(
        f'/physical_workload/upperbody_keypoints',
        UpperBodyKeypoints,
        upperbody_keypoints_callback,
    )

    # get the controller velocity
    # controller_velocity_sub = rospy.Subscriber('/my_gen3/in/cartesian_velocity', TwistCommand, callback_velocity, queue_size=1)

    # publish the realtime overall workload
    workload_pub = rospy.Publisher(
        '/physical_workload',
        Float64MultiArray,
        queue_size=1,
    )

    while not rospy.is_shutdown():

        try:

            global rh_x, rh_y, rh_z, lh_x, lh_y, lh_z
            global chest_x, chest_y, chest_z, lua_x, lua_y, lua_z, rua_x, rua_y, rua_z, lla_x, lla_y, lla_z, rla_x, rla_y, rla_z, waist_x, waist_y, waist_z

            dot_head, = ax.plot3D([], [], [], 'ko', ms=17, mec='k')
            dot_chest, = ax.plot3D([], [], [], 'ko', ms=12, mec='k')
            dot_lua, = ax.plot3D([], [], [], 'bo', ms=12, mec='k')
            dot_rua, = ax.plot3D([], [], [], 'go', ms=12, mec='k')
            dot_lla, = ax.plot3D([], [], [], 'bo', ms=12, mec='k')
            dot_rla, = ax.plot3D([], [], [], 'go', ms=12, mec='k')
            dot_lh, = ax.plot3D([], [], [], 'bo', ms=12, mec='k')
            dot_rh, = ax.plot3D([], [], [], 'go', ms=12, mec='k')
            dot_waist, = ax.plot3D([], [], [], 'ko', ms=12, mec='k')

            line_head_chest, = ax.plot3D([], [], [], 'k')
            line_chest_waist, = ax.plot3D([], [], [], 'k')
            line_left_shoulder, = ax.plot3D([], [], [], 'k')
            line_right_shoulder, = ax.plot3D([], [], [], 'k')
            line_left_upperarm, = ax.plot3D([], [], [], 'k')
            line_right_upperarm, = ax.plot3D([], [], [], 'k')
            line_left_lowerarm, = ax.plot3D([], [], [], 'k')
            line_right_lowerarm, = ax.plot3D([], [], [], 'k')

            # bars
            current_x = [0.35, 0.35, 0.35, 0.35]
            current_y = [-0.9, -0.7, -0.7, -0.9]
            current_z = [0.5, 0.5, -0.57, -0.57]
            current_verts = [list(zip(current_x, current_y, current_z))]
            ax.add_collection3d(
                Poly3DCollection(
                    current_verts,
                    facecolors='white',
                    linewidths=1,
                    edgecolors='r',
                    alpha=.20
                )
            )
            dot_current, = ax.plot3D([], [], [], 'ro', ms=17, mec='r')
            line_current, = ax.plot3D([], [], [], 'r', linewidth=7)
            accumulated_x = [0.35, 0.35, 0.35, 0.35]
            accumulated_y = [0.9, 1.1, 1.1, 0.9]
            accumulated_z = [0.5, 0.5, -0.57, -0.57]
            accumulated_verts = [
                list(zip(accumulated_x, accumulated_y, accumulated_z))
            ]
            ax.add_collection3d(
                Poly3DCollection(
                    accumulated_verts,
                    facecolors='white',
                    linewidths=1,
                    edgecolors='r',
                    alpha=.20
                )
            )
            dot_accumulated, = ax.plot3D([], [], [], 'ro', ms=17, mec='r')
            line_accumulated, = ax.plot3D([], [], [], 'r', linewidth=7)

            r_shouler_WL_text = ax.text(
                0.35,
                -0.9,
                -1.1,
                'R_Shoulder: ' + str(),
                color='green',
                fontsize=15
            )
            r_elbow_WL_text = ax.text(
                0.35,
                -0.9,
                -0.8,
                'R_Elbow: ' + str(),
                color='green',
                fontsize=15
            )
            l_shouler_WL_text = ax.text(
                0.35,
                0.2,
                -1.1,
                'L_Shoulder: ' + str(),
                color='blue',
                fontsize=15
            )
            l_elbow_WL_text = ax.text(
                0.35, 0.2, -0.8, 'L_Elbow: ' + str(), color='blue', fontsize=15
            )
            current_WL_text = ax.text(
                0.35, -0.9, 0.7, 'CURRENT: ' + str(), color='red', fontsize=15
            )
            accumulated_WL_text = ax.text(
                0.35,
                0.2,
                0.7,
                'ACCUMULATED: ' + str(),
                color='red',
                fontsize=15
            )

            def update(i):
                # global rh_x, rh_y, rh_z, lh_x, lh_y, lh_z
                # global chest_x, chest_y, chest_z, lua_x, lua_y, lua_z, rua_x, rua_y, rua_z, lla_x, lla_y, lla_z, rla_x, rla_y, rla_z, waist_x, waist_y, waist_z
                global grasp_condn, caught_workload, velocity_magnitude, physical_accumulated, accumulate_start_time, start_time, total_time_dwell, accumulated_workload, first_time_nf, accumulated_workload_bar, gripper_state_value, reset_condn

                # offset the data by waist
                chest_data_x = chest_x - waist_x
                chest_data_y = chest_y - waist_y
                chest_data_z = chest_z - waist_z
                lua_data_x = lua_x - waist_x
                lua_data_y = lua_y - waist_y - 0.0000001
                lua_data_z = lua_z - waist_z
                rua_data_x = rua_x - waist_x
                rua_data_y = rua_y - waist_y
                rua_data_z = rua_z - waist_z
                lla_data_x = lla_x - waist_x
                lla_data_y = lla_y - waist_y
                lla_data_z = lla_z - waist_z
                rla_data_x = rla_x - waist_x
                rla_data_y = rla_y - waist_y
                rla_data_z = rla_z - waist_z
                lh_data_x = lh_x - waist_x
                lh_data_y = lh_y - waist_y
                lh_data_z = lh_z - waist_z
                rh_data_x = rh_x - waist_x
                rh_data_y = rh_y - waist_y
                rh_data_z = rh_z - waist_z
                waist_data_x = waist_x - waist_x
                waist_data_y = waist_y - waist_y
                waist_data_z = waist_z - waist_z

                # print(rua_data_x, rua_data_y, rua_data_z, lua_data_x, lua_data_y, lua_data_z)

                # Calculate joint angles
                # Right Shoulder abd adduction
                r_shoulder_abd_add_baseline = []
                v1 = []
                v2 = []
                r_shoulder_abd_add = []
                r_shoulder_abd_add_baseline = (
                    -(rua_data_y)**2 + (rua_data_y) * (lua_data_y) - 0.6 *
                    (lua_data_z - rua_data_z)
                ) / (lua_data_y - rua_data_y)
                v1 = [r_shoulder_abd_add_baseline - rua_data_y, 0.6]
                v2 = [rla_data_y - rua_data_y, rla_data_z - rua_data_z]
                if norm(v1, 1) == 0.0:
                    div_value1 = 1.0
                else:
                    div_value1 = norm(v1, 1)
                if norm(v2, 1) == 0.0:
                    div_value2 = 1.0
                else:
                    div_value2 = norm(v2, 1)
                v1 = [v1[0] / div_value1, v1[1] / div_value1]
                # v5 /= div_value1
                v2 = [v2[0] / div_value2, v2[1] / div_value2]
                r_shoulder_abd_add = round(
                    math.degrees((math.acos(np.dot(v1, v2))))
                )
                if r_shoulder_abd_add > 120:
                    r_shoulder_abd_add = 120
                if r_shoulder_abd_add == 0:
                    r_shoulder_abd_add = 1
                if np.isnan(r_shoulder_abd_add):
                    r_shoulder_abd_add = np.nan_to_num(r_shoulder_abd_add)
                # print(r_shoulder_abd_add)
                # Right Shoulder ex flexion
                v4 = []
                r_shoulder_ex_flex = []
                v3 = [0, 0.6]
                v4 = [rla_data_x - rua_data_x, rla_data_z - rua_data_z]
                if norm(v3, 1) == 0.0:
                    div_value1 = 1.0
                else:
                    div_value1 = norm(v3, 1)
                if norm(v4, 1) == 0.0:
                    div_value2 = 1.0
                else:
                    div_value2 = norm(v4, 1)
                v3 = [v3[0] / div_value1, v3[1] / div_value1]
                # v5 /= div_value1
                v4 = [v4[0] / div_value2, v4[1] / div_value2]
                r_shoulder_ex_flex = round(
                    math.degrees(math.acos(np.dot(v3, v4)))
                )
                # print(norm(v3, 1), "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                if r_shoulder_ex_flex > 150:
                    r_shoulder_ex_flex = 150
                if r_shoulder_ex_flex == 0:
                    r_shoulder_ex_flex = 1
                if np.isnan(r_shoulder_ex_flex):
                    r_shoulder_ex_flex = np.nan_to_num(r_shoulder_ex_flex)
                # print(r_shoulder_ex_flex)
                # Right Elbow flexion
                v5 = []
                v6 = []
                r_elbow_flex = []
                v5 = [rua_data_x - rla_data_x, rua_data_z - rla_data_z]
                v6 = [rh_data_x - rla_data_x, rh_data_z - rla_data_z]
                if norm(v5, 1) == 0.0:
                    div_value1 = 1.0
                else:
                    div_value1 = norm(v5, 1)
                if norm(v6, 1) == 0.0:
                    div_value2 = 1.0
                else:
                    div_value2 = norm(v6, 1)
                v5 = [v5[0] / div_value1, v5[1] / div_value1]
                # v5 /= div_value1
                v6 = [v6[0] / div_value2, v6[1] / div_value2]
                # r_elbow_flex = 180-round(math.degrees(math.acos(np.dot(v5/div_value1, v6/div_value2))))
                r_elbow_flex = 150 - round(
                    math.degrees(math.acos(np.dot(v5, v6)))
                )
                if r_elbow_flex > 150:
                    r_elbow_flex = 150
                if r_elbow_flex == 0:
                    r_elbow_flex = 1
                if np.isnan(r_elbow_flex):
                    r_elbow_flex = np.nan_to_num(r_elbow_flex)
                # print(r_elbow_flex)
                # Left Shoulder abd adduction
                l_shoulder_abd_add_baseline = []
                v11 = []
                v12 = []
                l_shoulder_abd_add = []
                l_shoulder_abd_add_baseline = (
                    -(rua_data_y)**2 + (rua_data_y) * (lua_data_y) - 0.6 *
                    (rua_data_z - lua_data_z)
                ) / (rua_data_y - lua_data_y)
                v11 = [l_shoulder_abd_add_baseline - lua_data_y, 0.6]
                v12 = [lla_data_y - lua_data_y, lla_data_z - lua_data_z]
                if norm(v11, 1) == 0.0:
                    div_value1 = 1.0
                else:
                    div_value1 = norm(v11, 1)
                if norm(v12, 1) == 0.0:
                    div_value2 = 1.0
                else:
                    div_value2 = norm(v12, 1)
                v11 = [v11[0] / div_value1, v11[1] / div_value1]
                # v5 /= div_value1
                v12 = [v12[0] / div_value2, v12[1] / div_value2]
                l_shoulder_abd_add = round(
                    math.degrees(math.acos(np.dot(v11, v12)))
                )
                if l_shoulder_abd_add > 120:
                    l_shoulder_abd_add = 120
                if l_shoulder_abd_add == 0:
                    l_shoulder_abd_add = 1
                if np.isnan(l_shoulder_abd_add):
                    l_shoulder_abd_add = np.nan_to_num(l_shoulder_abd_add)
                # print(l_shoulder_abd_add)
                # Left Shoulder ex flexion
                v14 = []
                l_shoulder_ex_flex = []
                v13 = [0, 0.6]
                v14 = [lla_data_x - lua_data_x, lla_data_z - lua_data_z]
                if norm(v13, 1) == 0.0:
                    div_value1 = 1.0
                else:
                    div_value1 = norm(v13, 1)
                if norm(v14, 1) == 0.0:
                    div_value2 = 1.0
                else:
                    div_value2 = norm(v14, 1)
                v13 = [v13[0] / div_value1, v13[1] / div_value1]
                # v5 /= div_value1
                v14 = [v14[0] / div_value2, v14[1] / div_value2]
                l_shoulder_ex_flex = round(
                    math.degrees(math.acos(np.dot(v13, v14)))
                )
                if l_shoulder_ex_flex > 150:
                    l_shoulder_ex_flex = 150
                if l_shoulder_ex_flex == 0:
                    l_shoulder_ex_flex = 1
                if np.isnan(l_shoulder_ex_flex):
                    l_shoulder_ex_flex = np.nan_to_num(l_shoulder_ex_flex)
                # print(l_shoulder_ex_flex)
                # Left Elbow flexion
                v15 = []
                v16 = []
                l_elbow_flex = []
                v15 = [lua_data_x - lla_data_x, lua_data_z - lla_data_z]
                v16 = [lh_data_x - lla_data_x, lh_data_z - lla_data_z]
                if norm(v15, 1) == 0.0:
                    div_value1 = 1.0
                else:
                    div_value1 = norm(v15, 1)
                if norm(v16, 1) == 0.0:
                    div_value2 = 1.0
                else:
                    div_value2 = norm(v16, 1)
                v15 = [v15[0] / div_value1, v15[1] / div_value1]
                # v5 /= div_value1
                v16 = [v16[0] / div_value2, v16[1] / div_value2]
                l_elbow_flex = 150 - round(
                    math.degrees(math.acos(np.dot(v15, v16)))
                )
                if l_elbow_flex > 150:
                    l_elbow_flex = 150
                if l_elbow_flex == 0:
                    l_elbow_flex = 1
                if np.isnan(l_elbow_flex):
                    l_elbow_flex = np.nan_to_num(l_elbow_flex)
                # print(l_elbow_flex, r_elbow_flex)

                # Calculate workload
                Effort_AD_index = AD_mapping * (100 / max(AD_mapping))  # 150
                Effort_MD_index = MD_mapping * (100 / max(MD_mapping))  # 120
                Effort_BI_index = BI_mapping * (100 / max(BI_mapping))  # 150
                Effort_AD_R = []
                Effort_MD_R = []
                R_Shoulder_workload = []
                R_Elbow_workload = []

                Effort_AD_R = Effort_AD_index[int(r_shoulder_ex_flex) - 1]
                Effort_MD_R = Effort_MD_index[int(r_shoulder_abd_add) - 1]
                R_Shoulder_workload = round(
                    0.43 * Effort_AD_R + 0.57 * Effort_MD_R
                )  # ratio AD:MD = 3:4
                # print(R_Shoulder_workload)
                R_Elbow_workload = round(Effort_BI_index[int(r_elbow_flex) - 1])
                # print(R_Elbow_workload)
                Effort_AD_L = []
                Effort_MD_L = []
                L_Shoulder_workload = []
                L_Elbow_workload = []
                Effort_AD_L = Effort_AD_index[int(l_shoulder_ex_flex) - 1]
                Effort_MD_L = Effort_MD_index[int(l_shoulder_abd_add) - 1]
                L_Shoulder_workload = round(
                    0.43 * Effort_AD_L + 0.57 * Effort_MD_L
                )  # ratio AD:MD = 3:4
                L_Elbow_workload = round(Effort_BI_index[int(l_elbow_flex) - 1])

                # Calculate total workload
                workload_overall = round(
                    (0.7 * R_Shoulder_workload + 0.3 * R_Elbow_workload) * 0.9
                    + (0.7 * L_Shoulder_workload + 0.3 * L_Elbow_workload) * 0.1
                )
                realtime_physical = workload_overall
                realtime_physical_bar = 0.5 - (realtime_physical * 0.01)
                # print(realtime_physical_bar)
                # workload.publish(workload_overall)

                # for accumulated workload
                # print(physical_accumulated, accumulate_start_time)
                # # print(1)
                if (
                    physical_accumulated == 1
                    and (gripper_state_value == 3 or gripper_state_value == 1)
                ):

                    velocity_magnitude = 0.051
                    grasp_condn = 1
                    caught_workload = accumulated_workload

                if velocity_magnitude > 0.05:

                    physical_accumulated = 0
                    same_accumulated = 0
                    per_fixation_total = 0
                    accumulate_start_time = 0
                    total_time_dwell = None
                    if first_time_nf == 0:
                        first_time_nf = 1
                        start_time = rospy.get_time()
                    total_time = rospy.get_time() - start_time

                if physical_accumulated == 1:
                    if accumulate_start_time is not None:
                        if (rospy.get_time() - accumulate_start_time) > 0.1:
                            # if total_time_dwell is None:
                            #     total_time_dwell = rospy.get_time() - accumulate_start_time
                            # else:
                            #     total_time_dwell += rospy.get_time() - accumulate_start_time
                            total_time_dwell = rospy.get_time(
                            ) - accumulate_start_time
                    # accumulate_start_time = rospy.get_time()
                    total_time = rospy.get_time() - start_time

                if velocity_magnitude <= 0.05 and physical_accumulated == 0:
                    physical_accumulated = 1
                    accumulate_start_time = rospy.get_time()
                    per_fixation_total = rospy.get_time()
                    first_time_nf = 0

                # print(
                #     "##################", physical_accumulated, total_time,
                #     total_time_dwell
                # )

                # if (physical_accumulated == 1 and gripper_state_value == 3)  or (physical_accumulated == 1 and gripper_state_value == 2):
                #     reset_condn = 1

                # else:
                #     reset_condn = 0

                if (
                    physical_accumulated == 1
                    and (gripper_state_value == 3 or gripper_state_value == 1)
                ):
                    # Reset the calculation of physical workload
                    print("???????????????????????????????????????????")
                    accumulated_workload = 0
                    accumulated_workload_bar = 0.5
                    total_time_dwell = None

                # if physical_accumulated == 1:
                #     # Reset the calculation of physical workload
                #     print("???????????????????????????????????????????")
                #     accumulated_workload = 0
                #     accumulated_workload_bar = 0.5
                #     total_time_dwell = None

                if total_time_dwell is not None and total_time != 0:
                    if grasp_condn == 0:
                        accumulated_workload = round(
                            (((realtime_physical) *
                              (total_time_dwell)) / 1000) * 100
                        )
                    else:
                        accumulated_workload = round(
                            (((realtime_physical) *
                              (total_time_dwell)) / 1000) * 100
                        ) + caught_workload / 2
                    if accumulated_workload >= 100:
                        accumulated_workload = 100
                    accumulated_workload_bar = 0.5 - (
                        accumulated_workload * 0.01
                    )

                if realtime_physical is not None and accumulated_workload is not None:
                    physical_workload = Float64MultiArray()
                    physical_workload.data = [
                        realtime_physical, accumulated_workload
                    ]
                    workload_pub.publish(physical_workload)
                else:
                    physical_workload = Float64MultiArray()
                    physical_workload.data = [0, 0]
                    workload_pub.publish(physical_workload)
                # for accumulated workload

                r_shouler_WL_text.set_text(
                    'R_Shoulder: ' + str(R_Shoulder_workload)
                )
                r_elbow_WL_text.set_text('R_Elbow: ' + str(R_Elbow_workload))
                l_shouler_WL_text.set_text(
                    'L_Shoulder: ' + str(L_Shoulder_workload)
                )
                l_elbow_WL_text.set_text('L_Elbow: ' + str(L_Elbow_workload))
                current_WL_text.set_text('CURRENT: ' + str(realtime_physical))
                accumulated_WL_text.set_text(
                    'ACCUMULATED: ' + str(accumulated_workload)
                )

                dot_head.set_data(chest_data_x, chest_data_y)
                dot_head.set_3d_properties(chest_data_z - 0.3, 'z')
                dot_chest.set_data(chest_data_x, chest_data_y)
                dot_chest.set_3d_properties(chest_data_z, 'z')
                dot_lua.set_data(lua_data_x, lua_data_y)
                dot_lua.set_3d_properties(lua_data_z, 'z')
                dot_rua.set_data(rua_data_x, rua_data_y)
                dot_rua.set_3d_properties(rua_data_z, 'z')
                dot_lla.set_data(lla_data_x, lla_data_y)
                dot_lla.set_3d_properties(lla_data_z, 'z')
                dot_rla.set_data(rla_data_x, rla_data_y)
                dot_rla.set_3d_properties(rla_data_z, 'z')
                dot_lh.set_data(lh_data_x, lh_data_y)
                dot_lh.set_3d_properties(lh_data_z, 'z')
                dot_rh.set_data(rh_data_x, rh_data_y)
                dot_rh.set_3d_properties(rh_data_z, 'z')
                dot_waist.set_data(waist_data_x, waist_data_y)
                dot_waist.set_3d_properties(waist_data_z, 'z')

                line_head_chest.set_data(
                    [chest_data_x, chest_data_x], [chest_data_y, chest_data_y]
                )
                line_head_chest.set_3d_properties(
                    [chest_data_z, chest_data_z - 0.3], 'z'
                )
                line_chest_waist.set_data(
                    [chest_data_x, waist_data_x], [chest_data_y, waist_data_y]
                )
                line_chest_waist.set_3d_properties(
                    [chest_data_z, waist_data_z], 'z'
                )
                line_left_shoulder.set_data(
                    [chest_data_x, lua_data_x], [chest_data_y, lua_data_y]
                )
                line_left_shoulder.set_3d_properties(
                    [chest_data_z, lua_data_z], 'z'
                )
                line_right_shoulder.set_data(
                    [chest_data_x, rua_data_x], [chest_data_y, rua_data_y]
                )
                line_right_shoulder.set_3d_properties(
                    [chest_data_z, rua_data_z], 'z'
                )
                line_left_upperarm.set_data(
                    [lla_data_x, lua_data_x], [lla_data_y, lua_data_y]
                )
                line_left_upperarm.set_3d_properties(
                    [lla_data_z, lua_data_z], 'z'
                )
                line_right_upperarm.set_data(
                    [rla_data_x, rua_data_x], [rla_data_y, rua_data_y]
                )
                line_right_upperarm.set_3d_properties(
                    [rla_data_z, rua_data_z], 'z'
                )
                line_left_lowerarm.set_data(
                    [lla_data_x, lh_data_x], [lla_data_y, lh_data_y]
                )
                line_left_lowerarm.set_3d_properties(
                    [lla_data_z, lh_data_z], 'z'
                )
                line_right_lowerarm.set_data(
                    [rla_data_x, rh_data_x], [rla_data_y, rh_data_y]
                )
                line_right_lowerarm.set_3d_properties(
                    [rla_data_z, rh_data_z], 'z'
                )

                # for workload bars
                dot_current.set_data(0.35, -0.8)
                dot_current.set_3d_properties(realtime_physical_bar, 'z')
                line_current.set_data([0.35, 0.35], [-0.8, -0.8])
                line_current.set_3d_properties(
                    [0.47, realtime_physical_bar], 'z'
                )
                dot_accumulated.set_data(0.35, 1)
                dot_accumulated.set_3d_properties(accumulated_workload_bar, 'z')
                line_accumulated.set_data([0.35, 0.35], [1, 1])
                line_accumulated.set_3d_properties(
                    [0.47, accumulated_workload_bar], 'z'
                )

                rate.sleep()

                return dot_chest, dot_lua, dot_rua, dot_lla, dot_rla, dot_lh, dot_rh, dot_waist, dot_head, \
                line_head_chest, line_left_shoulder, line_chest_waist, line_right_shoulder, line_left_upperarm, line_right_upperarm, \
                line_left_lowerarm, line_right_lowerarm, \
                r_shouler_WL_text, r_elbow_WL_text, l_shouler_WL_text, l_elbow_WL_text, current_WL_text, accumulated_WL_text, \
                dot_current, line_current, dot_accumulated, line_accumulated

            ani = FuncAnimation(
                fig=fig, func=update, interval=1, blit=True, repeat=False
            )
            plt.axis('off')
            # plt.show()

        except Exception as e:
            print(f'{e}')


if __name__ == '__main__':
    run()
