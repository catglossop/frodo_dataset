import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os 
import argparse
import glob
from tqdm_multiprocess import TqdmMultiProcessPool
from multiprocessing import Pool
from functools import partial
import tqdm
import cv2 
import m3u8
from moviepy.editor import *
import utm 
import pypose as pp
import shutil
from PIL import Image
from scipy import ndimage as scipy

ROBOT_L = 0.3
ROBOT_WHEEL_R = 0.05
ARROW_FACTOR = 2
ASPECT_RATIO = 4/3
DOWNSAMPLE = 0.5


def get_traj_paths(input_path): 

    paths = glob.glob(os.path.join(input_path, "*/ride_*"), recursive=True)
    print(len(paths))
    print(paths[:10])

    return paths

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=False)

def load_ts_video(path: str, camera_data) -> np.ndarray:

    playlist_files = glob.glob(os.path.join(path, "recordings", "*1000__uid_e_video.m3u8"))
    print(playlist_files)

    frames = []
    frames_dict = {}
    frame = -1
    for file in playlist_files:
        playlist = m3u8.load(file)
        idx = 0 
        for segment in playlist.segments:
            clip = cv2.VideoCapture(os.path.join(path, "recordings", segment.uri))
            length = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(length):
                _, frame = clip.read()
                if frame is None:
                    continue
                frames.append(frame)
                idx += 1
    for timestamp in camera_data.timestamp:
        frames_dict[timestamp] = frames.pop(0)

    return frames_dict 

def convert_gps_to_utm(lats, longs):
    robot_utm = []
    for i, (lat, long) in enumerate(zip(lats, longs)):
        utm_coords = utm.from_latlon(lat, long)
        if i == 0:
            utm_zone_num = utm_coords[2]
            utm_zone_letter = utm_coords[3]
            init_utm_x = utm_coords[0]
            init_utm_y = utm_coords[1]
            robot_utm_x = 0
            robot_utm_y = 0
        else:
            if utm_coords[2] != utm_zone_num or utm_coords[3] != utm_zone_letter:
                print("Error: UTM zone number or letter changed")
            robot_utm_x = utm_coords[0] - init_utm_x
            robot_utm_y = utm_coords[1] - init_utm_y
        robot_utm.append([robot_utm_x, robot_utm_y])
    return np.array(robot_utm)

def convert_wheel_to_vel(rpm_1, rpm_2, rpm_3, rpm_4):
    # Integrate wheel encoders to get odometry
    wheel_vel_l = (rpm_1 + rpm_3) * np.pi * ROBOT_WHEEL_R / 60
    wheel_vel_r = (rpm_2 + rpm_4) * np.pi * ROBOT_WHEEL_R / 60

    # Get the linear and angular velocities
    w = (wheel_vel_r - wheel_vel_l)/ROBOT_L
    v = (wheel_vel_r + wheel_vel_l)/2

    return v,w

def diff_gps(utm): 

    v = np.sqrt(np.diff(utm[:,0])**2 + np.diff(utm[:,1])**2)
    w = np.arctan2(np.diff(utm[:,1]), np.diff(utm[:,0]))

    return v, w

def alignment_utm_control(utm, control_data, viz=False):

    utm_diffs = np.diff(utm[:,:2],axis=0)
    utm_vs = (np.sqrt(utm_diffs[:,0]**2 + utm_diffs[:,1]**2)/np.diff(utm[:,2]))[1:]
    utm_yaws = np.arctan2(utm_diffs[:,1], utm_diffs[:,0])
    utm_ws = np.diff(utm_yaws)/np.diff(utm[1:,2])
    approx_v, approx_w = convert_wheel_to_vel(control_data[:,0], control_data[:,1], control_data[:,2], control_data[:,3])
    approx_v = approx_v[2:]
    approx_w = approx_w[2:]

    shifts = list(range(-100, 100))
    correlations_utm_rpms = []
    for shift in shifts:
        shifted_approx_v = approx_v[shift:] if shift >= 0 else approx_v[:shift]
        shifted_approx_w = approx_w[shift:] if shift >= 0 else approx_w[:shift]
        shifted_utm_v = utm_vs[:-shift] if shift > 0 else utm_vs[-shift:]
        shifted_utm_w = utm_ws[:-shift] if shift > 0 else utm_ws[-shift:]
        correlations_utm_rpms.append([np.corrcoef(shifted_approx_v, shifted_utm_v)[0, 1], np.corrcoef(shifted_approx_w, shifted_utm_w)[0, 1]])
    correlations_utm_rpms = np.array(correlations_utm_rpms)

    gps_rpm_corr_v = np.max(correlations_utm_rpms[...,0])
    gps_rpm_corr_w = np.max(correlations_utm_rpms[...,1])
    gps_rpm_corr_shift = shifts[np.argmax(correlations_utm_rpms[...,0])]
    print("GPS RPM CORR V: ", gps_rpm_corr_v)
    print("GPS RPM CORR W: ", gps_rpm_corr_w)
    print("GPS RPM CORR SHIFT: ", gps_rpm_corr_shift)
    if viz: 
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(shifts, correlations_utm_rpms[:,0], label="v")
        ax.plot(shifts, correlations_utm_rpms[:,1], label="w")
        ax.axvline(shifts[np.argmax(correlations_utm_rpms[...,0])], color='r')
        ax.axvline(shifts[np.argmax(correlations_utm_rpms[...,1])], color='b')
        ax.legend()
        ax.set_title("Alignment between UTM and control data")
        plt.savefig("alignment_utm_control.png")
        plt.show()



def kalman_filter(control_data, gps_data, compass_data=None): 

    P_k = np.eye(3)
    Q = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    if compass_data is not None:
        R = np.array([[5.0, 0, 0, 0], [0, 5.0, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]])
    else:
        R = np.array([[5.0, 0], [0, 5.0]])
    x_k = np.zeros((3,))

    ## Find best alignment between initial datapoints

    # GPS and control  
    timestamp_diff = np.abs(control_data.timestamp - gps_data[0,2])
    init_control_idx = np.argmin(timestamp_diff)

    # control and GPS 
    timestamp_diff = np.abs(gps_data[:,2] - control_data.timestamp[init_control_idx])
    init_gps_idx = np.argmin(timestamp_diff)
    gps_idx = init_gps_idx
    gps_data[:,:2] = gps_data[:,:2] - gps_data[init_gps_idx,:2]

    # GPS and compass 
    if compass_data is not None:
        timestamp_diff = np.abs(compass_data[:,1] - gps_data[0,2])
        compass_idx = np.argmin(timestamp_diff)
        compass_data[:,0] = compass_data[:,0] - compass_data[compass_idx,0]
    
    filtered_odom = [np.vstack((x_k.reshape(-1,1), np.array(control_data.timestamp[init_control_idx]).reshape(-1,1)))]
    
    # Loop through data to get filtered odom
    for control_idx in range(init_control_idx + 1, len(control_data.timestamp[init_control_idx:])):
        # Compute the wheel odom for the current time step 
        v, w = convert_wheel_to_vel(control_data.rpm_1[control_idx], control_data.rpm_2[control_idx], control_data.rpm_3[control_idx], control_data.rpm_4[control_idx])

        if gps_idx >= gps_data.shape[0]:
            print("Reached end of GPS data")
            break
        # Check if GPS data is available for the current time step
        t_diff = control_data.timestamp[control_idx] - gps_data[gps_idx,2]
        if t_diff >= 0.5: 
            # control data is too far ahead 
            filtered_odom.append(np.vstack((x_k.reshape(-1,1), control_data.timestamp[control_idx].reshape(-1,1))))
            gps_idx +=1 
        
        if (t_diff < 0.5 and t_diff > 0) or (gps_idx == init_gps_idx):
            if v == 0 and w == 0:
                # control input is zero, so we don't have any new information: remove this data 
                gps_idx +=1 
                filtered_odom.append(np.vstack((x_k.reshape(-1,1), control_data.timestamp[control_idx].reshape(-1,1))))
                continue
            
            # Prediction step
            if compass_data is not None:
                timestamp_diff = np.abs(compass_data[:,1] - gps_data[gps_idx,2])
                compass_idx = np.argmin(timestamp_diff)
                compass_sample = compass_data[compass_idx,0]
                gps_sample = np.hstack((gps_data[gps_idx,:2], np.sin(compass_sample), np.cos(compass_sample)))
            else: 
                gps_sample = gps_data[gps_idx,:2]

            dt = control_data.timestamp[control_idx] - control_data.timestamp[control_idx-1]
            x_k_pred = np.array([x_k[0] + v*np.cos(x_k[2])*dt, x_k[1] + v*np.sin(x_k[2])*dt, x_k[2] + w*dt])
            J_fa = np.array([[1, 0, -v*np.sin(x_k[2])*dt], [0, 1, v*np.cos(x_k[2])*dt], [0, 0, 1]])
            P_k_pred = J_fa@P_k@J_fa.T + Q
 
            # Update step 
            if compass_data is not None:
                J_h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, v*np.cos(x_k[2])*dt], [0, 0, -v*np.sin(x_k[2])*dt]])
            else:
                J_h = np.array([[1, 0, 0], [0, 1, 0]])
            K_k = P_k_pred@J_h.T@np.linalg.inv(J_h@P_k_pred@J_h.T + R)
            if compass_data is not None:
                H = np.array([x_k[0], x_k[1], np.sin(x_k[2]), np.cos(x_k[2])])
            else:
                H = x_k_pred[:2]
            x_k = x_k_pred + K_k@(gps_sample - H)
            P_k = (np.eye(K_k.shape[0]) - K_k@J_h)@P_k_pred
            gps_idx += 1

            filtered_odom.append(np.vstack((x_k.reshape(-1,1), control_data.timestamp[control_idx].reshape(-1,1))))
    filtered_odom = np.array(filtered_odom)
    if filtered_odom.shape[0] < gps_data.shape[0]: 
        breakpoint()
    return filtered_odom
        
def convert_control_to_odom(lin_vels, ang_vels, rpms, timestamps):
    control_x = 0
    control_y = 0
    control_theta = 0
    wheel_x = 0 
    wheel_y = 0
    wheel_theta = 0
    robot_control_odom = []
    robot_wheel_odom = []
    lin_vels = lin_vels*0.75
    ang_vels = (ang_vels -0.02)*0.75
    
    for i in range(1,len(lin_vels)):
        # Integrate from control data to get odometry
        control_theta = control_theta + ang_vels[i-1] * (timestamps[i] - timestamps[i-1])
        control_x = control_x + lin_vels[i-1] * np.cos(control_theta) * (timestamps[i] - timestamps[i-1])
        control_y = control_y + lin_vels[i-1] * np.sin(control_theta) * (timestamps[i] - timestamps[i-1])

        # Integrate wheel encoders to get odometry
        wheel_vel_l = np.mean([rpms[i,0], rpms[i,2]])*2*np.pi*ROBOT_WHEEL_R/60
        wheel_vel_r = np.mean([rpms[i,1], rpms[i,3]])*2*np.pi*ROBOT_WHEEL_R/60

        # Get the linear and angular velocities
        w = (wheel_vel_r - wheel_vel_l)/ROBOT_L
        v = (wheel_vel_r + wheel_vel_l)/2 

        wheel_theta = wheel_theta + w * (timestamps[i] - timestamps[i-1])
        wheel_x = wheel_x + v * np.cos(wheel_theta) * (timestamps[i] - timestamps[i-1])
        wheel_y = wheel_y + v * np.sin(wheel_theta) * (timestamps[i] - timestamps[i-1])
        
        robot_control_odom.append([control_x, control_y, control_theta])
        robot_wheel_odom.append([wheel_x, wheel_y, wheel_theta])
    
    return np.array(robot_control_odom), np.array(robot_wheel_odom)

def compute_alignment(control_data, optical_flow_data, viz = False): 

    approx_v, approx_w = convert_wheel_to_vel(control_data.rpm_1, control_data.rpm_2, control_data.rpm_3, control_data.rpm_4) 
    control_vals = np.stack([control_data.linear, control_data.angular], axis=1)
    optical_flow_data[:,1] = optical_flow_data[:,1] * np.max(np.abs(approx_w))

    shifts = list(range(-100, 100))
    dt = np.mean(np.diff(control_data.timestamp[int(len(control_data.timestamp)*0.2):]))
    correlations_control_rpms = []
    for shift in shifts:
        shifted_approx_v = approx_v[shift:] if shift >= 0 else approx_v[:shift]
        shifted_approx_w = approx_w[shift:] if shift >= 0 else approx_w[:shift]
        shifted_control_v = control_vals[:, 0][:-shift] if shift > 0 else control_vals[:, 0][-shift:]
        shifted_control_w = control_vals[:, 1][:-shift] if shift > 0 else control_vals[:, 1][-shift:]
        correlations_control_rpms.append([np.corrcoef(shifted_approx_v, shifted_control_v)[0, 1], np.corrcoef(shifted_approx_w, shifted_control_w)[0, 1]])
    correlations_control_rpms = np.array(correlations_control_rpms)

    correlations_control_opt = []
    for shift in shifts:
        shifted_opt_flow_w = optical_flow_data[:,1][shift:] if shift >= 0 else optical_flow_data[:,1][:shift]
        shifted_control_w = control_vals[:, 1][:-shift] if shift > 0 else control_vals[:, 1][-shift:]

        correlations_control_opt.append([np.corrcoef(shifted_opt_flow_w, shifted_control_w)[0, 1], 0])
    correlations_control_opt = np.array(correlations_control_opt)

    correlations_rpms_opt = []
    for shift in shifts:
        shifted_opt_flow_w = optical_flow_data[:,1][shift:] if shift >= 0 else optical_flow_data[:,1][:shift]
        shifted_approx_w = approx_w[:-shift] if shift > 0 else approx_w[-shift:]
        correlations_rpms_opt.append([np.corrcoef(shifted_opt_flow_w, shifted_approx_w)[0, 1]])
    correlations_rpms_opt = np.array(correlations_rpms_opt)

    if viz: 
        fig, ax = plt.subplots(3, 1, figsize=(10, 10))
        ax[0].plot(shifts, correlations_control_rpms[:,0], label="v")
        ax[0].plot(shifts, correlations_control_rpms[:,1], label="w")
        ax[0].axvline(shifts[np.argmax(correlations_control_rpms[...,0])], color='r')
        ax[0].axvline(shifts[np.argmax(correlations_control_rpms[...,1])], color='b')
        ax[0].legend()

        ax[1].plot(shifts, correlations_control_opt[:,0], label="w")
        ax[1].axvline(shifts[np.argmax(correlations_control_opt[...,0])], color='r')
        ax[1].legend()

        ax[2].plot(shifts, correlations_rpms_opt[:,0], label="w")
        ax[2].axvline(shifts[np.argmax(correlations_rpms_opt[...,0])], color='r')

    control_rpm_corr = np.max(correlations_control_rpms[...,0])
    control_rpm_corr_shift = shifts[np.argmax(correlations_control_rpms[...,0])]
    control_opt_corr = np.max(correlations_control_opt[...,0])
    control_opt_corr_shift = shifts[np.argmax(correlations_control_opt[...,0])]
    rpm_opt_corr = np.max(correlations_rpms_opt[...,0])
    rpm_opt_corr_shift = shifts[np.argmax(correlations_rpms_opt[...,0])]
    print("CONTROL RPM CORR: ", control_rpm_corr)
    print("CONTROL RPM CORR SHIFT: ", control_rpm_corr_shift)
    print("CONTROL OPT CORR: ", control_opt_corr)
    print("CONTROL OPT CORR SHIFT: ", control_opt_corr_shift)
    print("RPM OPT CORR: ", rpm_opt_corr)
    print("RPM OPT CORR SHIFT: ", rpm_opt_corr_shift)
    print("DT: ", dt)

    # Check for threshold correlations and align the data accordingly
    if control_rpm_corr > 0.7: 
        final_shift = control_rpm_corr_shift

    # control_data.linear = control_data.linear[final_shift:] if final_shift >= 0 else control_data.linear[:final_shift]
    # control_data.angular = control_data.angular[final_shift:] if final_shift >= 0 else control_data.angular[:final_shift]
    # control_data.rpm_1 = control_data.rpm_1[:-final_shift] if final_shift > 0 else control_data.rpm_1[-final_shift:] 
    # control_data.rpm_2 = control_data.rpm_2[:-final_shift] if final_shift > 0 else control_data.rpm_2[-final_shift:]
    # control_data.rpm_3 = control_data.rpm_3[:-final_shift] if final_shift > 0 else control_data.rpm_3[-final_shift:]
    # control_data.rpm_4 = control_data.rpm_4[:-final_shift] if final_shift > 0 else control_data.rpm_4[-final_shift:]


    # if final_shift > 0:
    #     control_data.timestamp = control_data.timestamp[final_shift:]
    # else:   
    #     control_data.timestamp = control_data.timestamp[:final_shift]
    return control_data

def convert_mag_to_yaw(compass):
    x = compass[1]
    y = compass[2]

    ang = np.arctan2(x, y)

    return ang + np.pi/2

def visualize_data(abs_pos, abs_yaw, filtered_odom, frame=None, save = False, idx=0, folder_name=None, first = False):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].plot(abs_pos[:,0], abs_pos[:,1], label="UTM")
    ax[0].plot(filtered_odom[:,0], filtered_odom[:,1], label="Filtered")
    for i in range(0, len(filtered_odom)):
        ax[0].arrow(filtered_odom[i,0], filtered_odom[i,1], np.cos(filtered_odom[i,2])*ARROW_FACTOR, np.sin(filtered_odom[i,2])*ARROW_FACTOR, head_width=0.1, head_length=0.2, fc='r', ec='r')
    plt.legend()
    ax[0].set_title("Position estimates")
    if frame is not None:
        ax[1].imshow(frame)
    if save:
        os.makedirs(f"viz_{folder_name}", exist_ok=True)
        plt.savefig(f"viz_{folder_name}/viz_{idx}.png")
    if first: 
        plt.savefig("viz_estimates.png")
    plt.close()

def visualize_data_odom(vel_odom, wheel_odom):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(vel_odom[:,0], vel_odom[:,1], label="vel_odom")
    ax.plot(wheel_odom[:,0], wheel_odom[:,1], label="wheel_odom")
    for i in range(0, len(vel_odom), 10):
        ax.arrow(vel_odom[i,0], vel_odom[i,1], np.cos(vel_odom[i,2]), np.sin(vel_odom[i,2]), head_width=0.1, head_length=0.2, fc='r', ec='r')
        ax.arrow(wheel_odom[i,0], wheel_odom[i,1], np.cos(wheel_odom[i,2]), np.sin(wheel_odom[i,2]), head_width=0.1, head_length=0.2, fc='b', ec='b')
        plt.legend()
    plt.savefig("odom_estimates.png")
    
def transform_image(image):
    h,w = image.shape[:2]
    center = (h / 2, w / 2)
    if h > w: 
        new_w = w
        new_h = int(w / ASPECT_RATIO)
        crop_img = image[int(center[0] - new_h/2):int(center[0] + new_h/2),:]
    if w >= h: 
        new_h = h
        new_w = int(h * ASPECT_RATIO)
        crop_img = image[:,int(center[1] - new_w/2):int(center[1] + new_w/2)]
    new_h, new_w = new_h*DOWNSAMPLE, new_w*DOWNSAMPLE
    image = cv2.resize(crop_img, (new_h, new_w))

    return image

def visualize_control(control_data, left_right_optical_flow):
    rpm_vs = []
    rpm_ws = []
    input_vs = []
    input_ws = []
    for idx, timestamp in enumerate(control_data.timestamp):
        rpm_v, rpm_w = convert_wheel_to_vel(control_data.rpm_1[idx], control_data.rpm_2[idx], control_data.rpm_3[idx], control_data.rpm_4[idx])
        input_v = control_data.linear[idx]
        input_w = control_data.angular[idx]
        rpm_vs.append(rpm_v)
        rpm_ws.append(rpm_w)
        input_vs.append(input_v)
        input_ws.append(input_w)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(control_data.timestamp, rpm_vs, label="RPM v")
    ax[0].plot(control_data.timestamp, input_vs, label="Input v")
    ax[1].plot(control_data.timestamp, rpm_ws, label="RPM w")
    ax[1].plot(control_data.timestamp, input_ws, label="Input w")
    ax[1].plot(left_right_optical_flow[:,0], left_right_optical_flow[:,1], label="Optical flow left")
    plt.legend()
    plt.savefig("control_estimates.png")
    plt.close()

def get_optical_flow(video_data, frame_timestamps, control_data):
    feature_params = dict( maxCorners = 100,
        qualityLevel = 0.3,
        minDistance = 7,
        blockSize = 7)
    left_right_optical_flow = []
    for i in range(1, frame_timestamps.shape[0]):
        # Calculate optical flow
        prev = cv2.cvtColor(video_data[frame_timestamps[i-1]], cv2.COLOR_RGB2GRAY)
        next = cv2.cvtColor(video_data[frame_timestamps[i]], cv2.COLOR_RGB2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev, mask = None, **feature_params)
        if p0 is None:
            left_right_optical_flow.append(np.array([0, 0]))
            continue
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, next, p0, None)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        flow = good_new - good_old
        if flow.shape[0] == 0:
            left_right_optical_flow.append(np.array([0, 0]))
            continue
        left_right_optical_flow.append(flow.mean(axis=0))
    left_right_optical_flow = np.array(left_right_optical_flow)
    left_right_optical_flow = left_right_optical_flow/np.max(np.abs(left_right_optical_flow))
    left_right_optical_flow = np.hstack((frame_timestamps[1:].reshape(-1,1), left_right_optical_flow))
    optical_flow_with_control_timestamps = []
    cursor = 0
    for i in range(len(control_data.timestamp)):
        while frame_timestamps[cursor] < control_data.timestamp[i] and cursor < len(frame_timestamps):
            cursor += 1
        if cursor >= len(frame_timestamps):
            break
        optical_flow_with_control_timestamps.append(left_right_optical_flow[cursor])
    optical_flow_with_control_timestamps = np.array(optical_flow_with_control_timestamps)
    return optical_flow_with_control_timestamps

def process_traj(path, output_path, viz=False):

    folder_name = path.split("/")[-1]
    os.makedirs(f"{output_path}/{folder_name}")
                
    # load control csv file 
    control_path = glob.glob(path + "/control_data_*.csv")[0]
    control_data = load_csv(control_path)
    rpms = np.array([control_data.rpm_1, control_data.rpm_2, control_data.rpm_3, control_data.rpm_4]).T
    viz= True 
    if viz: 
        integrated_vel_odom, integrated_wheel_odom = convert_control_to_odom(control_data.linear, control_data.angular, rpms, control_data.timestamp)
        visualize_data_odom(integrated_vel_odom, integrated_wheel_odom)

    # load gps csv file 
    gps_path = glob.glob(path + "/gps_data_*.csv")[0]
    gps_data = load_csv(gps_path)
    robot_utm = convert_gps_to_utm(gps_data.latitude, gps_data.longitude)
    robot_utm = scipy.gaussian_filter1d(robot_utm, 1.5, axis=0)
    robot_utm_timestamps = gps_data.timestamp / 1000
    robot_utm = np.hstack((robot_utm, np.array(robot_utm_timestamps).reshape(-1,1)))
    robot_utm[:,2] = robot_utm[:,2] - 6

    control_data_with_gps_timestamps = []
    cursor = 0
    for i in range(robot_utm.shape[0]):
        while control_data.timestamp[cursor] < robot_utm[i,2] and cursor < len(control_data.timestamp)-1:
            cursor += 1
        if cursor >= len(control_data.timestamp):
            break
        control_data_with_gps_timestamps.append(np.array([control_data.rpm_1[cursor], control_data.rpm_2[cursor], control_data.rpm_3[cursor], control_data.rpm_4[cursor], control_data.timestamp[cursor]]))
    control_data_with_gps_timestamps = np.array(control_data_with_gps_timestamps)
    alignment_utm_control(robot_utm, control_data_with_gps_timestamps, viz=True)

    # load imu csv file 
    imu_path = glob.glob(path + "/imu_data_*.csv")[0]
    imu_data = load_csv(imu_path)

    # Put imu data into the correct format
    compass = [np.array(imu_data.compass[i].strip("][").replace("\"", "").split(",")).astype(np.float64) for i in range(len(imu_data.compass))]
    compass_data = np.array(compass)
    robot_abs_yaw = np.array([convert_mag_to_yaw(compass_data[i,:]) for i in range(compass_data.shape[0])])
    robot_abs_yaw = np.hstack((robot_abs_yaw.reshape(-1,1), np.array(compass_data[:,3]).reshape(-1,1)))

    # Perform the kalman filter on the data
    filtered_odom = kalman_filter(control_data, robot_utm).squeeze(axis=2)
    odom_yaw = np.array([np.arctan2(filtered_odom[i,1] - filtered_odom[i-1,1], filtered_odom[i,0] - filtered_odom[i-1,0]) for i in range(1, len(filtered_odom))])
    filtered_odom[1:,2] = odom_yaw
    visualize_data(robot_utm, robot_abs_yaw, filtered_odom, save=False, first=True)

    # load front camera csv file
    front_camera_path = glob.glob(path + "/front_camera_timestamps_*.csv")[0]
    front_camera_data = load_csv(front_camera_path)

    # load video data 
    video_data = load_ts_video(path, front_camera_data)
    odom_timestamps = filtered_odom[:,3]
    frame_timestamps = np.array(list(video_data.keys()))

    optical_flow_with_control_timestamps = get_optical_flow(video_data, frame_timestamps, control_data)

    # compute alignment between control and wheel odometry
    control_data = compute_alignment(control_data, optical_flow_with_control_timestamps)
    
    visualize_control(control_data, optical_flow_with_control_timestamps)


    for idx, timestamp in enumerate(odom_timestamps):
        time_diff = np.abs(frame_timestamps - timestamp)
        frame_idx = np.argmin(time_diff)
        frame = video_data[frame_timestamps[frame_idx]]
        visualize_data(robot_utm[:idx+1], robot_abs_yaw[:idx+1], filtered_odom[:idx+1], frame=frame, save=True, idx=idx, folder_name=folder_name)
        # frame = transform_image(frame)
        # cv2.imwrite(f"{output_path}/{folder_name}/{image_idx}.jpg", frame)
    
    traj_dict = {}
    traj_dict["pos"] = filtered_odom[:,:2]
    traj_dict["yaw"] = filtered_odom[:,-1]
    gif_frames = []
    paths = glob.glob(f"viz_{folder_name}/*")
    paths = sorted(paths, key=lambda x: int(x.strip(".png").split("/")[-1].split("_")[-1]))
    for file in paths:
        gif_frames.append(Image.open(file))
    
    gif_frames[0].save(f"trajectory_{folder_name}.gif", save_all=True, append_images=gif_frames[1:], duration=200, loop=0)

# save to the gnm format for a batch of trajectories
def save_to_gnm(paths, output_path, tqdm_func, global_tqdm):

    for path in paths: 
        process_traj(path, output_path)

        global_tqdm.update(1)
    global_tqdm.write(f"Finished {output_path}")

def save_to_gnm_single_process(paths, output_path):

    for path in paths: 
        print("Current path: ", path)
        process_traj(path, output_path)
        print("Finished path: ", path)  

    print(f"Finished {output_path}")

def main(args):

    paths = get_traj_paths(args.input_path)
    # shard paths
    shards = np.array_split(
        paths, np.ceil(len(paths) / args.num_workers)
    )

    # create output paths
    if args.overwrite:
        shutil.rmtree(args.output_path, ignore_errors=True)
        os.makedirs(args.output_path, exist_ok=False)
    else:
        os.makedirs(args.output_path, exist_ok=False)

    save_to_gnm_single_process(paths, args.output_path)
    # create tasks (see tqdm_multiprocess documenation)
    # tasks = [
    #     (save_to_gnm, (shards[i], args.output_path))
    #     for i in range(len(shards))
    #     ]

    # total_len = len(paths)

    # run tasks
    # pool = TqdmMultiProcessPool(args.num_workers)
    # with tqdm.tqdm(
    #     total=total_len,
    #     dynamic_ncols=True,
    #     position=0,
    #     desc="Total progress",
    # ) as pbar:
    #     pool.map(pbar, tasks, lambda _: None, lambda _: None) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")

    main(args=parser.parse_args())
