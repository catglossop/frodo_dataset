import os 
import glob 
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import time 
import shutil
import openai
from PIL import Image 
import io
import base64
from tqdm import tqdm
import copy

path = "/hdd/frodo/frodo_dataset_control_only_clean_no_pause"
output_path = "/hdd/frodo/frodo_dataset_control_only_clean_no_pause_resampled"

OVERWRITE = False
if not os.path.exists(output_path):
    os.makedirs(output_path)
if OVERWRITE:
    shutil.rmtree(output_path)
    os.makedirs(output_path)
paths = glob.glob(os.path.join(path, "*"))
total_stationary = 0
total_control = 0
skipped_folders = []
waypoint_spacing_straight = []
waypoint_spacing = []
avg_dt = []
straight_inds = {}
for folder in tqdm(paths): 
    print(f"Processing {folder}")
    data_pkl_file = os.path.join(folder, "traj_data.pkl")
    if not os.path.exists(data_pkl_file):
        print(f"Skipping {folder}")
        breakpoint()
        skipped_folders.append(folder)
        continue
    with open(data_pkl_file, "rb") as f:
        temp_data = np.load(f, allow_pickle=True)
    data = copy.deepcopy(temp_data)
    data_len = data["position"].shape[0]
    image_len = len(glob.glob(os.path.join(folder, "*.jpg")))
    if data_len != image_len:
        print(data_len, image_len)
        breakpoint()

    straight = False
    prev_straight = False
    straight_start = None
    straight_end = None
    for idx in range(1, data["position"].shape[0]):
        straight = np.abs(data["linear"][idx]) > 0.1 and data["angular"][idx] < 0.05
        if straight and not prev_straight:
            straight_start = idx
        if not straight and prev_straight:
            straight_end = idx
            if folder not in straight_inds.keys():
                straight_inds[folder] = [[straight_start, straight_end]]
            else:
                straight_inds[folder].append([straight_start, straight_end])
        if straight and idx == data["position"].shape[0] - 1:
            straight_end = idx
            if folder not in straight_inds.keys():
                straight_inds[folder] = [[straight_start, straight_end]]
            else:
                straight_inds[folder].append([straight_start, straight_end])
        prev_straight = straight
        spacing = np.linalg.norm(data["position"][idx] - data["position"][idx - 1])
        dt = data["timestamps"][idx] - data["timestamps"][idx - 1]
        waypoint_spacing.append(spacing)
        if straight:
            waypoint_spacing_straight.append(spacing)
        avg_dt.append(dt)


# Get the average spacing between waypoints
avg_waypoint_spacing = np.array(waypoint_spacing).mean()
print(f"Average waypoint spacing: {avg_waypoint_spacing:.2f} meters")
print(f"Average waypoint spacing straight: {np.array(waypoint_spacing_straight).mean():.2f} meters")
print(f"Total number of skipped folders: {len(skipped_folders)}")
print(f"Average dt: {np.array(avg_dt).mean()} seconds")
avg_dt = np.array(avg_dt).mean()
goal_spacing_straight = 0.75 # meters
straight_sampling_factor = np.ceil(goal_spacing_straight / np.array(waypoint_spacing_straight).mean())
# Resample the data to have sample rate of 4 Hz 
sampling_factor = 4

print(f"Sampling factor: {sampling_factor}")
print(f"Straight sampling factor: {straight_sampling_factor}")
for folder in tqdm(paths):
    folder_name = folder.split("/")[-1]
    if os.path.exists(os.path.join(output_path, folder_name, "traj_data.pkl")) and OVERWRITE == False and len(glob.glob(os.path.join(output_path, folder_name, "*.jpg"))) > 0:
        print("Folder already exists, skipping")
        continue
    traj_data_sampled = {"position": np.zeros((0,2)), "yaw": np.zeros((0,)), "timestamps": np.zeros((0,)), "linear": np.zeros((0,)), "angular": np.zeros((0,)), "control_timestamps": np.zeros((0,))}
    traj_data_path = os.path.join(folder, "traj_data.pkl")
    with open(traj_data_path, "rb") as f:
        temp_data = np.load(f, allow_pickle=True)
    traj_data = copy.deepcopy(temp_data)
    image_inds = np.arange(traj_data["position"].shape[0])
    image_inds_sampled = np.zeros((0,))
    prev_end = 0
    if folder not in straight_inds.keys() or len(straight_inds[folder]) == 0:
        traj_data_sampled = {
            "position": traj_data["position"][::int(sampling_factor)],
            "yaw": traj_data["yaw"][::int(sampling_factor)],
            "timestamps": traj_data["timestamps"][::int(sampling_factor)],
            "linear": traj_data["linear"][::int(sampling_factor)],
            "angular": traj_data["angular"][::int(sampling_factor)],
            "control_timestamps": traj_data["control_timestamps"][::int(sampling_factor)],
        }
        image_inds_sampled = image_inds[::int(sampling_factor)]
    else:
        for idx, straight_ind in enumerate(straight_inds[folder]):
            start, end = straight_ind
            if idx == 0 and start > 0: 
                # sample the non-straight part
                traj_data_sampled = {
                    "position": traj_data["position"][prev_end:start][::int(sampling_factor)],
                    "yaw": traj_data["yaw"][prev_end:start][::int(sampling_factor)],
                    "timestamps": traj_data["timestamps"][prev_end:start][::int(sampling_factor)],
                    "linear": traj_data["linear"][prev_end:start][::int(sampling_factor)],
                    "angular": traj_data["angular"][prev_end:start][::int(sampling_factor)],
                    "control_timestamps": traj_data["control_timestamps"][prev_end:start][::int(sampling_factor)],
                }
                image_inds_sampled = image_inds[prev_end:start][::int(sampling_factor)]
            elif idx != 0 and start > prev_end:
                # sample the non-straight part
                traj_data_sampled = {
                    "position": np.vstack((traj_data_sampled["position"], traj_data["position"][prev_end:start][::int(sampling_factor)])),
                    "yaw": np.hstack((traj_data_sampled["yaw"],traj_data["yaw"][prev_end:start][::int(sampling_factor)])),
                    "timestamps": np.hstack((traj_data_sampled["timestamps"],traj_data["timestamps"][prev_end:start][::int(sampling_factor)])),
                    "linear": np.hstack((traj_data_sampled["linear"],traj_data["linear"][prev_end:start][::int(sampling_factor)])),
                    "angular": np.hstack((traj_data_sampled["angular"],traj_data["angular"][prev_end:start][::int(sampling_factor)])),
                    "control_timestamps": np.hstack((traj_data_sampled["control_timestamps"],traj_data["control_timestamps"][prev_end:start][::int(sampling_factor)])),
                }
                image_inds_sampled = np.hstack((image_inds_sampled, image_inds[prev_end:start][::int(sampling_factor)]))
            traj_data_sampled = {
                "position": np.vstack((traj_data_sampled["position"], traj_data["position"][start:end][::int(straight_sampling_factor)])),
                "yaw": np.hstack((traj_data_sampled["yaw"], traj_data["yaw"][start:end][::int(straight_sampling_factor)])),
                "timestamps": np.hstack((traj_data_sampled["timestamps"], traj_data["timestamps"][start:end][::int(straight_sampling_factor)])),
                "linear": np.hstack((traj_data_sampled["linear"],traj_data["linear"][start:end][::int(straight_sampling_factor)])),
                "angular": np.hstack((traj_data_sampled["angular"],traj_data["angular"][start:end][::int(straight_sampling_factor)])),
                "control_timestamps": np.hstack((traj_data_sampled["control_timestamps"],traj_data["control_timestamps"][start:end][::int(straight_sampling_factor)])),
            }
            image_inds_sampled = np.hstack((image_inds_sampled, image_inds[start:end][::int(straight_sampling_factor)]))
            if idx == len(straight_inds[folder]) - 1:
                # sample the non-straight part
                traj_data_sampled = {
                    "position": np.vstack((traj_data_sampled["position"],traj_data["position"][end:][::int(sampling_factor)])),
                    "yaw": np.hstack((traj_data_sampled["yaw"],traj_data["yaw"][end:][::int(sampling_factor)])),
                    "timestamps": np.hstack((traj_data_sampled["timestamps"],traj_data["timestamps"][end:][::int(sampling_factor)])),
                    "linear": np.hstack((traj_data_sampled["linear"],traj_data["linear"][end:][::int(sampling_factor)])),
                    "angular": np.hstack((traj_data_sampled["angular"], traj_data["angular"][end:][::int(sampling_factor)])),
                    "control_timestamps": np.hstack((traj_data_sampled["control_timestamps"],traj_data["control_timestamps"][end:][::int(sampling_factor)])),
                }
                image_inds_sampled = np.hstack((image_inds_sampled, image_inds[end:][::int(sampling_factor)]))
            prev_end = end
    print("original data length", traj_data["position"].shape[0])
    print("sampled data length", traj_data_sampled["position"].shape[0])
    assert traj_data_sampled["position"].shape[0] == len(image_inds_sampled)

    # Copy all relevant images 
    os.makedirs(os.path.join(output_path, folder_name), exist_ok=True)
    for i, image_num in enumerate(image_inds_sampled):
        curr_image_path = os.path.join(path, folder_name, f"{image_num}.jpg")
        image = Image.open(curr_image_path)
        # Save to the output folder
        image.save(os.path.join(output_path, folder_name, f"{i}.jpg"))
    output_traj_data_path = os.path.join(output_path, folder_name, "traj_data.pkl")
    pkl.dump(traj_data_sampled, open(output_traj_data_path, "wb"))
    print(f"Resampled {folder}")











