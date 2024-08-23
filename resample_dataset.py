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

path = "/hdd/frodo/frodo_dataset_control_only_clean"

paths = glob.glob(os.path.join(path, "*"))
total_stationary = 0
total_control = 0
skipped_folders = []
waypoint_spacing = []
avg_dt = []
for folder in tqdm(paths): 
    shutil.rmtree("images", ignore_errors=True)
    os.makedirs(f"images")
    print(f"Processing {folder}")
    data_pkl_file = os.path.join(folder, "traj_data.pkl")
    if not os.path.exists(data_pkl_file):
        print(f"Skipping {folder}")
        shutil.rmtree(folder)
        skipped_folders.append(folder)
        continue
    try:
        with open(data_pkl_file, "rb") as f:
            data = pkl.load(f)
        data_len = data["position"].shape[0]
        image_len = len(glob.glob(os.path.join(folder, "*.jpg")))
        if data_len != image_len:
            breakpoint()
    except:
        print(f"Skipping {folder}")
        shutil.rmtree(folder)
        skipped_folders.append(folder)
        continue
    for idx in range(1, data["position"].shape[0]):
        spacing = np.linalg.norm(data["position"][idx] - data["position"][idx - 1])
        dt = data["timestamps"][idx] - data["timestamps"][idx - 1]
        waypoint_spacing.append(spacing)
        avg_dt.append(dt)


# Get the average spacing between waypoints
avg_waypoint_spacing = np.array(waypoint_spacing).mean()
print(f"Average waypoint spacing: {avg_waypoint_spacing:.2f} meters")
print(f"Total number of skipped folders: {len(skipped_folders)}")
print(f"Average dt: {np.array(avg_dt).mean()} seconds")
print(skipped_folders)

avg_dt = np.array(avg_dt).mean()
print(f"Average dt: {avg_dt}")


# Resample the data to have sample rate of 4 Hz 
RATE = 4
sampling_factor = np.ceil((1/RATE)/avg_dt)
sampling_factor = 2

print(f"Sampling factor: {sampling_factor}")
breakpoint()
for folder in tqdm(paths):
    traj_data_path = os.path.join(folder, "traj_data.pkl")
    with open(traj_data_path, "rb") as f:
        traj_data = pkl.load(f)
    traj_data_sampled = {
        "position": traj_data["position"][::int(sampling_factor)],
        "yaw": traj_data["yaw"][::int(sampling_factor)],
        "timestamps": traj_data["timestamps"][::int(sampling_factor)],
        "linear": traj_data["linear"],
        "angular": traj_data["angular"],
        "control_timestamps": traj_data["control_timestamps"],
    }
    pkl.dump(traj_data_sampled, open(traj_data_path, "wb"))
    print(f"Resampled {folder}")











