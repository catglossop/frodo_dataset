import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import pickle as pkl
import shutil
from tqdm import tqdm
import numpy.ma as ma

traj_paths = glob.glob('/hdd/frodo/frodo_dataset/*')
save_path = "/hdd/frodo/frodo_dataset_clean"
OVERWRITE = False 

if os.path.exists(save_path) and OVERWRITE:
    shutil.rmtree(save_path)
    os.makedirs(save_path)


image_paths = glob.glob('/hdd/frodo/frodo_dataset/*/*.jpg')

for image_path in tqdm(image_paths): 
    try:
        Image.open(image_path)
    except:
        print(f"Image {image_path} is corrupted, removing")
        breakpoint()

for traj_path in tqdm(traj_paths):
    if not os.path.exists(os.path.join(traj_path, "traj_data.pkl")):
        print(f"Trajectory {traj_path} does not have data, skipping")
        continue
    traj_name = traj_path.split("/")[-1]
    save_traj_path = os.path.join(save_path, traj_name)
    if os.path.exists(save_traj_path) and not OVERWRITE:
        data = pkl.load(open(os.path.join(save_traj_path, "traj_data.pkl"), "rb"))
        data_len = len(data["timestamps"])
        img_len = len(glob.glob(os.path.join(save_traj_path, "*.jpg")))
        if data_len == img_len:
            print(f"Trajectory {traj_name} already exists, skipping")
            continue
        else:
            print(f"Trajectory {traj_name} has mismatched data, cleaning")
            shutil.rmtree(save_traj_path)

    traj_data_path = os.path.join(traj_path, "traj_data.pkl")
    traj_data = np.load(traj_data_path, allow_pickle=True)
    traj_data["linear"] = np.array(traj_data["linear"])
    traj_data["angular"] = np.array(traj_data["angular"])
    traj_data["control_timestamps"] = np.array(traj_data["control_timestamps"])
    traj_data["rpm_1"] = np.array(traj_data["rpm_1"])
    traj_data["rpm_2"] = np.array(traj_data["rpm_2"])
    traj_data["rpm_3"] = np.array(traj_data["rpm_3"])
    traj_data["rpm_4"] = np.array(traj_data["rpm_4"])

    # Check if nans in control data and remove
    input_nan_mask = np.logical_or(ma.masked_invalid(traj_data["linear"]).mask, ma.masked_invalid(traj_data["angular"]).mask)
    rpm_nan_mask = np.logical_or(np.logical_or(ma.masked_invalid(traj_data["rpm_1"]).mask, ma.masked_invalid(traj_data["rpm_2"]).mask), np.logical_or(ma.masked_invalid(traj_data["rpm_3"]).mask, ma.masked_invalid(traj_data["rpm_4"]).mask))
    nan_mask = np.logical_not(np.logical_or(input_nan_mask, rpm_nan_mask))
    traj_data["linear"] = traj_data["linear"][nan_mask]
    traj_data["angular"] = traj_data["angular"][nan_mask]
    traj_data["rpm_1"] = traj_data["rpm_1"][nan_mask]
    traj_data["rpm_2"] = traj_data["rpm_2"][nan_mask]
    traj_data["rpm_3"] = traj_data["rpm_3"][nan_mask]
    traj_data["rpm_4"] = traj_data["rpm_4"][nan_mask]
    traj_data["control_timestamps"] = traj_data["control_timestamps"][nan_mask]
    print(f"Removed {np.sum(~nan_mask)} samples with nans from trajectory {traj_name}")                                                       

    print(f"Original data shape: {traj_data['pos'].shape}")

    if len(traj_data["linear"]) < 5:
        print(f"Trajectory {traj_name} is too short, skipping cleaning")
        continue

    # Get control timestamps aligned to odometry timestamps
    odom_timestamps = traj_data["timestamps"]
    control_timestamps = traj_data["control_timestamps"]
    control_odom_timestamps = []
    for idx, timestamp in enumerate(odom_timestamps):
        time_diff = np.abs(control_timestamps - timestamp, dtype=np.float64)
        control_idx = np.argmin(time_diff)
        control_odom_timestamps.append(control_idx)
    assert len(control_odom_timestamps) == len(odom_timestamps), f"Trajectory {traj_name} has mismatched timestamps"
    # Check if control is stationary 
    start_idx = 0 
    start_time = control_timestamps[control_odom_timestamps[start_idx]]
    pause = False
    traj_data_new = {}
    traj_data_new["pos"] = []
    traj_data_new["yaw"] = []
    traj_data_new["timestamps"] = []
    traj_data_new["linear"] = []
    traj_data_new["angular"] = []
    traj_data_new["control_timestamps"] = []
    remove_idx = []
    control_odom_idx = 0
    prev_control_odom_idx = 0
    while control_odom_idx < len(control_odom_timestamps):
        control_idx = control_odom_timestamps[control_odom_idx]
        if traj_data["linear"][control_idx] == 0.0 and traj_data["angular"][control_idx] == 0.0:
            pause_idx = control_idx
            while traj_data["linear"][pause_idx] == 0.0 and traj_data["angular"][pause_idx] == 0.0 and pause_idx < len(traj_data["linear"])-1:
                pause_idx += 1 
            elapsed_pause = control_timestamps[pause_idx] - control_timestamps[control_idx]
            if elapsed_pause > 10.0:
                print(f"Trajectory {traj_name} has a pause of {elapsed_pause} seconds at control index {control_idx} to {pause_idx}")
                closest_control_odom_idx = np.argmin(np.abs(pause_idx - control_odom_timestamps))
                remove_idx.append(np.arange(control_odom_idx, closest_control_odom_idx))
                if prev_control_odom_idx == 0:
                    traj_data_new["pos"].extend(traj_data["pos"][:control_odom_idx])
                    traj_data_new["yaw"].extend(traj_data["yaw"][:control_odom_idx])
                    traj_data_new["timestamps"].extend(traj_data["timestamps"][:control_odom_idx])
                    traj_data_new["linear"].extend(traj_data["linear"][:control_odom_timestamps[control_odom_idx]])
                    traj_data_new["angular"].extend(traj_data["angular"][:control_odom_timestamps[control_odom_idx]])
                    traj_data_new["control_timestamps"].extend(traj_data["control_timestamps"][:control_odom_timestamps[control_odom_idx]])
                else:
                    traj_data_new["pos"].extend(traj_data["pos"][prev_control_odom_idx:control_odom_idx])
                    traj_data_new["yaw"].extend(traj_data["yaw"][prev_control_odom_idx:control_odom_idx])
                    traj_data_new["timestamps"].extend(traj_data["timestamps"][prev_control_odom_idx:control_odom_idx])
                    traj_data_new["linear"].extend(traj_data["linear"][control_odom_timestamps[prev_control_odom_idx]:control_odom_timestamps[control_odom_idx]])
                    traj_data_new["angular"].extend(traj_data["angular"][control_odom_timestamps[prev_control_odom_idx]:control_odom_timestamps[control_odom_idx]])
                    traj_data_new["control_timestamps"].extend(traj_data["control_timestamps"][control_odom_timestamps[prev_control_odom_idx]:control_odom_timestamps[control_odom_idx]])
                control_odom_idx = closest_control_odom_idx
                prev_control_odom_idx = closest_control_odom_idx
        control_odom_idx += 1
    
    if prev_control_odom_idx == 0:
        traj_data_new["pos"].extend(traj_data["pos"])
        traj_data_new["yaw"].extend(traj_data["yaw"])
        traj_data_new["timestamps"].extend(traj_data["timestamps"])
        traj_data_new["linear"].extend(traj_data["linear"])
        traj_data_new["angular"].extend(traj_data["angular"])
        traj_data_new["control_timestamps"].extend(traj_data["control_timestamps"])
    else:
        traj_data_new["pos"].extend(traj_data["pos"][prev_control_odom_idx:])
        traj_data_new["yaw"].extend(traj_data["yaw"][prev_control_odom_idx:])
        traj_data_new["timestamps"].extend(traj_data["timestamps"][prev_control_odom_idx:])
        traj_data_new["linear"].extend(traj_data["linear"][control_odom_timestamps[prev_control_odom_idx]:])
        traj_data_new["angular"].extend(traj_data["angular"][control_odom_timestamps[prev_control_odom_idx]:])
        traj_data_new["control_timestamps"].extend(traj_data["control_timestamps"][control_odom_timestamps[prev_control_odom_idx]:])
    if len(remove_idx) == 0:
        print(f"No pauses found in trajectory {traj_name}")
    else:
        remove_idx = np.concatenate(remove_idx).flatten()
    print(f"Removed {len(remove_idx)} samples from trajectory {traj_name}")
    for key in traj_data_new.keys():
        traj_data_new[key] = np.array(traj_data_new[key])
    if len(traj_data_new["pos"]) < 5:
        print(f"Trajectory {traj_name} is too short, skipping cleaning")
        continue
    # Check data
    assert traj_data_new["pos"].shape[0] == traj_data_new["yaw"].shape[0] == traj_data_new["timestamps"].shape[0], f"Trajectory {traj_name} has mismatched data"
    assert traj_data_new["linear"].shape[0] == traj_data_new["angular"].shape[0] == traj_data_new["control_timestamps"].shape[0], f"Trajectory {traj_name} has mismatched control data"

    # Save the cleaned data
    save_traj_path = os.path.join(save_path, traj_name)

    os.makedirs(save_traj_path)
    pkl.dump(traj_data_new, open(os.path.join(save_traj_path,"traj_data.pkl"), "wb"))
    idx = 0
    for i in range(traj_data["timestamps"].shape[0]):
        if i not in remove_idx:
            img = Image.open(os.path.join(traj_path, f"{i}.jpg"))
            img.save(os.path.join(save_traj_path, f"{idx}.jpg"))
            idx += 1    
    assert idx == len(traj_data_new["timestamps"]), f"Trajectory {traj_name} has mismatched image data"




            


        
            

        