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

path = "/hdd/frodo/frodo_dataset_clean"

def PIL_to_base64(pil_image):
    pil_image = pil_image.convert("RGB")
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode() 

def relabelling_vlm(images, client):
    prompt = [{"type": "text", "text": "These images represent a trajectory of robot visual observations:"}]
    
    for i, image in enumerate(images):
        if i == 0: 
            image = image.convert("RGB")
            image.save("images/start.jpg")
        if i == len(images) - 1:
            image = image.convert("RGB")
            image.save("images/end.jpg")
        image_base64 = PIL_to_base64(image)
        prompt.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
            },
        })
    
    question = """ Given these series of frames, construct a descriptive label which describes the trajectory the robot has taken and where it ends. Keep the description simple and to the point, capturing key landmarks in the trajectory of the form "[instruction verb] (to the/the/into the etc.) [landmark]"
                    Only return the final label."""
    prompt.append({"type": "text", "text": question})

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    label = response.choices[0].message.content
    print(label)
    breakpoint()
    return label

paths = glob.glob(os.path.join(path, "*"))
total_stationary = 0
total_control = 0
skipped_folders = []
waypoint_spacing = []
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
        waypoint_spacing.append(spacing)


# Get the average spacing between waypoints
avg_waypoint_spacing = np.array(waypoint_spacing).mean()
print(f"Average waypoint spacing: {avg_waypoint_spacing:.2f} meters")
print(f"Total number of skipped folders: {len(skipped_folders)}")
print(skipped_folders)



