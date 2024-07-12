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

path = "/hdd/frodo/frodo_dataset"
client = openai.Client(api_key=os.environ["OPENAI_API_KEY"], 
                    organization=os.environ["OPENAI_ORG_ID"])

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
for folder in paths: 
    shutil.rmtree("images", ignore_errors=True)
    os.makedirs(f"images")
    print(folder)
    data_pkl_file = os.path.join(folder, "traj_data.pkl")
    stats_pkl_file = os.path.join(folder, "traj_stats.pkl")
    if not os.path.exists(data_pkl_file) or not os.path.exists(stats_pkl_file):
        print(f"Skipping {folder}")
        skipped_folders.append(folder)
        continue
    stats = np.load(stats_pkl_file, allow_pickle=True)
    total_stationary += stats["stationary_control"]
    total_control += stats["total_control"]

    # # plot the data and images
    # data = np.load(data_pkl_file, allow_pickle=True)
    # image_paths = glob.glob(os.path.join(folder, "*.jpg"))
    # pos = data["pos"]
    # yaw = data["yaw"]
    # fig, ax = plt.subplots(1, 2)
    # frames = []
    # images = []
    # if pos.shape[0] > 100:
    #     pos = pos[:100, :]
    #     yaw = yaw[:100]
    # for idx in range(pos.shape[0]):
    #     print(f"Plotting {idx}/{pos.shape[0]}")
    #     ax[0].plot(pos[:idx, 0], pos[:idx, 1])
    #     for i in range(idx):
    #         ax[0].arrow(pos[i, 0], pos[i, 1], 0.1*np.cos(yaw[i]), 0.1*np.sin(yaw[i]))
    #     ax[0].set_title(f"{folder}")
    #     images.append(Image.open(os.path.join(folder, f"{idx}.jpg")))
    #     ax[1].imshow(images[-1])
    #     plt.savefig(f"images/image_{idx}.png")

    # for i in range(pos.shape[0]):
    #     print(f"Creating gif {i}/{pos.shape[0]}")
    #     frames.append(Image.open(f"images/image_{i}.png"))

    # folder_name = folder.split("/")[-1]
    # frames[0].save(f'trajectory_viz/trajectory_{folder_name}.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

    # # Get a label for the first 30 frames in the trajectory 
    # sampled_frames = frames[:30][::3]
    # label = relabelling_vlm(sampled_frames, client)

print(f"Total stationary control: {total_stationary}")
print(f"Total control: {total_control}")
print(f"Percentage of stationary control: {total_stationary/total_control}")
print("Number of skipped folders: ", len(skipped_folders))
print(skipped_folders)


