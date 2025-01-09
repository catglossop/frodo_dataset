import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from pathlib import Path
import random
from matplotlib import animation
import geomag

timestamps = [float(t) for t in np.arange(1000) * 0.1]
dataset = LeRobotDataset(
    "frodobots_dataset",
    Path("/data/frodobots_export"),
    delta_timestamps={
        "observation.images.front": timestamps,
        "observation.relative_position": timestamps,
        "observation.compass_heading": timestamps,
        "action": timestamps,
        "observation.latitude": timestamps,
        "observation.longitude": timestamps,
    }
)

while True:
    # Generate an animation with matplotlib
    sample = random.choice(dataset)
    num_frames = np.count_nonzero(~sample["action_is_pad"])

    heading_raw = sample["observation.compass_heading"]
    declination = np.array([geomag.declination(lat, lon) for lat, lon in zip(sample["observation.latitude"], sample["observation.longitude"])])
    heading = heading_raw + np.deg2rad(declination)

    compass_diff = np.diff(heading)
    compass_diff = np.where(compass_diff > np.pi, compass_diff - 2 * np.pi, compass_diff)

    # Set up the figure and axis
    fig, axs = plt.subplots(2)
    axs[0].axis('off')  # Turn off axis labels

    # Create the plot
    im = axs[0].imshow(sample["observation.images.front"][0].permute((1, 2, 0)).numpy())
    wy, wx = sample["observation.images.front"][0].shape[1:3]
    compass = axs[0].arrow(100, 100, 80, 0, head_width=10, head_length=10, fc='r', ec='r', lw=5)
    compass_raw = axs[0].arrow(100, 100, 80, 0, head_width=10, head_length=10, fc='r', ec='r', lw=3, alpha=0.5)
    action = axs[0].arrow(wx//2, wy//2, 0, -150, head_width=10, head_length=10, fc='b', ec='b', lw=5)
    timestamp = axs[0].text(wx-20, 20, f"Timestamp: 0.00", fontsize=12, horizontalalignment='right', verticalalignment='top', color='white')

    gps_plot, = axs[1].plot(sample["observation.relative_position"][:, 0], sample["observation.relative_position"][:, 1])
    axs[1].set_xlim(min(sample["observation.relative_position"][:, 0]), max(sample["observation.relative_position"][:, 0]))
    axs[1].axis("equal")

    def animate(i):
        im.set_array(sample["observation.images.front"][i].permute((1, 2, 0)).numpy())
        heading_to_north = np.pi/2 - heading[i]
        heading_to_north_raw = np.pi/2 - heading_raw[i]
        compass.set_data(x=100, y=100, dx=-80*np.sin(heading_to_north), dy=-80*np.cos(heading_to_north))
        compass_raw.set_data(x=100, y=100, dx=-80*np.sin(heading_to_north_raw), dy=-80*np.cos(heading_to_north_raw))
        action.set_data(
            x=wx//2,
            y=wy//2,
            dx=-150*np.clip(sample["action"][i, 1], -1, 1),
            dy=-150*np.clip(sample["action"][i, 0], -1, 1),
        )
        timestamp.set_text(f"Timestamp: {i * 0.1:.2f}")

        gps_plot.set_data(sample["observation.relative_position"][:i, 0], sample["observation.relative_position"][:i, 1])

        return [im, compass, action, timestamp, gps_plot, compass_raw]

    # Create the animation
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=20, blit=True)

    plt.show()

# for _ in range(100):
#     idx = random.randint(0, len(dataset))
#     print(idx)
#     sample = dataset[idx]
    # heading = sample["observation.compass_heading"][sample["action_is_pad"] == 0]
    # position = sample["observation.relative_position"][sample["action_is_pad"] == 0]
    # action = sample["action"][sample["action_is_pad"] == 0]
    #
    # fig, axs = plt.subplots(3)
    #
    # axs[0].plot(position[:, 0], position[:, 1])
    # axs[0].quiver(position[::10, 0], position[::10, 1], np.cos(heading[::10]), np.sin(heading[::10]), np.arange(len(heading[::10])), scale=10)
    # axs[0].axis("equal")
    # axs[1].plot(action)
    # axs[1].plot(heading)
    # plt.show()
