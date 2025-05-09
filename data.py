from dataclasses import dataclass
from typing import Any, Dict,  Callable

from torch.utils.data import Dataset

import numpy as np

import zarr
import os
import torch 
from safetensors import safe_open
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random
import einops 

import logging
import torchvision

from itertools import accumulate
from typing import Iterator

import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
from typing import Union

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training





@dataclass
class TimestampedData:
    data: Any
    timestamps: np.ndarray


TimestampedDataDict = Dict[str, TimestampedData]


# LOADING HELPER FUNCTIONS
# Define Helper Functions 
# Kyle / Noriaki 
def load_frames_zarr(
    dataset: zarr.Array,
    index: int,
    episode_data_index: dict[str, torch.Tensor],
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
) -> dict[torch.Tensor]:
    """
    dataset: zarr.Array,
    index: int, which index to take as our 0 
    episode_data_index: dict[str, torch.Tensor], episode data index (used to calculate beginning and end of episode)
    delta_timestamps: dict[str, list[float]], says which keys to load, along with which timestamps around idx 
    tolerance_s: float,
    """
    # get indices of the frames associated to the episode, and their timestamps
    ep_id = dataset["episode_index"][index].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)

    # load timestamps
    ep_timestamps = torch.from_numpy(dataset["timestamp"][ep_data_id_from:ep_data_id_to]).float()

    # we make the assumption that the timestamps are sorted
    ep_first_ts = ep_timestamps[0]
    ep_last_ts = ep_timestamps[-1]
    current_ts = dataset["timestamp"][index]

    item = {}

    for key, delta_ts in delta_timestamps.items():
        # if it is a video frame
        timestamp_key = f"{key}.timestamp"
        path_key = f"{key}.path"
        is_video = timestamp_key in dataset.keys() and path_key in dataset.keys()

        # get timestamps used as query to retrieve data of previous/future frames
        if delta_ts is None:
            if key in dataset.keys():
                item[key] = torch.from_numpy(np.asarray(dataset[key][index]))
            elif is_video:
                item[key] = [
                    {"path": dataset[path_key][i.item()], "timestamp": dataset[timestamp_key][i.item()]}
                    for i in ep_data_ids
                ]
            else:
                raise ValueError(f"Timestamp key {timestamp_key} not found in dataset")
        else:
            query_ts = current_ts + torch.tensor(delta_ts)

            # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode                               
            dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
            min_, argmin_ = dist.min(1)

            # TODO(rcadene): synchronize timestamps + interpolation if needed
            is_pad = min_ > tolerance_s

            # Skipping check for now 
            # assert ((query_ts[is_pad] < ep_first_ts) | (ep_last_ts < query_ts[is_pad])).all(), (
            #     f"One or several timestamps unexpectedly violate the tolerance ({min_} > {tolerance_s=}) inside episode range."
            #     "This might be due to synchronization issues with timestamps during data collection."
            # )

            # get dataset indices corresponding to frames to be loaded
            data_ids = ep_data_ids[argmin_].numpy()

            if is_video:
                # video mode where frame are expressed as dict of path and timestamp
                item[key] = [
                    {"path": dataset[path_key][i], "timestamp": float(dataset[timestamp_key][i])}
                    for i in data_ids
                ]
            else:
                item[key] = torch.from_numpy(dataset[key][data_ids])

            item[f"{key}_is_pad"] = is_pad

    return item

# from LeRobot at some point 

def decode_video_frames_torchvision(
    video_path: str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    # assert is_within_tol.all(), (
    #     f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
    #     "It means that the closest frame that can be loaded from the video is too far away in time."
    #     "This might be due to synchronization issues with timestamps during data collection."
    #     "To be safe, we advise to ignore this item during training."
    #     f"\nqueried timestamps: {query_ts}"
    #     f"\nloaded timestamps: {loaded_ts}"
    #     f"\nvideo: {video_path}"
    #     f"\nbackend: {backend}"
    # )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames

def load_from_videos(
    item: dict[str, torch.Tensor],
    video_frame_keys: list[str],
    videos_dir: Path,
    tolerance_s: float,
    backend: str = "pyav",
):
    """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
    in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a Segmentation Fault.
    This probably happens because a memory reference to the video loader is created in the main process and a
    subprocess fails to access it.
    """
    # since video path already contains "videos" (e.g. videos_dir="data/videos", path="videos/episode_0.mp4")
    data_dir = videos_dir.parent

    for key in video_frame_keys:
        if isinstance(item[key], list):
            # load multiple frames at once (expected when delta_timestamps is not None)
            timestamps = [frame["timestamp"] for frame in item[key]]
            paths = [frame["path"] for frame in item[key]]
            if len(set(paths)) > 1:
                raise NotImplementedError("All video paths are expected to be the same for now.")
            video_path = data_dir / paths[0]

            frames = decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
            item[key] = frames
        else:
            # load one frame
            timestamps = [item[key]["timestamp"]]
            video_path = data_dir / item[key]["path"]

            frames = decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
            item[key] = frames[0]

    return item

# VINT DATASET 


def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {
        "image": ".jpg",
        # add more data types here
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

def trans_mat(pos: float | np.ndarray | torch.Tensor, yaw: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), pos[0]],
                [torch.sin(yaw), torch.cos(yaw), pos[1]],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ],
        )
    else:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), pos[0]],
                [np.sin(yaw), np.cos(yaw), pos[1]],
                [0.0, 0.0, 1.0],
            ],
        )



def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if isinstance(positions, torch.Tensor):
        rotmat = torch.from_numpy(rotmat)

    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError
    
    return (positions - curr_pos) @ rotmat

def to_local_coords_yaw(
    positions: np.ndarray | torch.Tensor, curr_pos: np.ndarray | torch.Tensor, curr_yaw: float | np.ndarray | torch.Tensor,  goal_yaw: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    cur_mat = trans_mat(curr_pos, curr_yaw)
    goal_mat = trans_mat(positions[0], goal_yaw)    
    cur_mat_inv = torch.linalg.inv(cur_mat)
    relative_mat = torch.matmul(cur_mat_inv, goal_mat)

    return relative_mat

def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate deltas between waypoints

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: deltas
    """
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    return viz_img, transf_img


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img


def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    Load an image from a path and transform it
    Args:
        path (str): path to the image
        image_resize_size (Tuple[int, int]): size to resize the image to
    Returns:
        torch.Tensor: resized image as tensor
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    return resize_and_aspect_crop(Image.open(path), image_resize_size)    

# Reward Computation 
def compute_rewards(
    curr_pos, 
    next_pos,
    goal_pos,
    discount,
    waypoint_spacing,
    will_reach,
):
    
    distance_to_goal = torch.norm(curr_pos - goal_pos, dim=-1).float()
    next_distance_to_goal = torch.norm(next_pos - goal_pos, dim=-1).float()
    
    reward = (distance_to_goal - next_distance_to_goal) / waypoint_spacing 

    if distance_to_goal == 0:
        reward = torch.tensor(1 / (1 - discount)) 

    if will_reach:
        mc_returns = torch.tensor(0.5 / (1 - discount))
    else:
        mc_returns = 0
    
    return reward, mc_returns

def compute_rewards_negatives(did_crash, crash_penalty, discount):
    reward = crash_penalty * did_crash.float()
    mc_returns = torch.zeros_like(did_crash, dtype=torch.float32)
    return reward, mc_returns


class ViNT_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        waypoint_spacing: int,
        metric_waypoint_spacing: float,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
        discount: float = 0.95,
    ):
        """
        Main ViNT dataset class

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names in the dataset split that are each seperated by a newline
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            negative_mining (bool): Whether to use negative mining from the ViNG paper (Shah et al.) (https://arxiv.org/abs/2012.09812)
            len_traj_pred (int): Length of trajectory of waypoints to predict if this is an action dataset
            learn_angle (bool): Whether to learn the yaw of the robot at each predicted waypoint if this is an action dataset
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
            goals_per_obs (int): Number of goals to sample per observation
            normalize (bool): Whether to normalize the distances or actions
            goal_type (str): What data type to use for the goal. The only one supported is "image" for now.
        """

        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.waypoint_spacing = waypoint_spacing
        self.metric_waypoint_spacing =  metric_waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, self.waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining
        if self.negative_mining:
            self.distance_categories.append(-1)
        self.len_traj_pred = len_traj_pred
        self.learn_angle = learn_angle
        self.discount = discount

        self.min_action_distance = min_action_distance
        self.max_action_distance = max_action_distance

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.goals_per_obs = goals_per_obs
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type

        # # load data/data_config.yaml
        # with open(
        #     os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        # ) as f:
        #     all_data_config = yaml.safe_load(f)
        # assert (
        #     self.dataset_name in all_data_config
        # ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        # dataset_names = list(all_data_config.keys())
        # dataset_names.sort()
        # # use this index to retrieve the dataset name from the data_config.yaml
        # self.dataset_index = dataset_names.index(self.dataset_name)
        # self.data_config = all_data_config[self.dataset_name]

        
        self.trajectory_cache = {}
        self._load_index()
        self._build_caches()
        
        if self.learn_angle:
            self.num_action_params = 3
        else:
            self.num_action_params = 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        """
        Build a cache of images for faster loading using LMDB
        """
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )

        # Load all the trajectories into memory. These should already be loaded, but just in case.
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        """
        If the cache file doesn't exist, create it by iterating through the dataset and writing each image to the cache
        """
        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())

        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)
                samples_index.append((traj_name, curr_time, max_goal_distance))

        return samples_index, goals_index

    def _sample_goal(self, trajectory_name, curr_time, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(0, max_goal_dist + 1)
        if goal_offset == 0:
            trajectory_name, goal_time = self._sample_negative()
            return trajectory_name, goal_time, True
        else:
            goal_time = curr_time + int(goal_offset * self.waypoint_spacing)
            return trajectory_name, goal_time, False

    def _sample_negative(self):
        """
        Sample a goal from a (likely) different trajectory.
        """
        return self.goals_index[np.random.randint(0, len(self.goals_index))]

    def _load_index(self) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            # load the index_to_data if it already exists (to save time)
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            # if the index_to_data file doesn't exist, create it
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            return img_path_to_data(image_bytes, self.image_size)
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]
        goal_yaw = traj_data["yaw"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(np.array([goal_yaw]).shape) == 2:
            goal_yaw = goal_yaw[0]
        #print('goal_yaw', goal_yaw)
                    
        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            const_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        assert yaw.shape == (self.len_traj_pred + 1,), f"{yaw.shape} and {(self.len_traj_pred + 1,)} should be equal"
        assert positions.shape == (self.len_traj_pred + 1, 2), f"{positions.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw_loc = goal_yaw - yaw[0]
        
        assert waypoints.shape == (self.len_traj_pred + 1, 2), f"{waypoints.shape} and {(self.len_traj_pred + 1, 2)} should be equal"

        if self.learn_angle:
            yaw = yaw[1:] - yaw[0]
            actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        else:
            actions = waypoints[1:]
        
        if self.normalize:
            actions[:, :2] /= self.metric_waypoint_spacing * self.waypoint_spacing
            goal_pos /= self.metric_waypoint_spacing * self.waypoint_spacing

        assert actions.shape == (self.len_traj_pred, self.num_action_params), f"{actions.shape} and {(self.len_traj_pred, self.num_action_params)} should be equal"

        return actions, goal_pos, goal_yaw_loc
    
    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            Tuple of tensors containing the context, observation, goal, transformed context, transformed observation, transformed goal, distance label, and action label
                obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the robot's observation
                goal_image (torch.Tensor): tensor of shape [3, H, W] containing the subgoal image 
                dist_label (torch.Tensor): tensor of shape (1,) containing the distance labels from the observation to the goal
                action_label (torch.Tensor): tensor of shape (5, 2) or (5, 4) (if training with angle) containing the action labels from the observation to the goal
                which_dataset (torch.Tensor): index of the datapoint in the dataset [for identifying the dataset for visualization when using multiple datasets]
        """
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        # Load images
        context = []
        if self.context_type == "temporal":
            # sample the last self.context_size times from interval [0, curr_time)
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1 + self.waypoint_spacing, # also get the NEXT obs 
                    self.waypoint_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context[:-1]
        ])

        next_obs_image = torch.cat([
            self._load_image(f, t) for f, t in context[1:]
        ])

        # Load goal image
        goal_image = self._load_image(f_goal, goal_time)

        # Load other trajectory data
        curr_traj_data = self._get_trajectory(f_curr)
        curr_traj_len = len(curr_traj_data["position"])
        assert curr_time < curr_traj_len, f"{curr_time} and {curr_traj_len}"

        goal_traj_data = self._get_trajectory(f_goal)
        goal_traj_len = len(goal_traj_data["position"])
        assert goal_time < goal_traj_len, f"{goal_time} an {goal_traj_len}"

        # Compute actions
        actions, goal_pos, goal_yaw = self._compute_actions(curr_traj_data, curr_time, goal_time)
        
        # Compute distances
        if goal_is_negative:
            distance = self.max_dist_cat
        else:
            distance = (goal_time - curr_time) // self.waypoint_spacing
            assert (goal_time - curr_time) % self.waypoint_spacing == 0, f"{goal_time} and {curr_time} should be separated by an integer multiple of {self.waypoint_spacing}"
        
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)
        
        action_mask = (
            (distance < self.max_action_distance) and
            (distance > self.min_action_distance) and
            (not goal_is_negative)
        )

        # Get Rewards 
        if goal_is_negative:
            reward = 0 
            mc_returns = 0 # won't get any progress 
        else:
            curr_pos =  curr_traj_data["position"][curr_time]
            next_pos =  curr_traj_data["position"][curr_time + 1] if curr_time < curr_traj_len else curr_pos
            goal_pos_raw =  goal_traj_data["position"][goal_time]
            
            reward, mc_returns = compute_rewards(curr_pos = torch.tensor(curr_pos[:2]),
                                    next_pos = torch.tensor(next_pos[:2]),
                                    goal_pos = torch.tensor(goal_pos_raw[:2]),
                                    discount = self.discount,
                                    waypoint_spacing = self.waypoint_spacing,
                                    will_reach = True)

        
        return {
            "obs_imgs": torch.as_tensor(obs_image, dtype=torch.float32),
            "next_obs_imgs": torch.as_tensor(next_obs_image, dtype=torch.float32),
            "goal_img": torch.as_tensor(goal_image, dtype=torch.float32),
            "actions": actions_torch,
            "distance":  torch.as_tensor(distance, dtype=torch.int64),
            "goal_pos": torch.as_tensor(goal_pos, dtype=torch.float32),
            "goal_yaw": torch.as_tensor(goal_yaw, dtype=torch.float32),
            "reward": torch.as_tensor(reward,  dtype=torch.float32),
            "mc_returns": torch.as_tensor(mc_returns, dtype=torch.float32),
            # torch.as_tensor(self.dataset_index, dtype=torch.int64),
            # torch.as_tensor(action_mask, dtype=torch.float32),
        }
    

# FRODOBOTS DATASET 
class Frodo_Dataset(Dataset):
    def __init__(
            self, 
            data_dir: str,
            split: str,
            action_horizon: int = 8,
            action_spacing: int = 1,
            goal_horizon: int = 20,
            context_spacing: int = 1,
            context_size: int = 5,
            dataset_framerate: int = 10,
            image_size: Tuple[int, int] = (120, 160),
            image_transforms: Callable | None = None,
            discount: float=0.95,
    ):
        # Set up dataset 
        self.data_dir = data_dir
        self.videos_dir = Path(self.data_dir) / "videos"
        self.video_backend = "pyav"

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.image_transforms = image_transforms
        self.discount = discount

        self.split = split 
        self.tolerance_s = 1e-4

        img_history_spacing = [i * context_spacing * self.dt for i in range(-context_size, 1 + 1)]  # get next image obs too 
        action_future_spacing = [i * action_spacing * self.dt for i in range(action_horizon)]
        self.delta_timestamps={
                "observation.filtered_position": [0.0],
                "observation.relative_position": [0.0],
                "observation.filtered_heading": [0.0],
                "observation.images.front": img_history_spacing,
                "action": action_future_spacing,                
            }

        # Load Dataset Cache
        self.dataset_cache = zarr.load(Path(data_dir) / "dataset_cache.zarr")
        self.dataset_cache = {
            k: np.asarray(v) for k, v in self.dataset_cache.items()
        }
        self.total_length = self.dataset_cache["action"].shape[0]
        print("Dataset Cache Loaded")

        # Compute Episode Data Index 
        self.episode_data_index = self.get_episode_data_index(self.dataset_cache["episode_index"])
        print("Episode Index Computed")
        

    def get_episode_data_index(self, episode_index: list[int]) -> dict[str, torch.Tensor]:
        episode_lengths = []
        current_episode = episode_index[0]
        count = 0

        # Compute Episode Lengths 
        for ep in episode_index:
            if ep == current_episode:
                count += 1
            else:
                episode_lengths.append(count)
                current_episode = ep
                count = 1
        episode_lengths.append(count)  # Append the last episode's length

        # Compute Cumulative Lengths
        cumulative_lengths = list(accumulate(episode_lengths))
        return {
            "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
            "to": torch.LongTensor(cumulative_lengths),
        }


    def _image_transforms(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        original_height, original_width = img.shape[-2:]
        target_aspect = 4 / 3
        img = TF.resize(img, self.image_size)
        if flip:
            img = torch.flip(img, dims=(-1,))
        return img

    def _image_transforms_depth(self, img: torch.Tensor, flip) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): image tensor
        Returns:
            torch.Tensor: transformed image
        """

        img_rsize = TF.resize(img, (128, 416)) #
        if flip:
            img_rsize = torch.flip(img_rsize, dims=(-1,))
            img = torch.flip(img, dims=(-1,))                      
        return img_rsize, img

    def viz_rollout(self, actions: torch.Tensor) -> torch.Tensor:
        positions = torch.zeros_like(actions)
        heading = torch.zeros_like(actions[..., 0, 0])

        for i in range(1, actions.shape[-2]):
            vel = actions[..., i - 1, 0]
            angvel = actions[..., i - 1, 1]

            direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
            positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
            heading = heading + angvel * self.dt

        return positions
    
    def __len__(self):
        return self.total_length // 4

    def __getitem__(self, idx):
            # Sample a goal timestamp        
            ep_id = self.dataset_cache["episode_index"][idx].item()
            episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
            goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
            goal_delta_ts = goal_dist * self.dt * self.action_spacing

            # Add the goal to the list of delta timestamps
            delta_timestamps = self.delta_timestamps 
            
            # get information for goal, as well 
            delta_timestamps = {
                k: list(v) + [goal_delta_ts] if v is not None else None for k, v in delta_timestamps.items()
            }
            # Getting position / heading information for up to goal_horizon actions 
            delta_timestamps["observation.filtered_position"] = delta_timestamps["observation.filtered_heading"] = delta_timestamps["observation.relative_position"] = [
                i * self.dt * self.action_spacing
                for i in range(self.goal_horizon)
            ] + [goal_delta_ts]

            # Load information 
            item = load_frames_zarr(
                self.dataset_cache,
                idx,
                self.episode_data_index,
                delta_timestamps,
                self.tolerance_s,
            )

            flip_tf = random.random() > 0.5 
            all_imgs = self._image_transforms(load_from_videos(
                {"observation.images.front": item["observation.images.front"]},
                ["observation.images.front"],
                self.videos_dir,
                self.tolerance_s,
                self.video_backend,
            )["observation.images.front"], flip_tf)

            image_obs = all_imgs[:-2]
            next_image_obs = all_imgs[1:-1]
            image_goal = all_imgs[-1]

            # image_obs = 
            # image_obs = self._image_transforms(load_from_videos(
            #     {"observation.images.front": item["observation.images.front"][:-2]},
            #     ["observation.images.front"],
            #     self.videos_dir,
            #     self.tolerance_s,
            #     self.video_backend,
            # )["observation.images.front"], flip_tf)

            # next_image_obs = self._image_transforms(load_from_videos(
            #     {"observation.images.front": item["observation.images.front"][1:-1]},
            #     ["observation.images.front"],
            #     self.videos_dir,
            #     self.tolerance_s,
            #     self.video_backend,
            # )["observation.images.front"], flip_tf)

            # image_goal = self._image_transforms(load_from_videos(
            #     {"observation.images.front": item["observation.images.front"][-1]},
            #     ["observation.images.front"],
            #     self.videos_dir,
            #     self.tolerance_s,
            #     self.video_backend,
            # )["observation.images.front"], flip_tf)

            unnorm_position = item["observation.filtered_position"][:-1]
            current_heading = item["observation.filtered_heading"][0]
            goal_heading = item["observation.filtered_heading"][-1]   
            heading = item["observation.filtered_heading"][:-1]        
            

            next_pos_relative = to_local_coords(item["observation.filtered_position"][1, None], unnorm_position[0], current_heading)[0]
            goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]
            relative_mat = to_local_coords_yaw(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading, goal_heading)
            
            if flip_tf:
                goal_pos_relative[1] *= -1
                goal_heading *= -1
                relative_mat[0,1] *= -1
                relative_mat[1,0] *= -1
                relative_mat[1,2] *= -1
            
            if flip_tf:                  
                future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)    
                future_positions_unfiltered[:,1] *= -1   

                direction = torch.stack([torch.cos(-heading), torch.sin(-heading)], dim=-1)
                action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(-heading))), -1, 1) * 5           
                unnorm_position[:,1] *= -1            
                action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
                action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing               
                
            else:
                future_positions_unfiltered = to_local_coords(item["observation.relative_position"][:-1], unnorm_position[0], current_heading)

                direction = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
                action_steer = torch.clip(torch.from_numpy(np.diff(np.unwrap(heading))), -1, 1) * 5
                action_forward = torch.sum(torch.diff(unnorm_position, dim=0) * direction[:-1], dim=-1)
                action = torch.stack([action_forward[:self.action_horizon], action_steer[:self.action_horizon]], dim=-1) / self.dt / self.action_spacing        

            which_dataset = 0

            image_flattened = einops.rearrange(image_obs, "... t c h w -> ... (t c) h w")
            next_image_flattened = einops.rearrange(next_image_obs, "... t c h w -> ... (t c) h w")

            curr_pos = torch.zeros(2,)
            next_pos = next_pos_relative
            goal_pos = goal_pos_relative
            
            reward, mc_returns = compute_rewards(curr_pos = curr_pos,
                                    next_pos = next_pos,
                                    goal_pos = goal_pos,
                                    discount = self.discount,
                                    waypoint_spacing = self.action_spacing,
                                    will_reach = True)

            return {
                "obs_imgs": torch.as_tensor(image_flattened, dtype=torch.float32),
                "next_obs_imgs": torch.as_tensor(next_image_flattened, dtype = torch.float32),
                "goal_img": torch.as_tensor(image_goal, dtype=torch.float32),
                "actions": action,
                "distance":  torch.as_tensor(goal_dist, dtype=torch.int64),
                "goal_pos": torch.as_tensor(goal_pos_relative, dtype=torch.float32),
                "goal_yaw": torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),
                "reward": torch.as_tensor(reward,  dtype=torch.float32),
                "mc_returns": torch.as_tensor(mc_returns, dtype=torch.float32),
                # torch.as_tensor(self.dataset_index, dtype=torch.int64),
                # torch.as_tensor(action_mask, dtype=torch.float32),
            }


            # return (
            #     torch.as_tensor(image_flattened, dtype=torch.float32),
            #     torch.as_tensor(image_goal, dtype=torch.float32),
            #     # torch.as_tensor(image_current, dtype=torch.float32),            
            #     torch.as_tensor(action, dtype=torch.float32),
            #     # torch.as_tensor(goal_dist/3.0, dtype=torch.int64),
            #     torch.as_tensor(goal_pos_relative, dtype=torch.float32),
            #     # torch.as_tensor(relative_mat, dtype=torch.float32),  
            #     torch.as_tensor(goal_heading - current_heading, dtype=torch.float32),                        
            #     # torch.as_tensor(which_dataset, dtype=torch.int64),
            #     # torch.as_tensor(future_positions_unfiltered, dtype=torch.float32),
            #     # torch.as_tensor(idx, dtype=torch.float32),
            #     # torch.as_tensor(image_raw, dtype=torch.float32),     
            # )  

class DatasetWeightingSampler(torch.utils.data.Sampler):

    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, dataset_weights, samples):
        assert np.isclose(sum(dataset_weights), 1.0), "Dataset weights must add up to 1"

        self.dataset_weights = dataset_weights
        self.cumulative_weights = list(accumulate(self.dataset_weights))

        self.dataset_lengths = np.array([len(d) for d in datasets])
        
        self.cumulative_sizes = np.array(list(accumulate(self.dataset_lengths)))
        self.start_sizes = np.insert(self.cumulative_sizes, 0, 0)[:-1]

        self.samples = samples


    def __iter__(self) -> Iterator:
        new_idxs = []

        rand_nums = np.random.rand(self.samples)

        datasets_picked = np.searchsorted(self.cumulative_weights, rand_nums)  # tells us which index it would need to be inserted at, which maps it to the appropriate dataset 
        indices_picked = self.start_sizes[datasets_picked] + np.random.randint(0, self.dataset_lengths[datasets_picked], size=len(datasets_picked))

        return iter(indices_picked)



class EpisodeSampler_IL(torch.utils.data.Sampler):
    
    def __init__(self, dataset, episode_index_from: int, episode_index_to: int, data_split_type: str):
        """
        Setting the specific episodes to sample from is how we handle train / test split. 
        """
        self.dataset = dataset

        from_idx = dataset.episode_data_index["from"][episode_index_from].item()
        to_idx = dataset.episode_data_index["to"][episode_index_to].item()
        self.frame_ids_range = range(from_idx, to_idx)
        print("from_idx", from_idx, "to_idx", to_idx)  

        self.yaw_list = dataset.dataset_cache["action"][:, 0] # don't trim because still indexed by original value                   
                                      
    def __iter__(self) -> Iterator:   
        indices_new = []
        yawangle_list = []
        
        for idx in tqdm(self.frame_ids_range):
            
            thres_rate = random.random()
            if self.yaw_list[idx] % (2*3.14) > 3.14:
                ang_yaw = self.yaw_list[idx] % (2*3.14) - 2.0*3.14
            else:
                ang_yaw = self.yaw_list[idx] % (2*3.14)   
            
            # 80% of the time, find a data point where the action has an angle of greater than 0.4 radians (23 degrees) and less than 2 radians (115 degrees)
            if thres_rate < 0.8:
                while not (abs(ang_yaw) > 0.4 and abs(ang_yaw) < 2.0):                  
                    idx = random.choice(self.frame_ids_range)            
                    if self.yaw_list[idx] % (2*3.14) > 3.14:
                        ang_yaw = self.yaw_list[idx] % (2*3.14) - 2.0*3.14
                    else:
                        ang_yaw = self.yaw_list[idx] % (2*3.14)             
            
            indices_new.append(idx)           

        indices_new_random = random.sample(indices_new, len(indices_new)) # shuffle 
        return iter(indices_new_random)

    def __len__(self) -> int:
        return len(self.frame_ids_range)   