from collections import defaultdict
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import os
import glob
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    get_default_encoding,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame
import m3u8
from utm import from_latlon
import torch
from pathlib import Path
import time
import ffmpeg
import geomag
import ast
import re
import json
import subprocess
import multiprocessing
import zarr

from datasets import Dataset, Features, Sequence, Value
import tqdm

from dataclasses import dataclass


from filtering import filter_data_postprocess, filter_data_preprocess
from interpolation_utils import (
    TimestampedData,
    TimestampedDataDict,
    cut_interp_clips,
)


def get_single_file(ride_dir: Path, glob_pattern: str):
    """
    Get a single file for the specified ride directory and glob pattern.

    Throws an error if no files are found or if multiple files are found.
    """
    files = list(ride_dir.glob(glob_pattern))
    if len(files) == 0:
        raise ValueError(f"No files found for {glob_pattern}")
    elif len(files) > 1:
        raise ValueError(f"Multiple files found for {glob_pattern}: {files}")
    return files[0]


def get_raw_ride_dfs(ride_dir: Path):
    """
    Load the raw CSV data (everything except video) from a ride directory.
    """
    csv_file_descriptors = [
        "control_data",
        "imu_data",
        "gps_data",
        "front_camera_timestamps",
        "rear_camera_timestamps",
        "speaker_audio_timestamps",
    ]

    return {
        k: pd.read_csv(get_single_file(ride_dir, f"{k}*.csv"))
        for k in csv_file_descriptors
    }


def get_frame_timestamps(video_path: str):
    # FFmpeg command to extract frame timestamps
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_entries', 'frame=pkt_pts_time',
        '-select_streams', 'v:0',
        video_path
    ]

    # Run the FFmpeg command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise ValueError(f"FFmpeg command failed with code {result.returncode} when executing: ```{cmd}```: {result.stderr}")

    # Parse the JSON output
    try:
        frames_data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing FFmpeg output: {result.stdout}")

    # Extract timestamps
    timestamps = [float(frame['pkt_pts_time']) for frame in frames_data['frames']]
    return timestamps

# def count_video_frames(video_path: str) -> int:
#     video_streams = [s for s in ffmpeg.probe(video_path)["streams"] if s["codec_type"] == "video"]
#     assert len(video_streams) == 1
#     return int(video_streams[0]["nb_frames"])


def run_commands(command_iterator, cores, num_items=None):
    processes = []

    # Start initial processes, one per core
    for core in cores:
        try:
            commands, data = next(command_iterator)
        except StopIteration:
            print("No more commands to run")
            break  # No more commands to run

        cmd = commands.pop(0)
        full_cmd = ['taskset', '-c', str(core)] + cmd
        proc = subprocess.Popen(full_cmd)
        processes.append({'process': proc, 'core': core, 'cmd': cmd, 'data': data, 'remaining_commands': commands})

    successful_data = []

    # Continuously check for finished processes and start new ones
    with tqdm.tqdm(total=num_items, dynamic_ncols=True) as pbar:
        while processes:
            for p in processes[:]:  # Iterate over a copy of the list
                proc = p['process']
                if proc.poll() is not None:  # Process has finished
                    core = p['core']

                    # Check the return code
                    if proc.returncode != 0:
                        print(f"Command {' '.join(p['cmd'])} failed with code {proc.returncode}")
                        p['cmd'] = None
                    elif not p['remaining_commands']:
                        successful_data.append(p['data'])
                        p['cmd'] = None
                    else:
                        p['cmd'] = p['remaining_commands'].pop(0)

                    if p['cmd'] is None:
                        pbar.update(1)
                        try:
                            commands, data = next(command_iterator)
                            p['cmd'] = commands.pop(0)
                            p['data'] = data
                            p['remaining_commands'] = commands
                        except StopIteration:
                            print(f"No more commands to run on core {core}")
                            processes.remove(p)
                            continue

                    if p['cmd'] is not None:
                        full_cmd = ['taskset', '-c', str(core)] + p['cmd']
                        p['process'] = subprocess.Popen(full_cmd)
            time.sleep(0.1)  # Sleep briefly to prevent CPU overuse

    return successful_data


def concat_videos_commands(all_rides, output_dir: Path):
    """
    Use ffmpeg to concatenate the video segments for a ride into a single video file.

    Returns a dictionary mapping camera names to the output video file names.
    """

    camera_specific_ffmpeg_args = {
        "front": {
            "filter:v": "scale=540:360", # "scale=224:128",
        },
        "rear": {
            "filter:v": "scale=192:128", # "scale=96:64",
        },
    }

    for ride_dir in all_rides:
        commands = []
        out_files = {}
        ride_id = get_ride_id(ride_dir)
        for camera, index in [("front", 1000), ("rear", 1001)]:
            # Open the playlist file
            try:
                playlist_filename = get_single_file(ride_dir, f"recordings/*{index}__uid_e_video.m3u8")
                playlist = m3u8.load(str(playlist_filename))
                files = [
                    str(ride_dir / "recordings" / segment.uri) for segment in playlist.segments
                ]
            except ValueError as e:
                print(f"Error processing {ride_dir}: {e}")
                continue

            # Concatenate the video with ffmpeg, using h264 codec
            basename = f"ride_{ride_id}_{camera}_camera.mp4"
            out_filename = output_dir / basename
            out_files[camera] = os.path.join("videos", basename)

            # Run ffmpeg to concatenate the video segments
            ffmpeg_args = {
                "i": f"concat:{'|'.join(files)}",
                "vsync": 0,
                "vcodec": "libx264",
                "pix_fmt": "yuv420p",
                "g": 2,
                "crf": 30,
                **camera_specific_ffmpeg_args[camera],
            }
            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "panic",
                "-hide_banner",
            ]
            for k, v in ffmpeg_args.items():
                command.append(f"-{k}")
                command.append(str(v))
            command.append(str(out_filename))
            commands.append(command)

        if len(commands) == 2:
            yield commands, out_files

def concat_videos(ride_dir: Path, output_dir: Path, prefix="", run_ffmpeg=True, restrict_cores=None):
    """
    Use ffmpeg to concatenate the video segments for a ride into a single video file.

    Returns a dictionary mapping camera names to the output video file names.
    """

    if prefix:
        prefix += "_"

    out_files = {}

    camera_specific_ffmpeg_args = {
        "front": {
            "filter:v": "scale=768:432",
        },
        "rear": {
            "filter:v": "scale=360:240",
        },
    }

    for camera, index in [("front", 1000), ("rear", 1001)]:
        # Open the playlist file
        playlist_filename = get_single_file(ride_dir, f"recordings/*{index}__uid_e_video.m3u8")
        playlist = m3u8.load(str(playlist_filename))
        files = [
            str(ride_dir / "recordings" / segment.uri) for segment in playlist.segments
        ]

        # Concatenate the video with ffmpeg, using h264 codec
        basename = f"{prefix}{camera}_camera.mp4"
        out_filename = output_dir / basename
        out_files[camera] = os.path.join("videos", basename)

        if run_ffmpeg:
            # Run ffmpeg to concatenate the video segments
            ffmpeg_args = {
                "i": f"concat:\"{'|'.join(files)}\"",
                "vsync": 0,
                "vcodec": "libx264",
                "pix_fmt": "yuv420p",
                "g": 2,
                "crf": 30,
                **camera_specific_ffmpeg_args[camera],
            }
            ffmpeg_command = " ".join([
                "ffmpeg -y -loglevel panic -hide_banner ",
                ' '.join(f'-{k} {v}' for k, v in ffmpeg_args.items()),
                str(out_filename),
            ])
            if restrict_cores is not None:
                cores_str = ",".join(str(c) for c in restrict_cores)
                ffmpeg_command = f"taskset -c {cores_str} {ffmpeg_command}"
            code = os.system(ffmpeg_command)

            if code != 0:
                raise ValueError(
                    f"ffmpeg command failed with code {code} when executing: ```{ffmpeg_command}```"
                )

    return out_files

def fast_parse_string_to_numpy(input_string, shape):
    # Remove brackets and split by commas
    cleaned = re.sub(r'[\[\]\s\'\"]', '', input_string)

    # Convert to numpy array and reshape
    return np.fromstring(cleaned, sep=',', dtype=np.float64).reshape(shape)

def get_imu_data(dfs: Dict[str, pd.DataFrame]) -> TimestampedDataDict:
    """
    Read all data and timestamps from the IMU CSV file
    """

    def parse_imu_data_entry(df: pd.DataFrame, column_name: str, dtype=np.float32):
        """
        IMU data is stored as a string in the CSV file. This function parses it into a numpy array.
        """
        np_arr = np.array([fast_parse_string_to_numpy(e, (-1, 4)) for e in df[column_name]])
        np_arr = np.reshape(np_arr, (-1, np_arr.shape[-1]))
        return np_arr[..., :-1].astype(dtype), np_arr[..., -1].astype(np.float64)

    accel, accel_timestamp = parse_imu_data_entry(dfs["imu_data"], "accelerometer")
    compass, compass_timestamp = parse_imu_data_entry(dfs["imu_data"], "compass")
    gyroscope, gyroscope_timestamp = parse_imu_data_entry(dfs["imu_data"], "gyroscope")

    return {
        "observation.accelerometer": TimestampedData(
            data=accel, timestamps=accel_timestamp
        ),
        "observation.magnetometer": TimestampedData(
            data=compass, timestamps=compass_timestamp
        ),
        "observation.gyroscope": TimestampedData(
            data=gyroscope, timestamps=gyroscope_timestamp
        ),
    }


def get_gps_data(dfs: Dict[str, pd.DataFrame]) -> TimestampedDataDict:
    """
    Parse all data and timestamps from the GPS CSV file.
    """
    lat: np.ndarray = dfs["gps_data"]["latitude"].values
    lon: np.ndarray = dfs["gps_data"]["longitude"].values
    easting, northing, zone_number, zone_letter = from_latlon(lat, lon)
    zone_number = np.array([int(zone_number)] * len(lat))
    zone_letter_id = np.array([ord(str(zone_letter))] * len(lat))

    utm_position = np.stack([easting, northing], axis=1)

    timestamp = dfs["gps_data"]["timestamp"].values.astype(np.float64) / 1000

    return {
        "observation.latitude": TimestampedData(data=lat, timestamps=timestamp),
        "observation.longitude": TimestampedData(data=lon, timestamps=timestamp),
        "observation.utm_position": TimestampedData(
            data=utm_position, timestamps=timestamp
        ),
        "observation.utm_zone_number": TimestampedData(
            data=zone_number, timestamps=timestamp
        ),
        "observation.utm_zone_letter": TimestampedData(
            data=zone_letter_id, timestamps=timestamp
        ),
    }


def get_control_data(dfs: Dict[str, pd.DataFrame]) -> TimestampedDataDict:
    """
    Parse all data and timestamps from the control CSV file.
    """
    control = dfs["control_data"]
    wheel_rpm = np.stack([control[f"rpm_{i}"] for i in range(1, 5)], axis=1)
    action = np.stack([control["linear"], control["angular"]], axis=1)

    return {
        "action": TimestampedData(data=action, timestamps=control["timestamp"].values),
        "observation.wheel_rpm": TimestampedData(
            data=wheel_rpm, timestamps=control["timestamp"].values
        ),
    }


def get_lowdim_data(dfs: Dict[str, pd.DataFrame]) -> TimestampedDataDict:
    """
    Read all low-dimensional data from the ride directory.
    """
    return {
        **get_control_data(dfs),
        **get_imu_data(dfs),
        **get_gps_data(dfs),
    }


def get_video_data(
    ride_dir: Path, video_output_dir: Path, dfs: Dict[str, pd.DataFrame], ride_id: str
) -> TimestampedDataDict:
    """
    Encode the video files and gather the correct timestamps for each video file.
    """
    video_files = concat_videos(ride_dir, video_output_dir, prefix=f"ride_{ride_id}", run_ffmpeg=False)

    out_data = {}

    for camera_name, cam_video in video_files.items():
        cam_file = str(video_output_dir.parent / cam_video)
        # num_frames = count_video_frames(cam_file)
        video_frame_timestamps = get_frame_timestamps(cam_file)
        num_frames = len(video_frame_timestamps)
        num_timestamps = len(dfs[f"{camera_name}_camera_timestamps"]["timestamp"].values)

        if num_frames != num_timestamps:
            raise ValueError(
                f"Number of frames ({num_frames}) does not match number of timestamps ({num_timestamps}) in {cam_video}"
            )

        timestamps = dfs[f"{camera_name}_camera_timestamps"]["timestamp"].values
        out_data[f"observation.images.{camera_name}"] = TimestampedData(
            data=[{"path": cam_video, "timestamp": t} for t in video_frame_timestamps],
            timestamps=timestamps,
        )

    return out_data


def add_synthesized_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add synthesized data (compass heading, etc.)
    """
    raw_compass_heading = np.arctan2(
        -data["observation.magnetometer"][:, 2],
        data["observation.magnetometer"][:, 1],
    )
    declination = [geomag.declination(lat, lon) for lat, lon in zip(data["observation.latitude"], data["observation.longitude"])]
    data["observation.compass_heading"] = raw_compass_heading + np.deg2rad(declination)
    data["observation.relative_position"] = data["observation.utm_position"] - data["observation.utm_position"][0]
    return data


def clip_to_torch(clip: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a clip to torch tensors
    """
    return {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in clip.items()
    }


def get_ride_id(ride_dir: Path) -> str:
    components = ride_dir.name.split("_")
    ride_id = components[-2]
    timestamp = components[-1]
    return f"{ride_id}_{timestamp}"


def _mp_concat_video(args):
    ride_dir, output_dir = args
    try:
        ride_id = get_ride_id(ride_dir)
        concat_videos(ride_dir, output_dir, f"ride_{ride_id}")
    except Exception as e:
        print(f"Error processing {ride_dir}: {e}")


def _mp_get_ride_data(args):
    ride_dir, videos_dir = args
    ride_id = get_ride_id(ride_dir)
    try:
        dfs = get_raw_ride_dfs(ride_dir)
        lowdim_data = get_lowdim_data(dfs)
        video_data = get_video_data(ride_dir, videos_dir, dfs, ride_id)
        all_data = {**lowdim_data, **video_data}
        return (ride_id, all_data)
    except Exception as e:
        print(f"Error processing {ride_dir}: {e}")
        return ride_id, None


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
):
    assert video, "Only video format is supported"
    assert episodes is None, "Can only convert all episodes at once"
    assert fps is None or fps == 10, "Only 10 fps is supported"

    ride_dirs = [ride_dir for ride_group in raw_dir.glob("output_rides_*") for ride_dir in ride_group.glob("ride_*")]

    clips = []

    all_ffmpeg_commands = concat_videos_commands(ride_dirs, videos_dir)
    all_valid_files = run_commands(all_ffmpeg_commands, list(range(250)), num_items=len(ride_dirs))
    # with multiprocessing.Pool(4) as pool:
    #     # First run ffmpeg on everything
    #     needs_convert = True
    #     if needs_convert:
    #          list(tqdm.tqdm(pool.imap_unordered(_mp_concat_video, map(lambda ride_dir: (ride_dir, videos_dir), ride_dirs)), total=len(ride_dirs), desc="Encoding videos", smoothing=0.02))

    with multiprocessing.Pool(16) as pool:
        ride_data_pairs = list(tqdm.tqdm(pool.imap_unordered(_mp_get_ride_data, map(lambda ride_dir: (ride_dir, videos_dir), ride_dirs)), total=len(ride_dirs), desc="Reading data", smoothing=0.02))
        ride_data = {ride_id: data for ride_id, data in ride_data_pairs if data is not None}

    # Cut and interpolate. This part must be sequential to maintain an episode index.
    with tqdm.tqdm(ride_dirs, desc="Interpolating and post-processing") as pbar:
        for ride_dir in pbar:
            try:
                ride_id = get_ride_id(ride_dir)

                if ride_id not in ride_data:
                    continue

                all_data = ride_data[ride_id]

                def max_diff_fn(k):
                    if k in [
                        "observation.utm_position",
                        "observation.latitude",
                        "observation.longitude",
                        "observation.utm_zone_letter",
                        "observation.utm_zone_number",
                    ]:
                        return 5.0
                    if k == "action":
                        return 1.0
                    if "image" in k:
                        return 1.0
                    return 2.1

                # Preprocess data
                all_data = filter_data_preprocess(all_data)
                new_clips = cut_interp_clips(all_data, len(clips), max_diff_fn=max_diff_fn)
                # new_clips = [add_synthesized_data(clip) for clip in new_clips]
                new_clips = [filter_data_postprocess(clip) for clip in new_clips]

                clips.extend(new_clips)
            except Exception as e:
                print(f"Error processing {ride_dir}: {e}")

    clips = [clip_to_torch(clip) for clip in clips]

    print(f"Concatenating {len(clips)} episodes...")
    dataset = concatenate_episodes(clips)
    # Save dataset dict as zarr
    print("Saving dataset dict...")
    dataset_numpy = {}
    for k, v in dataset.items():
        if isinstance(v, torch.Tensor):
            dataset_numpy[k] = v.numpy()
        else:
            dataset_numpy[f"{k}.timestamp"] = np.array([x["timestamp"] for x in v])
            dataset_numpy[f"{k}.path"] = np.array([x["path"] for x in v])
    dataset_dict_path = out_root_dir / "dataset_cache.zarr"
    zarr.save_group(dataset_dict_path, **dataset_numpy)

    features = {
        "observation.accelerometer": Sequence(length=3, feature=Value(dtype="float32")),
        "observation.magnetometer": Sequence(length=3, feature=Value(dtype="float32")),
        "observation.gyroscope": Sequence(length=3, feature=Value(dtype="float32")),
        "observation.latitude": Value(dtype="float64"),
        "observation.longitude": Value(dtype="float64"),
        "observation.utm_position": Sequence(length=2, feature=Value(dtype="float64")),
        "observation.utm_zone_number": Value(dtype="int32"),
        "observation.utm_zone_letter": Value(dtype="int32"),
        "observation.wheel_rpm": Sequence(length=4, feature=Value(dtype="float32")),
        "observation.images.front": VideoFrame(),
        "observation.images.rear": VideoFrame(),
        "action": Sequence(length=2, feature=Value(dtype="float32")),
        "episode_index": Value(dtype="int64"),
        "frame_index": Value(dtype="int64"),
        "index": Value(dtype="int64"),
        "timestamp": Value(dtype="float32"),

        "observation.magnetometer_filtered": Sequence(length=3, feature=Value(dtype="float32")),
        "observation.compass_heading": Value(dtype="float32"),
        "observation.relative_position": Sequence(length=2, feature=Value(dtype="float32")),
        "observation.filtered_position": Sequence(length=2, feature=Value(dtype="float32")),
        "observation.filtered_heading": Value(dtype="float32"),
        "action_original": Sequence(length=2, feature=Value(dtype="float32")),
    }

    print("Creating HF dataset...")
    hf_dataset = Dataset.from_dict(dataset, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)

    print("Calculating episode data index...")
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": 10,
        "video": True,
        "encoding": {
            "vcodec": "libx264",
            "pix_fmt": "yuv420p",
            "g": 2,
            "crf": 30,
            "fast_decode": 0,
        },
    }
    return hf_dataset, episode_data_index, info, dataset_numpy


if __name__ == "__main__":
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.compute_stats import compute_stats
    from push_dataset_to_hub import save_meta_data

    batch_size = 32
    num_workers = 8

    raw_data_dir = Path("/ephemeral/frodobots_v2")

    dataset_id = "frodobots_dataset"
    # out_root_dir = Path("/ephemeral/frodobots_v2_export") / dataset_id
    out_root_dir = Path("/mnt/ephemeral2/frodobots_v2_export") / dataset_id

    split_dir = out_root_dir / "train"
    videos_dir = out_root_dir / "videos"
    meta_data_dir = out_root_dir / "meta_data"

    videos_dir.mkdir(parents=True, exist_ok=True)
    meta_data_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    hf_dataset, episode_data_index, info, dataset_dict = from_raw_to_lerobot_format(
        raw_data_dir,
        videos_dir,
    )

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id="frodobots_dataset",
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    hf_dataset.with_format(None).save_to_disk(split_dir)

    # Partial stats save
    print("Computing partial stats...")
    stats = compute_stats(lerobot_dataset, batch_size, num_workers=1, max_num_samples=128 * batch_size)
    save_meta_data(info, stats, episode_data_index, meta_data_dir)

    print("Computing full stats...")
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)
    save_meta_data(info, stats, episode_data_index, meta_data_dir)
