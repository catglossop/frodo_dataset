from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import numpy as np

from data import TimestampedData, TimestampedDataDict


def search_nearest(indices: np.ndarray, target_indices: np.ndarray, max_threshold=None):
    """
    Find the index of the nearest timestamp for each target index
    """
    insert_indices = np.searchsorted(indices, target_indices, side="right")

    lefts = np.maximum(insert_indices - 1, 0)
    rights = np.minimum(insert_indices, len(indices) - 1)

    left_diff = target_indices - indices[lefts]
    right_diff = indices[rights] - target_indices

    if max_threshold is not None and np.any(np.maximum(left_diff, right_diff) > max_threshold):
        raise ValueError("Nearest timestamp is more than max_threshold away")

    return np.where(left_diff < right_diff, lefts, rights)


def interpolate_data(
    stamped_data: TimestampedDataDict, target_timestamps: np.ndarray
) -> TimestampedDataDict:
    """
    Interpolate the data to the target timestamps.

    Assumes that all data contains the target timestamps
    """
    data_interpolated = {}

    for k in stamped_data:
        data = stamped_data[k].data
        timestamps = stamped_data[k].timestamps
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                interp_data = np.interp(target_timestamps, timestamps, data)
            elif data.ndim == 2:
                interp_data = np.stack(
                    [
                        np.interp(target_timestamps, timestamps, data[:, i])
                        for i in range(data.shape[1])
                    ],
                    axis=-1,
                )
            else:
                raise ValueError(f"Unsupported data shape: {data.shape}")
        else:
            assert (
                "image" in k
            ), f"Data {k} is not a np.ndarray, but is also not an image"
            video_indices = search_nearest(timestamps, target_timestamps)
            interp_data = [data[i] for i in video_indices]

        data_interpolated[k] = TimestampedData(
            data=interp_data, timestamps=target_timestamps
        )

    return data_interpolated


Interval = Tuple[float, float]


def interval_union(intervals: List[Interval]) -> List[Interval]:
    """
    Given a list of (possibly overlapping) intervals, merge them into non-overlapping intervals.
    """
    intervals.sort()
    merged_intervals = []

    current_interval = None

    for start, end in intervals:
        if current_interval is None:
            current_interval = (start, end)

        current_start, current_end = current_interval
        if start <= current_end:
            current_interval = (current_start, max(current_end, end))
        else:
            merged_intervals.append(current_interval)
            current_interval = None

    if current_interval is not None:
        merged_intervals.append(current_interval)

    return merged_intervals


def interval_difference(
    positive_interval: Interval, negative_intervals: List[Interval]
) -> List[Interval]:
    """
    Given a positive interval and a list of sorted non-overlapping negative intervals, return the difference.
    """
    start, end = positive_interval

    while len(negative_intervals) and negative_intervals[0][1] < start:
        negative_intervals.pop(0)

    while len(negative_intervals) and negative_intervals[-1][0] > end:
        negative_intervals.pop()

    if len(negative_intervals) == 0:
        return [positive_interval]

    endpoints = [endpoint for interval in negative_intervals for endpoint in interval]

    if start < endpoints[0]:
        endpoints = [start] + endpoints
    else:
        endpoints.pop(0)

    if end > endpoints[-1]:
        endpoints = endpoints + [end]
    else:
        endpoints.pop()

    return [(endpoints[i], endpoints[i + 1]) for i in range(0, len(endpoints), 2)]


def find_gaps(timestamps: np.ndarray, max_diff: float) -> List[Interval]:
    """
    Find gaps in the timestamps that are longer than max_diff
    """
    diffs = np.diff(timestamps)
    gap_indices = np.nonzero(diffs > max_diff)[0]

    return [(timestamps[gap].item(), timestamps[gap + 1].item()) for gap in gap_indices]


def cut_interp_clips(
    all_data: TimestampedDataDict,
    start_episode_idx,
    max_diff_fn=lambda k: 1.0,
    sample_rate=10.0,
) -> List[Dict[str, Any]]:
    """
    Break up an episode into "clips" based on gaps in the timestamps with missing data, and interpolate all data within a single clip using the `master_key` (usually action).

    Returns a list of dictionaries, each corresponding to a single clip, with the following additional keys:
     - "episode_index": a unique identifier for the clip
     - "frame_index": the frame index within the clip
     - "timestamp": the timestamps of the newly-interpolated items in the clip
    """

    # First, find max start time and min end time
    start_time = max([data.timestamps[0] for data in all_data.values()])
    end_time = min([data.timestamps[-1] for data in all_data.values()])

    if start_time >= end_time:
        raise ValueError("No overlapping timestamps")

    # Find gaps in timestamps
    # Cut at any gaps longer than 1 second
    gap_intervals = []
    for k, v in all_data.items():
        max_diff = max_diff_fn(k)
        gap_intervals.extend(find_gaps(v.timestamps, max_diff))

    # Invert the gap intervals to get the clip intervals
    gap_intervals = interval_union(gap_intervals)
    clip_intervals = interval_difference((start_time, end_time), gap_intervals)

    # master_timestamps = all_data[master_key].timestamps
    master_timestamps = np.arange(start_time, end_time, 1 / sample_rate)

    clip_target_timestamps = [
        master_timestamps[(master_timestamps >= start) & (master_timestamps <= end)]
        for start, end in clip_intervals
    ]

    # Remove empty clips
    clip_target_timestamps = [ts for ts in clip_target_timestamps if len(ts) > 0]

    clips_with_timestamps = [
        interpolate_data(all_data, target_timestamps)
        for target_timestamps in clip_target_timestamps
    ]

    clips = []
    for clip_stamped in clips_with_timestamps:
        clip = {}

        for k, v in clip_stamped.items():
            if isinstance(v.data, np.ndarray):
                clip[k] = v.data
            else:
                assert "image" in k, f"Data {k} is not a np.ndarray, but is also not an image"
                video_start_time = all_data[k].data[0]["timestamp"]
                clip[k] = [
                    {"path": e["path"], "timestamp": e["timestamp"] - video_start_time}
                    for e in v.data
                ]

        # Remove any timestamps with zero action
        action_nonzero = np.any(np.abs(clip["action"]) > 0.01, axis=-1)
        convolve_padding = np.zeros((10,), dtype=bool)
        nearby_action_nonzero = np.convolve(
            np.concatenate([convolve_padding, action_nonzero, convolve_padding]),
            np.ones(10), mode="same"
        )[10:-10] > 0
        assert len(nearby_action_nonzero) == len(action_nonzero), f"Bad length {len(nearby_action_nonzero)} vs {len(action_nonzero)}"
        nonzero_action = np.where(nearby_action_nonzero)[0]

        for k, v in clip.items():
            if isinstance(v, np.ndarray):
                clip[k] = v[nonzero_action]
            else:
                clip[k] = [v[i] for i in nonzero_action]

        # Compute timestamps and indices after removing stationary frames
        clip_length = len(nonzero_action)
        clip["timestamp"] = np.arange(0, clip_length) / sample_rate
        clip["episode_index"] = np.full((clip_length,), start_episode_idx + len(clips))
        clip["frame_index"] = np.arange(0, clip_length)

        # Remove clips with fewer than 10 frames
        if clip_length < 10:
            continue

        # Remove clips with no/minimal GPS movement
        utm_positions = clip["observation.utm_position"]
        if np.abs(np.max(utm_positions, axis=0) - np.min(utm_positions, axis=0)).max() < 1.0:
            continue

        clips.append(clip)

    return clips
