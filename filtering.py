import numpy as np
from typing import Dict, Any
import geomag

from data import TimestampedData, TimestampedDataDict

def correct_magnetometer(magnetometer: np.ndarray) -> np.ndarray:
    """
    Correct for the bias on the magnetometer.
    """
    bias = (magnetometer.max(axis=0) + magnetometer.min(axis=0)) / 2
    return magnetometer - bias


def wheel_rpm_to_velocity(wheel_rpm: np.ndarray) -> np.ndarray:
    """
    Convert wheel RPM to angular velocity.
    """
    forward_speed = (wheel_rpm[:, 0] + wheel_rpm[:, 1] + wheel_rpm[:, 2] + wheel_rpm[:, 3])
    angular_speed = (wheel_rpm[:, 1] + wheel_rpm[:, 3] - wheel_rpm[:, 0] - wheel_rpm[:, 2])
    return forward_speed, angular_speed


def correct_actions(actions: np.ndarray, wheel_rpm: np.ndarray) -> np.ndarray:
    """
    Compute bias on steering actions and correct for it.

    We assume a constant bias on the curvature (steering/throttle), and perform
    a linear fit between the curvature from actions and the curvature from the
    wheel RPM, then correct the steering actions.
    """
    forward_speed, angular_speed = wheel_rpm_to_velocity(wheel_rpm)

    # Smooth out angular speed and steering to reduce noise
    window_size = 5
    kernel = np.ones(window_size) / window_size

    angular_speed_smooth = np.convolve(angular_speed, kernel, mode='valid')
    steering_smooth = np.convolve(actions[:, 1], kernel, mode='valid')
    throttle_smooth = np.convolve(actions[:, 0], kernel, mode='valid')
    forward_speed_smooth = np.convolve(forward_speed, kernel, mode='valid')

    # Compute curvature from actions and from wheel RPM
    action_curvature = steering_smooth / (np.abs(throttle_smooth) + 1e-3) * np.sign(throttle_smooth)
    sensor_curvature = angular_speed_smooth / (np.abs(forward_speed_smooth) + 1e-3) * np.sign(forward_speed_smooth)
    high_speed_mask = (throttle_smooth > 0.3) & (forward_speed_smooth > 50)

    action_curvature = action_curvature[high_speed_mask]
    sensor_curvature = sensor_curvature[high_speed_mask]

    if len(action_curvature) < 10:
        return actions

    slope, intercept = np.polyfit(action_curvature, sensor_curvature, 1)

    steering_corrected = actions[:, 1] + intercept / slope * actions[:, 0]
    action_corrected = np.stack([actions[:, 0], steering_corrected], axis=-1)

    print("Bias:", intercept / slope)

    return action_corrected


def align_timestamps(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the latency of action timestamps across the trajectory and correct for it.

    We do this by computing the cross-correlation between the action and the wheel RPM.
    """
    forward_speed, angular_speed = wheel_rpm_to_velocity(data["observation.wheel_rpm"])

    def find_max_cross_correlation(v1, v2, max_lag=50):
        if v1.ndim == 1:
            v1 = v1.reshape(-1, 1)
            v2 = v2.reshape(-1, 1)

        assert v1.shape == v2.shape, f"v1.shape: {v1.shape}, v2.shape: {v2.shape}"

        cross_correlation = sum([np.correlate(v1[:, i], v2[:, i], mode='full') / np.sqrt(np.sum(v1[:, i] ** 2) * np.sum(v2[:, i] ** 2)) for i in range(v1.shape[-1])])

        return np.argmax(cross_correlation[len(v1) - max_lag:len(v1) + max_lag]) - max_lag

    def adjust_for_lag(data, lag_key, lag_value):
        # Positive lag means the value lags behind the rest of the data
        # Negative lag means the value leads the rest of the data

        for key in data.keys():
            if key == lag_key:
                if lag_value > 0:
                    data[key] = data[key][:-lag_value]
                else:
                    data[key] = data[key][lag_value:]
            else:
                if lag_value > 0:
                    data[key] = data[key][lag_value:]
                else:
                    data[key] = data[key][:-lag_value]

    lag_frames_action = find_max_cross_correlation(angular_speed, data["action"][:, 1])
    adjust_for_lag(data, "action", lag_frames_action)

    return data


def do_ekf_filter(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Do an extended kalman filter on the data.

    State: [x, y, theta, compass_bias, gps_bias_x, gps_bias_y]
    Measurement: [x, y, cos(theta), sin(theta)]
    Input: [forward_speed, steering_angle]
    """
    dt = 0.1

    def _measurement_model(state: np.ndarray):
        """
        Measurement model for the EKF.

        Returns: expected measurement and jacobian of the measurement model
        """
        biased_compass_heading = state[..., 2] + state[..., 3]
        measurement = np.stack([
            state[..., 0] + state[..., 4],
            state[..., 1] + state[..., 5],
            np.cos(biased_compass_heading),
            np.sin(biased_compass_heading),
        ], axis=-1)

        jacobian = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, -np.sin(biased_compass_heading), -np.sin(biased_compass_heading), 0, 0],
            [0, 0, np.cos(biased_compass_heading), np.cos(biased_compass_heading), 0, 0],
        ])

        return measurement, jacobian

    def _process_model(state: np.ndarray, input: np.ndarray, dt: float):
        """
        Process model for the EKF.
        
        Returns: expected state and jacobian of the process model
        """
        forward_speed, steering_angle = input
        theta = state[..., 2]

        bias_decay = 0.0
        
        next_state = state + dt * np.stack([
            forward_speed * np.cos(theta),
            forward_speed * np.sin(theta),
            steering_angle,
            bias_decay * state[..., 3],
            bias_decay * state[..., 4],
            bias_decay * state[..., 5],
        ], axis=-1)

        jacobian = np.array([
            [1, 0, -dt * forward_speed * np.sin(theta), 0, 0, 0],
            [0, 1, dt * forward_speed * np.cos(theta), 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1 - bias_decay * dt, 0, 0],
            [0, 0, 0, 0, 1 - bias_decay * dt, 0],
            [0, 0, 0, 0, 0, 1 - bias_decay * dt],
        ])

        return next_state, jacobian
        
    def _make_initial_state(data: Dict[str, Any], index: int):
        state = np.zeros((6,))
        state[2] = data["observation.compass_heading"][index]
        state[:2] = data["observation.relative_position"][index]
        covariance = np.diag([3.0, 3.0, 0.5, 0.5, 3.0, 3.0]) ** 2
        return state, covariance

    def _Q(input: np.ndarray):
        return np.diag([
            1e-2,
            1e-2,
            1e-2 + 1e-1 * input[1] ** 2,
            1e-2,
            1e-2 * (0.1 + input[0] ** 2),
            1e-2 * (0.1 + input[0] ** 2),
        ])

    # Q = np.diag([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])
    R = np.diag([3.0, 3.0, 0.5, 0.5]) ** 2

    state_fwd, covariance_fwd = _make_initial_state(data, 0)
    means_fwd = [state_fwd]
    covariances_fwd = [covariance_fwd]

    for t in range(len(data["action"]) - 1):
        action = data["action"][t]

        # Predict
        state_fwd, A = _process_model(state_fwd, action, dt)
        covariance_fwd = A @ covariance_fwd @ A.T + _Q(action)
        means_fwd.append(state_fwd)
        covariances_fwd.append(covariance_fwd)

        # Update
        measurement, H = _measurement_model(state_fwd)
        z = np.concatenate([
            data["observation.relative_position"][t+1],
            np.cos(data["observation.compass_heading"][t+1])[..., None],
            np.sin(data["observation.compass_heading"][t+1])[..., None],
        ])

        S = H @ covariance_fwd @ H.T + R
        K = covariance_fwd @ H.T @ np.linalg.inv(S)
        state_fwd = state_fwd + K @ (z - measurement)
        covariance_fwd = (np.eye(6) - K @ H) @ covariance_fwd

    # Backwards pass
    state_rev, covariance_rev = _make_initial_state(data, -1)
    means_rev = [state_rev]
    covariances_rev = [covariance_rev]

    for t in reversed(range(1, len(data["action"]))):
        # Reversed dynamics
        action = data["action"][t]
        state_rev, A = _process_model(state_rev, action, -dt)
        covariance_rev = A @ covariance_rev @ A.T + _Q(action)
        means_rev.append(state_rev)
        covariances_rev.append(covariance_rev)

        # Update (normal)
        measurement, H = _measurement_model(state_rev)
        z = np.concatenate([
            data["observation.relative_position"][t-1],
            np.cos(data["observation.compass_heading"][t-1])[..., None],
            np.sin(data["observation.compass_heading"][t-1])[..., None],
        ])
        S = H @ covariance_rev @ H.T + R
        K = covariance_rev @ H.T @ np.linalg.inv(S)
        state_rev = state_rev + K @ (z - measurement)
        covariance_rev = (np.eye(6) - K @ H) @ covariance_rev

    means_rev = means_rev[::-1]
    covariances_rev = covariances_rev[::-1]

    # Combine forward and backward pass weighted by the covariances
    states_combined = []

    for mean_fwd, covariance_fwd, mean_rev, covariance_rev in zip(means_fwd, covariances_fwd, means_rev, covariances_rev):
        # Make sure mean_fwd true heading is within -pi to pi of mean_rev true heading
        mean_fwd[2] -= np.round((mean_fwd[2] - mean_rev[2]) / (2 * np.pi)) * 2 * np.pi

        precision_fwd = np.linalg.inv(covariance_fwd)
        precision_rev = np.linalg.inv(covariance_rev)

        precision = precision_fwd + precision_rev
        covariance = np.linalg.inv(precision)
        mean = (precision_fwd @ mean_fwd + precision_rev @ mean_rev) @ covariance

        states_combined.append(mean)

    states = np.stack(states_combined, axis=0)

    data["observation.filtered_position"] = states[..., :2]
    data["observation.filtered_heading"] = states[..., 2]

    return data


def filter_data_preprocess(data: TimestampedDataDict) -> TimestampedDataDict:
    """
    Filter the data before interpolation.
    """

    # Correct for magnetometer bias
    data["observation.magnetometer_filtered"] = TimestampedData(data=correct_magnetometer(data["observation.magnetometer"].data), timestamps=data["observation.magnetometer"].timestamps)

    # Correct for bias in actions
    data["action_original"] = data["action"]
    data["action"] = TimestampedData(data=correct_actions(data["action"].data, data["observation.wheel_rpm"].data), timestamps=data["action"].timestamps)
    return data


def compute_compass_heading(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the compass heading.
    """
    raw_compass_heading = np.arctan2(
        -data["observation.magnetometer_filtered"][:, 2],
        data["observation.magnetometer_filtered"][:, 1],
    )

    declination = [geomag.declination(lat, lon) for lat, lon in zip(data["observation.latitude"], data["observation.longitude"])]
    data["observation.compass_heading"] = raw_compass_heading + np.deg2rad(declination)

    return data


def compute_relative_position(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the relative position and rotate it to align with the initial heading.
    """
    data["observation.relative_position"] = data["observation.utm_position"] - data["observation.utm_position"][0]
    return data


def filter_data_postprocess(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter the data after interpolation.
    """
    data = compute_compass_heading(data)
    data = compute_relative_position(data)
    data = do_ekf_filter(data)
    return data