import numpy as np

class KalmanFilter:
    def __init__(self, R=1.0, Q=1.0, A=1.0, B=0.0, C=1.0):
        self.R = R
        self.Q = Q
        self.A = A
        self.B = B
        self.C = C
        self.cov = float('nan')
        self.x = float('nan')

    def filter(self, z, u=0.0):
        if np.isnan(self.x):
            self.x = (1.0 / self.C) * z
            self.cov = (1.0 / self.C) * self.Q * (1.0 / self.C)
        else:
            pred_x = self.A * self.x + self.B * u
            pred_cov = self.A * self.cov * self.A + self.R

            K = pred_cov * self.C * (1.0 / (self.C * pred_cov * self.C + self.Q))

            self.x = pred_x + K * (z - self.C * pred_x)
            self.cov = pred_cov - K * self.C * pred_cov
            
        return self.x

def extract_features(buffer_data):
    """
    Standard statistical feature extraction (30 features).
    buffer_data: list of dicts with keys: accelX, accelY, accelZ, gyroX, gyroY, gyroZ, speed
    Returns a numpy array of 30 features.
    """
    features = np.zeros(30, dtype=np.float32)
    N = len(buffer_data)
    if N == 0:
        return features

    # Extract arrays
    accelX = np.array([row['accelX'] for row in buffer_data])
    accelY = np.array([row['accelY'] for row in buffer_data])
    accelZ = np.array([row['accelZ'] for row in buffer_data])
    gyroX = np.array([row['gyroX'] for row in buffer_data])
    gyroY = np.array([row['gyroY'] for row in buffer_data])
    gyroZ = np.array([row['gyroZ'] for row in buffer_data])
    speed = np.array([row['speed'] for row in buffer_data])

    # Derived values
    mag = np.sqrt(accelX**2 + accelY**2 + accelZ**2)
    
    # Jerk X (Delta AccX / dt). dt = 0.05s
    jerkX = np.zeros(N)
    jerkX[1:] = (accelX[1:] - accelX[:-1]) / 0.05

    # Yaw-Speed Coupling
    yawSpeed = gyroZ * speed

    def calc_stats(arr):
        return np.mean(arr), np.std(arr, ddof=0), np.max(arr), np.min(arr)

    idx = 0
    # 1. Basic Stats for 6 Axes (24 Features)
    for axis_arr in [accelX, accelY, accelZ, gyroX, gyroY, gyroZ]:
        mean, std, max_val, min_val = calc_stats(axis_arr)
        features[idx:idx+4] = [mean, std, max_val, min_val]
        idx += 4

    # 24. Acc_Mag_Max
    features[idx] = np.max(mag)
    idx += 1

    # 25. Jerk_X_Max
    features[idx] = np.max(jerkX)
    idx += 1

    # 26. Lat_Long_Ratio: Mean(|Acc_y|) / (Mean(|Acc_x|) + 0.1)
    mean_abs_accY = np.mean(np.abs(accelY))
    mean_abs_accX = np.mean(np.abs(accelX))
    features[idx] = mean_abs_accY / (mean_abs_accX + 0.1)
    idx += 1

    # 27. Yaw_Speed_Coupling: Mean(|Gyro_z * Speed|)
    features[idx] = np.mean(np.abs(yawSpeed))
    idx += 1

    # 29. Speed_Mean  (index 28 in 0-indexed array)
    features[idx] = np.mean(speed)
    idx += 1

    # 30. Speed_Std (index 29 in 0-indexed array)
    features[idx] = np.std(speed, ddof=0)
    idx += 1

    return features

def extract_features_raw(buffer_data):
    """
    Raw feature extraction (210 features: 7 axes * 30 timesteps).
    buffer_data: list of dicts with 30 rows.
    Returns a flattened numpy array of 210 features.
    """
    # Order: accelX, accelY, accelZ, gyroX, gyroY, gyroZ, speed
    axes = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ', 'speed']
    output = []
    for axis in axes:
        axis_values = [row[axis] for row in buffer_data]
        # Pad or truncate to exactly 30 if needed (though buffer_data should be 30)
        if len(axis_values) < 30:
            axis_values.extend([axis_values[-1]] * (30 - len(axis_values)))
        elif len(axis_values) > 30:
            axis_values = axis_values[:30]
        output.extend(axis_values)
    
    return np.array(output, dtype=np.float32)
