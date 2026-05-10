import numpy as np

class RiskKalmanFilter:
    """
    Simple Kalman Filter to smooth risk scores and prevent jittery alerts.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1):
        # Initial state (risk score) and covariance
        self.x = 0.0  # Initial risk estimate
        self.P = 1.0  # Initial uncertainty

        # Model parameters
        self.Q = process_noise      # Process noise covariance (system uncertainty)
        self.R = measurement_noise  # Measurement noise covariance (sensor noise)
        self.A = 1.0                # State transition matrix (identity for constant model)
        self.H = 1.0                # Observation matrix (identity for direct measurement)
        self.K = 0.0                # Kalman Gain

    def update(self, measurement):
        # Prediction Step
        x_pred = self.A * self.x
        P_pred = self.A * self.P * self.A + self.Q

        # Update Step
        self.K = P_pred * self.H / (self.H * P_pred * self.H + self.R)
        self.x = x_pred + self.K * (measurement - self.H * x_pred)
        self.P = (1 - self.K * self.H) * P_pred

        # Clamp result between 0 and 1
        return max(0.0, min(1.0, self.x))
