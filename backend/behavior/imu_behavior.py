import random
import time

class IMUBehavior:
    """
    Simulates IMU data for driver behavior analysis.
    In a real system, this would read from an accelerometer/gyroscope.
    """
    def __init__(self):
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.gyro_z = 0.0
        self.last_update = time.time()

    def update(self):
        """Read sensors (simulated)"""
        # Simulate occasional harsh events
        if random.random() < 0.05:
            self.accel_x = random.uniform(-0.8, 0.8) # Harsh braking/accel
        else:
            self.accel_x = random.uniform(-0.1, 0.1)
            
        if random.random() < 0.05:
            self.gyro_z = random.uniform(-1.0, 1.0) # Aggressive turn
        else:
            self.gyro_z = random.uniform(-0.1, 0.1)

    def get_behavior_score(self):
        """
        Returns normalized score 0.0 (calm) to 1.0 (aggressive).
        """
        self.update()
        
        # Simple magnitude check
        score = (abs(self.accel_x) + abs(self.gyro_z)) / 2.0
        return min(max(score, 0.0), 1.0)
