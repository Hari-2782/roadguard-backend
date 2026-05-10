class RiskEngine:
    """
    Central Risk Fusion Engine.
    Combines signals from all modules to compute real-time risk.
    """
    def __init__(self):
        self.current_risk = 0.0

    def compute(self, hazards, potholes, lane_status, speed_kmh, speed_limit, behavior_score):
        """
        Calculate risk score based on inputs.
        """
        # 1. Pothole Severity
        pothole_score = 0.0
        if potholes:
            pothole_score = max([p['severity'] for p in potholes])

        # 2. Hazard Severity
        hazard_score = 0.0
        if hazards:
            hazard_score = max([h['severity'] for h in hazards])

        # 3. Speed Violation
        speed_score = 0.0
        if speed_limit and speed_kmh > speed_limit:
            diff = speed_kmh - speed_limit
            speed_score = min(diff / 20.0, 1.0) # Cap at 20km/h over limit

        # 4. Lane Violation
        lane_score = 0.0
        if lane_status.get('is_crossing', False):
            lane_score = 1.0
            
        # Base Formula
        # 0.25 * pothole + 0.25 * hazard + 0.20 * speed + 0.15 * lane + 0.15 * behavior
        
        risk = (0.25 * pothole_score) + \
               (0.25 * hazard_score) + \
               (0.20 * speed_score) + \
               (0.15 * lane_score) + \
               (0.15 * behavior_score)

        # Interaction Amplification
        if pothole_score > 0.5 and speed_score > 0.3:
            risk *= 1.4 # High speed near pothole is very dangerous
            
        if hazard_score > 0.5 and behavior_score > 0.5:
            risk *= 1.3 # Aggressive driving near hazard

        self.current_risk = min(risk, 1.0)
        return self.current_risk

    def get_risk_level(self):
        if self.current_risk < 0.3: return "LOW"
        if self.current_risk < 0.6: return "MEDIUM"
        if self.current_risk < 0.8: return "HIGH"
        return "CRITICAL"
