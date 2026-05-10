import os
import numpy as np
import joblib
import pickle

LABELS = [
    "Harsh_Brake",          # Index 0
    "Normal_Driving",       # Index 1
    "Sharp_Turn",           # Index 2
    "Sudden_Acceleration",  # Index 3
    "Sudden_Lane_Change"    # Index 4
]

class BehaviorInferenceService:
    def __init__(self, model_dir):
        self.model = None
        self.label_encoder = None
        
        # Paths to PKL files
        model_path = os.path.join(model_dir, "master_ensemble.pkl")
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        
        if os.path.exists(model_path):
            try:
                # Try joblib first (standard for sklearn), fall back to pickle
                try:
                    self.model = joblib.load(model_path)
                except:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                print(f"[BehaviorModel] Loaded PKL model from {model_path}")
            except Exception as e:
                print(f"[BehaviorModel] Error loading model: {e}")
        else:
            print(f"[BehaviorModel] Model not found at {model_path}")

        if os.path.exists(encoder_path):
            try:
                try:
                    self.label_encoder = joblib.load(encoder_path)
                except:
                    with open(encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                print(f"[BehaviorModel] Loaded Label Encoder from {encoder_path}")
            except Exception as e:
                print(f"[BehaviorModel] Error loading label encoder: {e}")

    def run_inference(self, features: np.ndarray) -> tuple[str, dict]:
        """
        Runs the hybrid inference pipeline with Physics Safety Layer and PKL model.
        Returns: (predicted_label, additional_info_dict)
        """
        if self.model is None:
            return "Normal_Driving", {"error": "Model not loaded"}

        try:
            # 1. EARLY PHYSICS OVERRIDE (Reflex Layer)
            if len(features) >= 28:
                magMax = float(features[24])
                jerkMax = float(features[25])
                yawSpeed = float(features[27])

                # Crash/Bump
                if magMax > 12.0:
                    return LABELS[0], {"override": "Early Reflex (Crash/Bump)"}
                
                # Sharp Turn
                if yawSpeed > 4.0:
                    return LABELS[2], {"override": "Early Reflex (Sharp Turn)"}

            # 2. RUN PKL PREDICTION
            # Reshape features for model: (1, n_features)
            # If the model expects 210 features but we have 30, it will error.
            # We try to use the features provided, but if it fails with a feature mismatch,
            # this indicates we should have used the other extraction method.
            # However, run_inference takes pre-extracted features.
            # Let's adjust the caller or make this robust.
            input_data = features.reshape(1, -1)
            
            try:
                # Predict
                label_idx = self.model.predict(input_data)[0]
                
                # Get probabilities if available
                probs = [0.0] * 5
                try:
                    probs_raw = self.model.predict_proba(input_data)[0]
                    probs = probs_raw.tolist()
                except:
                    pass
            except Exception as e:
                # If feature mismatch, return the error so the caller can retry with raw features
                if "expecting 210 features" in str(e) or "requests feature 209" in str(e):
                    return "Normal_Driving", {"error": "Feature mismatch", "expected": 210, "received": len(features)}
                raise e

            # Convert index to label using encoder if available, else static list
            if self.label_encoder is not None:
                try:
                    predicted_label = self.label_encoder.inverse_transform([label_idx])[0]
                except:
                    predicted_label = LABELS[label_idx] if label_idx < len(LABELS) else str(label_idx)
            else:
                predicted_label = LABELS[label_idx] if label_idx < len(LABELS) else str(label_idx)

            # Sensitivity Boost for Sudden Lane Changes
            if len(probs) > 4 and probs[4] > 0.70:
                return LABELS[4], {"probs": probs, "trigger": "AI Lane Change Boost"}

            # 3. LATE PHYSICS SAFETY LAYER (Overrides AI)
            ax_max = features[2]
            ax_min = features[3]
            ay_mean = features[4]
            ay_max = features[6]
            ay_min = features[7]
            gz_max = features[22]
            jerkMax = features[25]
            speedKmh = features[28]

            isLowSpeed = speedKmh < 30
            sensitivityMult = 1.5 if isLowSpeed else 1.0

            # Harsh Brake
            if ax_min < -5.0:
                return "Harsh_Brake", {"probs": probs, "override": "Physics (Harsh Brake)"}
            
            # Sudden Acceleration
            if ax_max > 2.5 or jerkMax > 15.0:
                return "Sudden_Acceleration", {"probs": probs, "override": "Physics (Sudden Accel)"}
            
            # Sharp Turn
            ay_threshold = 4.0 * sensitivityMult
            if (abs(ay_max) > ay_threshold or abs(ay_min) > ay_threshold) and abs(ay_mean) > (ay_threshold * 0.5):
                return "Sharp_Turn", {"probs": probs, "override": "Physics (Sharp Turn)"}

            # Sudden Lane Change
            gz_threshold = 2.0 * sensitivityMult
            if abs(gz_max) > gz_threshold:
                return "Sudden_Lane_Change", {"probs": probs, "override": "Physics (Sudden Lane Change)"}

            return predicted_label, {"probs": probs, "trigger": "Normal AI"}

        except Exception as e:
            return "Normal_Driving", {"error": str(e)}
