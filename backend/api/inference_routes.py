from flask import Blueprint, jsonify, request
import time
import os
import torch
import numpy as np

# This blueprint handles REST requests for model inference
inference_bp = Blueprint('inference', __name__, url_prefix='/api/inference')

# Device discovery
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@inference_bp.route('/health', methods=['GET'])
def health():
    """Returns the health and status of all ML models"""
    from api.api_server import hazard_model, pothole_model, sign_model, lane_model, road_model, MODELS_LOADED
    
    def _check(m):
        if m is None:
            return "not_loaded"
        # Check if the wrapper's inner model actually loaded
        if hasattr(m, 'model') and m.model is None:
            return "wrapper_only"
        return "ready"
    
    status = {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": MODELS_LOADED,
        "models": {
            "hazard": {"health": _check(hazard_model)},
            "lane_detector": {"health": _check(lane_model)},
            "pothole": {"health": _check(pothole_model)},
            "road_segmenter": {"health": _check(road_model)},
            "sign_detector": {"health": _check(sign_model)}
        },
        "cache": {
            "size": 0,
            "max_size": 5,
            "hits": 0,
            "misses": 0,
            "hit_rate": "0.0%",
            "total_requests": 0
        }
    }
    return jsonify(status)

@inference_bp.route('/models', methods=['GET'])
def list_models():
    """List all available ML models"""
    models = [
        {"id": "hazard", "name": "Hazard Detector", "type": "YOLOv8"},
        {"id": "lane", "name": "Lane Detector", "type": "PyTorch-DeepLabV3"},
        {"id": "pothole", "name": "Pothole Detector", "type": "YOLOv8"},
        {"id": "road", "name": "Road Segmenter", "type": "PyTorch-DeepLabV3"},
        {"id": "sign", "name": "Sign Detector", "type": "YOLOv8"}
    ]
    return jsonify(models)

@inference_bp.route('/hazard', methods=['POST'])
def infer_hazard():
    return jsonify({"error": "Endpoint not fully implemented in reconstruction"}), 501

@inference_bp.route('/lane', methods=['POST'])
def infer_lane():
    return jsonify({"error": "Endpoint not fully implemented in reconstruction"}), 501

@inference_bp.route('/pothole', methods=['POST'])
def infer_pothole():
    return jsonify({"error": "Endpoint not fully implemented in reconstruction"}), 501

@inference_bp.route('/road', methods=['POST'])
def infer_road():
    return jsonify({"error": "Endpoint not fully implemented in reconstruction"}), 501

@inference_bp.route('/sign', methods=['POST'])
def infer_sign():
    return jsonify({"error": "Endpoint not fully implemented in reconstruction"}), 501

@inference_bp.route('/frame', methods=['POST'])
def infer_frame():
    """Analyze a single frame for hazards, potholes, and signs."""
    import cv2
    import numpy as np
    from api.api_server import hazard_model, pothole_model, sign_model
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    in_memory_file = file.read()
    nparr = np.frombuffer(in_memory_file, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400
        
    results = {
        'hazards': [],
        'potholes': [],
        'signs': []
    }
    
    if hazard_model and hasattr(hazard_model, 'model') and hazard_model.model is not None:
        try:
            results['hazards'] = hazard_model.detect(frame)
        except Exception as e:
            pass
            
    if pothole_model and hasattr(pothole_model, 'model') and pothole_model.model is not None:
        try:
            results['potholes'] = pothole_model.detect(frame)
        except Exception as e:
            pass
            
    if sign_model and hasattr(sign_model, 'model') and sign_model.model is not None:
        try:
            results['signs'] = sign_model.detect(frame)
        except Exception as e:
            pass
            
    return jsonify(results)

@inference_bp.route('/batch', methods=['POST'])
def infer_batch():
    return jsonify({"error": "Endpoint not fully implemented in reconstruction"}), 501

@inference_bp.route('/model/<name>/version', methods=['POST'])
def model_version(name):
    return jsonify({"model": name, "version": "1.0.0", "status": "active"})
