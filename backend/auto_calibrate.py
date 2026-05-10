"""
Automated Speed Calibration Script
===================================
This script automatically calibrates the optical flow speed detection
using two videos of known speeds (30 km/h and 60 km/h).

Uses the same optical flow algorithm as road_monitor.py for consistency.
"""

import cv2
import numpy as np
import json
import sys

# === ROI Configuration (must match road_monitor.py) ===
FLOW_ROI_Y_START = 0.50   # Start at 50% down
FLOW_ROI_Y_END = 0.90     # End at 90%
FLOW_ROI_X_START = 0.20   # Wider for more road coverage
FLOW_ROI_X_END = 0.80     # Wider for more road coverage

# Optical flow parameters
FLOW_MIN_THRESHOLD = 0.5

def get_roi_mask(h, w):
    """Create a mask for the road ROI."""
    mask = np.zeros((h, w), dtype=np.uint8)
    y1 = int(h * FLOW_ROI_Y_START)
    y2 = int(h * FLOW_ROI_Y_END)
    x1 = int(w * FLOW_ROI_X_START)
    x2 = int(w * FLOW_ROI_X_END)
    mask[y1:y2, x1:x2] = 255
    return mask


def compute_optical_flow_speed(old_gray, frame_gray, p0, lk_params, h, w):
    """
    Compute optical flow - same logic as road_monitor.py
    Returns: (flow_value, new_p0)
    """
    if p0 is None or len(p0) < 10:
        roi_mask = get_roi_mask(h, w)
        p0 = cv2.goodFeaturesToTrack(
            old_gray, maxCorners=300, qualityLevel=0.2,
            minDistance=7, blockSize=7, mask=roi_mask
        )
        if p0 is None or len(p0) < 10:
            return 0.0, p0
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is None:
        return 0.0, None
    
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    if len(good_new) < 5:
        return 0.0, None
    
    # Filter points in ROI
    y1_roi = int(h * FLOW_ROI_Y_START)
    y2_roi = int(h * FLOW_ROI_Y_END)
    x1_roi = int(w * FLOW_ROI_X_START)
    x2_roi = int(w * FLOW_ROI_X_END)
    
    roi_mask_pts = (
        (good_old[:, 0] >= x1_roi) & (good_old[:, 0] <= x2_roi) &
        (good_old[:, 1] >= y1_roi) & (good_old[:, 1] <= y2_roi)
    )
    
    good_new = good_new[roi_mask_pts]
    good_old = good_old[roi_mask_pts]
    
    if len(good_new) < 5:
        return 0.0, None
    
    # Compute displacements (use median for robustness)
    displacements = np.linalg.norm(good_new - good_old, axis=1)
    flow_value = float(np.median(displacements))
    
    if flow_value < FLOW_MIN_THRESHOLD:
        flow_value = 0.0
    
    # Re-detect points for next frame
    roi_mask = get_roi_mask(h, w)
    new_p0 = cv2.goodFeaturesToTrack(
        frame_gray, maxCorners=300, qualityLevel=0.2,
        minDistance=7, blockSize=7, mask=roi_mask
    )
    
    return flow_value, new_p0


def process_video(video_path, known_speed, skip_start_frames=30, skip_end_frames=30):
    """
    Process a video and compute average optical flow.
    
    Args:
        video_path: Path to video file
        known_speed: Known speed in km/h
        skip_start_frames: Skip first N frames (vehicle accelerating)
        skip_end_frames: Skip last N frames (vehicle decelerating)
    
    Returns:
        (average_flow, std_flow, num_samples)
    """
    print(f"\n[CALIBRATION] Processing: {video_path} @ {known_speed} km/h")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return None, None, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {total_frames} frames @ {fps:.2f} FPS")
    
    # Calculate valid frame range
    valid_start = skip_start_frames
    valid_end = total_frames - skip_end_frames
    
    print(f"[INFO] Processing frames {valid_start} to {valid_end} (skipping start/end)")
    
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    
    # Read first frame
    ret, old_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read first frame")
        cap.release()
        return None, None, 0
    
    h, w = old_frame.shape[:2]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize tracking points
    roi_mask = get_roi_mask(h, w)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=300, qualityLevel=0.2,
                                  minDistance=7, blockSize=7, mask=roi_mask)
    
    flow_values = []
    frame_count = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip start and end frames
        if frame_count < valid_start or frame_count > valid_end:
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            continue
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        flow_value, p0 = compute_optical_flow_speed(old_gray, frame_gray, p0, lk_params, h, w)
        old_gray = frame_gray.copy()
        
        if flow_value > 0:
            flow_values.append(flow_value)
    
    cap.release()
    
    if len(flow_values) < 10:
        print(f"[ERROR] Not enough valid flow samples: {len(flow_values)}")
        return None, None, 0
    
    avg_flow = np.median(flow_values)
    std_flow = np.std(flow_values)
    
    print(f"[RESULT] Samples: {len(flow_values)}, Median Flow: {avg_flow:.4f}, Std: {std_flow:.4f}")
    
    return avg_flow, std_flow, len(flow_values)


def calibrate(video_30, video_60, output_file="calibration.json"):
    """
    Perform two-point linear calibration.
    
    speed = flow * kmh_per_pixel + offset
    
    Given:
      - flow_30 at 30 km/h
      - flow_60 at 60 km/h
    
    Solve:
      30 = flow_30 * k + b
      60 = flow_60 * k + b
    
    Therefore:
      k = (60 - 30) / (flow_60 - flow_30) = 30 / (flow_60 - flow_30)
      b = 30 - flow_30 * k
    """
    print("=" * 60)
    print("AUTOMATED SPEED CALIBRATION")
    print("=" * 60)
    
    # Process both videos
    flow_30, std_30, n_30 = process_video(video_30, 30, skip_start_frames=60, skip_end_frames=60)
    flow_60, std_60, n_60 = process_video(video_60, 60, skip_start_frames=60, skip_end_frames=60)
    
    if flow_30 is None or flow_60 is None:
        print("[ERROR] Calibration failed - could not process videos")
        return False
    
    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    
    print(f"\n30 km/h: Flow = {flow_30:.4f} px/frame (std = {std_30:.4f}, n = {n_30})")
    print(f"60 km/h: Flow = {flow_60:.4f} px/frame (std = {std_60:.4f}, n = {n_60})")
    
    # Check for proper separation
    flow_diff = flow_60 - flow_30
    print(f"\nFlow difference: {flow_diff:.4f} px/frame")
    
    if flow_diff < 0.5:
        print("[WARNING] Flow difference is very small - calibration may be inaccurate")
        print("          Consider checking camera mounting and road visibility")
    
    # Calculate calibration parameters
    kmh_per_pixel = 30.0 / flow_diff  # (60-30) / (flow_60 - flow_30)
    offset = 30.0 - flow_30 * kmh_per_pixel
    
    print(f"\n>>> CALIBRATION RESULT <<<")
    print(f"    kmh_per_pixel = {kmh_per_pixel:.4f}")
    print(f"    offset = {offset:.4f}")
    
    # Verify calibration
    predicted_30 = flow_30 * kmh_per_pixel + offset
    predicted_60 = flow_60 * kmh_per_pixel + offset
    
    print(f"\nVerification:")
    print(f"    At flow {flow_30:.4f}: Predicted = {predicted_30:.1f} km/h (Expected: 30)")
    print(f"    At flow {flow_60:.4f}: Predicted = {predicted_60:.1f} km/h (Expected: 60)")
    
    # Save calibration
    calib_data = {
        "kmh_per_pixel": round(kmh_per_pixel, 4),
        "offset": round(offset, 4),
        "method": "auto_calibration_two_point",
        "flow_30": round(flow_30, 4),
        "flow_60": round(flow_60, 4),
        "std_30": round(std_30, 4),
        "std_60": round(std_60, 4),
        "samples_30": n_30,
        "samples_60": n_60,
        "note": f"Calibrated from Videos_test. Flow diff = {flow_diff:.4f}"
    }
    
    with open(output_file, "w") as f:
        json.dump(calib_data, f, indent=2)
    
    print(f"\n[SUCCESS] Calibration saved to: {output_file}")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    # Default video paths
    video_30 = "Videos_test/30Kmph_10Sec.mp4"
    video_60 = "Videos_test/60Kmph_10Sec.mp4"
    
    # Allow command-line override
    if len(sys.argv) >= 3:
        video_30 = sys.argv[1]
        video_60 = sys.argv[2]
    
    calibrate(video_30, video_60)
