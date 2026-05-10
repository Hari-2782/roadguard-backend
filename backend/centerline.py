# centerline.py
# Detect lane lines (left/right), classify solid/dashed, and estimate color (yellow vs unknown).
# UPDATED: Now uses strict Color Filtering (White/Yellow) to ignore sand/dirt.

import cv2
import math
import numpy as np


def _filter_lane_colors(frame):
    """
    Keep only White and Yellow pixels. Black out everything else.
    This removes sand, grass, and dirt edges.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- WHITE MASK ---
    # High value, low-to-medium saturation
    # Sensitivity factor: lower to be stricter
    # Raised V-min to 200 to exclude bright sand/dirt.
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # --- YELLOW MASK ---
    # Hue ~20-30 (OpenCV scale 0-180)
    lower_yellow = np.array([15, 60, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # ... (rest of function) ...

# ...

    # Helper: Average segments
    def average_segments(segments):
        if not segments:
            return None

        X_coords = []
        Y_coords = []
        for (x1, y1, x2, y2) in segments:
            X_coords += [x1, x2]
            Y_coords += [y1, y2]

        if len(X_coords) < 2:
            return None

        # Robust fit
        try:
            m, b = np.polyfit(X_coords, Y_coords, 1)
        except Exception:
            return None

        if math.isinf(m) or abs(m) < 0.3: # ignore horizontal/flat lines
            return None

        y_bottom = height
        y_top = int(height * 0.6) # extend up to 60% of screen

        x_bottom = int((y_bottom - b) / m)
        x_top = int((y_top - b) / m)
        
        # --- CRITICAL FIX FOR BIKE "X" PATTERN ---
        # If the line starts (at the bottom) from the CENTER of the screen,
        # it is tracking the bike or its shadow.
        # Real lanes must start from the SIDES.
        # Reject if x_bottom is between 30% and 70% of width.
        if (width * 0.30) < x_bottom < (width * 0.70):
             return None
        # -----------------------------------------

        return (x_bottom, y_bottom, x_top, y_top, m, b)
    # Hue ~20-30 (OpenCV scale 0-180)
    lower_yellow = np.array([15, 60, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Combine
    mask_combined = cv2.bitwise_or(mask_white, mask_yellow)
    
    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask_combined


def _estimate_line_color(patch):
    """
    Roughly decide if this patch is 'yellow' or not using HSV.
    Returns 'yellow' or 'unknown' (white).
    """
    if patch is None or patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # --- YELLOW CHECK ---
    # Broaden range to catch faded yellow
    # Lower Saturation to 30 (from 50) and Value to 80 (from 100)
    lower_yellow = np.array([12, 30, 80]) 
    upper_yellow = np.array([45, 255, 255])
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = mask_yellow.mean() / 255.0

    if yellow_ratio > 0.05: # > 5% Yellow
        return "yellow"

    # --- WHITE CHECK ---
    # High Value, Low Saturation
    lower_white = np.array([0, 0, 180])   # Val > 180 (Bright)
    upper_white = np.array([180, 50, 255]) # Sat < 50 (Not colorful)
    
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    white_ratio = mask_white.mean() / 255.0
    
    if white_ratio > 0.05: # > 5% White
        return "white"

    # --- UNKNOWN / DIRT ---
    return "unknown"


def detect_lane_lines(frame, model=None):
    """
    Main lane detection.
    Returns:
        {
          'left':  {'type': 'solid'/'dashed'/None, ... },
          'right': { ... }
        }
    """
    height, width = frame.shape[:2]

    # --- REBUILT PIPELINE (Based on User Spec) ---
    # Pipeline: Grayscale → Blur → Canny → Color Filter → Hough → Slope Filter
    
    # 1. Grayscale + Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny Edge Detection
    edges = cv2.Canny(gray_blur, 50, 150)
    
    # 3. COLOR PRE-FILTER (Fixes platform edge false positives)
    # Only keep edges that are near WHITE or YELLOW pixels
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Yellow mask (for left lanes) - STRICTER to avoid picking up walls/objects
    # Real yellow road paint has specific hue (18-30) and decent saturation
    lower_yellow = np.array([15, 60, 120])  # Stricter: higher sat/val required
    upper_yellow = np.array([35, 255, 255])  # Narrower hue range
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # White mask (for center lines) - VERY RELAXED to catch faded paint
    lower_white = np.array([0, 0, 140])  # Very low to catch faded white
    upper_white = np.array([180, 60, 255])  # Allow more saturation
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Combine and dilate (allow nearby edges)
    color_mask = cv2.bitwise_or(mask_yellow, mask_white)
    kernel = np.ones((15, 15), np.uint8)  # Dilate to catch edges near paint
    color_mask_dilated = cv2.dilate(color_mask, kernel, iterations=2)
    
    # Apply color filter to edges
    edges = cv2.bitwise_and(edges, color_mask_dilated)

    # 4. ROI Masking (BALANCED - catch lanes but avoid far edges)
    # Use 18% - 88% width for good coverage
    roi_polygon = np.array([[
        (int(width * 0.18), height),             # Bottom-left
        (int(width * 0.40), int(height * 0.50)), # Top-left
        (int(width * 0.60), int(height * 0.50)), # Top-right
        (int(width * 0.88), height)              # Bottom-right (NARROWER)
    ]], dtype=np.int32)

    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, roi_polygon, 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)

    # 5. Hough Transform (sensitive settings for early detection)
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,       # Lower threshold for sensitivity
        minLineLength=20,   # Shorter segments OK
        maxLineGap=200      # Connect broken segments
    )

    # 6. Slope-based Classification
    left_segments = []
    right_segments = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                slope = math.inf
            else:
                slope = (y2 - y1) / (x2 - x1)

            # Filter horizontal lines
            if abs(slope) < 0.4:
                continue
            
            # Filter near-vertical lines (unlikely to be lanes)
            if abs(slope) > 10:
                continue

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            center_x = width / 2
            
            # === VEHICLE EXCLUSION ZONE ===
            # Reject short line segments in the center of the frame
            # These are likely vehicle edges, not lane markings
            vehicle_zone_left = width * 0.30
            vehicle_zone_right = width * 0.70
            vehicle_zone_top = height * 0.50
            
            line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # If line is in center zone AND is short, it's likely a vehicle edge
            if (mid_y > vehicle_zone_top and 
                vehicle_zone_left < mid_x < vehicle_zone_right and
                line_length < height * 0.20):  # Short segments in center = vehicle
                continue  # Skip this line
            # ===============================
            
            # POSITION-BASED classification (fixes slope misclassification issue)
            # Lines on LEFT half of screen = left boundary candidates
            # Lines on RIGHT half of screen = right boundary candidates
            if mid_x < center_x:  # Left side of screen
                left_segments.append((x1, y1, x2, y2))
            else:  # Right side of screen
                right_segments.append((x1, y1, x2, y2))

    # Helper: Average segments
    def average_segments(segments, side):
        # REDUCED requirement for EARLY DETECTION
        # Even 2 segments can indicate a lane when far away
        if not segments or len(segments) < 2:
            return None

        X_coords = []
        Y_coords = []
        for (x1, y1, x2, y2) in segments:
            X_coords += [x1, x2]
            Y_coords += [y1, y2]

        if len(X_coords) < 4:
            return None

        # Robust fit
        try:
            m, b = np.polyfit(X_coords, Y_coords, 1)
        except Exception:
            return None

        # Strict Slope: Real lanes are steep.
        # Ignore lines that are too horizontal (messy noise).
        if math.isinf(m) or abs(m) < 0.6: 
            return None

        # --- EVIDENCE-BASED DRAWING ---
        # Don't extrapolate to the sky. Draw only where we see paint.
        # Bottom is usually the bottom of screen (if close) or max detected Y.
        # Top is the highest point we actually saw.
        
        y_max_evidence = max(Y_coords)
        y_min_evidence = min(Y_coords)
        
        # FILTER: Reject lines that don't reach near the bottom of the frame
        # BALANCED at 65% - catches center lines while filtering noise
        # This filters out vehicle edges, shadows, etc. that appear in the middle
        if y_max_evidence < height * 0.65:
            print(f"[DEBUG-LANE] {side} REJECTED: Line too high (y_max={y_max_evidence}, need >{height * 0.65:.0f})")
            return None
        
        # Clamp to screen
        y_bottom = height
        # If the Evidence starts high up, don't draw from bottom of screen (floating line)
        if y_max_evidence < height * 0.90:
             y_bottom = y_max_evidence
        
        # Stop exactly where evidence stops
        y_top = max(int(height * 0.45), y_min_evidence)

        # Calculate X at bottom and top
        x_bottom = int((y_bottom - b) / m)
        x_top = int((y_top - b) / m)
        
        # FILTER: Reject lines where x_bottom is too close to frame edges
        # BALANCED: 15%-85% - reject far edges but allow center-ish lines
        # Real lane lines should be in the CENTER portion of the road surface
        if x_bottom < width * 0.15 or x_bottom > width * 0.85:
            print(f"[DEBUG-LANE] {side} REJECTED: x_bottom={x_bottom} outside valid range ({width*0.15:.0f}-{width*0.85:.0f})")
            return None
        
        # --- POST-FIT COLOR VERIFICATION ---
        # Sample pixels along the fitted line to verify it's actually a lane (yellow/white)
        # This rejects road edges (brown/gray) and dirt boundaries
        
        num_samples = 7  # Increased from 5 for more robust sampling
        yellow_count = 0
        white_count = 0
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for i in range(num_samples):
            # Sample at different Y positions along the line
            sample_y = int(y_bottom - (y_bottom - y_top) * (i / (num_samples - 1)))
            if m != 0:
                sample_x = int((sample_y - b) / m)
            else:
                sample_x = x_bottom
                
            # Clamp to frame bounds
            sample_x = max(0, min(width - 1, sample_x))
            sample_y = max(0, min(height - 1, sample_y))
            
            # Sample a small patch around the point
            patch_size = 8  # Smaller patch for more precise color check
            y1 = max(0, sample_y - patch_size)
            y2 = min(height, sample_y + patch_size)
            x1 = max(0, sample_x - patch_size)
            x2 = min(width, sample_x + patch_size)
            
            patch_hsv = hsv[y1:y2, x1:x2]
            if patch_hsv.size == 0:
                continue
                
            # Check for yellow (stricter saturation check)
            lower_y = np.array([10, 50, 120])  # Higher sat/val for real paint
            upper_y = np.array([45, 255, 255])
            mask_y = cv2.inRange(patch_hsv, lower_y, upper_y)
            if mask_y.mean() > 15:  # Require >15% yellow pixels
                yellow_count += 1
                
            # Check for white (VERY RELAXED to catch center lines)
            # Use same value (140) as pre-filter to ensure detection
            lower_w = np.array([0, 0, 140])  # Matches pre-filter value
            upper_w = np.array([180, 60, 255])  # Allow more saturation
            mask_w = cv2.inRange(patch_hsv, lower_w, upper_w)
            if mask_w.mean() > 5:  # Very relaxed for thin/faded lines
                white_count += 1
        
        # === SIDE-SPECIFIC VALIDATION ===
        # LEFT lanes can be YELLOW (centerline) or WHITE (edge marking)
        # RIGHT lanes are typically WHITE
        # Require minimum paint detection to filter out non-lane edges
        
        if side == 'right':
            # RIGHT lane MUST have white paint (at least 3 out of 7 samples - REDUCED for thin lines)
            if white_count < 3:
                print(f"[DEBUG-LANE] {side} REJECTED: Right lane needs white paint ({white_count}W < 3)")
                return None
            # RIGHT lane should not be too far left (must be on right side of frame)
            if x_bottom < width * 0.45:
                print(f"[DEBUG-LANE] {side} REJECTED: Right lane x_bottom too far left ({x_bottom} < {width*0.45:.0f})")
                return None
                
        elif side == 'left':
            # LEFT lane can be YELLOW (3+) OR WHITE (4+)
            # This allows detection of both yellow centerlines and white edge markings
            has_yellow = yellow_count >= 3
            has_white = white_count >= 4
            
            if not has_yellow and not has_white:
                print(f"[DEBUG-LANE] {side} REJECTED: Left lane needs yellow(3+) or white(4+) paint ({yellow_count}Y, {white_count}W)")
                return None
            # LEFT lane should not be too far right (must be on left side of frame)
            if x_bottom > width * 0.55:
                print(f"[DEBUG-LANE] {side} REJECTED: Left lane x_bottom too far right ({x_bottom} > {width*0.55:.0f})")
                return None
        
        center_x = width // 2
        print(f"[DEBUG-LANE] {side} ACCEPTED: x_bot={x_bottom} x_top={x_top} (Center={center_x}) Color=({yellow_count}Y, {white_count}W)")

        return (x_bottom, y_bottom, x_top, y_top, m, b)

    left_line = average_segments(left_segments, 'left')
    right_line = average_segments(right_segments, 'right')
    


    # Helper: Classify Solid vs Dashed
    def classify_line(segments, main_line):
        if not segments or not main_line:
            return None, "unknown"

        # Calculate Coverage
        y_vals = []
        for s in segments:
            y_vals.extend(range(min(s[1], s[3]), max(s[1], s[3])))
        
        y_vals = sorted(list(set(y_vals))) 
        
        if not y_vals:
            return None, "unknown"

        covered_height = len(y_vals)
        total_span = abs(main_line[1] - main_line[3])
        
        coverage_ratio = covered_height / max(1.0, total_span)

        # Color sampling
        lx1, ly1, lx2, ly2, _, _ = main_line
        h, w = frame.shape[:2]
        
        sample_y = int(h * 0.85)
        sample_x = int((sample_y - main_line[5]) / main_line[4]) if main_line[4] != 0 else 0
        
        patch_size = 20
        y_start, y_end = max(0, sample_y-patch_size), min(h, sample_y+patch_size)
        x_start, x_end = max(0, sample_x-patch_size), min(w, sample_x+patch_size)
        
        patch = frame[y_start:y_end, x_start:x_end]
        color = _estimate_line_color(patch)

        # Classification Logic
        # Raised threshold to 0.75 to STRICTLY identify solid lines.
        # Any gaps will drop coverage below 75%, forcing "dashed".
        if coverage_ratio > 0.75:
            return "solid", color
        else:
            return "dashed", color

    # Final Classification
    l_type, l_color = classify_line(left_segments, left_line)
    r_type, r_color = classify_line(right_segments, right_line)

    # --- CONTEXT-AWARE COLOR INFERENCE ---
    # Faded yellow lines often show up as "unknown" or "white" in harsh lighting.
    # If we have a SOLID line on the LEFT, it is statistically highly likely to be the Yellow Shoulder/Center.
    if l_type == "solid" and l_color == "unknown":
        l_color = "yellow"
        # print("[DEBUG-LANE] Faded Left Solid Line -> Inferred YELLOW")

    result = {
        'left':  {'type': l_type, 'x_bottom': left_line[0] if left_line else None, 'line_coords': left_line, 'color': l_color},
        'right': {'type': r_type, 'x_bottom': right_line[0] if right_line else None, 'line_coords': right_line, 'color': r_color}
    }

    return result


def draw_lane_lines(frame, lane_info, highlight=None):
    """
    Draw logical lanes on the frame.
    highlight: 'left' or 'right' (draw that line red)
    """
    out = frame.copy()
    
    # Left
    l_info = lane_info['left']
    if l_info['line_coords']:
        x1, y1, x2, y2, m, b = l_info['line_coords']
        color = (0, 0, 255) if highlight == "left" else ((0, 255, 255) if l_info['color'] == 'yellow' else (0, 255, 0))
        thickness = 5 if l_info['type'] == 'solid' else 2
        
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
        
        # Label with Sampling Point visualization
        h, w = frame.shape[:2]
        sample_y = int(h * 0.85)
        if m != 0:
            sample_x = int((sample_y - b) / m)
            cv2.circle(out, (sample_x, sample_y), 5, (0, 0, 255), -1) # Red dot = sample point

        # Label
        text = f"L: {l_info['type']} ({l_info['color']})"
        cv2.putText(out, text, (max(0, x1 - 50), y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Right
    r_info = lane_info['right']
    if r_info['line_coords']:
        x1, y1, x2, y2, m, b = r_info['line_coords']
        color = (0, 0, 255) if highlight == "right" else ((0, 255, 255) if r_info['color'] == 'yellow' else (0, 255, 0))
        thickness = 5 if r_info['type'] == 'solid' else 2
        
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
        
        # Label with Sampling Point visualization
        # Re-calculate sample point for display
        h, w = frame.shape[:2]
        sample_y = int(h * 0.85)
        if m != 0:
            sample_x = int((sample_y - b) / m)
            cv2.circle(out, (sample_x, sample_y), 5, (0, 0, 255), -1) # Red dot = sample point
            
        text = f"R: {r_info['type']} ({r_info['color']})"
        cv2.putText(out, text, (min(frame.shape[1], x1 + 10), y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return out


