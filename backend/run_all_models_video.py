import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.perception.hazard_detector import HazardDetector
from modules.perception.lane_detector import LaneDetector
from modules.perception.pothole_detector import PotholeDetector
from modules.perception.road_segmenter import RoadSegmenter
from modules.perception.sign_detector import SignDetector
from modules.utils.kalman_filter import RiskKalmanFilter
from sign_detector import YOLOSignDetector


def verify_speed_sign_circle(frame, x1, y1, x2, y2, cls_name, conf=0.5):
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    expand = int(max(bw, bh) * 0.20)
    x1 = max(0, int(x1) - expand)
    y1 = max(0, int(y1) - expand)
    x2 = min(w, int(x2) + expand)
    y2 = min(h, int(y2) + expand)

    if x2 <= x1 or y2 <= y1:
        return False

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 25 or roi_w < 25:
        return True

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity = 1.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0 and area > 100:
            circularity = (4 * 3.14159 * area) / (perimeter * perimeter)

    circ_threshold = 0.25 if conf >= 0.85 else 0.40
    if circularity < circ_threshold:
        return False

    if conf >= 0.85 and circularity >= 0.25:
        return True

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 80, 70]), np.array([12, 255, 255])),
        cv2.inRange(hsv, np.array([165, 80, 70]), np.array([180, 255, 255])),
    )
    red_ratio = cv2.countNonZero(mask_red) / (roi_h * roi_w)

    if circularity >= 0.55:
        if red_ratio < 0.005 and roi_w > 15:
            return False
        return True

    return red_ratio >= 0.02


def load_models(project_root, pothole_confirm_th, speed_conf):
    hazard_model = HazardDetector(model_path=os.path.join(project_root, "models", "hazard_model", "best.pt"))
    pothole_model = PotholeDetector(model_path=os.path.join(project_root, "models", "pothole_model", "best copy.pt"))
    pothole_model._CONFIRM_TH = pothole_confirm_th
    sign_model = SignDetector(model_path=os.path.join(project_root, "models", "sign_model", "bestS.pt"))
    lane_model = LaneDetector(model_path=os.path.join(project_root, "models", "lane_model", "lane_detector.pth"))
    road_model = RoadSegmenter(model_path=os.path.join(project_root, "models", "road_segmentation", "best.pth"))

    speed_model = None
    speed_weights = os.path.join(
        project_root,
        "runs",
        "detect",
        "runs",
        "detect",
        "speed_junction_v1",
        "weights",
        "best.pt",
    )
    if os.path.exists(speed_weights):
        speed_model = YOLOSignDetector(speed_weights, conf=speed_conf, imgsz=640)

    return hazard_model, pothole_model, sign_model, lane_model, road_model, speed_model


def process_video(source, output_video, output_json, pothole_confirm_th=0.25, speed_conf=0.45):
    (
        hazard_model,
        pothole_model,
        sign_model,
        lane_model,
        road_model,
        speed_model,
    ) = load_models(PROJECT_ROOT, pothole_confirm_th, speed_conf)

    if hasattr(pothole_model, "reset"):
        pothole_model.reset()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_src if 1 < fps_src < 120 else 25.0
    out_w, out_h = 960, 540
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_video}")

    alpha = 0.65
    prev_risk = 0.0
    frame_n = 0
    persist_frames = 3
    track_history = defaultdict(lambda: deque(maxlen=persist_frames))
    road_mask_cache = None
    kalman = RiskKalmanFilter()

    bottom_gate = 0.65
    green_th = 0.20
    orange_th = 0.45

    last_hazards = []
    last_potholes = []
    last_signs = []
    last_speed_signs = []
    last_lane_status = {}

    summary = {
        "source_video": os.path.abspath(source),
        "output_video": os.path.abspath(output_video),
        "pothole_confirm_threshold": pothole_confirm_th,
        "frames_processed": 0,
        "total_frames": total_frames,
        "max_counts": {
            "hazards": 0,
            "potholes": 0,
            "signs": 0,
            "speed_or_junction_signs": 0,
        },
        "frames_with": {
            "hazards": 0,
            "potholes": 0,
            "signs": 0,
            "lane_crossing": 0,
        },
        "max_risk": 0.0,
        "risk_level_peak": "LOW",
        "processing_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    risk_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (out_w, out_h))
            frame_n += 1
            h, w = frame.shape[:2]

            if frame_n % 6 == 0 or road_mask_cache is None:
                road_mask_cache = road_model.segment(frame)
            road_mask = road_mask_cache

            if road_mask is not None:
                overlay = frame.copy()
                overlay[road_mask > 0] = (
                    overlay[road_mask > 0] * 0.6
                    + np.array([180, 120, 50], dtype=np.uint8) * 0.4
                ).astype(np.uint8)
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

            if frame_n % 3 == 0:
                hazards = road_model.filter_detections(hazard_model.detect(frame), road_mask, threshold=0.15)
                last_hazards = hazards
            else:
                hazards = last_hazards

            if frame_n % 3 == 0:
                potholes = road_model.filter_detections(pothole_model.detect(frame), road_mask, threshold=0.15)
                last_potholes = potholes
            else:
                potholes = last_potholes

            if frame_n % 2 == 0:
                signs = road_model.filter_detections(sign_model.detect(frame), road_mask, threshold=0.05)
                last_signs = signs
            else:
                signs = last_signs

            if frame_n % 4 == 0 and speed_model is not None:
                raw_speed_signs = speed_model.detect(frame)
                speed_signs = []
                for det in raw_speed_signs:
                    cls_name = det["cls_name"]
                    if cls_name.startswith("Speed-"):
                        x1, y1, x2, y2 = det["xyxy"]
                        if not verify_speed_sign_circle(frame, x1, y1, x2, y2, cls_name, conf=det["conf"]):
                            continue
                    speed_signs.append(det)
                last_speed_signs = speed_signs
            else:
                speed_signs = last_speed_signs

            if frame_n % 3 == 0:
                lane_status = lane_model.detect(frame)
                last_lane_status = lane_status
            else:
                lane_status = last_lane_status

            haz_score = 0.0
            for det in hazards:
                x1, y1, x2, y2 = map(int, det["bbox"])
                raw_bottom = y2 / h
                bottom_position = max(0.0, (raw_bottom - bottom_gate) / (1.0 - bottom_gate))
                bottom_position = min(bottom_position, 1.0)

                key = (x1 // 80, y1 // 80)
                track_history[key].append(bottom_position)
                if len(track_history[key]) < persist_frames:
                    continue

                haz_score = max(haz_score, bottom_position)

                if bottom_position >= orange_th:
                    color = (0, 0, 255)
                    label = "HIGH RISK"
                elif bottom_position >= green_th:
                    color = (0, 165, 255)
                    label = "MEDIUM RISK"
                else:
                    color = (0, 255, 0)
                    label = "LOW RISK"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            pot_score = min(1.0, sum(p["severity"] for p in potholes))
            lane_score = 1.0 if lane_status.get("is_crossing") else 0.0
            raw_risk = min(0.35 * haz_score + 0.35 * pot_score + 0.15 * lane_score + 0.15 * prev_risk, 1.0)
            risk = kalman.update(raw_risk)
            prev_risk = alpha * prev_risk + (1 - alpha) * risk

            if risk > 0.8:
                risk_level, risk_color = "CRITICAL", (0, 0, 255)
            elif risk > 0.6:
                risk_level, risk_color = "HIGH", (0, 80, 255)
            elif risk > 0.3:
                risk_level, risk_color = "MEDIUM", (0, 200, 255)
            else:
                risk_level, risk_color = "LOW", (0, 220, 80)

            for det in potholes:
                x1, y1, x2, y2 = map(int, det["bbox"])
                label = f"Pothole s={det['severity']:.2f} c={det.get('confidence', 0.0):.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            for det in signs:
                x1, y1, x2, y2 = map(int, det["bbox"])
                label = f"{det['subtype']} {det.get('confidence', 0.0):.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
                cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 2)

            if speed_model is not None:
                frame = speed_model.draw(frame, speed_signs)

            frame = lane_model.draw_lanes(frame, lane_status)

            cv2.rectangle(frame, (0, 0), (w, 58), (15, 15, 20), -1)
            info = (
                f"Hazards:{len(hazards)}  Potholes:{len(potholes)}  "
                f"Signs:{len(signs)}  Speed/Junction:{len(speed_signs)}"
            )
            cv2.putText(frame, info, (18, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(
                frame,
                f"Risk:{risk:.2f}  Level:{risk_level}  LaneCross:{bool(lane_status.get('is_crossing'))}",
                (18, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                risk_color,
                2,
            )
            cv2.putText(
                frame,
                f"Pothole confirm threshold: {pothole_confirm_th:.2f}",
                (w - 285, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
            )

            writer.write(frame)

            summary["frames_processed"] = frame_n
            summary["max_counts"]["hazards"] = max(summary["max_counts"]["hazards"], len(hazards))
            summary["max_counts"]["potholes"] = max(summary["max_counts"]["potholes"], len(potholes))
            summary["max_counts"]["signs"] = max(summary["max_counts"]["signs"], len(signs))
            summary["max_counts"]["speed_or_junction_signs"] = max(
                summary["max_counts"]["speed_or_junction_signs"], len(speed_signs)
            )
            if hazards:
                summary["frames_with"]["hazards"] += 1
            if potholes:
                summary["frames_with"]["potholes"] += 1
            if signs or speed_signs:
                summary["frames_with"]["signs"] += 1
            if lane_status.get("is_crossing"):
                summary["frames_with"]["lane_crossing"] += 1
            summary["max_risk"] = max(summary["max_risk"], float(risk))
            if risk_rank[risk_level] > risk_rank[summary["risk_level_peak"]]:
                summary["risk_level_peak"] = risk_level

            if frame_n % 30 == 0:
                print(f"[INFO] Processed {frame_n}/{total_frames or '?'} frames")

    finally:
        cap.release()
        writer.release()

    summary["processing_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run all road-safety models on a video and save annotated output.")
    parser.add_argument("--source", required=True, help="Path to source video")
    parser.add_argument("--output-video", required=True, help="Path to annotated output video")
    parser.add_argument("--output-json", required=True, help="Path to JSON summary")
    parser.add_argument("--pothole-confirm-th", type=float, default=0.25, help="Lower value = more pothole detections")
    parser.add_argument("--speed-conf", type=float, default=0.45, help="Confidence for speed/junction sign model")
    args = parser.parse_args()

    summary = process_video(
        source=args.source,
        output_video=args.output_video,
        output_json=args.output_json,
        pothole_confirm_th=args.pothole_confirm_th,
        speed_conf=args.speed_conf,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
