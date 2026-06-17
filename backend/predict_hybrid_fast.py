import cv2
import numpy as np
import math
from collections import Counter
from ultralytics import YOLO
import mediapipe as mp
import os

# --- Configuration ---
# Path to Fire Model (Download and place here)
FIRE_MODEL_PATH = os.getenv("VISIONSAFE_FIRE_MODEL", os.path.join("weights", "fire_smoke.pt"))
FIRE_CONF_THRESH = 0.35

# Safe Activity Rules
ANGLE_THRESH_SITTING = 140

# MediaPipe Setup
try:
    mp_pose = mp.solutions.pose
except:
    mp_pose = None

class VisionSafePredictor:
    def __init__(self):
        print("Initializing VisionSafe Hybrid Fast V2 (YOLO Fire + Pose)...")
        
        # 1. Main Object Detection (Person, Vehicles)
        # Downloads automatically if not present
        self.yolo_coco = YOLO('yolov8n.pt') 
        
        # 2. Fire/Smoke Detection
        self.yolo_fire = None
        if os.path.exists(FIRE_MODEL_PATH):
            print(f"Loading Fire Model from {FIRE_MODEL_PATH}...")
            try:
                self.yolo_fire = YOLO(FIRE_MODEL_PATH)
            except Exception as e:
                print(f"ERROR loading fire model: {e}")
                self.yolo_fire = None
        else:
            print(f"WARNING: Fire model not found at {FIRE_MODEL_PATH}. Fire detection DISABLED.")
            
        # 3. Pose Keypoints
        self.pose = None
        if mp_pose:
            self.pose = mp_pose.Pose(
                static_image_mode=True, 
                model_complexity=1,
                min_detection_confidence=0.5
            )

    def _calc_angle(self, a, b, c):
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return abs(ang) if abs(ang) < 180 else 360 - abs(ang)

    def _classify_pose(self, landmarks):
        if not landmarks: return "UNKNOWN"
        pts = {i: (lm.x, lm.y) for i, lm in enumerate(landmarks.landmark)}
        
        required = [11, 12, 15, 16, 23, 24, 25, 26, 27, 28]
        if any(i not in pts for i in required): return "UNKNOWN"

        hands_up = (pts[15][1] < pts[11][1]) or (pts[16][1] < pts[12][1])
        if hands_up: return "YOGA"

        left_leg = self._calc_angle(pts[23], pts[25], pts[27])
        right_leg = self._calc_angle(pts[24], pts[26], pts[28])
        avg_leg = (left_leg + right_leg) / 2
        
        left_hip = self._calc_angle(pts[11], pts[23], pts[25])
        right_hip = self._calc_angle(pts[12], pts[24], pts[26])
        avg_hip = (left_hip + right_hip) / 2

        if avg_leg < ANGLE_THRESH_SITTING or avg_hip < ANGLE_THRESH_SITTING: return "SITTING"
        
        leg_diff = abs(left_leg - right_leg)
        if leg_diff > 30: return "WALKING"
            
        return "STANDING"

    def predict_frame(self, frame):
        h_img, w_img, _ = frame.shape
        detections = []
        is_unsafe = False
        detected_activities = []
        
        # --- 1. FIRE DETECT (YOLO) ---
        if self.yolo_fire:
            # Classes: assume model trained with 'fire', 'smoke'
            fire_res = self.yolo_fire(frame, verbose=False)[0]
            for box in fire_res.boxes:
                conf = float(box.conf[0])
                if conf >= FIRE_CONF_THRESH:
                    xyxy = [int(x) for x in box.xyxy[0].tolist()]
                    cls_id = int(box.cls[0])
                    # Try to get label name, default to 'fire'
                    label_name = self.yolo_fire.names.get(cls_id, 'fire').lower()
                    
                    if "smoke" in label_name: label_name = "smoke"
                    else: label_name = "fire"
                    
                    is_unsafe = True
                    detected_activities.append("FIRE")
                    detections.append({
                        "label": label_name,
                        "confidence": conf,
                        "bbox": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
                        "kind": "hazard"
                    })

        # --- 2. OBJECT DETECT (COCO) ---
        coco_res = self.yolo_coco(frame, verbose=False)[0]
        person_boxes = []
        # COCO IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
        UNSAFE_IDS = [2, 3, 5, 7] 
        
        for box in coco_res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [int(x) for x in box.xyxy[0].tolist()]
            label = self.yolo_coco.names[cls_id]

            if cls_id == 0: # person
                person_boxes.append(xyxy)
            elif cls_id in UNSAFE_IDS:
                is_unsafe = True
                detected_activities.append("VEHICLE")
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]},
                    "kind": "hazard"
                })

        # --- 3. POSE DETECT ---
        if self.pose and person_boxes:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)
            
            if res.pose_landmarks:
                action = self._classify_pose(res.pose_landmarks)
                detected_activities.append(action)
                
                xs = [lm.x for lm in res.pose_landmarks.landmark]
                ys = [lm.y for lm in res.pose_landmarks.landmark]
                bx1, bx2 = int(min(xs)*w_img), int(max(xs)*w_img)
                by1, by2 = int(min(ys)*h_img), int(max(ys)*h_img)
                
                detections.append({
                    "label": action,
                    "confidence": 0.9,
                    "bbox": {"x1": max(0, bx1), "y1": max(0, by1), "x2": min(w_img, bx2), "y2": min(h_img, by2)},
                    "kind": "person"
                })
            else:
                # YOLO person fallback
                for pb in person_boxes:
                    detections.append({
                       "label": "person", 
                       "confidence": 0.5,
                       "bbox": {"x1": pb[0], "y1": pb[1], "x2": pb[2], "y2": pb[3]},
                       "kind": "person"
                    })
                    detected_activities.append("STANDING")

        return {
            "is_unsafe": is_unsafe,
            "activities": detected_activities,
            "detections": detections,
            "frame_size": {"w": w_img, "h": h_img}
        }

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 30
        
        # Sample 5 frames
        sample_indices = np.linspace(0, total_frames-1, 5, dtype=int)
        results = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                results.append(self.predict_frame(frame))
        cap.release()
        
        if not results: return None
            
        unsafe_count = sum(1 for r in results if r['is_unsafe'])
        is_unsafe_final = (unsafe_count >= 1)
        
        all_acts = []
        for r in results: all_acts.extend(r['activities'])
        
        # Priority: Fire > Vehicle > Fighting > Safe
        hazards = [a for a in all_acts if a in ["FIRE", "VEHICLE"]]
        if hazards:
            final_activity = Counter(hazards).most_common(1)[0][0]
        else:
            safe_acts = [a for a in all_acts if a not in ["FIRE", "VEHICLE", "UNKNOWN", "PERSON"]]
            final_activity = Counter(safe_acts).most_common(1)[0][0] if safe_acts else "UNKNOWN"

        # Best frame for detections
        best_res = results[0]
        for r in results:
            if r['is_unsafe']:
                best_res = r
                break
        
        return {
            "safety_status": "UNSAFE" if is_unsafe_final else "SAFE",
            "activity": final_activity,
            "confidence": 0.9,
            "detections": best_res['detections'],
            "frame_size": best_res['frame_size']
        }
