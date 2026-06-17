import cv2
import numpy as np
import math
from collections import deque, Counter
from ultralytics import YOLO
import mediapipe as mp
import os

# --- Heuristic Configuration ---
FIRE_HSV_LOWER = np.array([0, 100, 200]) # Example: Yellow-ish/Red-ish
FIRE_HSV_UPPER = np.array([30, 255, 255])
FIRE_AREA_THRESH = 300 # Pixels

FIGHTING_PROXIMITY_RATIO = 0.15 # if dist < 15% of width -> fighting candidate

# Safe Import for MediaPipe to prevent crashes
try:
    mp_pose = mp.solutions.pose
except AttributeError:
    try:
        import mediapipe.python.solutions.pose as mp_pose
    except ImportError:
        print("CRITICAL: MediaPipe Pose not found. Disabling Safe Pose features.")
        mp_pose = None

class VisionSafePredictor:
    def __init__(self):
        print("Initializing VisionSafe Hybrid Engine (Fast Mode)...")
        # Load YOLO (downloads if needed) - Auto Safe/Unsafe Object Detection
        self.yolo = YOLO('yolov8n.pt') 
        
        # Load Pose
        self.pose = None
        if mp_pose:
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
        # Temporal Smoothing
        self.history = deque(maxlen=5) # Last 5 frames
        self.last_clean_activity = "Scanning..."

    def _calc_angle(self, a, b, c):
        """Angle at B"""
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return abs(ang) if abs(ang) < 180 else 360 - abs(ang)

    def _get_pose_label(self, landmarks):
        """Rule-based Safe Action Classification"""
        if not landmarks: return "UNKNOWN"
        
        # Landmarks map (MediaPipe Pose)
        # 11,12 shoulder | 23,24 hip | 25,26 knee | 27,28 ankle | 15,16 wrist
        pts = {i: (lm.x, lm.y) for i, lm in enumerate(landmarks.landmark)}
        
        # Heuristics
        # 1. Hands Up (Yoga/Surrender?) -> Wrist Y < Shoulder Y (0 is top)
        hands_up = (pts[15][1] < pts[11][1]) or (pts[16][1] < pts[12][1])
        
        # 2. Leg Angles
        left_knee_ang = self._calc_angle(pts[23], pts[25], pts[27])
        right_knee_ang = self._calc_angle(pts[24], pts[26], pts[28])
        avg_knee = (left_knee_ang + right_knee_ang) / 2
        
        # 3. Hip Angles
        left_hip_ang = self._calc_angle(pts[11], pts[23], pts[25])
        right_hip_ang = self._calc_angle(pts[12], pts[24], pts[26])
        
        if hands_up:
            return "YOGA"
            
        if avg_knee < 140 or (left_hip_ang < 130 or right_hip_ang < 130):
            # Bent knees/hips -> Sitting
            return "SITTING"
            
        # Differentiate Standing vs Walking
        # Walking implies asymmetry in legs usually
        if abs(left_knee_ang - right_knee_ang) > 25:
            return "WALKING"
            
        return "STANDING"

    def predict(self, frame):
        """
        Main Pipeline:
        1. Fire Check (HSV)
        2. YOLO Detect (Person, Vehicle) -> If Vehicle -> Unsafe
        3. Fighting Check (If >1 Person + Proximity) -> Unsafe
        4. Pose Check (If Person + Safe) -> Safe Activity
        """
        h_img, w_img, _ = frame.shape
        
        # Default Output State
        result = {
            "safety_status": "SAFE",
            "activity": "Scanning...",
            "category": "unknown",
            "confidence": 0.0,
            "frame_size": {"w": w_img, "h": h_img},
            "detections": [],
            "debug": {}
        }
        
        # --- 1. FIRE DETECTION ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_fire = cv2.inRange(hsv, FIRE_HSV_LOWER, FIRE_HSV_UPPER)
        cnts, _ = cv2.findContours(mask_fire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            if cv2.contourArea(c) > FIRE_AREA_THRESH:
                x, y, w, h = cv2.boundingRect(c)
                result["detections"].append({
                    "label": "fire",
                    "confidence": 0.95,
                    "bbox": {"x1": x, "y1": y, "x2": x+w, "y2": y+h},
                    "kind": "object"
                })
                # Critical Priority
                result["safety_status"] = "UNSAFE"
                result["activity"] = "FIRE"
                result["category"] = "fire"
                result["confidence"] = 0.95
                return result # Return immediately on fire

        # --- 2. YOLO OBJECT DETECTION ---
        yolo_res = self.yolo(frame, verbose=False)[0]
        person_boxes = []
        
        # Filter classes
        UNSAFE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'knife', 'scissors']
        
        for box in yolo_res.boxes:
            cls_id = int(box.cls[0])
            label = self.yolo.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            bbox = {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3]}

            # Check Vehicle / Weapon
            if label in UNSAFE_CLASSES:
                result["detections"].append({
                    "label": label,
                    "confidence": conf,
                    "bbox": bbox,
                    "kind": "object"
                })
                # If unhandled priority (so far usually vehicle is high unsafe)
                if result["safety_status"] == "SAFE":
                    result["safety_status"] = "UNSAFE"
                    result["activity"] = "VEHICLE"
                    result["category"] = "vehicle"
                    result["confidence"] = conf
            
            elif label == 'person':
                person_boxes.append(bbox)
        
        # If already unsafe due to vehicle, we can return or continue to add people.
        # Let's continue to detections but keep Unsafe status if set.
        
        if result["safety_status"] == "UNSAFE":
             # Still add persons for context
            for pb in person_boxes:
                result["detections"].append({"label": "person", "confidence": 0.8, "bbox": pb, "kind": "object"})
            return result

        # --- 3. FIGHTING DETECTION (Proximity) ---
        if len(person_boxes) >= 2:
            centers = [((b['x1']+b['x2'])/2, (b['y1']+b['y2'])/2) for b in person_boxes]
            fighting_found = False
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = math.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1])
                    if dist < (w_img * FIGHTING_PROXIMITY_RATIO):
                        fighting_found = True
                        break
                if fighting_found: break
            
            if fighting_found:
                result["safety_status"] = "UNSAFE"
                result["activity"] = "FIGHTING"
                result["category"] = "fighting"
                result["confidence"] = 0.85
                for pb in person_boxes:
                    result["detections"].append({"label": "fighting_person", "confidence": 0.85, "bbox": pb, "kind": "object"})
                return result

        # --- 4. SAFE POSE CLASSIFICATION ---
        current_pose_act = "scanning"
        pose_conf = 0.0

        if self.pose:
            # Process Frame for Pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)
            
            if res.pose_landmarks:
                current_pose_act = self._get_pose_label(res.pose_landmarks)
                pose_conf = 0.9
                
                # Use Pose BBox (min/max of landmarks)
                xs = [lm.x for lm in res.pose_landmarks.landmark]
                ys = [lm.y for lm in res.pose_landmarks.landmark]
                
                # Add padding
                bx1, bx2 = min(xs)*w_img, max(xs)*w_img
                by1, by2 = min(ys)*h_img, max(ys)*h_img
                
                result["detections"].append({
                    "label": "person_pose",
                    "confidence": 0.9,
                    "bbox": {"x1": bx1-20, "y1": by1-20, "x2": bx2+20, "y2": by2+20},
                    "kind": "pose"
                })
            else:
                # Fallback to YOLO person box if pose fail
                if person_boxes:
                    current_pose_act = "STANDING" # Default
                    pose_conf = 0.6
                    result["detections"].append({
                        "label": "person", 
                        "confidence": 0.6, 
                        "bbox": person_boxes[0], 
                        "kind": "object"
                    })
        else:
            # No MediaPipe -> Fallback
             if person_boxes:
                current_pose_act = "PERSON_DETECTED"
                result["detections"].append({
                    "label": "person", 
                    "confidence": 0.5, 
                    "bbox": person_boxes[0], 
                    "kind": "object"
                })

        # --- 5. SMOOTHING ---
        if current_pose_act not in ["scanning", "UNKNOWN"]:
            self.history.append(current_pose_act)
        
        final_act = current_pose_act
        if len(self.history) > 0:
            final_act = Counter(self.history).most_common(1)[0][0]

        if person_boxes or (self.pose and res.pose_landmarks):
            result["activity"] = final_act
            result["category"] = "pose"
            result["confidence"] = pose_conf
        else:
            result["activity"] = "Scanning..."
            
        return result
