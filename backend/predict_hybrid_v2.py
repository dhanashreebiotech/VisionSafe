
import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque, Counter
from ultralytics import YOLO
import math
import os

# --- MODEL DEFINITION ---
class ActivityClassifier(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=64, num_classes=4, num_layers=2):
        super(ActivityClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        x = output[:, -1, :] 
        x = self.dropout2(x)
        x = self.fc(x)
        return x

class VisionSafePredictorV2:
    def __init__(self):
        print("Initializing VisionSafe Enterprise V2 (High Accuracy Mode)...")
        
        # --- GPU SETUP ---
        if torch.cuda.is_available():
            self.device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Detected: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
            print(f"CUDA Version: {torch.version.cuda}")
            # Enable TF32 for faster computation on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            self.device = 'cpu'
            print("WARNING: CUDA not available. Running on CPU (slower).")
            print("To enable GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        
        # --- INFERENCE SETTINGS (Higher accuracy) ---
        self.INFER_SIZE = 832       # Higher resolution = better accuracy (was 640)
        self.POSE_CONF = 0.40       # Person detection confidence
        self.UNSAFE_CONF = 0.20     # Lower initial threshold, filter later
        self.IOU_THRESH = 0.45      # NMS IoU threshold
        self.MAX_DET = 50           # Max detections per frame
        
        # --- YOLO POSE MODEL (People detection + keypoints) ---
        # Priority: yolov8s-pose > yolov8m-pose > yolov8n-pose (larger = more accurate)
        pose_model_priority = [
            os.path.join(os.path.dirname(__file__), "yolov8s-pose.pt"),
            os.path.join(os.path.dirname(__file__), "yolov8m-pose.pt"),
            "yolov8s-pose.pt",
            "yolov8m-pose.pt",
            os.path.join(os.path.dirname(__file__), "yolov8n-pose.pt"),
            "yolov8n-pose.pt",
        ]
        
        self.yolo_pose = None
        for mp in pose_model_priority:
            if os.path.exists(mp):
                try:
                    self.yolo_pose = YOLO(mp)
                    self.yolo_pose.to(self.device)
                    print(f"Loaded Pose Model: {mp}")
                    break
                except Exception as e:
                    print(f"Failed to load {mp}: {e}")
        
        if not self.yolo_pose:
            # Download yolov8s-pose for better accuracy (small model, good balance)
            print("Downloading yolov8s-pose.pt for higher accuracy...")
            try:
                self.yolo_pose = YOLO('yolov8s-pose.pt')
                self.yolo_pose.to(self.device)
            except:
                print("Falling back to yolov8n-pose.pt...")
                self.yolo_pose = YOLO('yolov8n-pose.pt')
                self.yolo_pose.to(self.device)
        
        # --- ACTIVITY MODEL (LSTM) ---
        self.act_model = ActivityClassifier(input_dim=51).to(self.device)
        
        model_paths = [
            os.path.join(os.path.dirname(__file__), "training", "activity_model.pth"),
            os.path.join(os.path.dirname(__file__), "activity_model.pth")
        ]
        
        self.activity_enabled = False
        for mp in model_paths:
            if os.path.exists(mp):
                try:
                    self.act_model.load_state_dict(torch.load(mp, map_location=self.device))
                    self.act_model.eval()
                    print(f"Loaded Activity (LSTM) Model: {mp}")
                    self.activity_enabled = True
                    break
                except Exception as e:
                    print(f"Error loading activity model from {mp}: {e}")
        
        if not self.activity_enabled:
            print("WARNING: Activity Model not found. Using Enhanced Rule-Based Fallback.")
            
        # --- UNSAFE MODEL (Fire/Smoke) ---
        self.yolo_unsafe = None
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "models", "fire.pt"),
            "models/fire.pt",
            "fire.pt",
            "fire_smoke.pt"
        ]
        
        for p in possible_paths:
            if os.path.exists(p):
                print(f"Loading Unsafe (Fire/Smoke) Model from: {p}")
                try:
                    self.yolo_unsafe = YOLO(p)
                    self.yolo_unsafe.to(self.device)
                    break
                except Exception as e:
                    print(f"Failed to load {p}: {e}")

        # --- GENERAL OBJECT DETECTOR (for vehicles, etc.) ---
        # Use a separate, better model for vehicle/object detection
        self.yolo_general = None
        general_model_priority = [
            os.path.join(os.path.dirname(__file__), "yolov8s.pt"),
            os.path.join(os.path.dirname(__file__), "yolov8m.pt"),
            "yolov8s.pt",
            "yolov8m.pt",
            os.path.join(os.path.dirname(__file__), "yolov8n.pt"),
            "yolov8n.pt",
        ]
        
        for mp in general_model_priority:
            if os.path.exists(mp):
                try:
                    self.yolo_general = YOLO(mp)
                    self.yolo_general.to(self.device)
                    print(f"Loaded General Object Model: {mp}")
                    break
                except Exception as e:
                    print(f"Failed to load {mp}: {e}")
        
        if not self.yolo_general:
            print("Downloading yolov8s.pt for better object detection...")
            try:
                self.yolo_general = YOLO('yolov8s.pt')
                self.yolo_general.to(self.device)
            except:
                self.yolo_general = YOLO('yolov8n.pt')
                self.yolo_general.to(self.device)

        # If no dedicated fire model, use general model as fallback
        if not self.yolo_unsafe:
            print("WARNING: No dedicated Fire/Smoke model found. Using general model as fallback.")
            self.yolo_unsafe = self.yolo_general
              
        # --- TRACKING STATE ---
        self.tracks = {} 
        self.next_track_id = 0
        self.frame_count = 0
        self.class_map = {0: "Sitting", 1: "Standing", 2: "Walking", 3: "Yoga"}

        # Temporal buffers for unsafe events
        self.fire_history = deque(maxlen=15)
        self.smoke_history = deque(maxlen=15)
        self.fall_history = deque(maxlen=15)
        
        # COCO vehicle class IDs (for yolov8 general model)
        self.VEHICLE_CLASSES = {
            2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
            1: "bicycle"
        }
        
        print(f"VisionSafe V2 initialized on [{self.device.upper()}] | Inference size: {self.INFER_SIZE}px")
        print("=" * 60)

    def _calc_angle_kpt(self, pts, a, b, c):
        if pts[a][2] < 0.5 or pts[b][2] < 0.5 or pts[c][2] < 0.5: return 180.0
        ang = math.degrees(math.atan2(pts[c][1]-pts[b][1], pts[c][0]-pts[b][0]) - math.atan2(pts[a][1]-pts[b][1], pts[a][0]-pts[b][0]))
        return abs(ang) if abs(ang) < 180 else 360 - abs(ang)

    def _classify_pose_rule_based(self, kpts_norm):
        l_hip_ang = self._calc_angle_kpt(kpts_norm, 5, 11, 13)
        r_hip_ang = self._calc_angle_kpt(kpts_norm, 6, 12, 14)
        avg_hip = (l_hip_ang + r_hip_ang) / 2
        
        l_knee_ang = self._calc_angle_kpt(kpts_norm, 11, 13, 15)
        r_knee_ang = self._calc_angle_kpt(kpts_norm, 12, 14, 16)
        avg_knee = (l_knee_ang + r_knee_ang) / 2
        
        # Yoga logic — hands raised above shoulders
        hands_up = False
        if kpts_norm[9][2] > 0.5 and kpts_norm[10][2] > 0.5:
             sho_y = (kpts_norm[5][1] + kpts_norm[6][1]) / 2
             if kpts_norm[9][1] < sho_y and kpts_norm[10][1] < sho_y:
                 hands_up = True
                 
        if avg_hip < 140 and avg_knee < 140: return "Sitting"
        if (hands_up and avg_knee < 160) or (abs(l_knee_ang - r_knee_ang) > 60): return "Yoga"
            
        ankle_dist = abs(kpts_norm[15][0] - kpts_norm[16][0])
        if avg_knee > 160 and avg_hip > 160:
            if ankle_dist > 0.15: return "Walking"
            return "Standing"
            
        if 130 < avg_knee < 160: return "Walking"
        return "Unknown"

    def _calc_iou(self, boxA, boxB):
        """Calculate Intersection over Union between two [x1,y1,x2,y2] boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
    
    def _calc_overlap_ratio(self, boxA, boxB):
        """Calculate how much of boxA is inside boxB (overlap / area_of_A)."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        if boxAArea <= 0: return 0
        return interArea / float(boxAArea)
    
    def _match_tracks(self, detections):
        matched_ids = {} 
        used_tracks = set()
        
        for i, det in enumerate(detections):
            best_iou = 0
            best_tid = -1
            box = det['bbox']
            
            for tid, tdata in self.tracks.items():
                if tid in used_tracks: continue
                prev_box = tdata.get('last_bbox', box)
                iou = self._calc_iou(box, prev_box)
                
                if iou > 0.3 and iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid != -1:
                matched_ids[i] = best_tid
                used_tracks.add(best_tid)
            else:
                self.tracks[self.next_track_id] = {
                    'buffer': deque(maxlen=32),
                    'last_seen': self.frame_count,
                    'last_bbox': box,
                    'label_hist': deque(maxlen=15)
                }
                matched_ids[i] = self.next_track_id
                self.next_track_id += 1
                
        # cleanup stale tracks
        dead = []
        for tid, tdata in self.tracks.items():
            if self.frame_count - tdata['last_seen'] > 30:
                dead.append(tid)
        for d in dead: del self.tracks[d]
        
        return matched_ids

    def predict_frame(self, frame):
        self.frame_count += 1
        h, w, _ = frame.shape
        
        # --- CONFIDENCE THRESHOLDS (tuned for accuracy) ---
        CONF_THRESH = {
            "fire": 0.35, "smoke": 0.30, "fall": 0.45, 
            "vehicle": 0.35, "person": 0.35, "default": 0.40
        }

        detections_out = []
        person_activities = []
        person_boxes = []  # Track person bounding boxes
        vehicle_boxes = [] # Track vehicle bounding boxes
        fall_detected_now = False
        fire_detected_now = False
        smoke_detected_now = False

        # ====================================================
        # STEP 1: General Object Detection (Vehicles, etc.)
        # Run this FIRST so we can use vehicle info for person classification
        # ====================================================
        if self.yolo_general:
            try:
                gen_res = self.yolo_general(
                    frame, 
                    verbose=False, 
                    conf=CONF_THRESH["vehicle"],
                    imgsz=self.INFER_SIZE,
                    iou=self.IOU_THRESH,
                    max_det=self.MAX_DET,
                    half=True if self.device == 'cuda' else False,
                    agnostic_nms=True
                )[0]
                
                for box in gen_res.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = [int(x) for x in box.xyxy[0].tolist()]
                    
                    # Check if it's a vehicle class in COCO
                    if cls_id in self.VEHICLE_CLASSES:
                        label_name = self.VEHICLE_CLASSES[cls_id]
                        
                        # Only add if confidence is high enough
                        if conf >= CONF_THRESH["vehicle"]:
                            vehicle_boxes.append({
                                'bbox': xyxy,
                                'label': label_name,
                                'confidence': conf
                            })
                            detections_out.append({
                                "label": "vehicle",
                                "sub_label": label_name,
                                "confidence": conf,
                                "bbox": xyxy,
                                "type": "unsafe"
                            })
                            
            except Exception as e:
                print(f"General Detection Error: {e}")

        # ====================================================
        # STEP 2: YOLO Pose (People Detection + Keypoints)
        # ====================================================
        try:
            pose_res = self.yolo_pose(
                frame, 
                verbose=False, 
                conf=CONF_THRESH["person"],
                imgsz=self.INFER_SIZE,
                iou=self.IOU_THRESH,
                max_det=self.MAX_DET,
                half=True if self.device == 'cuda' else False
            )[0]
            
            current_dets = []
            if pose_res.boxes:
                for i, box in enumerate(pose_res.boxes):
                    if int(box.cls[0]) == 0:  # Person class only
                        xyxy = [int(x) for x in box.xyxy[0].tolist()]
                        kpts = pose_res.keypoints.data[i].cpu().numpy()
                        current_dets.append({'bbox': xyxy, 'kpts': kpts, 'conf': float(box.conf[0])})

            matches = self._match_tracks(current_dets)
            
            for i, det in enumerate(current_dets):
                tid = matches[i]
                track = self.tracks[tid]
                track['last_seen'] = self.frame_count
                track['last_bbox'] = det['bbox']
                
                kpts = det['kpts']
                kpts_norm = kpts.copy()
                kpts_norm[:, 0] /= w
                kpts_norm[:, 1] /= h
                
                if self.activity_enabled:
                    feats = kpts_norm.flatten()
                    track['buffer'].append(feats)
                
                # === VEHICLE-PERSON OVERLAP CHECK ===
                # If person is on/near a vehicle, classify as "Riding" not "Walking/Standing"
                is_on_vehicle = False
                overlapping_vehicle = None
                person_box = det['bbox']
                
                for veh in vehicle_boxes:
                    veh_box = veh['bbox']
                    # Check if person overlaps significantly with vehicle
                    overlap = self._calc_overlap_ratio(person_box, veh_box)
                    iou = self._calc_iou(person_box, veh_box)
                    
                    # Person is "on" vehicle if:
                    # 1. High overlap (person bbox is mostly inside vehicle bbox), OR
                    # 2. Significant IoU and person center is near vehicle
                    person_center_x = (person_box[0] + person_box[2]) / 2
                    person_center_y = (person_box[1] + person_box[3]) / 2
                    
                    in_vehicle_x = veh_box[0] <= person_center_x <= veh_box[2]
                    in_vehicle_y = veh_box[1] <= person_center_y <= veh_box[3]
                    
                    if overlap > 0.25 or iou > 0.15 or (in_vehicle_x and in_vehicle_y):
                        is_on_vehicle = True
                        overlapping_vehicle = veh
                        break
                
                # --- FALL DETECTION LOGIC ---
                b_w = det['bbox'][2] - det['bbox'][0]
                b_h = det['bbox'][3] - det['bbox'][1]
                aspect_ratio = b_w / (b_h + 1e-6)
                
                is_falling = False
                if not is_on_vehicle:  # Don't detect fall if person is on vehicle
                    if aspect_ratio > 1.4:  # Stricter threshold (was 1.2)
                        is_falling = True
                    elif kpts_norm[0][2] > 0.5 and (kpts_norm[11][2] > 0.5 or kpts_norm[12][2] > 0.5):
                        hip_y = (kpts_norm[11][1] + kpts_norm[12][1]) / 2
                        if kpts_norm[0][1] > hip_y + 0.05:  # Head clearly below hips (with margin)
                            is_falling = True

                label = "Unknown"
                conf = det['conf']
                
                if is_on_vehicle:
                    # Person is on/near a vehicle → classify as Riding
                    veh_type = overlapping_vehicle['label'] if overlapping_vehicle else 'vehicle'
                    label = f"Riding ({veh_type.capitalize()})"
                elif is_falling:
                    label = "FALL"
                    if conf >= CONF_THRESH["fall"]:
                        fall_detected_now = True
                elif self.activity_enabled and len(track['buffer']) >= 32:
                    # LSTM prediction
                    try:
                        seq = np.array(list(track['buffer'])) 
                        inp = torch.tensor(seq).float().unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            logits = self.act_model(inp)
                            probs = torch.softmax(logits, dim=1)
                            max_prob = torch.max(probs).item()
                            idx = torch.argmax(probs, dim=1).item()
                        
                        # Only use LSTM if confidence is high enough
                        if max_prob > 0.4:
                            label = self.class_map.get(idx, "Unknown")
                        else:
                            label = self._classify_pose_rule_based(kpts_norm)
                    except:
                        label = self._classify_pose_rule_based(kpts_norm)
                else:
                    label = self._classify_pose_rule_based(kpts_norm)
                    
                track['label_hist'].append(label)
                
                # Temporal Smoothing (weighted: recent labels matter more)
                if len(track['label_hist']) >= 3:
                    # Use last 5 frames for smoothing
                    recent = list(track['label_hist'])[-5:]
                    c = Counter(recent)
                    label = c.most_common(1)[0][0]
                    
                person_activities.append(label)
                person_boxes.append(det['bbox'])
                
                # Determine type
                det_type = "safe"
                if label == "FALL":
                    det_type = "unsafe"
                elif "Riding" in label:
                    det_type = "info"
                
                detections_out.append({
                    "label": label.lower() if label not in ["Unknown", "FALL"] and "Riding" not in label else (label.lower() if label == "FALL" else label.lower()),
                    "confidence": float(conf),
                    "bbox": det['bbox'],
                    "type": det_type,
                    "track_id": tid
                })
        except Exception as e:
            print(f"Pose Error: {e}")

        self.fall_history.append(fall_detected_now)

        # ====================================================
        # STEP 3: Fire/Smoke Detection (dedicated model)
        # ====================================================
        if self.yolo_unsafe and self.yolo_unsafe != self.yolo_general:
            try:
                u_res = self.yolo_unsafe(
                    frame, 
                    verbose=False, 
                    conf=self.UNSAFE_CONF,
                    imgsz=self.INFER_SIZE,
                    iou=self.IOU_THRESH,
                    half=True if self.device == 'cuda' else False
                )[0]
                
                for box in u_res.boxes:
                    c = int(box.cls[0])
                    conf = float(box.conf[0])
                    label_name = self.yolo_unsafe.names[c].lower()
                    xyxy = [int(x) for x in box.xyxy[0].tolist()]
                    
                    is_unsafe_obj = False
                    
                    if "fire" in label_name:
                        label_name = "fire"
                        if conf >= CONF_THRESH["fire"]:
                            fire_detected_now = True
                            is_unsafe_obj = True
                    elif "smoke" in label_name:
                        label_name = "smoke"
                        if conf >= CONF_THRESH["smoke"]:
                            smoke_detected_now = True
                            is_unsafe_obj = True
                    
                    if is_unsafe_obj:
                         detections_out.append({
                            "label": label_name,
                            "confidence": conf,
                            "bbox": xyxy,
                            "type": "unsafe" 
                        })
            except Exception as e:
                print(f"Fire/Smoke Detection Error: {e}")

        self.fire_history.append(fire_detected_now)
        self.smoke_history.append(smoke_detected_now)

        # ====================================================
        # STEP 4: Temporal Logic & Status Determination
        # ====================================================
        status = "SAFE"
        main_activity = "None"
        reason = "Normal activity detected"
        
        def check_consecutive(hist, min_frames=2):
            """Check if last N frames all had detection."""
            if len(hist) < min_frames: return False
            return all(hist[-i] for i in range(1, min_frames + 1))

        is_fire_confirmed = check_consecutive(self.fire_history, 2)
        is_smoke_confirmed = check_consecutive(self.smoke_history, 2)
        is_fall_confirmed = check_consecutive(self.fall_history, 2)
        
        # Priority: Fire > Smoke > Fall > Vehicle > Riding > Person Activity
        if is_fire_confirmed or fire_detected_now:
            status = "UNSAFE"
            main_activity = "Fire"
            reason = "Fire detected in video stream" if is_fire_confirmed else "Possible fire detected"
        elif is_smoke_confirmed or smoke_detected_now:
            status = "UNSAFE"
            main_activity = "Smoke"
            reason = "Smoke detected in video stream"
        elif is_fall_confirmed or fall_detected_now:
            status = "UNSAFE"
            main_activity = "Fall"
            reason = "Person fall detected (abnormal posture)"
        else:
            # Check for vehicles
            if vehicle_boxes:
                # Check if any person is riding
                riding_activities = [a for a in person_activities if "Riding" in a]
                if riding_activities:
                    status = "UNSAFE"
                    main_activity = riding_activities[0]
                    reason = f"Person detected riding vehicle"
                else:
                    status = "UNSAFE"
                    main_activity = "Vehicle"
                    reason = "Vehicle detected near safety zone"
            else:
                # Check people
                if person_activities:
                    c = Counter(person_activities)
                    top_act = c.most_common(1)[0][0]
                    
                    if top_act == "FALL":
                        status = "UNSAFE"
                        main_activity = "Fall"
                    elif "Riding" in top_act:
                        status = "UNSAFE"
                        main_activity = top_act
                        reason = f"Person detected riding vehicle"
                    else:
                        status = "SAFE"
                        main_activity = top_act
                    
                    if "FALL" not in top_act and "Riding" not in top_act:
                        reason = f"Person detected: {main_activity}"
                else:
                    status = "SAFE"
                    main_activity = "None"
                    reason = "No activity detected"

        # ====================================================
        # STEP 5: Final Output Formatting
        # ====================================================
        max_conf = 0.0
        main_act_lower = main_activity.lower()
        relevant_dets = [d for d in detections_out if d['label'].lower() == main_act_lower]
        if not relevant_dets and "fire" in main_act_lower:
            relevant_dets = [d for d in detections_out if "fire" in d['label'].lower()]
        if not relevant_dets and "riding" in main_act_lower:
            relevant_dets = [d for d in detections_out if "riding" in d['label'].lower()]
        
        if relevant_dets:
            max_conf = max(d['confidence'] for d in relevant_dets)
        elif detections_out:
            max_conf = max(d['confidence'] for d in detections_out)

        return {
            "status": status,
            "activity": main_activity.capitalize() if "Riding" not in main_activity else main_activity,
            "confidence": f"{int(max_conf * 100)}%",
            "detections": detections_out,
            "bounding_boxes": [
                {
                    "class": d['label'],
                    "confidence": f"{d['confidence']:.2f}",
                    "box": d['bbox']
                } for d in detections_out
            ],
            "reason": reason,
            
            # Legacy/Compat Support
            "is_unsafe": status == "UNSAFE", 
            "activities": [main_activity.upper()] if main_activity != "None" else [],
            "frame_size": {"w": w, "h": h}
        }

    def predict_video(self, video_path, frames_to_sample=16):
        """Analyze video by sampling frames. Uses more frames for better accuracy."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return None
        
        step = max(1, total_frames // frames_to_sample)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= frames_to_sample: break
        cap.release()
        
        if not frames: return None
        
        # Analyze all frames
        results = []
        for f in frames:
            results.append(self.predict_frame(f))
            
        # Aggregation: Return UNSAFE if ANY frame was unsafe
        final_status = "SAFE"
        final_activity = "None"
        max_conf = 0.0
        unsafe_reasons = []
        
        def parse_conf(s):
            try:
                return float(s.strip('%')) / 100.0
            except:
                return 0.0

        for r in results:
            if r["status"] == "UNSAFE":
                final_status = "UNSAFE"
                unsafe_reasons.append(r["reason"])
            
            c_val = parse_conf(r["confidence"])
            if c_val > max_conf:
                max_conf = c_val
                final_activity = r["activity"]
        
        return {
            "status": final_status,
            "activity": final_activity,
            "confidence": f"{int(max_conf * 100)}%",
            "detections": results[-1]["detections"],
            "bounding_boxes": results[-1]["bounding_boxes"],
            "reason": "; ".join(list(set(unsafe_reasons))) if unsafe_reasons else "Normal activity",
            "meta": {"source_type": "video", "frames_analyzed": len(frames)},
            
            "is_unsafe": final_status == "UNSAFE",
            "activities": [final_activity.upper()]
        }
