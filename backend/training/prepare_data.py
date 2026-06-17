import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import GroupShuffleSplit
import torch
from pathlib import Path
import shutil
import argparse

# Config — parse dataset root from command line or use default
parser = argparse.ArgumentParser(description="Prepare pose dataset for activity classification")
parser.add_argument("--dataset", type=str, default=None,
                    help="Path to dataset root containing class folders (Sitting, Standing, Walking, Yoga)")
args, _ = parser.parse_known_args()

DATASET_ROOT = args.dataset or os.environ.get("DATASET_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "safe"))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_artifacts")
SEQ_LEN = 32
STRIDE = 10 
TARGET_FPS = 10
CLASSES = ["Sitting", "Standing", "Walking", "Yoga"]
CLASS_MAP = {
    "Sitting": 0,
    "Standing still": 1,
    "Standing": 1, 
    "Walking": 2,
    "Yoga": 3
}

# Init Models
# Use YOLO Pose instead of MediaPipe
model_pose = YOLO('yolov8n-pose.pt')

def extract_features_from_video(video_path, label_id):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    frame_skip = int(max(1, fps / TARGET_FPS))
    
    features = []
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if count % frame_skip != 0:
            count += 1
            continue
        count += 1
        
        # Detect & Pose
        results = model_pose(frame, verbose=False, conf=0.5)
        if not results or len(results[0].boxes) == 0:
            continue 
            
        # Take largest person
        # results[0].boxes.data -> [x1, y1, x2, y2, conf, cls]
        # results[0].keypoints.data -> [N, 17, 3] (x, y, conf)
        
        # Sort by confidence or area?
        # Let's take the one with highest conf
        best_idx = 0
        best_conf = -1
        for i, box in enumerate(results[0].boxes):
            if box.conf[0] > best_conf:
                best_conf = box.conf[0]
                best_idx = i
                
        # Extract keypoints
        # [17, 3]
        kpts = results[0].keypoints.data[best_idx].cpu().numpy() # x, y, conf
        
        # Normalize:
        # We need these to be invariant to scale/position
        # Method: Center around mid-hip (idx 11=left_hip, 12=right_hip in COCO format? No wait)
        # COCO: 0:nose, 1:Leye, 2:Reye, 3:Lear, 4:Rear, 5:Lsho, 6:Rsho, 7:Lelb, 8:Relb, 9:Lwri, 10:Rwri, 11:Lhip, 12:Rhip, 13:Lkne, 14:Rkne, 15:Lank, 16:Rank
        
        # Center: avg of hips (11, 12)
        mid_hip_x = (kpts[11, 0] + kpts[12, 0]) / 2
        mid_hip_y = (kpts[11, 1] + kpts[12, 1]) / 2
        
        # Scale: Distance between shoulders (5, 6) or hips? Or BBox height?
        # Pose model gives bbox too.
        # Let's use image dims to normalize to [0,1] first?
        # YOLO keypoints are in pixels.
        
        h, w = frame.shape[:2]
        
        # Normalize to [0,1]
        kpts_norm = kpts.copy()
        kpts_norm[:, 0] /= w
        kpts_norm[:, 1] /= h
        
        # Flatten [17, 3] -> 51
        features.append(kpts_norm.flatten())
        
    cap.release()
    return np.array(features)

def prepare_dataset():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    X_all = []
    y_all = []
    groups_all = [] 
    
    video_id_counter = 0
    
    root = Path(DATASET_ROOT)
    for class_name, label_id in CLASS_MAP.items():
        class_dir = root / class_name
        if not class_dir.exists(): continue
        
        print(f"Processing {class_name}...")
        videos = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mkv"))
        videos = videos[:3] # Limit for speed
        
        for vid in videos:
            print(f"  {vid.name}")
            feats = extract_features_from_video(vid, label_id)
            if feats is None or len(feats) < SEQ_LEN:
                continue
                
            num_seqs = (len(feats) - SEQ_LEN) // STRIDE + 1
            for i in range(num_seqs):
                start = i * STRIDE
                end = start + SEQ_LEN
                seq = feats[start:end]
                
                X_all.append(seq)
                y_all.append(label_id)
                groups_all.append(video_id_counter)
            
            video_id_counter += 1

    if len(X_all) == 0:
        print("No data found!")
        return

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.int64)
    groups_all = np.array(groups_all)
    
    print(f"Total Sequences: {len(X_all)}")
    print(f"Class counts: {np.bincount(y_all)}")
    
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(gss.split(X_all, y_all, groups_all))
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_temp, y_temp, groups_temp = X_all[temp_idx], y_all[temp_idx], groups_all[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.66, random_state=42)
    val_sub_idx, test_sub_idx = next(gss2.split(X_temp, y_temp, groups_temp))
    
    X_val, y_val = X_temp[val_sub_idx], y_temp[val_sub_idx]
    X_test, y_test = X_temp[test_sub_idx], y_temp[test_sub_idx]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    np.savez(f"{OUTPUT_DIR}/train.npz", X=X_train, y=y_train)
    np.savez(f"{OUTPUT_DIR}/val.npz", X=X_val, y=y_val)
    np.savez(f"{OUTPUT_DIR}/test.npz", X=X_test, y=y_test)
    
    import json
    with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
        json.dump({"classes": ["Sitting", "Standing", "Walking", "Yoga"], "feature_dim": 51}, f)

if __name__ == "__main__":
    prepare_dataset()
