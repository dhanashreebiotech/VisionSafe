import cv2
import os
import glob
import yaml
import numpy as np
from pathlib import Path

# --- Configuration ---
SOURCE_DATASET = r"c:\Users\N DHANASHREE\OneDrive\Desktop\Dataset"
OUTPUT_DATASET = r"c:\Users\N DHANASHREE\OneDrive\Desktop\Dataset\processed_data"
TARGET_SIZE = 640
EXTRACT_FPS = 1.0  # Safe for low-resource (1 frame per second)

# Strict Class Mapping (Do not change ID order)
# We map folder names to these IDs
CLASS_MAP = {
    "Sitting": 0,
    "Standing still": 1,
    "Walking": 2,
    "Yoga": 3,
    "Fighting": 4,
    "Fire": 5,
    "Smoking": 6,
    "Vehicles": 7
}

def letterbox_resize(img, target_size=640):
    """Resizes image to target_size with padding (black bars) to maintain aspect ratio."""
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Create blank canvas (YOLO grey 114)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    
    # Center
    y_offset = (target_size - nh) // 2
    x_offset = (target_size - nw) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    return canvas

def process_split_folder(split_name):
    """
    Process a specific split folder (train/val/test) independently.
    """
    source_split_path = os.path.join(SOURCE_DATASET, split_name)
    output_img_dir = os.path.join(OUTPUT_DATASET, 'images', split_name)
    
    # Create output directories
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DATASET, 'labels', split_name), exist_ok=True)

    print(f"\n--- Processing Split: {split_name.upper()} ---")
    if not os.path.exists(source_split_path):
        print(f"Skipping {split_name}: Folder not found.")
        return 0

    total_frames_saved = 0
    
    # Walk through split folder -> category (safe/unsafe) -> class -> video
    # We use rglob to find all videos recursively inside the split folder
    videos = list(Path(source_split_path).rglob("*.mp4"))
    
    if not videos:
        print(f"No videos found in {split_name}")
        return 0

    print(f"Found {len(videos)} videos in {split_name}")

    for vid_path in videos:
        # Determine Class Name from parent folder
        # Structure: Dataset/train/safe/Sitting/video.mp4 -> parent.name = Sitting
        class_name = vid_path.parent.name
        
        # Fallback: sometimes structure is different, check grandparent if needed
        if class_name not in CLASS_MAP:
             # Try grandparent (e.g. Dataset/train/Sitting/video.mp4 is unlikely based on logs, but safe to check)
             if vid_path.parent.parent.name in CLASS_MAP:
                 class_name = vid_path.parent.parent.name
        
        if class_name not in CLASS_MAP:
            print(f"Skipping video: {vid_path.name} (Unknown class folder '{class_name}')")
            continue

        # Unique ID for filename: Class_VideoName_Frame.jpg
        # specific cleaning of filename
        clean_vid_name = vid_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
        prefix = f"{class_name}_{clean_vid_name}"
        
        # Processing Video
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened(): continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25 # fallback
        
        frame_interval = int(fps / EXTRACT_FPS)
        if frame_interval < 1: frame_interval = 1
        
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % frame_interval == 0:
                final_img = letterbox_resize(frame, TARGET_SIZE)
                
                out_name = f"{prefix}_f{frame_idx}.jpg"
                out_path = os.path.join(output_img_dir, out_name)
                
                # Quality 85 is good tradeoff
                cv2.imwrite(out_path, final_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved_count += 1
                
            frame_idx += 1
        
        cap.release()
        total_frames_saved += saved_count
        # Optional: Print progress every few videos or just total at end
        
    print(f"Completed {split_name}: {total_frames_saved} frames extracted.")
    return total_frames_saved

def main():
    print("--- Starting Strict Split Processing ---")
    
    # 1. Process each official split folder
    total = 0
    for split in ['train', 'val', 'test']:
        total += process_split_folder(split)
        
    # 2. Generate data.yaml
    # We must list the classes in exact order of 0..N
    sorted_names = [k for k, v in sorted(CLASS_MAP.items(), key=lambda item: item[1])]
    
    yaml_config = {
        'path': OUTPUT_DATASET,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(sorted_names),
        'names': sorted_names
    }
    
    with open(os.path.join(OUTPUT_DATASET, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_config, f, sort_keys=False)
        
    print(f"\nAll Done! Total images: {total}")
    print(f"Config saved to: {os.path.join(OUTPUT_DATASET, 'data.yaml')}")

if __name__ == "__main__":
    main()
