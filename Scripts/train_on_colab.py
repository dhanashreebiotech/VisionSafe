# --- INSTRUCTIONS FOR GOOGLE COLAB TRAINING ---
# 1. Upload your "YOLO_Dataset" folder to Google Drive root directory.
# 2. Open https://colab.research.google.com/
# 3. Create a new Notebook.
# 4. Copy-paste this entire script into a cell.
# 5. Runtime -> Change runtime type -> T4 GPU (Important!).
# 6. Run the cell.

import os
import glob
from google.colab import drive
from ultralytics import YOLO
import yaml
from tqdm import tqdm

# 1. Mount Drive
drive.mount('/content/drive')

# Paths
DATASET_ROOT = '/content/drive/MyDrive/YOLO_Dataset'

if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f"Please upload YOLO_Dataset to {DATASET_ROOT}")

# --- STEP 2: AUTO-LABELING (Since local PC failed) ---
print("\n--- Starting Auto-Labeling on Cloud ---")

# Define mapping (Same as before)
CLASS_LOGIC = {
    "Sitting":  {"coco_ids": [0], "target_id": 0},
    "Standing still": {"coco_ids": [0], "target_id": 1},
    "Walking":  {"coco_ids": [0], "target_id": 2},
    "Yoga":     {"coco_ids": [0], "target_id": 3},
    "Fighting": {"coco_ids": [0], "target_id": 4},
    "Vehicles": {"coco_ids": [2, 3, 5, 7], "target_id": 7},
}

model_label = YOLO('yolov8n.pt')

for split in ['train', 'val', 'test']:
    img_dir = os.path.join(DATASET_ROOT, 'images', split)
    label_dir = os.path.join(DATASET_ROOT, 'labels', split)
    
    # Ensure label dir exists
    os.makedirs(label_dir, exist_ok=True)
    
    images = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"Labeling {split} ({len(images)} images)...")
    
    for img_path in tqdm(images):
        filename = os.path.basename(img_path)
        # Parse class (Sitting_video_...)
        class_name = filename.split('_')[0]
        if class_name == "Standing": class_name = "Standing still"
        
        if class_name not in CLASS_LOGIC: continue
        
        logic = CLASS_LOGIC[class_name]
        
        # Check if label already exists (don't overwrite if you restart)
        txt_name = filename.replace(".jpg", ".txt")
        txt_path = os.path.join(label_dir, txt_name)
        if os.path.exists(txt_path): continue
        
        results = model_label(img_path, verbose=False, conf=0.4)
        
        if results[0].boxes:
            with open(txt_path, "w") as f:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in logic['coco_ids']:
                         xywhn = box.xywhn[0].tolist()
                         line = f"{logic['target_id']} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n"
                         f.write(line)

print("Auto-labeling Complete!")

# --- STEP 3: TRAINING ---
print("\n--- Starting Training ---")

yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')

# Fix YAML for Colab paths
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
data['path'] = DATASET_ROOT
data['train'] = 'images/train'
data['val'] = 'images/val'
data['test'] = 'images/test'

with open('colab_config.yaml', 'w') as f:
    yaml.dump(data, f)

# Train using GPU
model_train = YOLO('yolov8n.pt')
results = model_train.train(
    data='colab_config.yaml',
    epochs=20, 
    imgsz=640,
    batch=16,
    device=0 
)
print("Training Complete!")
