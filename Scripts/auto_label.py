from ultralytics import YOLO
import os
import glob
from tqdm import tqdm

# --- Config ---
DATASET_ROOT = r"c:\Users\N DHANASHREE\OneDrive\Desktop\Dataset\processed_data"
CONF_THRESHOLD = 0.4

# Mapping our Folder Classes to COCO Classes (for Auto-Labeling)
# COCO IDs: 0=Person, 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
CLASS_LOGIC = {
    "Sitting":  {"coco_ids": [0], "target_id": 0},
    "Standing still": {"coco_ids": [0], "target_id": 1},
    "Walking":  {"coco_ids": [0], "target_id": 2},
    "Yoga":     {"coco_ids": [0], "target_id": 3},
    "Fighting": {"coco_ids": [0], "target_id": 4},
    # Fire (5) - Auto label not possible with standard COCO
    # Smoking (6) - Auto label not possible
    "Vehicles": {"coco_ids": [2, 3, 5, 7], "target_id": 7},
}

def auto_label():
    print("Loading YOLOv8n model for auto-labeling...")
    model = YOLO('yolov8n.pt')  # Will download automatically
    
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(DATASET_ROOT, 'images', split)
        label_dir = os.path.join(DATASET_ROOT, 'labels', split)
        
        print(f"\nLabeling {split}...")
        images = glob.glob(os.path.join(img_dir, "*.jpg"))
        
        for img_path in tqdm(images):
            # Filename format: ClassName_VideoName_Frame.jpg
            # Parse ClassName
            filename = os.path.basename(img_path)
            class_name = filename.split('_')[0]
            
            # Handling "Standing still" which has a space, but our filename replaced space with _
            # "Standing_still_..."
            if class_name == "Standing":
                class_name = "Standing still"
            
            if class_name not in CLASS_LOGIC:
                # Skip Fire/Smoking or unknown
                continue
            
            logic = CLASS_LOGIC[class_name]
            
            # Inference
            results = model(img_path, verbose=False, conf=CONF_THRESHOLD)
            
            label_lines = []
            if results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    
                    if cls_id in logic['coco_ids']:
                        # Convert to YOLO format: class x_center y_center width height
                        # Normalized 0-1
                        xywhn = box.xywhn[0].tolist()
                        line = f"{logic['target_id']} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}"
                        label_lines.append(line)
            
            # Write label file
            if label_lines:
                label_name = filename.replace(".jpg", ".txt")
                with open(os.path.join(label_dir, label_name), "w") as f:
                    f.write("\n".join(label_lines))

if __name__ == "__main__":
    # Install ultralytics if needed: pip install ultralytics
    auto_label()
