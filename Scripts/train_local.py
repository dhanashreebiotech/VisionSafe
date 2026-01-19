from ultralytics import YOLO
import os

# --- Configuration ---
DATASET_ROOT = r"c:\Users\N DHANASHREE\OneDrive\Desktop\Dataset\processed_data"
YAML_PATH = os.path.join(DATASET_ROOT, 'data.yaml')

def train_local():
    print("--- Starting Local Training ---")
    print(f"Dataset Config: {YAML_PATH}")

    if not os.path.exists(YAML_PATH):
        print(f"Error: {YAML_PATH} not found. Did you run process_existing_splits.py?")
        return

    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # We use batch=8 to be safe on local memory. Increase to 16 if safe.
    # imgsz=640 is standard.
    results = model.train(
        data=YAML_PATH,
        epochs=10,
        imgsz=640,
        batch=8,
        name='yolo_activity_det',
        exist_ok=True # overwrite existing run if exists
    )
    
    print("Training Complete.")
    print(f"Results saved to: {results.save_dir}")

if __name__ == "__main__":
    train_local()
