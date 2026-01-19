# Dataset Preprocessing & Training Scripts

This folder contains the Python scripts automatically generated to process your video dataset for YOLO training.

## 📂 Folder Structure
- **Scripts/**: Contains all the python code.
- **train/val/test**: Your original video data.

## 🚀 How to Use

### 1. Preparation (`process_existing_splits.py`)
Extracts frames from your videos (1 FPS), resizes them to 640x640, and saves them to `Desktop/YOLO_Dataset`.
```powershell
cd Scripts
python process_existing_splits.py
```

### 2. Annotation (`auto_label.py`)
Automatically detects people in your extracted images and creates YOLO labels for classes like Walking, Standing, Sitting, etc.
**Note:** This requires `yolov8n.pt` and `ultralytics`. If you have the "WinError 1114", use the Colab script instead.
```powershell
cd Scripts
python auto_label.py
```

### 3. Training on Colab (`train_on_colab.py`)
**RECOMMENDED FOR YOU**. Since your local PC has missing DLLs and no GPU, use this.
1. Upload `YOLO_Dataset` (from Desktop) to Google Drive.
2. Open Google Colab.
3. Copy the content of `Scripts/train_on_colab.py` into Colab.
4. Run it to Label AND Train in the cloud.

### 4. Analysis (`analyze_dataset.py`)
Run this to see how many videos you have per class.
```powershell
cd Scripts
python analyze_dataset.py
```
