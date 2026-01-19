import os
import cv2
import glob
from pathlib import Path

# Config
DATASET_ROOT = r"c:\Users\N DHANASHREE\OneDrive\Desktop\Dataset"
VIDEO_EXT = "*.mp4"

def analyze_dataset(root_path):
    print(f"Analyzing dataset at: {root_path}")
    
    # Classes are strictly defined based on your folder structure
    # We look for the lowest level folders that contain videos
    all_videos = []
    class_counts = {}
    
    # Walk through the directory
    for path in Path(root_path).rglob(VIDEO_EXT):
        # Parent folder is the class (e.g., 'Walking', 'Fire')
        # Grandparent is split (e.g., 'safe', 'unsafe') - we might need to flatten this
        
        # Assumption: Structure is Split/Category/Class/Video OR Split/Class/Video
        # Let's use the immediate parent folder as the specific Activity Class
        class_name = path.parent.name
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
        all_videos.append(str(path))

    print("\n--- Video Distribution (Current) ---")
    for cls, count in class_counts.items():
        print(f"Class '{cls}': {count} videos")
        
    print(f"\nTotal Videos Found: {len(all_videos)}")
    
    # Check a few random videos for resolution/fps
    if all_videos:
        print("\n--- Sample Video Properties ---")
        sample_vid = all_videos[0]
        cap = cv2.VideoCapture(sample_vid)
        if cap.isOpened():
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            print(f"Sample: {os.path.basename(sample_vid)}")
            print(f"Resolution: {int(width)}x{int(height)}")
            print(f"FPS: {fps}")
            print(f"Duration: {duration:.2f} seconds")
            cap.release()
        else:
            print(f"Error: Could not open sample video {sample_vid}")

if __name__ == "__main__":
    analyze_dataset(DATASET_ROOT)
