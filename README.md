# VisionSafe Enterprise V2

VisionSafe Enterprise V2 is a state-of-the-art, high-accuracy hybrid AI system designed for real-time industrial safety monitoring, physical security, and activity classification. It combines advanced human pose estimation with a custom temporal deep learning model (Conv1D-BiLSTM) and object detection to identify unsafe situations, fires, and track worker activity.

---

## Key Features

- **Real-Time Webcam & Video Streaming**: Process live camera feeds with low latency and instant bounding-box overlay.
- **Human Activity Recognition**: Classify complex human actions (Sitting, Standing, Walking, Yoga/Tripping) using a custom trained Conv1D-BiLSTM network.
- **Unsafe Hazard Detection**: Detect critical hazards (such as fire, smoke, and unauthorized objects) in real time.
- **GPU Acceleration (CUDA)**: Support for NVIDIA GPUs (utilizing TF32, cuDNN auto-tuning, and batch-first LSTM) for high-performance deployment.
- **Premium User Interface**: Built with a responsive glassmorphic dark-theme UI, interactive charts, and live FPS tracking.

---

## Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.10+)
- **Deep Learning**: PyTorch 2.6+, Ultralytics YOLOv8
- **Data Science**: NumPy, Scikit-learn (for dataset prep)
- **API Server**: Uvicorn

### Frontend
- **Framework**: React 19, Vite 7
- **Styling**: Tailwind CSS v4 (native theme directives)
- **Icons**: Lucide React
- **Routing**: React Router v7

---

## Directory Structure

```
VisionSafe/
├── README.md                    # Project documentation
├── .gitignore                   # Git exclusion rules
├── backend/
│   ├── app.py                   # FastAPI server
│   ├── predict_hybrid_v2.py     # High-accuracy predictor core
│   ├── activity_model.pth       # Custom trained LSTM activity classifier
│   ├── requirements.txt         # Backend Python packages
│   ├── models/
│   │   ├── README_FIRE.txt
│   │   └── fire.pt              # Unsafe hazard/fire detection weights
│   └── training/
│       ├── model.py             # LSTM model structure
│       ├── prepare_data.py      # Data preparation & split
│       └── train_activity.py    # Training pipeline with noise-injection
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── postcss.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css            # Tailwind theme colors & animations
        ├── components/          # Reusable UI widgets
        ├── layouts/             # App structure layouts
        ├── pages/               # Routing page components
        ├── routes/              # Protected routing logic
        └── utils/               # Storage and API utilities
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Node.js (v18+) and npm

### 1. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch for maximum performance:*
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

### 2. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Run the Vite development server:
   ```bash
   npm run dev
   ```
4. Open the displayed URL (usually `http://localhost:5173`) in your browser.

---

## API Endpoints

- **GET `/health`**: Returns backend and predictor availability status.
- **POST `/detect`**: Upload an image or video file to run complete hybrid inference.
- **POST `/detect_frame`**: Process a single frame payload (e.g. from live webcam stream) and return bounding boxes, keypoints, and activity classes.

---

## Model Training

If you want to re-train or fine-tune the human activity classifier:
1. Ensure your training dataset is inside `backend/training/dataset_artifacts/`.
2. Run data preparation:
   ```bash
   python training/prepare_data.py --dataset_path <path_to_dataset>
   ```
3. Train the model:
   ```bash
   python training/train_activity.py
   ```
This will train the Conv1D-BiLSTM model, utilizing early stopping, learning rate scheduling, and data augmentation, and export a fresh `activity_model.pth`.
