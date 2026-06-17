
import uvicorn
import shutil
import os
import uuid
import traceback
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any

# Use VisionSafe Hybrid Fast V2
try:
    from predict_hybrid_v2 import VisionSafePredictorV2
except ImportError:
    # Fallback if running from root or different structure
    try:
        from backend.predict_hybrid_v2 import VisionSafePredictorV2
    except:
        print("CRITICAL: Could not import VisionSafePredictorV2")
        VisionSafePredictorV2 = None

app = FastAPI(title="VisionSafe Enterprise API", version="2.0")

# --- CORS & MIDDLEWARE ---
origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:5176",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel
from datetime import datetime

# --- GLOBALS ---
predictor = None
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
DETECTION_HISTORY = []

# --- MODELS ---
class LoginRequest(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class DetectionRecord(BaseModel):
    id: str
    timestamp: str
    source: str
    safety_status: str
    activity: str
    confidence: str # Changed to str to support "XX%" format
    reason: Optional[str] = None

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        if VisionSafePredictorV2:
            predictor = VisionSafePredictorV2()
            print("VisionSafe V2 Model Loaded Successfully")
        else:
            print("VisionSafe V2 Class unavailable.")
    except Exception as e:
        print(f"Failed to load V2 model: {e}")
        traceback.print_exc()

@app.get("/")
def root():
    return health()

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "predictor_available": predictor is not None
    }

@app.post("/auth/login", response_model=Token)
def login(creds: LoginRequest):
    # predefined admin/password
    if creds.email == "admin@visionsafe.ai" and creds.password == "password":
        return {"access_token": f"fake-jwt-token-{uuid.uuid4()}", "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/detections", response_model=List[DetectionRecord])
def get_detections():
    # Return last 50, reversed (newest first)
    return list(reversed(DETECTION_HISTORY[-50:]))

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Handle Image OR Video file uploads.
    Returns ONE summary result.
    """
    global predictor
    
    # Default Safe Response
    response_payload = {
        "status": "SAFE",
        "activity": "Unknown",
        "confidence": "0%",
        "bounding_boxes": [],
        "detections": [],
        "reason": "Initializing...",
        "meta": {"source_type": "unknown", "frames_analyzed": 0, "error": None}
    }
    
    if not predictor:
        response_payload["meta"]["error"] = "Predictor not initialized"
        return JSONResponse(status_code=503, content=response_payload)
    
    fpath = None
    try:
        # Save Upload
        ext = file.filename.split('.')[-1].lower()
        fid = str(uuid.uuid4())
        fpath = os.path.join(UPLOAD_DIR, f"{fid}.{ext}")
        
        with open(fpath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        is_video = ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']
        response_payload["meta"]["source_type"] = "video" if is_video else "image"
        
        if is_video:
            # Analyze video (N frames)
            result = predictor.predict_video(fpath, frames_to_sample=12)
            if result:
                response_payload.update(result)
        else:
            # Analyze image
            frame = cv2.imread(fpath)
            if frame is None:
                raise ValueError("Could not decode image")
                
            raw = predictor.predict_frame(frame)
            
            # Update strictly from predictor result
            response_payload.update(raw)
            if "meta" not in response_payload: response_payload["meta"] = {}
            response_payload["meta"]["frames_analyzed"] = 1

        # LOG TO HISTORY
        conf_val = response_payload.get("confidence", "0%")
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "source": response_payload["meta"]["source_type"],
            "safety_status": response_payload.get("status", "SAFE"),
            "activity": response_payload.get("activity", "Unknown"),
            "confidence": str(conf_val),
            "reason": response_payload.get("reason", "")
        }
        DETECTION_HISTORY.append(record)

        return response_payload

    except Exception as e:
        print(f"Detect Error: {e}")
        traceback.print_exc()
        response_payload["meta"]["error"] = str(e)
        return response_payload
        
    finally:
        # Cleanup upload
        if fpath and os.path.exists(fpath):
            try:
                os.remove(fpath)
            except:
                pass

@app.get("/debug/sample_test")
def debug_sample_test():
    """
    Returns a sample fake detection to verify input/output contract without ML.
    """
    return {
        "status": "UNSAFE",
        "activity": "Fire",
        "confidence": "98%",
        "bounding_boxes": [
            {"class": "fire", "confidence": "0.98", "box": [100, 100, 200, 200]}
        ],
        "reason": "Detection based on visible flame",
        "meta": {"source_type": "generated", "frames_analyzed": 1}
    }

@app.post("/detect_frame")
async def detect_frame(frame: UploadFile = File(...)):
    """
    Handle single frame from Live Monitor or Video Playback.
    """
    global predictor
    
    start_ts = cv2.getTickCount()
    
    # Default response
    response_payload = {
        "status": "SAFE",
        "activity": "Unknown",
        "confidence": "0%",
        "bounding_boxes": [],
        "detections": [],
        "reason": "Processing...",
        "meta": {"fps": 0.0, "latency_ms": 0}
    }

    if not predictor:
        return JSONResponse(status_code=503, content=response_payload)

    try:
        # Read bytes to CV2
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return response_payload

        # Predict
        raw = predictor.predict_frame(img)
        
        # Calculate latency
        end_ts = cv2.getTickCount()
        time_sec = (end_ts - start_ts) / cv2.getTickFrequency()
        fps = 1.0 / time_sec if time_sec > 0 else 0.0
        
        # Update payload from raw result
        response_payload.update(raw)
        
        # Ensure meta exists
        if "meta" not in response_payload: response_payload["meta"] = {}
        response_payload["meta"].update({
            "fps": round(fps, 1),
            "latency_ms": int(time_sec * 1000)
        })
        
        return response_payload

    except Exception as e:
        print(f"Frame Error: {e}")
        return response_payload

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
