import os
import sys
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np

def sanitize_json(data):
    if isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(v) for v in data]
    elif isinstance(data, float) or isinstance(data, np.floating):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.ndarray):
        return sanitize_json(data.tolist())
    else:
        return data

# Add project root to sys.path so 'src' can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.inference.pipeline import DeepGuardXInference

# Initialize detector globally so models stay in memory
config_path = os.path.join(PROJECT_ROOT, "configs", "ensemble_config.yaml")

try:
    detector = DeepGuardXInference(
        config_path=config_path,
        use_onnx=True,
        device="cuda",
    )
    print("Detector initialized successfully.")
except Exception as e:
    print(f"Error initializing detector: {e}")
    detector = None

app = FastAPI(title="DeepGuard API")

@app.get("/")
def read_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.post("/api/analyze/")
async def analyze_video(
    file: UploadFile = File(...),
    threshold: float = Form(None)
):
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector failed to initialize")

    print(f"[AnalyzeView] Received threshold: {threshold}")

    # Save to temp file
    suffix = Path(file.filename).suffix
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp_file.name
    tmp_file.close()  # Close the initial file handle to release the lock on Windows
    
    try:
        # FastAPI UploadFile allows async read, or we can use shutil for simplicity
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run inference
        results = detector.predict(video_path=tmp_path, audio_path=None, threshold=threshold)
        print(f"[AnalyzeView] Detector results (raw): {results}")
        
        if "error" in results:
            return JSONResponse(status_code=500, content={"error": results["error"]})
        
        clean_results = sanitize_json(results)
        
        try:
            print(f"[AnalyzeView] Clean results summary: final_score={clean_results.get('final_score')}, threshold={clean_results.get('threshold')}, final_label={clean_results.get('final_label')}")
        except Exception:
            print("[AnalyzeView] Failed to print clean results summary")
            
        return clean_results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
