import os
import sys
import tempfile
from pathlib import Path
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.inference.pipeline import DeepGuardXInference

# Initialize detector globally so models stay in memory
# Assuming run from backend/ dir, so configs/ensemble_config.yaml is at ../configs/ensemble_config.yaml
# We should specify absolute path for safety
config_path = os.path.join(PROJECT_ROOT, "configs", "ensemble_config.yaml")

try:
    detector = DeepGuardXInference(
        config_path=config_path,
        use_onnx=True,
        device="cuda",
    )
except Exception as e:
    print(f"Error initializing detector: {e}")
    detector = None

class AnalyzeView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        if detector is None:
            return Response({"error": "Detector failed to initialize"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save to temp file
        suffix = Path(file_obj.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            for chunk in file_obj.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name

        # Extract optional threshold
        threshold = request.data.get('threshold')
        if threshold is not None:
            try:
                threshold = float(threshold)
            except ValueError:
                threshold = None
        # Log the threshold for debugging
        print(f"[AnalyzeView] Received threshold: {threshold}")

        try:
            results = detector.predict(video_path=tmp_path, audio_path=None, threshold=threshold)
            # Log the raw results for debugging
            print(f"[AnalyzeView] Detector results (raw): {results}")
        except Exception as e:
            results = {"error": str(e)}
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if "error" in results:
            return Response(results, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        clean_results = sanitize_json(results)
        # Log the cleaned results summary
        try:
            print(f"[AnalyzeView] Clean results summary: final_score={clean_results.get('final_score')}, threshold={clean_results.get('threshold')}, final_label={clean_results.get('final_label')}")
        except Exception:
            print("[AnalyzeView] Failed to print clean results summary")
        return Response(clean_results, status=status.HTTP_200_OK)
