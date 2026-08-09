import base64
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "billboard_best.pt"
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI(title="Billboard Ad Replacement")

# Model is loaded lazily on first request / on startup so the process can
# still boot (and serve /api/health) even if the weights aren't present yet.
_model = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def download_model_if_needed() -> None:
    """
    If models/billboard_best.pt isn't in the repo (common — weight files are
    often too large / awkward to commit to git) and a MODEL_URL env var is
    set, download it on startup. Otherwise this is a no-op.
    """
    if MODEL_PATH.exists():
        return
    model_url = os.environ.get("MODEL_URL")
    if not model_url:
        return

    print(f"MODEL_URL set — downloading model weights to {MODEL_PATH} ...")
    import requests

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(model_url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print("Model download complete.")


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No trained model found at {MODEL_PATH}. Run the training "
                    "pipeline (1_prepare_dataset.py -> 2_train_model.py) and "
                    "commit/upload models/billboard_best.pt, or set MODEL_URL."
                ),
            )
        from ultralytics import YOLO

        print(f"Loading model from {MODEL_PATH} ...")
        try:
            _model = YOLO(str(MODEL_PATH))
        except Exception as e:
            # Handle PyTorch 2.6+ weights_only security issue
            print(f"Standard load failed: {e}")
            print("Attempting load with safe_globals workaround...")
            import torch
            from ultralytics.nn.tasks import DetectionModel
            
            torch.serialization.add_safe_globals([DetectionModel])
            _model = YOLO(str(MODEL_PATH))
    return _model


@app.on_event("startup")
def on_startup():
    download_model_if_needed()
    # Don't hard-fail startup if weights are missing — /api/health will
    # report it, and the frontend shows a clear message instead of a 500.
    if MODEL_PATH.exists():
        try:
            get_model()
        except Exception as e:  # pragma: no cover
            print(f"Warning: model failed to load at startup: {e}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def read_upload_as_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
    return img


def bgr_to_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Detection / quad-fitting / compositing (ported from test.py)
# ---------------------------------------------------------------------------
def detect_billboards(model, image_bgr: np.ndarray, conf_threshold: float):
    results = model.predict(image_bgr, conf=conf_threshold, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
    return boxes


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def box_to_quad(box, image_shape) -> np.ndarray:
    x1, y1, x2, y2, _ = box
    h, w = image_shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def find_billboard_quad(scene_bgr: np.ndarray, box, padding_ratio: float = 0.06):
    x1, y1, x2, y2, _ = box
    h, w = scene_bgr.shape[:2]
    box_w, box_h = x2 - x1, y2 - y1
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(w, x2 + pad_x)
    crop_y2 = min(h, y2 + pad_y)

    crop = scene_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    fallback_quad = box_to_quad(box, scene_bgr.shape)

    if crop.size == 0:
        return fallback_quad, True

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_quad, True

    crop_area = crop.shape[0] * crop.shape[1]
    best_quad = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.15 * crop_area or area > 0.98 * crop_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best_quad = approx

    if best_quad is None:
        return fallback_quad, True

    quad = order_points(best_quad.astype(np.float32))
    quad[:, 0] += crop_x1
    quad[:, 1] += crop_y1
    return quad, False


def warp_and_composite(scene_bgr, ad_bgr, quad, blend_edge: bool = True):
    h_scene, w_scene = scene_bgr.shape[:2]
    h_ad, w_ad = ad_bgr.shape[:2]

    src_pts = np.array([[0, 0], [w_ad, 0], [w_ad, h_ad], [0, h_ad]], dtype=np.float32)
    homography_matrix = cv2.getPerspectiveTransform(src_pts, quad)
    warped_ad = cv2.warpPerspective(
        ad_bgr, homography_matrix, (w_scene, h_scene),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    mask = np.zeros((h_scene, w_scene), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    if blend_edge:
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    composite = (warped_ad.astype(np.float32) * mask_3ch
                 + scene_bgr.astype(np.float32) * (1 - mask_3ch))
    return composite.astype(np.uint8)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/api/detect")
async def api_detect(scene: UploadFile = File(...), conf: float = Form(0.25)):
    model = get_model()
    scene_bytes = await scene.read()
    scene_bgr = read_upload_as_bgr(scene_bytes)

    boxes = detect_billboards(model, scene_bgr, conf)

    preview = scene_bgr.copy()
    for i, (x1, y1, x2, y2, c) in enumerate(boxes):
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(preview, f"#{i+1} {c:.2f}", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    h, w = scene_bgr.shape[:2]
    return JSONResponse({
        "boxes": [
            {"index": i, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "confidence": b[4]}
            for i, b in enumerate(boxes)
        ],
        "preview": bgr_to_data_url(preview),
        "width": w,
        "height": h,
    })


@app.post("/api/composite")
async def api_composite(
    scene: UploadFile = File(...),
    ad: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
    fit_shape: bool = Form(True),
    blend_edges: bool = Form(True),
):
    scene_bytes = await scene.read()
    ad_bytes = await ad.read()
    scene_bgr = read_upload_as_bgr(scene_bytes)
    ad_bgr = read_upload_as_bgr(ad_bytes)

    box = (int(x1), int(y1), int(x2), int(y2), 1.0)

    if fit_shape:
        quad, used_fallback = find_billboard_quad(scene_bgr, box)
    else:
        quad, used_fallback = box_to_quad(box, scene_bgr.shape), True

    result_bgr = warp_and_composite(scene_bgr, ad_bgr, quad, blend_edge=blend_edges)

    preview = scene_bgr.copy()
    cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.polylines(preview, [quad.astype(np.int32)], isClosed=True, color=(0, 140, 255), thickness=3)

    return JSONResponse({
        "result": bgr_to_data_url(result_bgr),
        "preview": bgr_to_data_url(preview),
        "used_fallback": used_fallback,
    })


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
