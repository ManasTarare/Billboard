"""
4_test_replace.py
--------------------
Streamlit app: upload a scene image containing a billboard + a custom ad
image, detect the billboard with your trained model, and composite the
custom ad onto it with a perspective warp so it matches the billboard's
angle in the scene.

USAGE:
    streamlit run 4_test_replace.py

Requires: models/billboard_best.pt to exist (run 2_train_model.py first).
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "billboard_best.pt"

st.set_page_config(page_title="Billboard Ad Replacement", layout="wide")


# ---------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


# ---------------------------------------------------------------------------
# Core CV pipeline
# ---------------------------------------------------------------------------
def detect_billboards(model: YOLO, image_bgr: np.ndarray, conf_threshold: float):
    """Run YOLO detection, return list of boxes as (x1, y1, x2, y2, confidence)."""
    results = model.predict(image_bgr, conf=conf_threshold, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
    return boxes


def box_to_quad(box, image_shape):
    """
    Convert an axis-aligned YOLO box into a quadrilateral (4 corner points)
    that we can perspective-warp the ad image onto.

    This is a straightforward version: since YOLO only gives us an
    axis-aligned box (not the true skewed corners of the billboard surface),
    we use the box corners directly. This gives a correct, front-on-looking
    replacement. It will NOT correct for a billboard that is heavily angled
    away from the camera — that would require a dedicated corner/keypoint
    model or contour-based quad fitting on top of the box (a natural next
    upgrade, not included here to keep this pipeline dependable and simple).
    """
    x1, y1, x2, y2, _ = box
    h, w = image_shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    quad = np.array([
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
    ], dtype=np.float32)
    return quad


def warp_and_composite(scene_bgr: np.ndarray, ad_bgr: np.ndarray, quad: np.ndarray, blend_edge: bool = True):
    """
    Perspective-warp the ad image onto the quad region of the scene, then
    composite it in. Uses a soft-edge mask to blend seams naturally instead
    of a hard cut.
    """
    h_scene, w_scene = scene_bgr.shape[:2]
    h_ad, w_ad = ad_bgr.shape[:2]

    src_pts = np.array([
        [0, 0],
        [w_ad, 0],
        [w_ad, h_ad],
        [0, h_ad],
    ], dtype=np.float32)

    homography_matrix = cv2.getPerspectiveTransform(src_pts, quad)

    warped_ad = cv2.warpPerspective(
        ad_bgr, homography_matrix, (w_scene, h_scene),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )

    # Build a mask of the warped region (white where the ad now covers)
    mask = np.zeros((h_scene, w_scene), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)

    if blend_edge:
        # Feather the mask edges slightly so the composite doesn't look
        # like a hard paste — a few px of gaussian blur on the mask border.
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0

    composite = (warped_ad.astype(np.float32) * mask_3ch
                 + scene_bgr.astype(np.float32) * (1 - mask_3ch))
    composite = composite.astype(np.uint8)

    return composite


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.title("Billboard Ad Replacement")
    st.caption("Detects billboards in a scene image and replaces them with your custom ad.")

    if not MODEL_PATH.exists():
        st.error(
            f"No trained model found at `{MODEL_PATH}`.\n\n"
            "Run these first, in order:\n"
            "1. `python 1_prepare_dataset.py`\n"
            "2. `python 2_train_model.py`\n"
            "3. `python 3_evaluate_model.py` (optional, but recommended)"
        )
        st.stop()

    model = load_model(str(MODEL_PATH))

    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)
        blend_edges = st.checkbox("Feather edges for smoother blend", value=True)
        st.divider()
        st.caption(
            "Note: this uses the raw detection box as the ad's target region. "
            "It looks best on billboards that are close to front-on in the "
            "scene photo. Steeply angled billboards will look less natural "
            "since this version doesn't yet fit a true skewed quad."
        )

    col1, col2 = st.columns(2)
    with col1:
        scene_file = st.file_uploader(
            "Scene image (contains the billboard)", type=["jpg", "jpeg", "png"]
        )
    with col2:
        ad_file = st.file_uploader(
            "Custom ad image (to place on the billboard)", type=["jpg", "jpeg", "png"]
        )

    if not scene_file or not ad_file:
        st.info("Upload both a scene image and an ad image to continue.")
        return

    scene_pil = Image.open(scene_file)
    ad_pil = Image.open(ad_file)
    scene_bgr = pil_to_bgr(scene_pil)
    ad_bgr = pil_to_bgr(ad_pil)

    st.subheader("Inputs")
    c1, c2 = st.columns(2)
    c1.image(scene_pil, caption="Scene", use_container_width=True)
    c2.image(ad_pil, caption="Custom ad", use_container_width=True)

    with st.spinner("Detecting billboards..."):
        boxes = detect_billboards(model, scene_bgr, conf_threshold)

    if not boxes:
        st.warning(
            "No billboard detected above the current confidence threshold. "
            "Try lowering the threshold in the sidebar, or use a clearer scene image."
        )
        return

    st.success(f"Detected {len(boxes)} billboard(s).")

    # If multiple billboards are found, let the user choose which one to replace.
    if len(boxes) > 1:
        options = [
            f"Detection {i+1}  (confidence {b[4]:.2f}, box=({b[0]},{b[1]})-({b[2]},{b[3]}))"
            for i, b in enumerate(boxes)
        ]
        chosen_idx = st.selectbox("Multiple billboards found — pick one to replace:",
                                   range(len(options)), format_func=lambda i: options[i])
    else:
        chosen_idx = 0
        st.caption(f"Detection confidence: {boxes[0][4]:.2f}")

    chosen_box = boxes[chosen_idx]
    quad = box_to_quad(chosen_box, scene_bgr.shape)

    with st.spinner("Compositing ad onto billboard..."):
        result_bgr = warp_and_composite(scene_bgr, ad_bgr, quad, blend_edge=blend_edges)

    # Draw detection box on a preview copy for transparency about what was detected
    preview_bgr = scene_bgr.copy()
    x1, y1, x2, y2, conf = chosen_box
    cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

    st.subheader("Result")
    r1, r2 = st.columns(2)
    r1.image(bgr_to_pil(preview_bgr), caption="Detected region (green box)", use_container_width=True)
    r2.image(bgr_to_pil(result_bgr), caption="Final composited output", use_container_width=True)

    result_pil = bgr_to_pil(result_bgr)
    import io
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button(
        "Download result",
        data=buf.getvalue(),
        file_name="billboard_replaced.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
