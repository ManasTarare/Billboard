# Billboard Ad Replacement — Local Pipeline

Four scripts, run in order.

## 0. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install PyTorch with CUDA support FIRST (pick your CUDA version):
# https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then everything else:
pip install -r requirements.txt

# Verify GPU is detected:
python -c "import torch; print(torch.cuda.is_available())"
```

## 1. Prepare the dataset

Put your Roboflow-exported zip in this folder, then:

```bash
python 1_prepare_dataset.py --zip your-dataset.zip
```

This extracts it into `dataset/{train,valid,test}/{images,labels}` and rewrites
`dataset/data.yaml` with correct absolute paths. It also validates that every
image has a matching label file.

## 2. Train the model

```bash
python 2_train_model.py --model yolov8m.pt --epochs 150 --batch 16
```

Adjust `--batch` down (e.g. 8) if you hit CUDA out-of-memory on your GPU.
Best weights are copied to `models/billboard_best.pt`. Full logs, loss curves,
and all checkpoints are under `runs/billboard_yolov8m/`.

## 3. Evaluate

```bash
python 3_evaluate_model.py --split test
```

Prints mAP@0.5, mAP@0.5:0.95, precision, recall, F1, and real inference speed
(ms/image, FPS) measured on your own hardware. Saves a JSON summary plus
confusion matrix / PR curve plots under `metrics/`.

## 4. Test it — upload a scene + ad, get the composited result

```bash
streamlit run 4_test_replace.py
```

Opens a browser UI. Upload a scene photo containing a billboard and a custom
ad image. It detects the billboard, warps your ad onto it, and shows/downloads
the result.

---

## Known limitation (by design, for now)

The compositing step uses the YOLO detection box directly as the target
region for the perspective warp. That means:

- **Works well**: billboards that are roughly front-on or mildly angled in
  the photo.
- **Looks less natural**: billboards viewed from a steep angle, since a box
  is axis-aligned and can't capture true skew.

The natural next upgrade is contour-based quad fitting inside the detected
box (find the actual 4 corners of the billboard surface) instead of using
the box corners as-is. Worth doing once you confirm detection accuracy is
solid — no point refining the warp on a model that isn't finding billboards
reliably yet.
