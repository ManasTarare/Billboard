# Billboard Ad Replacement — FastAPI + Render

This replaces the old Streamlit app (`test.py` / `4_test_replace.py`) with a
FastAPI backend and a plain HTML/JS frontend, deployable on Render as a
standard web service.

## What changed

| Before | After |
|---|---|
| `test.py` / `4_test_replace.py` (Streamlit) | `app.py` (FastAPI) + `static/index.html`, `static/style.css`, `static/script.js` |
| Deployed on Streamlit Community Cloud | Deployed on Render as a Python web service |
| `requirements.txt` included `streamlit` | `requirements.txt` is web-only (fastapi/uvicorn); training deps moved to `requirements-training.txt` |

`1_prepare_dataset.py`, `2_train_model.py`, and `3_evaluate_model.py` are
unchanged — you still run those locally/on a GPU machine to produce
`models/billboard_best.pt`, same as before.

## Project layout

```
app.py                     FastAPI app (detection + compositing API, serves the frontend)
static/
  index.html                Frontend markup
  style.css                 Styling
  script.js                 Upload/detect/composite flow
models/
  billboard_best.pt          Trained weights (see "Getting your model onto Render" below)
requirements.txt            Web service deps (installed by Render)
requirements-training.txt   Training pipeline deps (install locally, not on Render)
render.yaml                 Render service definition
1_prepare_dataset.py
2_train_model.py
3_evaluate_model.py
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000

## API

- `GET /api/health` — `{"status": "ok", "model_loaded": true|false}`
- `POST /api/detect` — form fields: `scene` (file), `conf` (float). Returns detected boxes + an annotated preview image (base64 PNG).
- `POST /api/composite` — form fields: `scene` (file), `ad` (file), `x1`, `y1`, `x2`, `y2` (the chosen box), `fit_shape` (bool), `blend_edges` (bool). Returns the composited image + a preview showing the detection box and warp shape (base64 PNGs).

## Deploying on Render

1. Push this repo to GitHub/GitLab.
2. In Render: **New +** → **Web Service** → connect the repo. Render will
   pick up `render.yaml` automatically (or set these manually):
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Deploy.

### Getting your model onto Render

`models/billboard_best.pt` is a binary checkpoint, often too large to
comfortably commit to git. You have two options:

- **Commit it anyway** (fine if it's under a couple hundred MB and you're
  okay with it living in git history) — just make sure it's at
  `models/billboard_best.pt` in the repo.
- **Host it externally** (S3, Cloudflare R2, a GitHub Release asset, etc.)
  and set the `MODEL_URL` environment variable in Render's dashboard.
  `app.py` downloads it to `models/billboard_best.pt` automatically on
  startup if the file isn't already present.

### Notes on Render's free plan

- Free web services spin down after inactivity and cold-start on the next
  request — the first detection after idle time will be slower (model
  reload).
- CPU-only inference: `requirements.txt` installs the CPU build of
  `torch`/`torchvision`. Expect slower inference than a local GPU; fine for
  occasional single-image use, less fine for high traffic.
- If you need consistent low-latency inference, use a paid instance type
  with more CPU, or move inference to a GPU-backed host.
