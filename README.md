# Billboard Ad Replacement Tool

A FastAPI-based web application that detects billboards in images using YOLO and enables seamless ad replacement with perspective-aware compositing.

---

## 🎯 Features

- **Billboard Detection**: Uses YOLOv8 to automatically detect billboards in uploaded images
- **Perspective-Aware Compositing**: Warps and blends replacement ads onto detected billboards with proper perspective transformation
- **Edge Blending**: Smooth blending at billboard boundaries for realistic results
- **Shape Fitting**: Automatic contour detection to refine billboard boundaries
- **Web Interface**: Clean, intuitive UI for detection preview and ad compositing
- **REST API**: Full API endpoints for programmatic access
- **Model Auto-Download**: Automatic model weight download from cloud storage via environment variables

---

## 📋 Prerequisites

- Python 3.8+
- pip or conda
- ~4GB disk space (for model weights)
- CUDA-capable GPU (optional, but recommended for faster inference)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd billboard-replacement
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Model Weights

**Option A: Provide model file directly**
```bash
mkdir -p models
# Copy billboard_best.pt to models/billboard_best.pt
```

**Option B: Set environment variable for auto-download**
```bash
export MODEL_URL="https://your-storage-url/billboard_best.pt"
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MODEL_URL` | Cloud URL to download model weights if not found locally | No |
| `YOLO_CONFIG_DIR` | Directory for YOLO cache (default: `/tmp/Ultralytics`) | No |

**Example:**
```bash
export MODEL_URL="https://s3.amazonaws.com/my-bucket/billboard_best.pt"
```

---

## 🏃 Running the Application

### Development (with auto-reload)
```bash
uvicorn app:app --reload
```

### Production
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Custom Port
```bash
uvicorn app:app --port 8080
```

The app will be available at:
- **Web UI**: `http://localhost:8000`
- **Health Check**: `http://localhost:8000/api/health`

---

## 📡 API Endpoints

### Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_file_present": true
}
```

### Detect Billboards
```http
POST /api/detect
```

**Parameters:**
- `scene` (file): Input image containing billboards
- `conf` (float, optional): Confidence threshold (0-1, default: 0.25)

**Response:**
```json
{
  "boxes": [
    {
      "index": 0,
      "x1": 100,
      "y1": 50,
      "x2": 400,
      "y2": 300,
      "confidence": 0.95
    }
  ],
  "preview": "data:image/png;base64,...",
  "width": 1920,
  "height": 1080
}
```

### Composite Ad onto Billboard
```http
POST /api/composite
```

**Parameters:**
- `scene` (file): Image with billboard
- `ad` (file): Advertisement image to composite
- `x1`, `y1`, `x2`, `y2` (float): Bounding box coordinates
- `fit_shape` (bool, optional): Auto-fit to billboard shape (default: true)
- `blend_edges` (bool, optional): Smooth edge blending (default: true)

**Response:**
```json
{
  "result": "data:image/png;base64,...",
  "preview": "data:image/png;base64,...",
  "used_fallback": false
}
```

---

## 🎨 Web Interface

1. **Upload Scene Image**: Select an image containing billboards
2. **Detect Billboards**: Click "Detect" to find all billboards
3. **Preview**: Review detected billboards with bounding boxes
4. **Upload Ad**: Select your replacement ad image
5. **Composite**: Click "Composite" to replace the billboard
6. **Download**: Save the final result

---

## 📁 Project Structure

```
billboard-replacement/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── static/
│   ├── index.html        # Web UI
│   ├── css/
│   └── js/
├── models/
│   └── billboard_best.pt # YOLO model (auto-downloaded if needed)
└── .env.example          # Environment variables template
```

---

## 🔧 How It Works

### Detection Pipeline
1. Load uploaded image
2. Run YOLO inference to detect billboards
3. Return bounding boxes with confidence scores
4. Generate preview with annotations

### Compositing Pipeline
1. Parse billboard bounding box from detection
2. **Shape Fitting** (optional):
   - Extract region around billboard with padding
   - Apply edge detection (Canny)
   - Find best-fitting contour
   - Order corners for perspective transform
3. **Warping**: Apply perspective transform to ad image
4. **Blending**: Composite warped ad onto scene with smooth edges
5. Return result image

---

## 🐛 Troubleshooting

### Model Not Found
```
Error: No trained model found at models/billboard_best.pt
```
**Solution:**
- Set `MODEL_URL` environment variable, OR
- Place `billboard_best.pt` in `models/` directory

### "Failed to Load Page" / ERR_ADDRESS_INVALID
```
Problem: Trying to access http://0.0.0.0:8000
```
**Solution:** Use `http://localhost:8000` or `http://127.0.0.1:8000`

### CUDA Out of Memory
```
Error: CUDA out of memory
```
**Solution:**
- Use CPU: `export CUDA_VISIBLE_DEVICES=""`
- Reduce image size before upload
- Reduce batch size

### PyTorch Security Warning
```
Warning: PyTorch 2.6+ weights_only security change
```
**Solution:** App automatically handles this with safe_globals fallback

---

## 📊 Performance Notes

- **Inference Time**: ~200-500ms per image (GPU), ~1-2s (CPU)
- **File Sizes**: Supports images up to 50MB
- **Concurrent Requests**: Recommended max 4 workers for typical GPU

---

## 🚀 Deployment

### Render.com
```bash
# Set environment variable in Render dashboard
MODEL_URL=https://your-storage-url/billboard_best.pt
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### AWS EC2 / DigitalOcean
```bash
# Install Python & dependencies
sudo apt update
sudo apt install python3.10 python3.10-venv

# Create venv and install
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with systemd or supervisor
```

---

## 📝 Requirements Details

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.104.1 | Web framework |
| uvicorn | 0.24.0 | ASGI server |
| opencv-python | 4.8.1.78 | Image processing |
| numpy | 1.24.3 | Numerical computing |
| torch | 2.1.1 | Deep learning |
| ultralytics | 8.0.211 | YOLOv8 models |
| requests | 2.31.0 | HTTP downloads |

**For server deployments** (no display), use `opencv-python-headless` instead.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚡ Performance Tips

- **GPU Acceleration**: CUDA automatically detected and used by PyTorch
- **Batch Processing**: API accepts one image at a time; queue externally for bulk jobs
- **Image Optimization**: Downscale large images before upload for faster processing
- **Caching**: Model loads once on first request, then reused

---

## 📄 License

This project is licensed under the MIT License — see LICENSE file for details.

---

## 🆘 Support

For issues or questions:
1. Check [Troubleshooting](#-troubleshooting) section
2. Review `/api/health` endpoint status
3. Check console logs for detailed error messages
4. Open an issue on GitHub with:
   - Error message
   - Input image (if applicable)
   - Environment details (OS, Python version, GPU info)

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection
- [FastAPI](https://fastapi.tiangolo.com/) — Web framework
- [OpenCV](https://opencv.org/) — Computer vision library

---

**Last Updated**: August 2024  
**Version**: 1.0.0
