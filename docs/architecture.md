# System Architecture — Brain MRI Tumor Detector

## End-to-End MLOps Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
│                                                                     │
│  MRI Dataset          Preprocessing         Model Training          │
│  ──────────           ────────────          ──────────────          │
│  data_raw/            Resize 224×224        ResNet18                │
│  ├── yes/    ──────►  Normalize      ──────►VGG16                   │
│  └── no/              Augment               DenseNet121             │
│  (Kaggle)             (torchvision)         EfficientNetB0          │
│                                             MobileNetV2             │
│                                             (PyTorch)               │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       EVALUATION & ENSEMBLE                         │
│                                                                     │
│  Per-Model Metrics         Ensemble Layer                           │
│  ─────────────────         ──────────────                           │
│  Accuracy / AUC            Average softmax outputs                  │
│  F1 / Sensitivity  ──────► across all 5 models                      │
│  Confusion Matrix          Ensemble AUC: ~92–95%                    │
│  reports/*.json            Sensitivity:   95.7%                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         INFERENCE API                               │
│                                                                     │
│  FastAPI Application (src/api.py + src/inference.py)                │
│  ─────────────────────────────────────────────────                  │
│  POST /predict  ◄── User uploads MRI image                         │
│       │                                                             │
│       ▼                                                             │
│  Preprocess → Load model weights → Forward pass → Ensemble          │
│       │                                                             │
│       ▼                                                             │
│  Return JSON: { "prediction": "tumor" | "no tumor",                 │
│                 "confidence": 0.97 }                                │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTAINERIZATION & DEPLOYMENT                    │
│                                                                     │
│  Local Build              Azure Container Registry                  │
│  ────────────             ───────────────────────                   │
│  Dockerfile      ──────►  acr push (Docker image)  ──────►         │
│  (Python + deps)          tagged image stored                       │
│                                                                     │
│                           Azure App Service (Linux)                 │
│                           ────────────────────────                  │
│                           Pulls from ACR ──────────► Live URL       │
│                           Uvicorn / Gunicorn         (eastus-01)    │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Summary

| Component | Technology | Purpose |
|---|---|---|
| Training framework | PyTorch | Model training and evaluation |
| Model architectures | ResNet18, VGG16, DenseNet121, EfficientNetB0, MobileNetV2 | Binary classification (tumor / no tumor) |
| Ensemble strategy | Softmax averaging | Improve accuracy by 2–5% over single model |
| Inference API | FastAPI + Uvicorn | Serve predictions via HTTP |
| Containerization | Docker | Reproducible runtime environment |
| Container registry | Azure Container Registry (ACR) | Store and version Docker images |
| Cloud deployment | Azure App Service (Linux) | Host live web application |
| Web interface | Jinja2 HTML templates | Upload MRI, display result |

## Key Metrics

| Metric | Value |
|---|---|
| Sensitivity (Tumor Recall) | **95.7%** |
| F1 Score | **83.0%** |
| Ensemble AUC | **~92–95%** |
| Models trained | **5** |
| Accuracy improvement (ensemble vs single) | **+2–5%** |
