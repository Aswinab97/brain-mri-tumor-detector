# Brain MRI Tumor Detector

Brain MRI Analyzer – a Dockerized Python web app that uses a deep learning model to detect brain tumors from MRI images. Users upload a brain MRI scan through a simple web interface, and the app classifies it as **tumor** or **no tumor**. The application is containerized with Docker and can be deployed to Azure App Service (Linux) via Azure Container Registry.

---

## Demo

Below is a screenshot of the web interface used to upload MRI images and view predictions:

![Brain MRI tumor detector UI](images/app-screenshot.png)

---

## Features

- Upload brain MRI images through a simple web UI.
- Deep learning–based classification (**tumor** vs **no tumor**).
- Pretrained PyTorch models:
  - `resnet18_brain_mri_mps.pth`
  - `simple_cnn_baseline_mps.pth`
- JSON reports with evaluation metrics for models.
- Dockerized for reproducible deployment.
- Ready to run on **Azure App Service (Linux)** using an image from **Azure Container Registry (ACR)**.

---

## Project Structure

```text
.
├── data_raw/                 # Raw MRI images (ignored in git)
│   ├── no/
│   └── yes/
├── data_processed/           # Processed / prepared data (ignored in git)
├── models/
│   ├── resnet18_brain_mri_mps.pth
│   └── simple_cnn_baseline_mps.pth
├── notebooks/
│   └── 00_explore_data.ipynb
├── reports/
│   ├── resnet18_results.json
│   └── simple_cnn_results.json
├── src/
│   ├── __init__.py
│   ├── api.py                # Web API / app entry point
│   └── inference.py          # Model loading & prediction utilities
├── images/
│   └── app-screenshot.png    # Screenshot of the web app UI
├── Dockerfile                # Docker image definition
├── startup.sh                # Startup script (for Azure App Service)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Requirements

- Python 3.9+ (or compatible with your dependencies)
- Docker (for containerized runs)
- Optional: Azure CLI / Azure portal access (for cloud deployment)

Python dependencies are listed in `requirements.txt`.

---

## Setup & Local Development

### 1. Clone the repository

```bash
git clone git@github.com:Aswinab97/brain-mri-tumor-detector.git
cd brain-mri-tumor-detector
```

(or use the HTTPS URL if you’re not using SSH.)

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API locally

This assumes `src/api.py` exposes a FastAPI or similar app named `app`:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Then open in your browser:

- App UI: `http://localhost:8000`
- API docs (if FastAPI): `http://localhost:8000/docs`

> If your app entry point is different (e.g., Flask), adjust the command accordingly.

---

## Inference Logic

The main inference utilities live in `src/inference.py`. At a high level:

1. **Load model weights** from `models/*.pth`.
2. **Preprocess** incoming MRI images to the expected size/normalization.
3. **Run the model** to get logits/probabilities.
4. **Map predictions** to labels (`tumor`, `no tumor`) and return them to the API.

The API layer in `src/api.py`:

- Exposes an endpoint that receives image uploads.
- Calls the inference functions to get predictions.
- Returns a JSON response and/or renders an HTML page with the result.

---

## Docker Usage

### 1. Build the Docker image (local)

```bash
docker build -t brain-mri-app:latest .
```

If you specifically need a Linux/amd64 image (for Azure App Service):

```bash
docker buildx build \
  --platform linux/amd64 \
  -t brain-mri-app-amd64:latest \
  .
```

### 2. Run the container locally

```bash
docker run -p 8000:8000 brain-mri-app:latest
```

Now the app should be available at:

- `http://localhost:8000`

---

## Azure Deployment (High-Level Overview)

1. **Build & tag image for Azure Container Registry (ACR)**

   ```bash
   # Example values – update with your real registry / image names
   ACR_NAME=brainmriregaswin
   IMAGE_NAME=brain-mri-app
   TAG=amd64

   az acr login --name $ACR_NAME

   docker buildx build \
     --platform linux/amd64 \
     -t $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG \
     .

   docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG
   ```

2. **Create Azure Web App for Containers**

   - Runtime stack: Linux
   - Container source: Azure Container Registry
   - Image: `$ACR_NAME.azurecr.io/$IMAGE_NAME`
   - Tag: `$TAG`
   - Port: `8000` (or your app port)

3. **Configure startup command (if needed)**

   In the Web App configuration, set the startup command (if Azure doesn’t infer it):

   ```bash
   ./startup.sh
   ```

   or directly:

   ```bash
   gunicorn -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 src.api:app
   ```

4. **Browse the deployed app**

   - `https://<your-app-name>.azurewebsites.net`

---

## Data & Model Files

Large data and model files are **not** committed to the repository by default (see `.gitignore`). To reproduce training or inference end‑to‑end:

- Place raw MRI images under `data_raw/no` and `data_raw/yes`.
- Place trained model weights under `models/`.

If you want fully reproducible experiments, consider:

- Using DVC or similar tools to track datasets.
- Publishing model weights in a release or external storage (e.g. Azure Blob Storage).

---

## Notebooks

The `notebooks/00_explore_data.ipynb` notebook contains initial data exploration and possibly some training experiments. To run it:

```bash
jupyter notebook
```

Then open the notebook from the browser UI.

---

## Future Improvements

- Add unit tests for inference and API endpoints.
- Integrate GitHub Actions for automatic testing and image builds.
- Add Grad‑CAM or other explainability visualizations.
- Extend classification to multiple tumor types.

---

## License

_Add your chosen license here (e.g. MIT, Apache 2.0)._

If you haven’t selected one yet, you can generate a license file on GitHub by clicking **“Add file → Create new file → LICENSE”** and choosing an open‑source license template.