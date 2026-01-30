# Brain MRI Tumor Detector

Brain MRI Analyzer – a Dockerized Python web app that uses a deep learning model to detect brain tumors from MRI images. Users upload a brain MRI scan through a simple web interface, and the app classifies it as “tumor” or “no tumor.” The application is containerized with Docker, stored in Azure Container Registry, and deployed on Azure App Service (Linux) for scalable cloud hosting.

---

## Project Structure

- `src/api.py` – Web API / app entry point (FastAPI/Flask).
- `src/inference.py` – Model loading and prediction utilities.
- `models/` – Trained model weights (not committed by default).
- `notebooks/` – Jupyter notebooks for data exploration and training.
- `reports/` – JSON files with model results/metrics.
- `Dockerfile` – Docker image definition for deployment.
- `startup.sh` – Startup script used by Azure App Service.
- `requirements.txt` – Python dependencies.

---

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Example: run API with uvicorn (adjust if needed)
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser (or `http://localhost:8000/docs` for FastAPI docs).

---

## Docker

Build the image (for linux/amd64):

```bash
docker buildx build \
  --platform linux/amd64 \
  -t brain-mri-app-amd64:latest \
  .
```

Run locally:

```bash
docker run -p 8000:8000 brain-mri-app-amd64:latest
```

---

## Azure Deployment (Summary)

1. Push image to Azure Container Registry:
   - Registry: `brainmriregaswin`
   - Image: `brain-mri-app:amd64`
2. Create Azure Web App (Linux) named `brain-mri-analyzer`.
3. Configure Web App to use the ACR image:
   - Image: `brain-mri-app`
   - Tag: `amd64`
   - Port: `8000`
4. Restart the Web App and access:
   - `https://<app-name>.azurewebsites.net/`