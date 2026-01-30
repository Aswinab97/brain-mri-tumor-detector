import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image

from .inference import (
    BrainTumorClassifier,
    InvalidImageError,
    NotBrainMRIError,
)

app = FastAPI(title="Brain MRI Tumor Detection API", version="0.1.0")

classifier = BrainTumorClassifier(
    model_path="../models/resnet18_brain_mri_mps.pth"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <title>Brain MRI Tumor Detection</title>
        <meta charset="utf-8" />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, shrink-to-fit=no"
        />

        <!-- Google Fonts -->
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />

        <!-- Bootstrap CSS -->
        <link
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
          rel="stylesheet"
        />

        <style>
          :root {
            --bg-gradient-start: #0f172a;
            --bg-gradient-end: #020617;
            --card-bg: rgba(15, 23, 42, 0.9);
            --accent: #4ade80;
            --accent-soft: rgba(74, 222, 128, 0.15);
            --accent-strong: #22c55e;
            --danger: #f97373;
            --text-main: #e5e7eb;
            --text-muted: #9ca3af;
            --border-soft: rgba(148, 163, 184, 0.3);
          }

          * {
            box-sizing: border-box;
          }

          body {
            margin: 0;
            min-height: 100vh;
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont,
              "Segoe UI", sans-serif;
            background: radial-gradient(circle at top left, #1d283a, #020617 45%),
              radial-gradient(circle at bottom right, #0f172a, #020617 55%);
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px 12px;
          }

          .page-wrapper {
            width: 100%;
            max-width: 980px;
          }

          .hero-card {
            background: linear-gradient(
              135deg,
              rgba(15, 23, 42, 0.96),
              rgba(15, 23, 42, 0.92)
            );
            border-radius: 18px;
            border: 1px solid var(--border-soft);
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.8);
            overflow: hidden;
            position: relative;
            padding: 28px 26px;
          }

          @media (min-width: 768px) {
            .hero-card {
              padding: 34px 32px;
            }
          }

          .hero-card::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: radial-gradient(
              circle at top right,
              rgba(74, 222, 128, 0.16),
              transparent 55%
            );
            opacity: 0.7;
          }

          .glass-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 11px;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--accent);
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(74, 222, 128, 0.4);
            box-shadow: 0 5px 16px rgba(15, 23, 42, 0.8);
            margin-bottom: 14px;
          }

          .glow-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: var(--accent);
            box-shadow: 0 0 0 4px rgba(74, 222, 128, 0.35);
          }

          .hero-title {
            font-size: clamp(26px, 3vw, 32px);
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-bottom: 8px;
          }

          .hero-subtitle {
            font-size: 14px;
            color: var(--text-muted);
            max-width: 460px;
          }

          .hero-subtitle span {
            color: var(--accent);
            font-weight: 500;
          }

          .layout-grid {
            margin-top: 24px;
          }

          @media (min-width: 992px) {
            .layout-grid {
              display: grid;
              grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
              gap: 26px;
              align-items: stretch;
            }
          }

          .upload-panel {
            background: rgba(15, 23, 42, 0.92);
            border-radius: 16px;
            padding: 18px 16px 18px;
            border: 1px solid rgba(148, 163, 184, 0.4);
            position: relative;
            overflow: hidden;
          }

          .upload-panel::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(
              135deg,
              rgba(148, 163, 184, 0.15),
              transparent 60%
            );
            opacity: 0.4;
            pointer-events: none;
          }

          .upload-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
          }

          .upload-title {
            font-size: 16px;
            font-weight: 500;
          }

          .upload-pill {
            font-size: 11px;
            padding: 3px 9px;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.5);
            color: var(--text-muted);
          }

          .upload-dropzone {
            position: relative;
            margin-top: 10px;
            padding: 16px 14px;
            border-radius: 14px;
            border: 1px dashed rgba(148, 163, 184, 0.6);
            background: radial-gradient(
              circle at top,
              rgba(15, 23, 42, 0.9),
              rgba(15, 23, 42, 0.94)
            );
            cursor: pointer;
            transition: border-color 0.2s ease, background 0.2s ease,
              transform 0.1s ease;
          }

          .upload-dropzone.hover {
            border-color: var(--accent);
            background: radial-gradient(
              circle at top,
              rgba(34, 197, 94, 0.1),
              rgba(15, 23, 42, 0.96)
            );
            transform: translateY(-1px);
          }

          .upload-icon-circle {
            width: 38px;
            height: 38px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 8px;
            background: var(--accent-soft);
            color: var(--accent-strong);
          }

          .upload-icon-circle svg {
            width: 20px;
            height: 20px;
          }

          .upload-main-text {
            font-size: 14px;
            font-weight: 500;
          }

          .upload-secondary-text {
            font-size: 12px;
            color: var(--text-muted);
          }

          .upload-hint-list {
            margin-top: 14px;
            padding-left: 0;
            list-style: none;
            font-size: 12px;
            color: var(--text-muted);
          }

          .upload-hint-list li {
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 4px;
          }

          .hint-dot {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.8);
          }

          .hint-key {
            color: var(--accent);
            font-weight: 500;
          }

          .hidden-input {
            display: none;
          }

          .primary-btn {
            margin-top: 16px;
            width: 100%;
            border-radius: 999px;
            border: none;
            outline: none;
            padding: 10px 16px;
            font-size: 14px;
            font-weight: 500;
            background: linear-gradient(135deg, #4ade80, #22c55e);
            color: #022c22;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            cursor: pointer;
            box-shadow: 0 14px 40px rgba(34, 197, 94, 0.35);
            transition: transform 0.12s ease, box-shadow 0.15s ease,
              filter 0.15s ease;
          }

          .primary-btn span.loader {
            display: none;
            width: 16px;
            height: 16px;
            border-radius: 999px;
            border: 2px solid rgba(15, 23, 42, 0.5);
            border-top-color: #022c22;
            animation: spin 0.9s linear infinite;
          }

          .primary-btn.loading span.loader {
            display: inline-block;
          }

          .primary-btn.loading span.btn-label {
            opacity: 0.7;
          }

          .primary-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 50px rgba(34, 197, 94, 0.45);
            filter: brightness(1.03);
          }

          .primary-btn:active {
            transform: translateY(0);
            box-shadow: 0 10px 28px rgba(34, 197, 94, 0.32);
          }

          @keyframes spin {
            to {
              transform: rotate(360deg);
            }
          }

          .preview-panel {
            margin-top: 18px;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.5);
            background: radial-gradient(
              circle at top,
              rgba(15, 23, 42, 0.88),
              rgba(15, 23, 42, 0.96)
            );
            padding: 12px;
          }

          .preview-header {
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
          }

          .preview-chip {
            font-size: 11px;
            border-radius: 999px;
            padding: 2px 8px;
            background: rgba(31, 41, 55, 0.9);
          }

          #preview {
            display: none;
            width: 100%;
            max-height: 260px;
            object-fit: contain;
            border-radius: 10px;
            border: 1px solid rgba(30, 64, 175, 0.4);
            background: radial-gradient(
              circle at top,
              rgba(15, 23, 42, 0.9),
              rgba(15, 23, 42, 1)
            );
          }

          .right-panel {
            margin-top: 22px;
          }

          @media (min-width: 992px) {
            .right-panel {
              margin-top: 0;
            }
          }

          .result-card {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.45);
            background: radial-gradient(
              circle at top,
              rgba(15, 23, 42, 0.88),
              rgba(15, 23, 42, 0.98)
            );
            padding: 18px 16px;
            position: relative;
            overflow: hidden;
          }

          .result-card::before {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            border-radius: 999px;
            background: radial-gradient(
              circle,
              rgba(74, 222, 128, 0.16),
              transparent 60%
            );
            right: -40px;
            top: -40px;
            opacity: 0.9;
          }

          .result-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
          }

          .result-title-text {
            font-size: 15px;
            font-weight: 500;
          }

          .badge-soft {
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.5);
            color: var(--text-muted);
          }

          #result {
            margin-top: 6px;
            font-size: 14px;
          }

          .result-neutral {
            color: var(--text-muted);
          }

          .result-success {
            color: var(--accent-strong);
          }

          .result-danger {
            color: var(--danger);
          }

          .pill-label {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
          }

          .pill-label.good {
            background: rgba(22, 163, 74, 0.1);
            color: #bbf7d0;
            border: 1px solid rgba(34, 197, 94, 0.6);
          }

          .pill-label.bad {
            background: rgba(248, 113, 113, 0.08);
            color: #fecaca;
            border: 1px solid rgba(248, 113, 113, 0.6);
          }

          .pill-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
          }

          .pill-dot.good {
            background: #22c55e;
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.4);
          }

          .pill-dot.bad {
            background: #ef4444;
            box-shadow: 0 0 0 3px rgba(248, 113, 113, 0.4);
          }

          .footer-note {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 10px;
          }
        </style>
      </head>
      <body>
        <div class="page-wrapper mx-auto">
          <div class="hero-card">
            <div class="glass-tag">
              <span class="glow-dot"></span>
              REAL‑TIME MRI IMAGE CHECK
            </div>

            <div class="d-flex flex-column flex-md-row justify-content-between">
              <div>
                <h1 class="hero-title">Brain MRI Tumor Detection Demo</h1>
                <p class="hero-subtitle mb-0">
                  Upload a
                  <span>clear brain MRI slice</span> (JPG/PNG) and get a quick,
                  AI‑powered indication. This tool is for learning and
                  experimentation only — <strong>not</strong> for medical use.
                </p>
              </div>
            </div>

            <div class="layout-grid">
              <!-- LEFT: Upload + preview -->
              <div>
                <div class="upload-panel">
                  <div class="upload-header">
                    <div>
                      <div class="upload-title">Upload MRI image</div>
                      <div class="text-muted" style="font-size: 11px;">
                        Please upload a single axial brain MRI slice.
                      </div>
                    </div>
                    <div class="upload-pill">JPG / PNG · &lt; 5 MP · &lt; 10 MB</div>
                  </div>

                  <form id="upload-form">
                    <label
                      for="file"
                      class="upload-dropzone"
                      id="dropzone"
                    >
                      <div class="upload-icon-circle">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="1.7"
                            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M8 12l4-4m0 0l4 4m-4-4v12"
                          />
                        </svg>
                      </div>
                      <div class="upload-main-text mb-1">
                        Click to browse, or drag &amp; drop an MRI image
                      </div>
                      <div class="upload-secondary-text">
                        Accepted: <strong>.jpg</strong>, <strong>.jpeg</strong>,
                        <strong>.png</strong> · Recommended: square, clear brain
                        MRI slice.
                      </div>

                      <ul class="upload-hint-list mt-2 mb-0">
                        <li>
                          <span class="hint-dot"></span>
                          <span class="hint-key">Clarity:</span> in‑focus brain
                          MRI, minimal noise.
                        </li>
                        <li>
                          <span class="hint-dot"></span>
                          <span class="hint-key">Size:</span> up to ~5 megapixels
                          (e.g. 2200×2200).
                        </li>
                        <li>
                          <span class="hint-dot"></span>
                          <span class="hint-key">Content:</span> single axial
                          slice, no text overlays or screenshots.
                        </li>
                      </ul>
                    </label>

                    <input
                      class="hidden-input"
                      type="file"
                      id="file"
                      name="file"
                      accept="image/*"
                      required
                    />

                    <button type="submit" class="primary-btn mt-2" id="submit-btn">
                      <span class="loader"></span>
                      <span class="btn-label">Run prediction</span>
                    </button>
                  </form>

                  <div class="preview-panel mt-3">
                    <div class="preview-header">
                      <span>Image preview</span>
                      <span class="preview-chip" id="preview-meta">No file selected</span>
                    </div>
                    <img id="preview" alt="MRI preview" />
                  </div>

                  <div class="footer-note">
                    This demo does <strong>not</strong> store your images. All
                    processing happens in memory for the current session only.
                  </div>
                </div>
              </div>

              <!-- RIGHT: Result -->
              <div class="right-panel">
                <div class="result-card">
                  <div class="result-title">
                    <div class="result-title-text">Model output</div>
                    <span class="badge-soft" id="status-pill">Waiting for image…</span>
                  </div>
                  <div id="result" class="result-neutral">
                    Upload a suitable MRI image to see the predicted label and confidence.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Bootstrap JS (optional) -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

        <script>
          const fileInput = document.getElementById('file');
          const dropzone = document.getElementById('dropzone');
          const previewImg = document.getElementById('preview');
          const previewMeta = document.getElementById('preview-meta');
          const form = document.getElementById('upload-form');
          const resultDiv = document.getElementById('result');
          const statusPill = document.getElementById('status-pill');
          const submitBtn = document.getElementById('submit-btn');

          function setButtonLoading(isLoading) {
            if (isLoading) {
              submitBtn.classList.add('loading');
              submitBtn.disabled = true;
            } else {
              submitBtn.classList.remove('loading');
              submitBtn.disabled = false;
            }
          }

          // Update preview when a file is selected
          fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) {
              previewImg.style.display = 'none';
              previewMeta.textContent = 'No file selected';
              return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
              previewImg.src = e.target.result;
              previewImg.style.display = 'block';
            };
            reader.readAsDataURL(file);

            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            previewMeta.textContent = `${file.name} · ${sizeMB} MB`;
          });

          // Drag & drop behavior
          ;['dragenter', 'dragover'].forEach(evtName => {
            dropzone.addEventListener(evtName, (e) => {
              e.preventDefault();
              e.stopPropagation();
              dropzone.classList.add('hover');
            });
          });

          ;['dragleave', 'dragend', 'drop'].forEach(evtName => {
            dropzone.addEventListener(evtName, (e) => {
              e.preventDefault();
              e.stopPropagation();
              if (evtName !== 'drop') {
                dropzone.classList.remove('hover');
              }
            });
          });

          dropzone.addEventListener('drop', (e) => {
            dropzone.classList.remove('hover');
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files[0]) {
              fileInput.files = files;
              fileInput.dispatchEvent(new Event('change'));
            }
          });

          // Handle form submit via JavaScript fetch
          form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const file = fileInput.files[0];
            if (!file) {
              alert('Please select an image first.');
              return;
            }

            const formData = new FormData();
            formData.append('file', file);

            setButtonLoading(true);
            resultDiv.className = 'result-neutral';
            resultDiv.innerHTML = '<span class="text-muted">Running prediction…</span>';
            statusPill.textContent = 'Processing…';

            try {
              const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
              });

              if (!response.ok) {
                let message = 'Unexpected error';
                try {
                  const data = await response.json();
                  if (data && data.error) {
                    message = data.error;
                  }
                } catch (_) {
                  const text = await response.text();
                  message = text || message;
                }
                resultDiv.className = 'result-danger';
                resultDiv.textContent = message;
                statusPill.textContent = 'Upload issue';
                return;
              }

              const data = await response.json();
              const probPercent = (data.probability * 100).toFixed(1);

              const isTumor = data.label_name === 'tumor';
              const pillClass = isTumor ? 'pill-label bad' : 'pill-label good';
              const dotClass = isTumor ? 'pill-dot bad' : 'pill-dot good';
              const labelText = isTumor ? 'Tumor detected' : 'No tumor detected';

              resultDiv.className = isTumor ? 'result-danger' : 'result-success';
              statusPill.textContent = 'Result ready';

              resultDiv.innerHTML = `
                <div class="${pillClass}">
                  <span class="${dotClass} pill-dot"></span>
                  <span>${labelText}</span>
                </div>
                <div class="mt-2">
                  <div><strong>Confidence:</strong> ${probPercent}%</div>
                  <div><strong>File:</strong> ${data.filename}</div>
                </div>
              `;
            } catch (err) {
              console.error(err);
              resultDiv.className = 'result-danger';
              resultDiv.textContent = 'Unexpected error. Please try again.';
              statusPill.textContent = 'Error';
            } finally {
              setButtonLoading(false);
            }
          });
        </script>
      </body>
    </html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Reject very large files up front (e.g. screenshots / photos > 10 MB)
        max_bytes = 10 * 1024 * 1024  # 10 MB
        if len(contents) > max_bytes:
            return JSONResponse(
                {
                    "error": "File is too large. Please upload a brain MRI image under 10 MB."
                },
                status_code=400,
            )

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            return JSONResponse(
                {
                    "error": "Invalid image file. Please upload a JPG or PNG brain MRI image."
                },
                status_code=400,
            )

        try:
            result = classifier.predict_image_from_pil(image)
        except InvalidImageError:
            return JSONResponse(
                {
                    "error": "Invalid image file. Please upload a clear JPG or PNG image."
                },
                status_code=400,
            )
        except NotBrainMRIError as e:
            # Include the specific reason from the classifier
            return JSONResponse(
                {"error": str(e)},
                status_code=400,
            )

        return JSONResponse(
            {
                "filename": file.filename,
                "label": result["label"],
                "label_name": result["label_name"],
                "probability": result["probability"],
            }
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"Unexpected server error: {str(e)}"},
            status_code=500,
        )