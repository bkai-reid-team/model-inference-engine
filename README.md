# Ray Serve Multi-Model Docker Setup
# Ray Serve Multi-Model — Quick Guide

Minimal instructions to run the Ray Serve multi-model API.

Prerequisites
- Docker Desktop or Python 3.10+ with Ray installed
- (Optional) NVIDIA GPU + CUDA for GPU models

Deploy (recommended)
- Use the Router DAG (deploys and binds model deployments):

```bash
serve run serve_app:app
```

Developer options
- Deploy each model separately (useful for debugging):

```bash
serve run qwen_service:app
serve run embedding_service:app
serve run mobilenet_service:app
serve run efficientnetb0_service:app
serve run efficientnetb4_service:app
serve run serve_app:app
```

- Or run the convenience script (optional):

Docker
- Build and start with Compose:

```bash
docker-compose up --build -d
```

Run `setup.sh` (Linux / macOS)
- Make the script executable and run it:

```bash
chmod +x setup.sh
./setup.sh
# or
bash setup.sh
```

- Quick pre-checks (script will also perform these):

```bash
docker --version
docker info
docker compose version || docker-compose version
```

- On Windows use the provided `setup.bat` from an elevated `cmd.exe` or PowerShell:

```cmd
setup.bat
```

- After setup completes, verify services and logs:

```bash
docker-compose ps
docker ps
docker-compose logs -f
# Quick API test
curl -X POST http://localhost:8000/api/generate -H "Content-Type: application/json" -d '{"prompt":"test"}'
```

API (base path)
- Router uses `route_prefix="/api"` — endpoints are under `/api`.
- Common endpoints:
  - `POST /api/generate`  (text generation)
  - `POST /api/embed`     (text embeddings)
  - `POST /api/predict`   (unified endpoint for text and images)
  - `POST /api/image_classification` (MobileNet)
  - `POST /api/efficientnet_b0`
  - `POST /api/efficientnet_b4`

Examples
- Text generation:

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":64}'
```

- Text embedding:

```bash
curl -X POST http://localhost:8000/api/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world"}'
```

- Image classification (multipart form):

```bash
curl -X POST http://localhost:8000/api/predict -F "file=@image.jpg"
```

Notes
- `serve_app.py` returns a DAG that binds the model deployments — deploying the DAG (`serve run serve_app:app`) is the simplest way to start everything.
- `deploy_all_services.py` is an optional convenience script that runs `serve run` per module; keep it if you prefer per-service deployments.

Files
- `serve_app.py` — Router and DAG binding
- `qwen_service.py`, `embedding_service.py`, `mobilenet_service.py`, `efficientnetb0_service.py`, `efficientnetb4_service.py` — model services
- `docker-compose.yml`, `Dockerfile`, `serve_config.yaml`

Quick checks
- API: http://localhost:8000
- Ray Dashboard: http://localhost:8265


That's it — this README is intentionally short. Keep `deploy_all_services.py` only if you want the per-service deploy convenience.
```bash
curl -X POST http://localhost:8000/classification \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product"}'
```

### Image Classification
```bash
# MobileNetV2 - Multiple tasks
curl -X POST http://localhost:8000/mobilenet/predict \
  -F "file=@image.jpg" \
  -F "task=gender"




### Manual API Testing
