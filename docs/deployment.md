# NovaScientist Production Deployment Guide

## 1. Quickstart via Docker Compose

### Prerequisites
- Docker Engine 24.0+
- Docker Compose v2+

### Running the Services
```bash
# Clone the repository
git clone https://github.com/chsushen/novascientist.git
cd novascientist

# Configure environment
cp .env.example .env

# Build and start services
docker compose up --build -d
```

Services will be available at:
- **FastAPI Backend Server**: `http://localhost:8000` (Docs at `/docs`)
- **Interactive Streamlit UI**: `http://localhost:8501`

---

## 2. Standalone Python Setup

### Prerequisites
- Python 3.11+
- Tectonic LaTeX engine:
  ```bash
  curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh
  sudo mv tectonic /usr/local/bin/
  ```

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Starting the Production API Server
```bash
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Starting the Streamlit Frontend
```bash
streamlit run streamlit_app.py --server.port 8501
```

---

## 3. Health & Readiness Probes
- **Liveness Probe**: `GET http://localhost:8000/health` (returns `{"status": "healthy"}`)
- **Diagnostics Probe**: `GET http://localhost:8000/diagnostics` (returns queue depth, Git SHA, and storage status)
