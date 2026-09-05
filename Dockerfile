# ==============================================================================
# NovaScientist v2.3.0 — Production Multi-Stage Dockerfile
# ==============================================================================

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (including Tectonic and build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libfontconfig1 \
    libgraphite2-3 \
    libharfbuzz0b \
    libicu-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Tectonic standalone binary for PDF compilation
RUN curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh \
    && mv tectonic /usr/local/bin/

# Copy dependencies lock
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application source code
COPY . .

# Create non-root system user for secure sandboxed execution
RUN useradd -m -u 1000 novascientist \
    && mkdir -p /app/.novascientist_data /app/.matplotlib \
    && chown -R novascientist:novascientist /app

USER novascientist

ENV MPLCONFIGDIR=/app/.matplotlib \
    NOVASCIENTIST_DATA_DIR=/app/.novascientist_data

EXPOSE 8000 8501

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
