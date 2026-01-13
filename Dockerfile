# ------------------------------
# Base Image (Small & Fast)
# ------------------------------
FROM python:3.11-slim

# ------------------------------
# Environment Variables
# ------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ------------------------------
# System Dependencies
# ------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Working Directory
# ------------------------------
WORKDIR /usr/app

# ------------------------------
# Copy dependency files first
# (Better Docker layer caching)
# ------------------------------
# Install Python Dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install the custom package in editable mode
COPY traditionalVSml/ ./traditionalVSml/
COPY setup.py .
RUN pip install -e .

# ------------------------------
# Copy Application FrontEnd Code
# ------------------------------
COPY app.py .

# ------------------------------
# Expose Streamlit Port
# ------------------------------
EXPOSE 8501

# ------------------------------
# Health Check (K8s / ECS Friendly)
# ------------------------------
# HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
#     CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ------------------------------
# Run Streamlit App
# ------------------------------
CMD ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
