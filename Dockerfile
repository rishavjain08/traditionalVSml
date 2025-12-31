# ------------------------------
# Base Image
# ------------------------------
FROM python:3.10-slim

# ------------------------------
# Environment variables
# ------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# # ------------------------------
# # System dependencies
# # ------------------------------
# RUN add --no-cache \
#         gcc \
#         g++ \
#         musl-dev \
#         mariadb-connector-c-dev \
#         curl \
#         jq \
#         unzip \
#         bash \
#         net-tools \
#         make \
#         openssl \
#         libffi-dev \

# ------------------------------
# Working directory
# ------------------------------
WORKDIR /usr/app

# ------------------------------
# Copy dependency files first
# (better Docker cache usage)
# ------------------------------
COPY setup.py requirements.txt README.md ./

# ------------------------------
# Install Python dependencies
# ------------------------------
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e .

# ------------------------------
# Copy application source code
# ------------------------------

COPY utils ./utils
COPY app.py .

# ------------------------------
# Expose Streamlit port
# ------------------------------
EXPOSE 8501

# ------------------------------
# Start Streamlit
# ------------------------------
CMD ["streamlit", "run", "app.py"]
