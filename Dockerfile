# Use Python 3.10 with Debian Bullseye as base image
FROM python:3.10-bullseye

LABEL maintainer="patelshreyansh376@gmail.com"
LABEL description="Vitamin B12 Hand Analysis Application - CPU Only"

# Install system dependencies for MediaPipe & OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libprotobuf-dev \
    protobuf-compiler \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create directories for file storage
RUN mkdir -p uploads processed stored_images/originals stored_images/processed

# Expose Flask port
EXPOSE 5002

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV MEDIAPIPE_DISABLE_GPU=1 

# Optional: Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5002/health')" || exit 1

# Run the application
CMD ["python", "app.py"]
