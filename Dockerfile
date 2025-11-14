# Use Python 3.10 with Debian Bullseye as base image
FROM python:3.10-bullseye

# Set metadata
LABEL maintainer="your-email@example.com"
LABEL description="Vitamin B12 Hand Analysis Application with MinIO"

# Install system dependencies required for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libprotobuf-dev \
    protobuf-compiler \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for temporary file storage
RUN mkdir -p uploads processed

# Expose Flask port
EXPOSE 5002

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Health check to ensure container is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5002/health')" || exit 1

# Run the application
CMD ["python", "app.py"]