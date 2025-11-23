# Use Python 3.10 slim image (lighter weight)
FROM python:3.13-slim

# Set working directory
WORKDIR /

# Install system dependencies needed for OpenCV and other packages
# included: libgl1 (GL), libglib2.0-0 (GObject), ffmpeg (Video processing)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Ensure uvicorn and python-multipart are installed if not in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn python-multipart

# Copy all application code
# Using COPY . . ensures video_keyframe_processor.py and others are included
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(1); result=s.connect_ex(('localhost',8000)); s.close(); exit(0 if result==0 else 1)" || exit 1

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]