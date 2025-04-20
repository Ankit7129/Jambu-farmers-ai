# Use Python 3.10 with slim-buster for better compatibility
FROM python:3.10-slim-buster

WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set default port (Cloud Run will override with $PORT)
ENV PORT=8080
ENV HOST=0.0.0.0

# Run with Gunicorn for production stability
RUN pip install gunicorn
CMD exec gunicorn --bind $HOST:$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 240 main:app