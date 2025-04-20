# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with uvicorn
# Use PORT environment variable and multiple workers
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]