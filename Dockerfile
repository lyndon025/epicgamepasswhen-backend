FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if required (e.g., for pandas/numpy compilation if wheels missing)
# RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default port for Fly.io and Render
ENV PORT=8080
EXPOSE 8080

# Use gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
