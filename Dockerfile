# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Flask for inference server
RUN pip install --no-cache-dir flask

# Copy the full project into the container
COPY . /app

# Ensure supply-chain-rl is on the Python path
ENV PYTHONPATH=/app:/app/supply-chain-rl
ENV PYTHONUNBUFFERED=1

# Expose port 7860 (Gradio / inference default)
EXPOSE 7860

# Run inference server by default
CMD ["python", "inference.py"]
