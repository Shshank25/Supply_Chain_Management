# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Since we don't have a requirements.txt, we'll install them directly.
# openenv, gymnasium, stable-baselines3, pydantic, openai, gradio, matplotlib, numpy
RUN pip install --no-cache-dir \
    gradio \
    numpy \
    matplotlib \
    pydantic \
    openai \
    stable-baselines3 \
    shimmy>=0.2.1

# Make port 7860 available to the world outside this container (Gradio default)
EXPOSE 7860

# Define environment variable
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run app.py when the container launches
CMD ["python", "app.py"]
