FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV INPUT_TASKS_PATH=/input/tasks.json
ENV OUTPUT_RESULTS_PATH=/output/results.json
ENV TEMP_DIR=/tmp

# Install system dependencies for OpenCV headless
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Create input and output directories (to match evaluation volumes)
RUN mkdir -p /input /output

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy main agent code
COPY main.py .

# Command to run on container startup
CMD ["python", "main.py"]
