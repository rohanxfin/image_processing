# Use a lightweight Python base image
FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

# Set environment variables to avoid tzdata prompts during install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 8080

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Command to run the application
CMD ["python", "app.py"]
