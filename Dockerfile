# Use the official Python base image with a specific version
FROM python:3.9-slim

# Set a working directory for your app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for ONNX and other requirements
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/app/src:$PYTHONPATH"
# Expose the Flask app port (default Flask port is 5000, but we're using 5001)
EXPOSE 5001

# Set environment variables for Flask
ENV FLASK_APP=app.app
ENV FLASK_ENV=production

# Command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
