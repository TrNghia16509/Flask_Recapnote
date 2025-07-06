# Flask API - recapnote-api
FROM python:3.10-slim

# Install system dependencies (ffmpeg for audio)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Flask will listen on
EXPOSE 8000

# Run the Flask app
CMD ["python", "flask_api.py"]
