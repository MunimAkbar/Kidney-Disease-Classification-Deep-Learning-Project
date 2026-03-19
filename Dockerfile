# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set up a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Upgrade pip and install vital OS dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy the entire project (must happen before pip install because of -e .)
COPY --chown=user:user . .

# Install dependencies (ignoring the GPU version of TF for deployment)
# We use standard tensorflow for CPU inference on HF Spaces
RUN pip install --no-cache-dir --upgrade pip && \
    sed -i 's/tensorflow-gpu==2.10.0/tensorflow==2.10.0/' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Environment variables setup for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860

# Command to run the application using Waitress (production WSGI server)
CMD ["python", "app.py"]
