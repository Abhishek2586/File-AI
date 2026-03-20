# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies for PyMuPDF and ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for docker cache layer efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create data directories
RUN mkdir -p data/sample_pdfs data/chroma embeddings_cache

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:5000/api/me || exit 1

# Run the API server
ENTRYPOINT ["python", "src/api_server.py"]
