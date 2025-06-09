# Multi-stage build untuk optimasi ukuran image
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies untuk building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_ENV=production
ENV MODEL_PATH=output/models/best_model.pkl
ENV THRESHOLD=0.2
ENV PORT=5000
ENV HOST=0.0.0.0

# Create non-root user untuk security
RUN groupadd -r modelapi && useradd -r -g modelapi modelapi

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies dari builder stage
COPY --from=builder /root/.local /home/modelapi/.local

# Make sure scripts in .local are usable
ENV PATH=/home/modelapi/.local/bin:$PATH

# Create directories
RUN mkdir -p /app/output/models /app/logs && \
    chown -R modelapi:modelapi /app

# Copy application files
COPY --chown=modelapi:modelapi app.py .
COPY --chown=modelapi:modelapi output/ ./output/

# Switch to non-root user
USER modelapi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Command untuk menjalankan aplikasi
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000", "--threshold", "0.2", "--model", "output/models/best_model.pkl"]