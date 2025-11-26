# Chillbot - AI Memory Infrastructure
# Multi-stage build for small production images

# ===========================================
# BUILD STAGE
# ===========================================
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===========================================
# PRODUCTION STAGE
# ===========================================
FROM python:3.12-slim as production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY chillbot/ ./chillbot/
COPY pyproject.toml README.md ./

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash chillbot
RUN chown -R chillbot:chillbot /app
USER chillbot

# Create data directory
RUN mkdir -p /app/data

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV DATABASE_PATH=/app/data/krnx.db
ENV REDIS_URL=redis://localhost:6379
ENV QDRANT_URL=http://localhost:6333

# Expose API port
EXPOSE 6380

# Default command
CMD ["python", "-m", "chillbot.server"]

# ===========================================
# DEVELOPMENT STAGE
# ===========================================
FROM production as development

USER root

# Install dev dependencies
RUN pip install pytest pytest-asyncio black ruff ipython

# Switch back to chillbot user
USER chillbot

# Mount point for live code
VOLUME ["/app"]

CMD ["python", "-m", "chillbot.server", "--reload"]
