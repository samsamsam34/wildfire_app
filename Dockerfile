FROM python:3.12-slim

# System dependencies for geospatial stack used by rasterio/pyproj/shapely/geopandas
# and optional PDF rendering dependencies (Cairo/Pango runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    libspatialindex-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    libxml2 \
    libxslt1.1 \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and config.
COPY backend/ ./backend/
COPY config/ ./config/
COPY frontend/ ./frontend/
COPY docs/ ./docs/

# Runtime data directories (mounted or populated during deployment workflows).
RUN mkdir -p data/national data/regions

# Drop root privileges.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
