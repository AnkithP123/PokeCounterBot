FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Install Tesseract OCR, Leptonica dev headers, and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and test samples
COPY . .

# Ensure custom traineddata is available in system tessdata paths without overwriting system eng.traineddata
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata /usr/share/tessdata && \
    cp tessdata/number.traineddata /usr/share/tesseract-ocr/5/tessdata/ 2>/dev/null || true && \
    cp tessdata/number.traineddata /usr/share/tessdata/ 2>/dev/null || true

# Run as non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "src.bot"]
