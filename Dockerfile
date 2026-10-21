# -------------------------------------------------------------------
# STEP 1: Base image — lightweight official Python image
# -------------------------------------------------------------------
FROM python:3.11-slim

# -------------------------------------------------------------------
# STEP 2: Set environment variables
#   - Prevent .pyc files & enable unbuffered logging (good for containers)
# -------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# -------------------------------------------------------------------
# STEP 3: Set working directory inside the container
# -------------------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------------------
# STEP 4: Install OS-level build dependencies required by XGBoost/numpy
# -------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# STEP 5: Copy only requirements first (leverages Docker layer caching)
# -------------------------------------------------------------------
COPY requirements.txt .

# -------------------------------------------------------------------
# STEP 6: Install Python dependencies
# -------------------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------------
# STEP 7: Copy the full project source code into the container
# -------------------------------------------------------------------
COPY . .

# -------------------------------------------------------------------
# STEP 8: Create required directories (if not already present via .gitkeep)
# -------------------------------------------------------------------
RUN mkdir -p data/raw data/processed models metrics

# -------------------------------------------------------------------
# STEP 9: Expose the port FastAPI/Uvicorn will run on
# -------------------------------------------------------------------
EXPOSE 8000

# -------------------------------------------------------------------
# STEP 10: Healthcheck (optional but recommended for orchestration)
# -------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# -------------------------------------------------------------------
# STEP 11: Run the FastAPI app with Uvicorn
# -------------------------------------------------------------------
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]