FROM python:3.12-slim

# gmsh (STEP import) needs a few X/GL libraries even when running headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglu1-mesa libgl1 libxcursor1 libxinerama1 libxft2 libxrender1 libxi6 libxkbcommon0 \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    U2NET_HOME=/app/data/models

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt

# Optional DINOv2 embeddings: docker compose build --build-arg INSTALL_ML=true
ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-ml.txt; \
    fi

COPY . .

EXPOSE 7860
CMD ["python", "-m", "app.main", "--host", "0.0.0.0", "--port", "7860"]
