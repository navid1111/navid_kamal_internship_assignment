FROM apache/airflow:3.1.6

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*
USER airflow

# Install ML/DL dependencies (headless OpenCV for container use)
# Upgrade numpy+pandas first to avoid binary incompatibility with base image
RUN pip install --no-cache-dir --upgrade numpy pandas && \
    pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    ultralytics==8.4.18 \
    opencv-python-headless==4.13.0.92 \
    pyyaml==6.0.3 \
    pillow==12.1.1 \
    matplotlib==3.10.8 \
    scipy==1.17.1 \
    wandb
