FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PROJECT_PATH=/workspace \
    PYTHONPATH=/workspace

# System packages — Python 3.10 is the default on Ubuntu 22.04, no PPA needed
RUN apt-get update --fix-missing && apt-get install -y --fix-missing \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    gnupg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /workspace

# PyTorch with CUDA 11.8 — separate layer for cache efficiency
RUN pip install \
    torch==2.5.1+cu118 \
    torchvision==0.20.1+cu118 \
    torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Remaining dependencies
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Project source
COPY . .
RUN pip install -e .

ENTRYPOINT ["python", "scripts/train_TAS.py"]
