# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# vlm-orchestrator Docker image.
# Lightweight: Python 3.10 + all deps pre-installed, CPU-only torch.
# ~1.5 GB total (no CUDA — orchestrator doesn't need GPU).
#
# Build:
#   docker build -t ghcr.io/chicychen/vlm-orchestrator:latest .
#
# Push:
#   docker push ghcr.io/chicychen/vlm-orchestrator:latest

FROM python:3.10-slim AS base

# System deps for opencv + general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        git tmux vim iputils-ping curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vlm-orchestrator

# ── Install Python dependencies first (cache-friendly) ──────────────
# Copy only dependency specs so this layer is cached unless deps change.
COPY pyproject.toml ./
# CPU-only torch first (from pytorch index)
RUN pip install --no-cache-dir \
        "torch>=2.0,<3" --index-url https://download.pytorch.org/whl/cpu

# Everything else from PyPI
RUN pip install --no-cache-dir \
        "websockets>=13.0" \
        "msgpack>=1.0" \
        "numpy>=1.24,<2" \
        "openai>=1.0" \
        "pillow>=10.0" \
        "requests>=2.28" \
        "opencv-python-headless>=4.8" \
        "imageio>=2.30" \
        "imageio-ffmpeg>=0.5" \
        "scipy>=1.10"

# ── Copy source and install package ─────────────────────────────────
COPY . .
RUN pip install --no-cache-dir -e .

# Verify CLI is available
RUN vlm-orchestrator --help > /dev/null 2>&1

# Default: show help. Override the command to launch the orchestrator.
ENTRYPOINT ["vlm-orchestrator"]
CMD ["--help"]
