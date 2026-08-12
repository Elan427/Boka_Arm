#!/usr/bin/env bash
set -euo pipefail

# Template for Phase 0 environment bootstrap.
# Adapt to the exact system and Python toolchain before use.

echo "[1/4] Updating system packages"
sudo apt-get update
sudo apt-get install -y build-essential git curl ca-certificates

echo "[2/4] Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "[3/4] Creating project environment"
uv venv .venv
source .venv/bin/activate

echo "[4/4] Installing core Python packages"
uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install "lerobot[all]" --no-deps

echo "Environment template complete. Verify the CUDA stack explicitly before proceeding."
