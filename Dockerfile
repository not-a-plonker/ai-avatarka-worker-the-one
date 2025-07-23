# Minimal Dockerfile for RunPod serverless with network storage
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Essential environment variables only
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    curl \
    wget \
    git \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify Python version
RUN python --version

# Copy only your project files (no models, no dependencies)
COPY workflow/ /workspace/workflow/
COPY prompts/ /workspace/prompts/
COPY src/handler.py /workspace/src/handler.py

# Create startup script that uses network storage environment
RUN cat > /workspace/start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting AI-Avatarka with Network Storage..."
echo "🔍 Debugging network storage contents..."

echo "Contents of /workspace:"
ls -la /workspace/

echo "Looking for ComfyUI directory:"
find /workspace -name "ComfyUI" -type d 2>/dev/null

echo "Looking for venv directory:"
find /workspace -name "venv" -type d 2>/dev/null

echo "Looking for main.py (ComfyUI indicator):"
find /workspace -name "main.py" 2>/dev/null

echo "Looking for activate script (venv indicator):"
find /workspace -name "activate" 2>/dev/null

if [ -d "/workspace/ComfyUI" ]; then
    echo "✅ Found ComfyUI at /workspace/ComfyUI"
    echo "Contents of ComfyUI directory:"
    ls -la /workspace/ComfyUI/
else
    echo "❌ ComfyUI directory not found at /workspace/ComfyUI"
fi

if [ -f "/workspace/venv/bin/activate" ]; then
    echo "✅ Found venv at /workspace/venv"
    echo "Contents of venv/bin:"
    ls -la /workspace/venv/bin/ | head -10
else
    echo "❌ Virtual environment not found at /workspace/venv"
    if [ -d "/workspace/venv" ]; then
        echo "venv directory exists but missing activate script:"
        ls -la /workspace/venv/
    fi
fi

# Try to continue anyway for debugging
echo "🔧 Network storage analysis complete"
echo "🎯 Attempting to start handler anyway..."
cd /workspace
python -c "import sys; sys.path.append('/workspace/src'); from handler import handler; import runpod; print('🚀 Starting AI-Avatarka handler...'); runpod.serverless.start({'handler': handler})"
EOF

RUN chmod +x /workspace/start.sh

# Set working directory
WORKDIR /workspace

# Use the startup script
CMD ["/workspace/start.sh"]