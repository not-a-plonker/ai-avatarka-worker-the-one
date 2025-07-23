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
RUN echo '#!/bin/bash\n\
echo "🚀 Starting AI-Avatarka with Network Storage..."\n\
echo "📁 Checking network storage..."\n\
if [ ! -d "/workspace/ComfyUI" ]; then\n\
    echo "❌ Network storage not found at /workspace/ComfyUI"\n\
    echo "Make sure your RunPod pod is using the correct network storage"\n\
    exit 1\n\
fi\n\
\n\
if [ ! -f "/workspace/venv/bin/activate" ]; then\n\
    echo "❌ Virtual environment not found at /workspace/venv"\n\
    echo "Make sure you completed the network storage setup"\n\
    exit 1\n\
fi\n\
\n\
echo "✅ Network storage found"\n\
echo "🔧 Activating virtual environment..."\n\
source /workspace/venv/bin/activate\n\
\n\
echo "🐍 Python version: $(python --version)"\n\
echo "📦 Torch version: $(python -c \"import torch; print(torch.__version__)\" 2>/dev/null || echo \"Not installed\")"\n\
echo "🎮 CUDA available: $(python -c \"import torch; print(torch.cuda.is_available())\" 2>/dev/null || echo \"Unknown\")"\n\
echo "🧠 SageAttention: $(python -c \"import sageattention; print(sageattention.__version__)\" 2>/dev/null || echo \"Not installed\")"\n\
\n\
echo "🎯 Starting RunPod handler..."\n\
cd /workspace\n\
python -c "import sys; sys.path.append(\"/workspace/src\"); from handler import handler; import runpod; print(\"🚀 Starting AI-Avatarka handler with network storage...\"); runpod.serverless.start({\"handler\": handler})"\n\
' > /workspace/start.sh && chmod +x /workspace/start.sh

# Set working directory
WORKDIR /workspace

# Use the startup script
CMD ["/workspace/start.sh"]