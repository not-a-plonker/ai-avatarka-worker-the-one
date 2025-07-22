#!/usr/bin/env bash

set -e  # Exit on any error

echo "[start.sh] 🚀 Worker initiated"

# Try to setup workspace, but don't fail if it doesn't work
echo "[start.sh] Setting up workspace..."
if [ -L /workspace ]; then
    echo "[start.sh] /workspace is already a symlink, skipping"
elif [ -d /workspace ]; then
    echo "[start.sh] /workspace exists as directory, trying to backup..."
    mv /workspace /workspace_backup 2>/dev/null || echo "[start.sh] Could not backup /workspace"
fi

# Create runpod volume if it doesn't exist
mkdir -p /runpod-volume/logs

# Try to create symlink
if ln -sf /runpod-volume /workspace 2>/dev/null; then
    echo "[start.sh] ✅ Workspace symlink created"
else
    echo "[start.sh] ⚠️ Could not create symlink, using /tmp for logs"
    mkdir -p /tmp/logs
    ln -sf /tmp /workspace 2>/dev/null || echo "[start.sh] Using direct paths"
fi

# Ensure log directory exists
mkdir -p /workspace/logs /tmp/logs

echo "[start.sh] 📝 Environment info:"
echo "  Python: $(python --version)"
echo "  Working dir: $(pwd)"
echo "  ComfyUI exists: $(ls -la /ComfyUI/ 2>/dev/null | head -3 || echo 'NOT FOUND')"
echo "  Models dir: $(ls -la /workspace/models/ 2>/dev/null | head -3 || echo 'NOT FOUND')"

echo "[start.sh] 🔥 Launching ComfyUI on port 8188"
cd /ComfyUI

# Start ComfyUI and capture BOTH stdout and stderr
echo "[start.sh] Starting ComfyUI with full error capture..."
python main.py --port 8188 --listen 0.0.0.0 > /tmp/comfyui_full.log 2>&1 &
COMFY_PID=$!

echo "[start.sh] ComfyUI started with PID: $COMFY_PID"

# Wait a moment for ComfyUI to either start or crash
sleep 5

# Check if process is still alive
if kill -0 $COMFY_PID 2>/dev/null; then
    echo "[start.sh] ✅ ComfyUI process still running after 5 seconds"
    
    # Show what ComfyUI is doing
    echo "[start.sh] 📝 ComfyUI startup logs:"
    tail -20 /tmp/comfyui_full.log 2>/dev/null || echo "No logs yet"
    
    # Function to check if ComfyUI is ready
    check_comfy_ready() {
        local max_attempts=60
        local attempt=0
        
        echo "[start.sh] ⏳ Waiting for ComfyUI to be ready..."
        
        while [ $attempt -lt $max_attempts ]; do
            # Check if process is still running
            if ! kill -0 $COMFY_PID 2>/dev/null; then
                echo "[start.sh] ❌ ComfyUI process died during startup!"
                echo "[start.sh] 📋 Full ComfyUI logs:"
                cat /tmp/comfyui_full.log
                return 1
            fi
            
            # Try to connect to ComfyUI API
            if curl -s -f http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
                echo "[start.sh] ✅ ComfyUI is ready!"
                return 0
            fi
            
            # Show progress every 10 attempts
            if [ $((attempt % 10)) -eq 0 ]; then
                echo "[start.sh] Still waiting... attempt $attempt/$max_attempts"
                echo "[start.sh] Recent ComfyUI logs:"
                tail -5 /tmp/comfyui_full.log 2>/dev/null || echo "  No logs yet"
            fi
            
            sleep 5
            attempt=$((attempt + 1))
        done
        
        echo "[start.sh] ❌ Timeout waiting for ComfyUI to start"
        return 1
    }
    
    # Wait for ComfyUI to be ready
    if check_comfy_ready; then
        echo "[start.sh] 🧪 ComfyUI startup complete. Final log check:"
        tail -10 /tmp/comfyui_full.log
        
        echo "[start.sh] 🔍 Testing ComfyUI endpoints:"
        echo "  System stats: $(curl -s http://127.0.0.1:8188/system_stats | jq -r '.system.comfyui_version // "ERROR"' 2>/dev/null || echo "FAILED")"
        echo "  Queue status: $(curl -s http://127.0.0.1:8188/queue | jq -r '.queue_pending | length' 2>/dev/null || echo "FAILED")"
        
        echo "[start.sh] 🧠 ComfyUI is ready! Keeping container alive..."
        # Keep container running (for regular pod)
        tail -f /tmp/comfyui_full.log
    else
        echo "[start.sh] ❌ ComfyUI failed to start properly"
        exit 1
    fi
    
else
    echo "[start.sh] ❌ ComfyUI process died immediately!"
    echo "[start.sh] 📋 Full error output:"
    cat /tmp/comfyui_full.log 2>/dev/null || echo "No logs captured"
    echo "[start.sh] 📋 Process list:"
    ps aux | grep -E "(python|comfy)" || true
    exit 1
fi