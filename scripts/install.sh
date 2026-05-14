#!/usr/bin/env bash
set -e

VENV_NAME="gomicro"
VENV_DIR="$HOME/venvs/$VENV_NAME"

TORCH_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl"
TORCHVISION_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl"
ORT_GPU_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl"

echo "==> [1/7] System packages (venv + TensorRT python)"
sudo apt update
sudo apt install -y python3.10-venv python3-pip \
  python3-libnvinfer python3-libnvinfer-dev

echo "==> [2/7] Recreate venv with system site packages enabled"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "==> [3/7] Install Ultralytics (no deps)"
pip install --no-deps ultralytics==8.4.13

echo "==> [4/7] Remove any wrong torch packages"
pip uninstall -y torch torchvision torchaudio || true

echo "==> [5/7] Install Jetson CUDA PyTorch wheels"
pip install "$TORCH_WHL"
pip install "$TORCHVISION_WHL"

echo "==> [6/7] Install export dependencies (ONNX + ORT-GPU)"
pip install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.71"
pip install "$ORT_GPU_WHL"

echo "==> [7/7] Final checks"
python -c "import ultralytics, torch; print('ultralytics', ultralytics.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -c "import tensorrt as trt; print('tensorrt', trt.__version__)"
python -c "import onnxruntime as ort; print('onnxruntime-gpu', ort.__version__)"

echo ""
echo "DONE ✅"
echo "To activate later: source $VENV_DIR/bin/activate"
echo "To export engine: yolo export model=best.pt format=engine device=0 half=True"
