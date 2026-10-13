# GFL 运行环境

## 操作系统
- OS：windows11 专业版
- 版本：26H2

## 硬件
- CPU：AMD Ryzen 5 6600H with Radeon Graphics (3.30 GHz)
- GPU：NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)
- GPU 显存：4GB
- RAM：16.0 GB (15.2 GB 可用)

## 软件
- Python：3.10.22
- PyTorch：2.3.1+cu121
- TorchVision：0.18.1+cu121
- CUDA：12.1
- NVIDIA Driver：617.14

## OpenMMLab 组件
- MMEngine：0.10.7
- MMCV：2.2.0
- MMDetection：3.3.0

## 验证命令

```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -c "import torchvision; print('TorchVision:', torchvision.__version__)"
python -c "import mmengine; print('MMEngine:', mmengine.__version__)"
python -c "import mmcv; print('MMCV:', mmcv.__version__)"
python -c "import mmdet; print('MMDetection:', mmdet.__version__)"