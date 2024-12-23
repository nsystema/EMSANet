@echo off
REM ============================================================
REM Setup Script: setup.bat
REM Description: Creates and configures the 'emsanet' Conda environment
REM ============================================================

REM Option 4: Create new Conda environment manually (follow-up work)
echo Creating Conda environment 'emsanet' with Python 3.8 and Anaconda...
call conda create -n emsanet python=3.8 anaconda -y

REM Activate the Conda environment
echo Activating the 'emsanet' environment...
call conda activate emsanet

REM Install remaining Conda dependencies
echo Installing Conda dependencies: PyTorch, torchvision, torchaudio, and CUDA support...
call conda install pytorch=1.13.0 torchvision=0.14.0 torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

REM Install remaining pip dependencies
echo Installing pip dependencies: OpenCV, torchmetrics, and wandb...
python -m pip install "opencv-python>=4.2.0.34"
python -m pip install torchmetrics==0.10.2
python -m pip install wandb==0.13.6

REM Install optional dependencies
REM Test dependencies and ./external only
echo Installing optional dependencies for ONNX support...
call conda install "protobuf<=3.19.1" -y
python -m pip install onnx==1.13.1
python -m pip install git+https://github.com/cocodataset/panopticapi.git

REM Dependencies for ./external only
echo Installing Detectron2...
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"

REM Install project-specific packages
echo Installing dataset package...
python -m pip install "git+https://github.com/TUI-NICR/nicr-scene-analysis-datasets.git@v0.7.0"

echo Installing multitask scene analysis package...
python -m pip install "git+https://github.com/TUI-NICR/nicr-multitask-scene-analysis.git" 

REM Final message
echo.
echo ============================================================
echo Environment 'emsanet' has been successfully set up.
echo ============================================================
echo.
pause

