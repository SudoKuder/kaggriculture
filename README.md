# Kaggriculture Training

This repository contains the strategic decision layer for the Kaggriculture environment. 

## Requirements
To install the dependencies, run:
```bash
python -m pip install -r requirements.txt
```

## Training on a GPU
The training script is designed to automatically detect and use an NVIDIA GPU if available. 
To ensure PyTorch can see your GPU, you must install the CUDA-enabled version of PyTorch. If `python -m pip install -r requirements.txt` does not install the CUDA version for your system, please visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for the exact installation command for your environment.

You can verify your GPU is working by running the quick smoke test:
```bash
python train_strategy.py --test
```

## Running Training
You can start a full training run and automatically export the weights for submission using the provided scripts:

### Windows
```cmd
train.bat
```

### Linux / macOS
```bash
chmod +x train.sh
./train.sh
```

### Manual Command
If you prefer running it manually:
```bash
# Full training run (2000 episodes)
python train_strategy.py --episodes 2000

# Export weights for submission
python -m strategy.export_weights strategy_checkpoints/actor_net_final.npz -o strategy_weights_inline.py
```
