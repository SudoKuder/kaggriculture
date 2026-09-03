#!/bin/bash
set -e

echo "Installing requirements..."
python3 -m pip install -r requirements.txt

echo -e "\nStarting training..."
python3 train_strategy.py --episodes 2000

echo -e "\nExporting weights..."
python3 -m strategy.export_weights strategy_checkpoints/actor_net_final.npz -o strategy_weights_inline.py

echo -e "\nTraining complete! You can now submit main.py along with strategy_weights_inline.py"
