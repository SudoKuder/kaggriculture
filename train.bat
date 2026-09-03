@echo off
echo Installing requirements...
pip install -r requirements.txt
echo.
echo Starting training...
python train_strategy.py --episodes 2000
echo.
echo Exporting weights...
python -m strategy.export_weights strategy_checkpoints/actor_net_final.npz -o strategy_weights_inline.py
echo.
echo Training complete! You can now submit main.py along with strategy_weights_inline.py
pause
