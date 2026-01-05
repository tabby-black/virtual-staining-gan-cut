set -ex
python train.py --dataroot ./datasets/histology --name hsi_to_he_cyclegan --model cycle_gan --pool_size 50 --no_dropout  --use_wandb
```