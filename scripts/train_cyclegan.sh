set -ex
python train.py --dataroot ./datasets/histology --dataset_mode hsi_unaligned --name hsi_to_rgb_cyclegan --model cycle_gan --pool_size 50 --no_dropout
```
