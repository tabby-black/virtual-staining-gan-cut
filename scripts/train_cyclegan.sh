set -ex
# --pool_size and --no_dropout flags have been added to example structure of training command
# have commented out CycleGAN training command for now to check if CUT training is error free
python train.py --dataroot ./datasets/histology --dataset_mode hsi_unaligned --name hsi_to_rgb_cyclegan --model cycle_gan --pool_size 50 --no_dropout --lambda_identity 0

# have commented out CUT training command for now - trying to train CycleGAN model
#python train.py --dataroot ./datasets/histology --dataset_mode hsi_unaligned --name hsi_to_rgb_CUT --CUT_mode CUT --nce_idt 0 --pool_size 50 --no_dropout

```
