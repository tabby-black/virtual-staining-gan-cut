set -ex
python test.py --dataroot ./datasets/histology --name hsi_to_he_cyclegan --model cycle_gan --phase test --no_dropout
