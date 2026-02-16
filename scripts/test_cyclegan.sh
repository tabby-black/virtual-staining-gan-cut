set -ex
#python test.py --dataroot ./datasets/histology --name hsi_to_he_cyclegan --model cycle_gan --phase test --no_dropout

# test command to test model with weights from 5th epoch
python test.py --dataroot ./datasets/histology --dataset_mode hsi_unaligned --name hsi_to_rgb_cyclegan_run2 --model cycle_gan --phase train --gpu_ids 3 --epoch 55 --num_threads 0 --preprocess resize_and_crop --load_size 286 --crop_size 256 --no_dropout