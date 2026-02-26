set -ex
#python test.py --dataroot ./datasets/histology --name hsi_to_he_cyclegan --model cycle_gan --phase test --no_dropout

# test command to test model with weights from 5th epoch
# CHANGE THIS LINE
python test.py --dataroot ./datasets/histology --dataset_mode hsi_unaligned --name hsi_to_rgb_cyclegan_bs3 --model cycle_gan --gpu_ids 3 --epoch str(ep) --num_threads 0 --preprocess resize_and_crop --load_size 286 --batch_size 1 --crop_size 256 --no_dropout