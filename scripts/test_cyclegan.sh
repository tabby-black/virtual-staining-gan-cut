set -ex
#python test.py --dataroot ./datasets/histology --name hsi_to_he_cyclegan --model cycle_gan --phase test --no_dropout


# CHANGE THIS LINE
python test.py --dataroot /content/data/testB_8b --checkpoints_dir /content/checkpoints/hsi_to_rgb_cyclegan_rep2quant8 --dataset_mode hsi_unaligned --model cycle_gan --num_test 846 --name hsi_to_rgb_cyclegan_rep2quant8 --epoch str(ep) --num_threads 0 --preprocess resize_and_crop --load_size 286 --batch_size 1 --crop_size 256 --no_dropout
# --CUT_mode CUT 
# --model cycle_gan
# --dataroot ./datasets/histology_full

# patched test images are in histology_full