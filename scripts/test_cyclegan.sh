set -ex
#python test.py --dataroot ./datasets/histology --name hsi_to_he_cyclegan --model cycle_gan --phase test --no_dropout


# CHANGE THIS LINE
python test.py --fp16 --dataroot ./datasets/histology_full_quant --dataset_mode hsi_unaligned --name hsi_to_rgb_cyclegan_rep2 --model cycle_gan --gpu_ids 2 --num_test 846 --epoch str(ep) --num_threads 0 --preprocess resize_and_crop --load_size 286 --batch_size 1 --crop_size 256 --no_dropout
# --CUT_mode CUT 
# --model cycle_gan
# --dataroot ./datasets/histology_full

# patched test images are in histology_full