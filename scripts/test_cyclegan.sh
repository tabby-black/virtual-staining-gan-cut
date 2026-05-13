set -ex
#python test.py  --dataroot ./datasets/histology --name hsi_to_he_cyclegan --model cycle_gan --phase test --no_dropout


# CHANGE THIS LINE
python test.py --fp16 --result_suffix _8b_16fp_visuals --dataroot ./datasets/histology_full_visuals --dataset_mode hsi_unaligned --name hsi_to_rgb_CUTrep2a --CUT_mode CUT  --gpu_ids 0 --num_test 846 --epoch str(ep) --num_threads 0 --preprocess resize_and_crop --load_size 286 --batch_size 1 --crop_size 256 --no_dropout
# --CUT_mode CUT 
# --model cycle_gan
# --dataroot ./datasets/histology_full
# --result_suffix 6b_16fp

# patched test images are in histology_full