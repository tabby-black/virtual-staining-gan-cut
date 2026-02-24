set -ex
# --pool_size and --no_dropout flags have been added to example structure of training command
# have commented out CycleGAN training command for now to check if CUT training is error free
# have disabled visdom for now
python train.py --dataroot ./datasets/histology --dataset_mode hsi_unaligned --name hsi_to_rgb_cyclegan_run2 --model cycle_gan --use_wandb --wandb_project_name hyperspectral_image_reconstruction --gpu_ids 3 --num_threads 0 --pool_size 50 --no_dropout --lambda_identity 0 --batch_size 1 --preprocess resize_and_crop --load_size 286 --crop_size 256 --display_id -1 --print_freq 100 --display_freq 1000 --save_epoch_freq 5 --no_html --amp
# have commented out CUT training command for now - trying to train CycleGAN model

# get initial CUT training underway!
# expecting some debugging to get dual encoder to work
#python train.py --dataroot ./datasets/histology --dataset_mode hsi_unaligned --name hsi_to_rgb_CUT1 --CUT_mode CUT --use_wandb --wandb_project_name hyperspectral_image_reconstruction --nce_idt False --gpu_ids 2 --nce_layers 4,8,12,16 --pool_size 50 --no_dropout --print_freq 100 --display_id -1 --no_html
