set -ex
# --pool_size and --no_dropout flags have been added to example structure of training command
# have commented out CycleGAN training command for now to check if CUT training is error free
# have disabled visdom for now

# new cycleGAN run6.1 command - learning rate reduced by one order of magnitude
#python train.py --dataroot ./datasets/histology --dataset_mode hsi_unaligned --name hsi_to_rgb_cyclegan_run6.1 --model cycle_gan --continue_train --epoch 20 --epoch_count 21 --use_wandb --wandb_project_name hyperspectral_image_reconstruction --gpu_ids 1 --num_threads 0  --pool_size 50 --no_dropout --lambda_identity 0 --lr 0.00002 --batch_size 3 --preprocess resize_and_crop --load_size 286 --crop_size 256 --display_id -1 --print_freq 100 --display_freq 1000 --save_epoch_freq 1 --no_html --amp

# cyclegan bs8 run command
#python train.py --dataroot ./datasets/histology_full --dataset_mode hsi_unaligned --name hsi_to_rgb_cyclegan_bs8 --model cycle_gan --continue_train --epoch 36 --epoch_count 37 --use_wandb --wandb_project_name hyperspectral_image_reconstruction --gpu_ids 0 --num_threads 0 --pool_size 50 --no_dropout --lambda_identity 0 --batch_size 8 --preprocess resize_and_crop --load_size 286 --crop_size 160 --display_id -1 --print_freq 100 --display_freq 1000 --save_epoch_freq 1 --no_html --amp

# cyclegan decay2 run command
#python train.py --dataroot ./datasets/histology_full --dataset_mode hsi_unaligned --name hsi_to_rgb_cyclegan_run7 --model cycle_gan --continue_train --epoch 21 --epoch_count 22 --lr_policy linear --n_epochs 100 --n_epochs_decay 100 --use_wandb --wandb_project_name hyperspectral_image_reconstruction --gpu_ids 3 --num_threads 0 --lr 0.0002 --pool_size 50 --no_dropout --lambda_identity 0 --batch_size 1 --preprocess resize_and_crop --load_size 286 --crop_size 256 --display_id -1 --print_freq 100 --display_freq 1000 --save_epoch_freq 1 --no_html --amp

# first CycleGAN reporting run command!!! woohoo
python train.py --dataroot ./datasets/histology_full --dataset_mode hsi_unaligned --patch_mode overlapping --name hsi_to_rgb_cyclegan_rep1 --model cycle_gan --continue_train --epoch 163 --epoch_count 164 --lr_policy linear --n_epochs 100 --n_epochs_decay 100 --use_wandb --wandb_project_name hyperspectral_image_reconstruction --gpu_ids 0 --num_threads 0 --lr 0.0002 --pool_size 50 --no_dropout --lambda_identity 0 --batch_size 1 --preprocess resize_and_crop --load_size 286 --crop_size 256 --display_id -1 --print_freq 100 --display_freq 1000 --save_epoch_freq 1 --no_html --amp


# have commented out CUT training command for now - trying to train CycleGAN model
# --lr 0.0001

# get initial CUT training underway!
# expecting some debugging to get dual encoder to work
# I am wokring on starting a new CUT run with the new dataloader 
#python train.py --dataroot ./datasets/histology_full --dataset_mode hsi_unaligned --patch_mode overlapping --name hsi_to_rgb_CUT4 --CUT_mode CUT --use_wandb --wandb_project_name hyperspectral_image_reconstruction --nce_idt False --gpu_ids 3 --nce_layers 4,8,12,16 --lambda_NCE 0.5 --pool_size 50 --no_dropout --print_freq 100 --save_epoch_freq 1 --display_id -1 --num_threads 0 --no_html --batch_size 1
# --continue_train --epoch 75 --epoch_count 76