# calculates the ssim and psnr values for each of the output images in an experiment, and then averages over these
# ssim = Structural Similarity Index
# psnr = Peak Signal-Noise Ratio

import os
import numpy as np
import imageio.v2 as imageio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr



def evaluate_experiment(results_dir, data_range=1.0, channel_axis=2):
    """
    Computes SSIM and PSNR for all fake_B vs real_B image pairs in the results directory for a given CycleGAN/CUT experiment.

    Parameters:
    results_dir (str): path to results/[experiment name]/test_[epoch name]/images directory
    data_range (float): 1.0 for normalised images (I normalised images in calibration)
    channel_axis (int): 2 for RGB images 
    """

    fake_images = sorted([f for f in os.listdir(results_dir) if f.endswith("_fake_B.png")])

    # add the ssim and psnr scores for each image to a list
    # will then calculate the mean of this to give experiment results
    ssim_scores = []
    psnr_scores = []

    i = 0
    for fake_image_name in fake_images:
        i += 1
        prefix = fake_image_name.replace("_fake_B.png", "")
        real_image_name = prefix + "_real_B.png"

        fake_image_path = os.path.join(results_dir, fake_image_name)
        real_image_path = os.path.join(results_dir, real_image_name)

        # open real and generated/fake images
        fake_image = imageio.imread(fake_image_path).astype(np.float32)
        real_image = imageio.imread(real_image_path).astype(np.float32)

        # check that real and generated output stained rgb images have the same shape
        if fake_image.shape != real_image.shape:
            print("Error: shape mismatch")

        # generate the ssim score for the current image pair and add to list of ssim scores
        ssim_score = ssim(fake_image, real_image, channel_axis=channel_axis, data_range=data_range)
        ssim_scores.append(ssim_score)

        # generate the psnr score for the current image pair and add to list of psnr scores
        psnr_score = psnr(real_image, fake_image, data_range=data_range)
        psnr_scores.append(psnr_score)

        print("SSIM and PSNR scores for output RGB image", i, "computed!")

    # calculate the mean and standard deviations of the ssim and psnr scores to get experiment results
    ssim_mean = np.mean(ssim_scores)
    ssim_std = np.std(ssim_scores)

    psnr_mean = np.mean(psnr_scores)
    psnr_std = np.std(psnr_scores)

    # print the evaluation results for the experiment
    print("Evaluation Results")
    print(f"Images evaluated: {len(ssim_scores)}")
    print(f"SSIM : {ssim_mean:.4f} ± {ssim_std:.4f}")
    # the unit of PSNR is decibels (dB)
    print(f"PSNR : {psnr_mean:.2f} ± {psnr_std:.2f} dB")



# leave the results path blank for now - can populate after testing
results_path = ""
evaluate_experiment(results_path)