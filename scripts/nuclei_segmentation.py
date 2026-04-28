# use cellpose to create nuclei segmentation mask on RGB images

import os
from cellpose import models, io
import numpy as np
from PIL import Image

def create_nuclei_segmentation_mask(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # specify that we are segmenting images by nuclei
    model = models.CellposeModel(model_type="nuclei", gpu=True)

    i = 0
    for filename in os.listdir(input_dir):
        i += 1
        # all files will end in .png since they are all RGB images
        path = os.path.join(input_dir, filename)

        img = io.imread(path)

        # just extracting masks from this
        masks, flows, styles = model.eval(
            img,
            channels=[1, 0],
            diameter=None,
            invert=True
        )

        save_name = os.path.splitext(filename)[0] + "_mask.npy"
        np.save(os.path.join(output_dir, save_name), masks)
        print("Nuclei segmentation mask", i, " saved!")

# create nuclei segmentation masks for all real RGB images
# input directory is the same as using the real_B folder inside one of the model epoch results folders
input_dir = "../datasets/histology_full/test_B"
output_dir = "../results/histology_analysis/real_RGB_nuclei_masks"
create_nuclei_segmentation_mask(input_dir, output_dir)

# and then for each of the 8 reporting runs (4 CycleGAN, 4 CUT)
# where x is the optimal epoch
# input_dir = ./results/hsi_to_rgb_cyclegan_rep1/test_x/images/fake_B
# output_dir = ./results/histology_analysis/cganrep1_RGB_nuclei_masks