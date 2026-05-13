# Step 1: use cellpose to create nuclei segmentation mask on RGB images

# Step 2: extract features from the folder of segmentation masks and save to a csv file

# Step 3: Go to nuclei_features.ipynb notebook to plot graphs of these features

import os
from cellpose import models, io
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import regionprops_table

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


# saves nuclei counts, mean nuclei areas, mean nuclei eccentricities and image names to a csv in histology_analysis
def extract_features_to_csv(mask_dir, output_csv_dir):

    image_names = []
    # array of nuclei count of each real RGB test image
    nuclei_counts = []
    mean_areas = []
    mean_eccentricities = []

    i = 0
    for filename in os.listdir(mask_dir):
        i += 1
        mask_path = os.path.join(mask_dir, filename)
        mask = np.load(mask_path)
    
        count = mask.max()
        # add image names to array in file
        image_names.append(filename.replace("_mask.npy", ""))
        # add nuclei count to array in file
        nuclei_counts.append(count)
        print("Nuclei count in image", i, "added to list!")

        # calculate the mean nuclei area and eccentricities
        props = regionprops_table(
            mask,
            properties=["area", "eccentricity"]
        )
        areas = props["area"]
        eccentricities = props["eccentricity"]
        # handle edge case: no nuclei detected
        if len(areas) > 0:
            mean_area = np.mean(areas)
            mean_eccentricity = np.mean(eccentricities)
        else:
            mean_area = np.nan
            mean_eccentricity = np.nan

        # add the mean nuclei area and eccentricity to their relevant lists
        mean_areas.append(mean_area)
        mean_eccentricities.append(mean_eccentricity)
        print("Mean area and eccentricity in image", i, "added to list!")
    

    #nuclei_counts = np.array(nuclei_counts)
    #print(nuclei_counts)

    # save nuclei counts so don't have to recompute each time
    df = pd.DataFrame({
        "image_name": image_names,
        "nuclei_count": nuclei_counts,
        "mean_areas": mean_areas,
        "mean_eccentricities": mean_eccentricities
    })

    # csv file = real_RGB_nuclei_counts.csv
    df.to_csv(output_csv_dir, index=False)


# create nuclei segmentation masks for all real RGB images
# input directory is the same as using the real_B folder inside one of the model epoch results folders
# calling this script from main project folder
input_dir = "results/hsi_to_rgb_CUTrep2a/test_75_6b/images/fake_B"
output_dir = "results/histology_analysis/cut_6b_fake_RGB_nuclei_masks"
create_nuclei_segmentation_mask(input_dir, output_dir)

# and then for each of the 8 reporting runs (4 CycleGAN, 4 CUT)
# where x is the optimal epoch
# input_dir = ./results/hsi_to_rgb_cyclegan_rep1/test_x/images/fake_B
# output_dir = ./results/histology_analysis/cganrep1_RGB_nuclei_masks


mask_dir = "results/histology_analysis/cut_6b_fake_RGB_nuclei_masks"
output_csv_dir = "results/histology_analysis/cut_6b_fake_RGB_nuclei_features.csv"
extract_features_to_csv(mask_dir, output_csv_dir)