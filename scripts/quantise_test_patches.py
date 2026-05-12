# generate quantised testB datasets

import os
import numpy as np
from PIL import Image
import spectral


def quantise_patch(patch, bits):
    levels = 2 ** bits - 1
    img_q = np.round(patch * levels) / levels

    return img_q.astype(np.float32)


current_test_directory = "datasets/histology_full/testA"
new_test_directory = "datasets/testA_8b"

os.makedirs(new_test_directory, exist_ok=True)

i = 0
for filename in os.listdir(current_test_directory):
    i += 1
    # for every patch in the current test directory, quantise it and save to new test directory
    if not filename.lower().endswith(".hdr"):
        continue

    hdr_input_path = os.path.join(current_test_directory, filename)
    # corresponding .img path is inferred automatically
    img = spectral.envi.open(hdr_input_path)

    # load hyperspectral cube
    patch = img.load().astype(np.float32)

    # calibration has already normalised hsi to [0,1]

    # quantise patch
    patch_q = quantise_patch(patch, 8)

    # output paths
    base_name = os.path.splitext(filename)[0]
    hdr_output_path = os.path.join(new_test_directory, base_name + ".hdr")
    

    # save new quantised .img and .hdr pair - with same names as originals
    spectral.envi.save_image(
        hdr_output_path,
        patch_q,
        dtype=np.float32,
        ext='.img',
        force=True
    )


    print("Quantised hyperspectral patch", i, " stored!")

