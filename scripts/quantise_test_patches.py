# generate quantised testB datasets

import os
import numpy as np
from PIL import Image


def quantise_patch(patch, bits):
    levels = 2 ** bits - 1
    img_q = np.round(patch * levels) / levels

    return img_q.astype(np.float32)


current_test_directory = "datasets/histology_full/testB"
new_test_directory = "datasets/histology_full/testB_6b_16"

os.makedirs(new_test_directory, exist_ok=True)

i = 0
for filename in os.listdir(current_test_directory):
    i += 1
    # for every patch in the current test directory, quantise it and save to new test directory
    
    input_path = os.path.join(current_test_directory, filename)
    output_path = os.path.join(new_test_directory, filename)

    # load patch
    img = Image.open(input_path).convert("RGB")
    # normalise to [0,1]
    # calibration normalises hsi images but I haven't yet normalised rgb images - this is done by the data loader to [-1,1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_fp16 = img_np.astype(np.float16)
    # quantise patch
    img_q = quantise_patch(img_fp16.astype(np.float32), 6).astype(np.float16)
    # convert back to [0,255]
    img_q = (img_q * 255).clip(0, 255).astype(np.uint8)

    # save patch - with same name as original
    Image.fromarray(img_q).save(output_path)
    print("Quantised patch", i, " stored!")

