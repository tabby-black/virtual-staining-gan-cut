# data preprocessing step 3

from spectral import open_image, envi
import numpy as np
import os
import glob

from PIL import Image
#from itertools import product


def patch_hsi(hdr_path, i=0):
    i+= 1
    # load hyperspectral cube
    img = open_image(hdr_path).load()
    H, W, B = img.shape
    patch_metadata = img.metadata.copy()

    dirname = os.path.dirname(hdr_path)
    basename = os.path.splitext(os.path.basename(hdr_path))[0]

    # compute 4 non-overlapping patches
    # I need to make patch width 500 rather than 502 so it is divisible by 4 for generator
    # save half heights and widths that are divisble by 4
    half_h = (H // 2) // 4 * 4
    half_w = (W // 2) // 4 * 4
    patches = {
        "top_left": img[0:half_h,   0:half_w, :],
        "top_right": img[0:half_h,   W-half_w:W, :],
        "bottom_left": img[H-half_h:H,   0:half_w, :],
        "bottom_right": img[H-half_h:H,   W-half_w:W, :]
    }
    print("Patches for hyperspectral image", i, "computed!")

    # change H and W in metadata before saving new hdr and cube files
    patch_metadata['lines'] = str(half_h)
    patch_metadata['samples'] = str(half_w)
    print("Hyperspectral image", i, "metadata amended for patches!")

    # save the 4 patches with the correct metadata
    for key, patch in patches.items():
        out_hdr = os.path.join(dirname, f"{basename}_{key}.hdr")
        envi.save_image(out_hdr, patch.astype(np.float32), dtype=np.float32, force=True, interleave='bsq', metadata=patch_metadata)

    print("Patches for hyperspectral image", i, "saved!")

    # remove the original full image
    # ran it once with this but don't want to remove the full images - not the end of the world because I have them all in datasets/raw
    #os.remove(hdr_path)
    #raw_path = hdr_path.replace(".hdr", "")
    #os.remove(raw_path)


def patch_rgb(image_path, i=0):
    i += 1
    rgb = Image.open(image_path)
    W, H = rgb.size

    dirname = os.path.dirname(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # compute 4 non-overlapping patches
    # safe half heights and half widths that are divisible by 4
    half_h = (H // 2) // 4 * 4
    half_w = (W // 2) // 4 * 4
    patches = {
        "top_left": rgb.crop((0,    0,  half_w,   half_h)),
        "top_right": rgb.crop((W-half_w, 0,  W,  half_h)),
        "bottom_left": rgb.crop((0, H-half_h,   half_w,   H)),
        "bottom_right": rgb.crop((W-half_w, H-half_h,   W,  H))
    }

    print("Patches for rgb image", i, "computed!")

    # save the 4 patches
    for key, patch in patches.items():
        out_png = os.path.join(dirname, f"{basename}_{key}.png")
        patch.save(out_png)

    print("Patches for rgb image", i, "saved!")

    # remove the original full image
    # ran it once with this but don't want to remove the full images - really don't want to remove full rgb images because I don't have a copy of them anywhere
    #os.remove(image_path)



# patch hsi images
# call this function on all images in trainA and testA - using glob
train_directory = "datasets/histology/trainA"
test_directory = "datasets/histology/testA"

#hdr_train_files = sorted(glob.glob(os.path.join(train_directory, "*.hdr")))

#hdr_test_files = sorted(glob.glob(os.path.join(test_directory, "*.hdr")))

hdr_test_files = ["P12_ROI_01_C10_T_raw_preprocessed.hdr", "P12_ROI_01_C11_T_raw_preprocessed.hdr", "P12_ROI_01_C12_T_raw_preprocessed.hdr", "P13_ROI_01_C01_T_raw_preprocessed.hdr", "P13_ROI_01_C02_T_raw_preprocessed.hdr", "P13_ROI_01_C03_T_raw_preprocessed.hdr", "P13_ROI_01_C04_T_raw_preprocessed.hdr", "P13_ROI_01_C05_T_raw_preprocessed.hdr", "P13_ROI_01_C06_T_raw_preprocessed.hdr", "P13_ROI_01_C07_T_raw_preprocessed.hdr", "P13_ROI_01_C08_T_raw_preprocessed.hdr", "P13_ROI_01_C09_T_raw_preprocessed.hdr", "P13_ROI_01_C10_T_raw_preprocessed.hdr", "P13_ROI_01_C11_T_raw_preprocessed.hdr", "P13_ROI_01_C12_T_raw_preprocessed.hdr", "P7_ROI_01_C07_NT_raw_preprocessed.hdr", "P7_ROI_01_C08_NT_raw_preprocessed.hdr", "P7_ROI_01_C09_NT_raw_preprocessed.hdr", "P7_ROI_01_C10_NT_raw_preprocessed.hdr", "P7_ROI_01_C11_NT_raw_preprocessed.hdr", "P7_ROI_01_C12_NT_raw_preprocessed.hdr", "P7_ROI_02_C01_NT_raw_preprocessed.hdr", "P7_ROI_02_C02_NT_raw_preprocessed.hdr", "P7_ROI_02_C03_NT_raw_preprocessed.hdr", "P7_ROI_02_C04_NT_raw_preprocessed.hdr", "P7_ROI_02_C05_NT_raw_preprocessed.hdr", "P7_ROI_02_C06_NT_raw_preprocessed.hdr", "P7_ROI_02_C07_NT_raw_preprocessed.hdr", "P7_ROI_02_C08_NT_raw_preprocessed.hdr", "P7_ROI_02_C09_NT_raw_preprocessed.hdr", "P7_ROI_02_C10_NT_raw_preprocessed.hdr", "P7_ROI_02_C11_NT_raw_preprocessed.hdr", "P7_ROI_02_C12_NT_raw_preprocessed.hdr", "P7_ROI_02_C13_NT_raw_preprocessed.hdr", "P7_ROI_02_C14_NT_raw_preprocessed.hdr", "P7_ROI_02_C15_NT_raw_preprocessed.hdr", "P7_ROI_02_C16_NT_raw_preprocessed.hdr", "P7_ROI_02_C17_NT_raw_preprocessed.hdr", "P7_ROI_02_C18_NT_raw_preprocessed.hdr", "P7_ROI_02_C19_NT_raw_preprocessed.hdr", "P8_ROI_01_C01_NT_raw_preprocessed.hdr", "P8_ROI_01_C02_NT_raw_preprocessed.hdr", "P8_ROI_01_C03_NT_raw_preprocessed.hdr", "P8_ROI_01_C04_NT_raw_preprocessed.hdr", "P8_ROI_01_C05_NT_raw_preprocessed.hdr", "P8_ROI_01_C06_NT_raw_preprocessed.hdr", "P8_ROI_01_C07_NT_raw_preprocessed.hdr", "P8_ROI_01_C08_NT_raw_preprocessed.hdr", "P8_ROI_01_C09_NT_raw_preprocessed.hdr", "P8_ROI_01_C10_NT_raw_preprocessed.hdr", "P8_ROI_01_C11_NT_raw_preprocessed.hdr", "P8_ROI_01_C12_NT_raw_preprocessed.hdr", "P8_ROI_01_C13_NT_raw_preprocessed.hdr", "P8_ROI_01_C14_NT_raw_preprocessed.hdr", "P8_ROI_01_C15_NT_raw_preprocessed.hdr", "P8_ROI_01_C16_NT_raw_preprocessed.hdr", "P8_ROI_01_C17_NT_raw_preprocessed.hdr", "P8_ROI_01_C18_NT_raw_preprocessed.hdr", "P8_ROI_01_C19_NT_raw_preprocessed.hdr", "P8_ROI_01_C20_NT_raw_preprocessed.hdr", "P8_ROI_02_C01_NT_raw_preprocessed.hdr", "P8_ROI_02_C02_NT_raw_preprocessed.hdr", "P8_ROI_02_C03_NT_raw_preprocessed.hdr", "P8_ROI_02_C04_NT_raw_preprocessed.hdr", "P8_ROI_02_C05_NT_raw_preprocessed.hdr", "P8_ROI_02_C06_NT_raw_preprocessed.hdr", "P8_ROI_02_C07_NT_raw_preprocessed.hdr", "P8_ROI_02_C08_NT_raw_preprocessed.hdr", "P8_ROI_02_C09_NT_raw_preprocessed.hdr", "P8_ROI_02_C10_NT_raw_preprocessed.hdr", "P8_ROI_02_C11_NT_raw_preprocessed.hdr", "P8_ROI_02_C12_NT_raw_preprocessed.hdr", "P8_ROI_02_C13_NT_raw_preprocessed.hdr", "P8_ROI_02_C14_NT_raw_preprocessed.hdr", "P8_ROI_02_C15_NT_raw_preprocessed.hdr", "P8_ROI_02_C16_NT_raw_preprocessed.hdr"]


#for hdr in hdr_train_files:
    #patch_hsi(hdr)

for hdr in hdr_test_files:
    patch_hsi(hdr)


# patch rgb images
# call this function on all images in trainB and testB - using glob
train_directory = "datasets/histology/trainB"
test_directory = "datasets/histology/testB"

png_train_files = sorted(glob.glob(os.path.join(train_directory, "*.png")))

png_test_files = sorted(glob.glob(os.path.join(test_directory, "*.png")))

for png in png_train_files:
    patch_rgb(png)

for png in png_test_files:
    patch_rgb(png)