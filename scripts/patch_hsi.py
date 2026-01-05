# data preprocessing step 3

from spectral import open_image, envi
import numpy as np
import os
import glob

from PIL import Image
#from itertools import product

# patch hyperspectral images
def patch_hsi(hdr_path):
    # load hyperspectral cube
    img = open_image(hdr_path).load
    H, W, B = img.shape
    patch_metadata = img.metadata.copy()

    dirname = os.path.dirname(hdr_path)
    basename = os.path.splitext(os.path.basename(hdr_path))[0]

    # compute 4 non-overlapping patches
    patches = {
        "top_left": img[0:H//2,   0:W//2, :],
        "top_right": img[0:H//2,   W//2:W, :],
        "bottom_left": img[H//2:H,   0:W//2, :],
        "bottom_right": img[H//2:H,   W//2:W, :]
    }

    # change H and W in metadata before saving new hdr and cube files
    patch_metadata['lines'] = str(H//2)
    patch_metadata['samples'] = str(W//2)

    # save the 4 patches with the correct metadata
    for key, patch in patches.items():
        out_hdr = os.path.join(dirname, f"{basename}_{key}.hdr")
        envi.save_image(out_hdr, patch.astype(np.float32), dtype=np.float32, force=True, interleave='bsq', metadata=patch_metadata)

    # remove the original full image
    os.remove(hdr_path)
    raw_path = hdr_path.replace(".hdr", "")
    os.remove(raw_path)

# call this function on all images in trainA and testA - using glob
train_directory = "/datasets/trainA"
test_directory = "/datasets/testA"

hdr_train_files = sorted(glob.glob(os.path.join(train_directory, "*.hdr")))

hdr_test_files = sorted(glob.glob(os.path.join(test_directory, "*.hdr")))

for hdr in hdr_train_files:
    patch_hsi(hdr)

for hdr in hdr_test_files:
    patch_hsi(hdr)



# patch rgb images
def patch_rgb(image_path):
    rgb = Image.open(image_path)
    W, H = rgb.size

    dirname = os.path.dirname(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # compute 4 non-overlapping patches
    patches = {
        "top_left": rgb.crop((0,    0,  W//2,   H//2)),
        "top_right": rgb.crop((W//2, 0,  W,  H//2)),
        "bottom_left": rgb.crop((0, H//2,   W//2,   H)),
        "bottom_right": rgb.crop((W//2, H//2,   W,  H))
    }

    # save the 4 patches
    for key, patch in patches.items():
        out_png = os.path.join(dirname, f"{basename}_{key}.png")
        patch.save(out_png)

    # remove the original full image
    os.remove(image_path)


# call this function on all images in trainB and testB - using glob
train_directory = "/datasets/trainB"
test_directory = "/datasets/testB"

png_train_files = sorted(glob.glob(os.path.join(train_directory, "*.png")))

png_test_files = sorted(glob.glob(os.path.join(test_directory, "*.png")))

for png in png_train_files:
    patch_rgb(png)

for png in png_test_files:
    patch_rgb(png)