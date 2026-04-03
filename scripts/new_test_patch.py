# data preprocessing step 3

from spectral import open_image, envi
import numpy as np
import os
import glob

from PIL import Image


def patch_hsi(hdr_path, i=0):
    i+= 1
    # load hyperspectral cube
    img = open_image(hdr_path).load()
    # don't need to transpose because this is to pre-compute patches
    # don't need compatability with PyTorch
    patch_metadata = img.metadata.copy()

    dirname = os.path.dirname(hdr_path)
    basename = os.path.splitext(os.path.basename(hdr_path))[0]

    # compute 12 non-overlapping patches
    patches = {
        1 : img[0:256, 0:256, :],
        2 : img[0:256, 256:512, :],
        3 : img[0:256, 512:778, :],
        4 : img[0:256, 778:1004, :],
        5: img[256:512, 0:256, :],
        6: img[256:512, 256:512, :],
        7: img[256:512, 512:778, :],
        8: img[256:512, 778:1004, :],
        9: img[512:778, 0:256, :]
    }

  
    print("Patches for hyperspectral image", i, "computed!")

    # change H and W in metadata before saving new hdr and cube files
    patch_metadata['lines'] = str(256)
    patch_metadata['samples'] = str(256)
    print("Hyperspectral image", i, "metadata amended for patches!")

    # save the 4 patches with the correct metadata
    for key, patch in patches.items():
        out_hdr = os.path.join(dirname, f"{basename}_{key}.hdr")
        # this automatically saves the patches with .img extensions even though 
        # the original hyperspectral images don't have extensions
        envi.save_image(out_hdr, patch.astype(np.float32), dtype=np.float32, force=True, interleave='bsq', metadata=patch_metadata)

    print("Patches for hyperspectral image", i, "saved!")


def patch_rgb(image_path, i=0):
    i += 1
    rgb = Image.open(image_path)

    dirname = os.path.dirname(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # compute 12 non-overlapping patches
    # (left, upper, right, lower)
    patches = {
        1 : rgb.crop((0, 0, 256, 256)),
        2 : rgb.crop((256, 0, 512, 256)),
        3 : rgb.crop((512, 0, 768, 256)),
        4 : rgb.crop((0, 256, 256, 512, )),
        5: rgb.crop((256, 256, 512, 512)),
        6: rgb.crop((512, 256, 768, 512)),
        7: rgb.crop((0, 512, 256, 768)),
        8: rgb.crop((256, 512, 512, 768)),
        9: rgb.crop((512, 512, 768, 768))
    }

    print("Patches for rgb image", i, "computed!")

    # save the 4 patches
    for key, patch in patches.items():
        out_png = os.path.join(dirname, f"{basename}_{key}.png")
        patch.save(out_png)

    print("Patches for rgb image", i, "saved!")



# patch hsi images
# call this function on all images in testA - using glob
test_directory = "datasets/histology/testA"
hdr_test_files = sorted(glob.glob(os.path.join(test_directory, "*.hdr")))

for fname in hdr_test_files:
    #hdr_path = os.path.join(test_directory, fname)
    patch_hsi(hdr_test_files)


# patch rgb images
# call this function on all images in testB - using glob
test_directory = "datasets/histology/testB"
png_test_files = sorted(glob.glob(os.path.join(test_directory, "*.png")))

for png in png_test_files:
    patch_rgb(png)