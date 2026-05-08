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

    # H, W, C

    # compute 12 non-overlapping patches
    patches = {
        1 : img[16:272, 118:374, :],
        2 : img[16:272, 374:630, :],
        3 : img[16:272, 630:886, :],
        4 : img[272:528, 118:374, :],
        5: img[272:528, 374:630, :],
        6: img[272:528, 630:886, :],
        7: img[528:784, 118:374, :],
        8: img[528:784, 374:630, :],
        9: img[528:784, 630:886, :]
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
        1 : rgb.crop((118, 16, 374, 272)),
        2 : rgb.crop((374, 16, 630, 272)),
        3 : rgb.crop((630, 16, 886, 272)),
        4 : rgb.crop((118, 272, 374, 528, )),
        5: rgb.crop((374, 272, 630, 528)),
        6: rgb.crop((630, 272, 886, 528)),
        7: rgb.crop((118, 528, 374, 784)),
        8: rgb.crop((374, 528, 630, 784)),
        9: rgb.crop((630, 528, 886, 784))
    }

    print("Patches for rgb image", i, "computed!")

    # save the 4 patches
    for key, patch in patches.items():
        out_png = os.path.join(dirname, f"{basename}_{key}.png")
        patch.save(out_png)

    print("Patches for rgb image", i, "saved!")



# patch hsi images
# call this function on all images in testA - using glob
#test_directory = "datasets/histology_full/testA"
#hdr_test_files = sorted(glob.glob(os.path.join(test_directory, "*.hdr")))

#for fname in hdr_test_files:
    #hdr_path = os.path.join(test_directory, fname)
    #patch_hsi(fname)


# patch rgb images
# call this function on all images in testB - using glob
test_directory = "datasets/extra_test/new"
png_test_files = sorted(glob.glob(os.path.join(test_directory, "*.png")))

for png in png_test_files:
    patch_rgb(png)