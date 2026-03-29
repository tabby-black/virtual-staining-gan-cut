# this script will eventually be combined into preprocess.py

# this script is to segment background pixels out of an image
# I may then go to average this over tiles, depending on my new patching strategy
# I will discuss my new patching strategy with Tiago

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_tissue_mask(image):
    """
    image: H x W x 3 (RGB), values in [0, 255]
    returns: H x W binary mask (1 = tissue, 0 = background)
    """

    # convert to grayscale using standard luminance formula
    gray = (
        0.299 * image[:, :, 0] +
        0.587 * image[:, :, 1] +
        0.114 * image[:, :, 2]
    )

    # normalize to [0, 1]
    gray = gray / 255.0

    # threshold
    # grayscale white value = 1
    # any pixels with grayscale values < 0.85 will be classified as tissue
    # any pixels with grayscale values >= 0.85 are light, so background
    threshold = 0.85

    # tissue = darker pixels = closer to 0
    mask = (gray < threshold).astype(np.uint8)

    # pixelwise binary mask classifying every pixel in the image as either
    # tissue (1) or background (0)
    return mask

# can check the tissue proportions of images
# inside "" is an absolute path to an image
# trainB because this contains RGB images
img = np.array(Image.open("/local/scratch-3/projects/virtual-staining-gan-cut/datasets/histology/trainB/P4_ROI_02_C31_NT_rgb.png/"))
mask = create_tissue_mask(img)

#print(mask.shape)  # (H, W)
#print(mask.mean())

# then visualise the mask applied to some RGB images to check this looks right
# better to visualise here or in a Jupyter notebook?
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Tissue mask")

plt.show()