# jupyter notebook transferred to python file for submission

from PIL import Image
import matplotlib.pyplot as plt


# Patch filenames in correct order
patch_paths = [
    "/local/scratch-3/tb789/projects/virtual-staining-gan-cut/results/hsi_to_rgb_CUTrep2a/test_75_16fp_visuals/images/fake_B/P2_ROI_01_C01_T_raw_preprocessed_1.png",
    "/local/scratch-3/tb789/projects/virtual-staining-gan-cut/results/hsi_to_rgb_CUTrep2a/test_75_16fp_visuals/images/fake_B/P2_ROI_01_C01_T_raw_preprocessed_2.png",
    "/local/scratch-3/tb789/projects/virtual-staining-gan-cut/results/hsi_to_rgb_CUTrep2a/test_75_16fp_visuals/images/fake_B/P2_ROI_01_C01_T_raw_preprocessed_3.png",
    "/local/scratch-3/tb789/projects/virtual-staining-gan-cut/results/hsi_to_rgb_CUTrep2a/test_75_16fp_visuals/images/fake_B/P2_ROI_01_C01_T_raw_preprocessed_4.png",
    "/local/scratch-3/tb789/projects/virtual-staining-gan-cut/results/hsi_to_rgb_CUTrep2a/test_75_16fp_visuals/images/fake_B/P2_ROI_01_C01_T_raw_preprocessed_5.png",
    "/local/scratch-3/tb789/projects/virtual-staining-gan-cut/results/hsi_to_rgb_CUTrep2a/test_75_16fp_visuals/images/fake_B/P2_ROI_01_C01_T_raw_preprocessed_6.png",
    "/local/scratch-3/tb789/projects/virtual-staining-gan-cut/results/hsi_to_rgb_CUTrep2a/test_75_16fp_visuals/images/fake_B/P2_ROI_01_C01_T_raw_preprocessed_7.png",
    "/local/scratch-3/tb789/projects/virtual-staining-gan-cut/results/hsi_to_rgb_CUTrep2a/test_75_16fp_visuals/images/fake_B/P2_ROI_01_C01_T_raw_preprocessed_8.png",
    "/local/scratch-3/tb789/projects/virtual-staining-gan-cut/results/hsi_to_rgb_CUTrep2a/test_75_16fp_visuals/images/fake_B/P2_ROI_01_C01_T_raw_preprocessed_9.png",
]

# Load images
patches = [Image.open(p) for p in patch_paths]

# Assume all patches same size
w, h = patches[0].size

# Create blank canvas
stitched = Image.new("RGB", (w * 3, h * 3))

# Paste patches into grid
for idx, patch in enumerate(patches):
    x = (idx % 3) * w
    y = (idx // 3) * h
    stitched.paste(patch, (x, y))

# Save or display
#stitched.save("stitched_image.png")

plt.imshow(stitched)
plt.axis("off")
plt.show()