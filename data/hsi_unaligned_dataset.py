# I have imported my unaligned_dataset code which I will edit to create my custom dataset loader
# custom dataset loader is the only edit I need to make for hyperspectral data to be loaded properly - as long my tensors have the right shape

# what I need this custom dataset loader to do:
# load (H, W, 275)
# transpose to (275, H, W) - this is the tensor convention that PyTorch models expect
# convert to torch.float32

import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util

# extra imports for custom data loader
import numpy as np
import torch
from spectral import open_image, envi
import glob

# need to make sure I use this new class instead of UnalignedDataset in the rest of my code / specify in training
class HSIUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It loads datasets where:
    A = hyperspectral (.hdr/.img)
    B = RGB image
    (rather than typical UnalignedDataset which assumes A and B are both rgb image)

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    # this function is the same as in UnalignedDataset except for assertions to make it obvious if HSI images haven't been loaded properly so aren't recognised by the model
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
      
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        # load images from '/path/to/data/trainA'
        # don't use make_dataset() because this only loads certain file types
        self.A_paths = sorted([
            os.path.join(self.dir_A, f)
            for f in os.listdir(self.dir_A)
            if f.endswith(".img")
        ])
        # load images from '/path/to/data/trainB'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        # assertions to make it obvious is HSI data hasn't been loaded as intended
        assert self.A_size > 0, f"No HSI files found in {self.dir_A}"
        assert self.B_size > 0, f"No RGB files found in {self.dir_B}"

    # add load_hsi and load_rgb functions to separate the loading of these different image types
    # load rgb images in the same way we load hyperspectral images for consistency - e.g. transposee explicitly rather than inside ToTensor()
    # be explicit about replacing .img with .hdr to ensure that spectral is loading the cube from the .hdr path not the .img path - this won't work
    def load_hsi(self, img_path):
      """
      Load ENVI hyperspectral cube
      returns tensor (C, H, W) - ie. tensor transposed to meet convention PyTorch expects
      """
      hdr_path = img_path.replace(".img", ".hdr")

      # add an error message to notify if there is a missing .hdr file
      if not os.path.exists(hdr_path):
          raise FileNotFoundError(f"Missing .hdr file for {img_path}")
          
      cube = spectral.open_image(hdr_path).load()
      # make cube float and writable
      cube = cube.astype(np.float32, copy=True)
      
      cube = np.transpose(cube, (2, 0, 1)) # (C, H, W)
      
      return torch.from_numpy(cube)
      

    def load_rgb(self, path):
      """
      Load RGB image as in UnalignedDataset, but in same way as hyperspectral image for consistency
      """
      img = Image.open(path).convert('RGB')
      img = np.array(img, dtype=np.float32) / 255.0
      img = np.transpose(img, (2, 0, 1))
      return torch.from_numpy(img)


    # helper function to random crop patches for training
    def random_crop(self, tensor, crop_size):
        # PyTorch compatible tensor is (C, H, W)
        _, h, w = tensor.shape

        if h < crop_size or w < crop_size:
            raise ValueError(f"Crop size {crop_size} is larger than image size {(h, w)}")

        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        patch = tensor[:, top:top + crop_size, left:left + crop_size]

        # use tissue mask to check if patch is >= 80% tissue
        mask = self.create_tissue_mask(patch)

        tissue_ratio = mask.mean()

        if tissue_ratio < 0.8:
            patch = self.random_crop(tensor, crop_size)

        # return a patch that is >= 80% tissue
        return patch


    # helper function to get centre crop coords for testing
    # these coords will then be used to crop both A and B
    def get_centre_crop_coords(self, tensor, crop_size):
        _, h, w = tensor.shape
        
        if h < crop_size or w < crop_size:
            raise ValueError(f"Crop size {crop_size} is larger than image size {(h, w)}")
        
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        
        return top, left
    
    
    # helper function to apply a centre crop for testing
    # allows same coordinates to be used to crop A and B patches
    def apply_crop(self, tensor, top, left, crop_size):
        return tensor[:, top:top + crop_size, left:left + crop_size]

    # compute overlapping 256x256 patches
    # called on both hsi and rgb images
    def get_overlapping_patch(tensor, patch_num=None):

        # tensor has been transposed so: (C, H, W) rather than (H, W, C)

        # randomly select the index of the overlapping patch to use if not provided i.e. in training
        if patch_num is None:
            patch_num = random.randint(1, 20)

        # compute the relevant patch
        # 256 x 256
        if patch_num == 1:
            patch = tensor[:, 0:256,   0:256]
        elif patch_num == 2:
            patch = tensor[:, 0:256,   187:443]
        elif patch_num == 3:
            patch = tensor[:, 0:256,   374:630]
        elif patch_num == 4:
            patch = tensor[:, 0:256,   561:817]
        elif patch_num == 5:
            patch = tensor[:, 0:256,   748:1004]
        elif patch_num == 6:
            patch = tensor[:, 181:437,   0:256]
        elif patch_num == 7:
            patch = tensor[:, 181:437,   187:443]
        elif patch_num == 8:
            patch = tensor[:, 181:437,   374:630]
        elif patch_num == 9:
            patch = tensor[:, 181:437,   561:817]
        elif patch_num == 10:
            patch = tensor[:, 181:437,   748:1004]
        elif patch_num == 11:
            patch = tensor[:, 362:618,   0:256]
        elif patch_num == 12:
            patch = tensor[:, 362:618,   187:443]
        elif patch_num == 13:
            patch = tensor[:, 362:618,   374:630]
        elif patch_num == 14:
            patch = tensor[:, 362:618,   561:817]
        elif patch_num == 15:
            patch = tensor[:, 362:618,   748:1004]
        elif patch_num == 16:
            patch = tensor[:, 543:799,   0:256]
        elif patch_num == 17:
            patch = tensor[:, 543:799,   187:443]
        elif patch_num == 18:
            patch = tensor[:, 543:799,   374:630]
        elif patch_num == 19:
            patch = tensor[:, 543:799,   561:817]
        else:
            patch = tensor[:, 543:799,   748:1004]
        

        # return this tensor patch
        return patch


    def create_tissue_mask(image):
        """
        image: 3 x H x W (RGB), values in [0, 255]
        returns: H x W binary mask (1 = tissue, 0 = background)
        """

        # tensor is (C, H, W)

        # convert rgb image to grayscale using standard luminance formula
        gray = (
            0.299 * image[0, :, :] +
            0.587 * image[1, :, :] +
            0.114 * image[2, :, :]
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
        # i.e. tissue is white, background is black
        # mask is an array of 1s and 0s
        return mask


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # make sure index is within range
        
        
        # TODO: Change A and B paths to point to dataset of
        # whole images rather than pre-patched
        A_path = self.A_paths[index % self.A_size]
        # use this B_path because I have paired (even if not perfectly) data
        # rather than totally unpaired data
        # B_path = self.B_paths[index % self.B_size]

        # make sure index is within range
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
             # randomize the index for domain B to avoid fixed pairs
            index_B = random.randint(0, self.B_size - 1)
        
        B_path = self.B_paths[index_B]

        # use load_hsi function because this includes transposing tensors from (H, W, C) to (C, H, W) so it is compatable with PyTorch
        A = self.load_hsi(A_path)
        B = self.load_hsi(B_path)

        # commented this part out for now
        # reducing late-stage randomness in CUT/CycleGAN, if in finetuning phase (learning rate is decaying)

        # is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        # transform = get_transform(modified_opt)

        
        crop_size = self.opt.crop_size

        try:
            # training - get patches independently from A and B for unpaired training
            if self.opt.isTrain:

                if self.opt.patch_mode == 'overlapping':
                    # randomly choose 1 of 20 overlapping patches per image
                    A = self.overlapping_patch(A)
                    B = self.get_overlapping_patch(B)

                else:
                    # segment out background and randomly crop a patch from the tissue
                    # random_crop calls create_tissue_mask to ensure the random patch it returns is >= 80% tissue
                    A = self.random_crop(A, crop_size)
                    B = self.random_crop(B, crop_size)

                    # can include in here to use centre_crop in decay phase of training

            else:
                # testing
                # patch the same part of the hsi and rgb image pairs
                # use overlapping patching strategy and test on all overlapping pairs

                
                top, left = self.get_centre_crop_coords(A, crop_size)
                
                A = self.apply_crop(A, top, left, crop_size)
                B = self.apply_crop(B, top, left, crop_size)

        except Exception as e:
            print(f"Bad sample: A_path={A_path}, B_path={B_path}, error={e}")
            # resample a different index
            new_index = random.randint(0, self.__len__() - 1)
            return self.__getitem__(new_index)

        # sanity check to check that cropping has happened
        if index == 0:
            print("A shape:", A.shape)
            print("B shape:", B.shape)

        # A is the hsi patch tensor - cropped part of a loaded hsi image
        return {'A': A,
                'B': B,
                'A_paths': A_path,
                'B_paths': B_path
               }

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
