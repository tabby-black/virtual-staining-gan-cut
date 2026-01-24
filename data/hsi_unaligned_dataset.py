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
import spectral

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
    def load_hsi(self, hdr_path):
      """
      Load ENVI hyperspectral cube
      returns tensor (C, H, W) - ie. tensor transposed to meet convention PyTorch expects
      """
      cube = spectral.open_image(hdr_path).load()
      cube = np.asarray(cube, dtype=np.float32) # (H, W, C) - needs transposing
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
        A_path = self.A_paths[index % self.A_size]

        # make sure index is within range
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
             # randomize the index for domain B to avoid fixed pairs
            index_B = random.randint(0, self.B_size - 1)
        
        B_path = self.B_paths[index_B]


        # commented this part out for now
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        # is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        # transform = get_transform(modified_opt)

        # replacing transforms with calls to load_hsi() and load_rgb() functions - transforms were carried out in these functions
        A = self.load_hsi(A_path)
        B = self.load_rgb(B_path)

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
