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
        _, h, w = tensor.shape

        if h < crop_size or w < crop_size:
            raise ValueError(f"Crop size {crop_size} is larger than image size {(h, w)}")

        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        return tensor[:, top:top + crop_size, left:left + crop_size]

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
    def patch_hsi(hdr_path, i=0):
        i+= 1
        # load hyperspectral cube
        img = open_image(hdr_path).load()
        H, W, B = img.shape
        patch_metadata = img.metadata.copy()

        dirname = os.path.dirname(hdr_path)
        basename = os.path.splitext(os.path.basename(hdr_path))[0]

        # compute 20 overlapping patches
        # 256 x 256
        patches = {
            1 : img[0:256,   0:256, :],
            2 : img[0:256,   187:443, :],
            3 : img[0:256,   374:630, :],
            4 : img[0:256,   561:817, :],
            5 : img[0:256,   748:1004, :],
            6 : img[181:437,   0:256, :],
            7 : img[181:437,   187:443, :],
            8 : img[181:437,   374:630, :],
            9 : img[181:437,   561:817, :],
            10 : img[181:437,   748:1004, :],
            11 : img[362:618,   0:256, :],
            12 : img[362:618,   187:443, :],
            13 : img[362:618,   374:630, :],
            14 : img[362:618,   561:817, :],
            15 : img[362:618,   748:1004, :],
            16 : img[543:799,   0:256, :],
            17 : img[543:799,   187:443, :],
            18 : img[543:799,   374:630, :],
            19 : img[543:799,   561:817, :],
            20 : img[543:799,   748:1004, :]
        }
        print("Patches for hyperspectral image", i, "computed!")

        # change H and W in metadata before saving new hdr and cube files
        patch_metadata['lines'] = str(256)
        patch_metadata['samples'] = str(256)
        print("Hyperspectral image", i, "metadata amended for patches!")

        # save the 20 patches with the correct metadata
        for key, patch in patches.items():
            out_hdr = os.path.join(dirname, f"{basename}_{key}.hdr")
            # this automatically saves the patches with .img extensions even though 
            # the original hyperspectral images don't have extensions
            envi.save_image(out_hdr, patch.astype(np.float32), dtype=np.float32, force=True, interleave='bsq', metadata=patch_metadata)

        print("Patches for hyperspectral image", i, "saved!")


    def patch_rgb(image_path, i=0):
        i += 1
        rgb = Image.open(image_path)
        W, H = rgb.size

        dirname = os.path.dirname(image_path)
        basename = os.path.splitext(os.path.basename(image_path))[0]

        # compute 20 overlapping patches
        # 256 x 256

        patches = {
            1 : rgb.crop((0,   0,   256,   256)),
            2 : rgb.crop((0, 187, 443, 256)),
            3 : rgb.crop((0, 374, 630, 256)),
            4 : rgb.crop((0, 561, 817, 256)),
            5 : rgb.crop((0, 748, 1004, 256)),
            6 : rgb.crop((181, 0, 256, 437)),
            7 : rgb.crop((181, 187, 443, 437)),
            8 : rgb.crop((181, 374, 630, 437)),
            9 : rgb.crop((181, 561, 817, 437)),
            10 : rgb.crop((181, 748, 1004, 437)),
            11 : rgb.crop((362, 0, 256, 618)),
            12 : rgb.crop((362, 187, 443, 618)),
            13 : rgb.crop((362, 374, 630, 618)),
            14 : rgb.crop((362, 561, 817, 618)),
            15 : rgb.crop((362, 748, 1004, 618)),
            16 : rgb.crop((543, 0, 256, 799)),
            17 : rgb.crop((543, 187, 443, 799)),
            18 : rgb.crop((543, 374, 630, 799)),
            19 : rgb.crop((543, 561, 817, 799)),
            20 : rgb.crop((543, 748, 1004, 799))
        }

        # (1:4, 2:3)
        # 1 2 3 4

        print("Patches for rgb image", i, "computed!")

        # save the 4 patches
        for key, patch in patches.items():
            out_png = os.path.join(dirname, f"{basename}_{key}.png")
            patch.save(out_png)

        print("Patches for rgb image", i, "saved!")


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


        # commented this part out for now
        # reducing late-stage randomness in CUT/CycleGAN, if in finetuning phase (learning rate is decaying)

        # is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        # transform = get_transform(modified_opt)

        # replacing transforms with calls to load_hsi() and load_rgb() functions - transforms were carried out in these functions
        
        A = self.load_hsi(A_path)
        B = self.load_rgb(B_path)

        crop_size = self.opt.crop_size

        try:
            # safety check to make sure patches A and B are the same size
            #if A.shape[1:] != B.shape[1:]:
                #raise ValueError(f"A and B have different spatial sizes: {A.shape} vs {B.shape}")
            if self.opt.isTrain:

                if self.opt.patch_mode == 'overlapping':
                    # extract overlapping patches online
                    # insert patch_hsi.py logic in here
                    # compute dictionary of 20 patches of hsi (A) and rgb (B)
                    # will know what to do with these once I have heard back from Tiago
                    # don't need to save them all to disk each time likein patch script?
                    pass

                else:
                    # insert random crop logic here
                    A = self.random_crop(A, crop_size)
                    B = self.random_crop(B, crop_size)

            else:
                # for testing we need to crop the same part of the hsi and rgb image pairs
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
