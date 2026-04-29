"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import torch

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # from cycleGAN repo
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # this line is not used later in the code!
    #train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # added code to manually handle where the generated output images get saved now I have removed the
    # html pipeline
    # directory for saved test images
    #results_dir = os.path.join(
        #opt.results_dir,
        #opt.name,
        #f'{opt.phase}_{opt.epoch}'
    #)

    # 'images' subfolder inside each epoch's test folder
    #images_dir = os.path.join(results_dir, "images")
    
    #util.mkdirs(images_dir)
    #print('Saving test results to', images_dir)

    if opt.eval:
        model.eval()
    
    for i, data in enumerate(dataset):
        if i == 0:
            # added selection statement in here - only CUT / FastCUT training need a data-dependent initialisation step, not CycleGAN
            # same fix as in train.py
            if opt.model in ['cut', 'fastcut']:
                model.data_dependent_initialize(data)
            
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            
            if opt.eval:
                model.eval()
        
        # removed restriction on number of testing images
        # want to test on all testing images in final tests, not a subset
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        
        # this is needed for saving images, not just for visualizer
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        # debug statement to check that we have all 4 dictionaries for visualiser
        print("visual keys:", visuals.keys())

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        # aspect_ratio from cycleGAN repo
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        
        # added code to save output images directly, without html
        # keep full original filename
        img_name = os.path.basename(img_path[0])
        # specify the file extension to save images with
        img_stem, _ = os.path.splitext(img_name)
        img_name = f"{img_stem}.png"
        

        # explicitly only save real_B and fake_B images to avoid running into problems with 
        # hyperspectral image dimensions
        #for label in ['real_B', 'fake_B']:
            #if label not in visuals:
                #continue

            #image = visuals[label]
            #image_numpy = util.tensor2im(image)

            #label_dir = os.path.join(results_dir, "images", label)
            #util.mkdirs(label_dir)

            #save_path = os.path.join(label_dir, img_name)

            #print(f"[DEBUG] Saving {label} -> {save_path}")
            #util.save_image(image_numpy, save_path, aspect_ratio=opt.aspect_ratio)

            #assert os.path.exists(save_path), f"File not saved: {save_path}"

    webpage.save()  # save the HTML


