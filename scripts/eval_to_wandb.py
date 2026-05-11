import os
import re
import subprocess
from pathlib import Path
from PIL import Image

import numpy as np
import imageio.v2 as imageio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import wandb
from collections import defaultdict

import time
import json


def evaluate_epoch(results_dir, data_range=255.0, channel_axis=2):
    fake_dir = os.path.join(results_dir, "fake_B")
    real_dir = os.path.join(results_dir, "real_B")

    if not os.path.isdir(fake_dir) or not os.path.isdir(real_dir):
        raise FileNotFoundError(f"Expected {fake_dir} and {real_dir}")

    fake_names = sorted(os.listdir(fake_dir))
    
    # amended calculation of evaluation metrics
    # there are now 9 patches per testing image
    # evaluate ssim and psnr per image, then average to get scores for the dataset
    # store patch metrics grouped by original image
    image_ssim = defaultdict(list)
    image_psnr = defaultdict(list)
    
    # adding in debug counters to see why testing is only using 30 not 50 images like before
    missing_real = 0
    shape_mismatch = 0
    used_patches = 0

    for name in fake_names:
        fake_path = os.path.join(fake_dir, name)
        real_path = os.path.join(real_dir, name)
        
        if not os.path.exists(real_path):
            missing_real += 1
            continue

        fake_img = imageio.imread(fake_path).astype(np.float32)
        real_img = imageio.imread(real_path).astype(np.float32)

        if fake_img.shape != real_img.shape:
            shape_mismatch += 1
            continue

        used_patches += 1
        
        # compute patch-level metrics
        # removed channel_axis argument from PSNR
        patch_ssim = ssim(fake_img, real_img, channel_axis=channel_axis, data_range=data_range)
        patch_psnr = psnr(fake_img, real_img, data_range=data_range)

        # group by original image
        # e.g. name = P12_ROI_01_C10_T_raw_preprocessed_7.img
        # e.g. stem = P12_ROI_01_C10_T_raw_preprocessed_7
        stem = os.path.splitext(name)[0]
        # e.g. image_id = P12_ROI_01_C10_T_raw_preprocessed
        image_id = stem.rsplit("_", 1)[0]

        # add the patch's metrics to the dictionary, grouped with other patches from the
        # same original image
        image_ssim[image_id].append(patch_ssim)
        image_psnr[image_id].append(patch_psnr)

    # average patch metrics within each image
    ssim_scores = [float(np.mean(scores)) for scores in image_ssim.values() if scores]
    psnr_scores = [float(np.mean(scores)) for scores in image_psnr.values() if scores]

    print("Missing real:", missing_real)
    print("Shape mismatch:", shape_mismatch)
    print("Used patches:", used_patches)

    # average over images to get dataset-level average metrics
    # return the average of the metrics for this epoch to be logged in wandb
    return {
        "eval/n_images": int(len(ssim_scores)),
        "eval/ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else float("nan"),
        "eval/ssim_std": float(np.std(ssim_scores)) if ssim_scores else float("nan"),
        "eval/psnr_mean": float(np.mean(psnr_scores)) if psnr_scores else float("nan"),
        "eval/psnr_std": float(np.std(psnr_scores)) if psnr_scores else float("nan"),
    }


def log_sample_images(results_dir, epoch, target_images):
    fake_dir = os.path.join(results_dir, "fake_B")
    real_dir = os.path.join(results_dir, "real_B")

    all_names = sorted(os.listdir(fake_dir))

    table = wandb.Table(columns=[
        "image_id",
        "patch_name",
        "real_B",
        "fake_B"
    ])

    for name in all_names:

        # cycles through all 9 patches in an image
        stem = os.path.splitext(name)[0]
        image_id = stem.rsplit("_", 1)[0]

        # only log the 9 patches for the 4 images I am visually reporting in diss
        # then can stitch these together from table
        if image_id not in target_images:
            continue

        real_path = os.path.join(real_dir, name)
        fake_path = os.path.join(fake_dir, name)

        if not os.path.exists(real_path):
            continue

        real = Image.open(real_path).convert("RGB")
        fake = Image.open(fake_path).convert("RGB")

        table.add_data(
            image_id,
            name,
            wandb.Image(real),
            wandb.Image(fake)
        )

    wandb.log(
        {f"eval_samples_epoch_{epoch}": table},
        step=epoch
    )

     
def run_test_py(repo_root, experiment_name, dataroot, ep, results_dir):
    # add --phase train to this command too to get results of testing on training set alongside validation set
    cmd = [
        "python", "-u", "test.py",
        "--name", experiment_name,
        "--dataroot", dataroot,
        "--fp16",
        #"--CUT_mode", "CUT",
        "--model", "cycle_gan",
        "--num_test", "846",
        "--dataset_mode", "hsi_unaligned",
        "--gpu_ids", "2",
        "--epoch", str(ep),
       # CHANGE THIS LINE
        # only include this line for eval on training dataset
        #"--phase", "train",
        "--results_dir", str(results_dir),
        "--no_dropout",
        "--preprocess", "resize_and_crop",
        "--load_size", "286",
        "--crop_size", "256",
    ]

    # proves we are in this function
    print("\n[run_test_py] Running:", " ".join(cmd))

    try:
        res = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
        print("[run_test_py] Return code:", res.returncode)
        if res.stdout:
            print("\n--- STDOUT ---\n", res.stdout)
        if res.stderr:
            print("\n--- STDERR ---\n", res.stderr)

        if res.returncode != 0:
            raise RuntimeError(f"test.py failed with return code {res.returncode}")

        return res

    except Exception as e:
        print("\n[run_test_py] Exception raised:", repr(e))
        raise

# measure checkpoint size to compare fp16 to fp32 model size
def checkpoint_size_mb(checkpoint_paths):
    total_bytes = sum(Path(p).stat().st_size for p in checkpoint_paths if Path(p).exists())
    return total_bytes / (1024 ** 2)

if __name__ == "__main__":
    repo_root = "/local/scratch-3/tb789/projects/virtual-staining-gan-cut"
    # CHANGE THIS LINE
    experiment_name = "hsi_to_rgb_cyclegan_rep2"
    # so run this from /scripts
    # test on new test patches
    dataroot = "./datasets/histology_full_quant/"
    results_root = Path(repo_root) / "results" / experiment_name
    # CUT1 epochs: 0-101
    # CHANGE THIS LINE
    # plot every epoch for reporting runs
    epochs = list(range(190, 195, 5))
    wandb_project = "hyperspectral_image_reconstruction"
    # set to training run id each time so I can see metrics on the same run
    # currently set to id of run2 i.e. cyclegan_initial
    # set to None to avoid non-monotonical step issue
    #wandb_run_id = None

    wandb.init(
        project=wandb_project,
        #name=f"{experiment_name}_eval_{Path(dataroot).name}",
        # CHANGE THIS LINE
        #id="c4fgacxj",
        #resume = "allow"
    )

    for ep in epochs:
        # 1: generate outputs (skip if already exists)
        # CHANGE THIS LINE
        out_images_dir = results_root / f"test_{ep}" / "images"
        if not out_images_dir.exists():
            ckpt_dir = Path(repo_root) / "checkpoints" / experiment_name
            # CHANGE THIS LINE
            # expected files for CycleGAN
            expected_files = [
                ckpt_dir / f"{ep}_net_D_A.pth",
                ckpt_dir / f"{ep}_net_D_B.pth",
                ckpt_dir / f"{ep}_net_G_A.pth",
                ckpt_dir / f"{ep}_net_G_B.pth",
            ]
            #generator_files = [
                #ckpt_dir / f"{ep}_net_G_A.pth",
                #ckpt_dir / f"{ep}_net_G_B.pth",
            #]
            # for 16fp
            generator_files = [
                results_root / f"test_{ep}" / "fp16_netG_A.pth",
                results_root / f"test_{ep}" / "fp16_netG_B.pth",
            ]
            # expected files for CUT
            #expected_files = [
                #ckpt_dir / f"{ep}_net_D.pth",
                #ckpt_dir / f"{ep}_net_E_B.pth",
                #ckpt_dir / f"{ep}_net_F.pth",
                #ckpt_dir / f"{ep}_net_G.pth",
            #]
            #generator_files = [
                #ckpt_dir / f"{ep}_net_E_B.pth",
                #ckpt_dir / f"{ep}_net_G.pth",
            #]
            print("Checking: ", expected_files)
            missing = [p.name for p in expected_files if not p.exists()]
            if missing:
                print(f"Checkpoint missing for epoch {ep}: {missing}; skipping.")
                continue
        
            run_test_py(repo_root, experiment_name, dataroot, ep, results_dir=str(results_root.parent))
        
        else:
            print(f"Found existing outputs for epoch {ep}, skipping test.py")

        # define generator files outside if statement so it is defined for runs where the test has already been done
        ckpt_dir = Path(repo_root) / "checkpoints" / experiment_name
        #generator_files = [
                #ckpt_dir / f"{ep}_net_G_A.pth",
                #ckpt_dir / f"{ep}_net_G_B.pth",
            #]
        # 2: evaluate and log
        # out_images_dir is the directory of the RGB output images produced during testing
        metrics = evaluate_epoch(str(out_images_dir))
        metrics["epoch"] = ep
        # measure the generator checkpoint size to compare between fp16 and fp32
        # INCLUDE THIS FOR REPORTING IMAGE RUNS
        metrics["eval/checkpoint_size_mb"] = checkpoint_size_mb(generator_files)
        # measure end-to-end test time to compare between fp16 and fp32
        # INCLUDE THIS FOR REPORTING IMAGE RUNS
        timing_path = results_root / f"test_{ep}" / "timing_metrics.json"

        with open (timing_path, "r") as f:
            timing_metrics = json.load(f)

        metrics["eval/inference_mean_s"] = timing_metrics["mean_inference_time_s"]
        metrics["eval/inference_std_s"] = timing_metrics["std_inference_time_s"]
        
        
        wandb.log(metrics)  
        print(ep, metrics)

        # 3: visualise and log
        # visualise relevant patches for reporting
        target_images = [
            "P2_ROI_01_C1_T_raw_preprocessed",
            "P2_ROI_03_C03_NT_raw_preprocessed",
            "P3_ROI_01_C04_T_raw_preprocessed",
            "P12_ROI_01_C03_T_raw_preprocessed",
            ]

        log_sample_images(str(out_images_dir), ep, target_images)

    wandb.finish()