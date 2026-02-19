import os
import re
import subprocess
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import wandb

def evaluate_epoch(results_dir, data_range=255.0, channel_axis=2):
    fake_dir = os.path.join(results_dir, "fake_B")
    real_dir = os.path.join(results_dir, "real_B")

    if not os.path.isdir(fake_dir) or not os.path.isdir(real_dir):
        raise FileNotFoundError(f"Expected {fake_dir} and {real_dir}")

    fake_names = sorted(os.listdir(fake_dir))
    ssim_scores, psnr_scores = [], []

    for name in fake_names:
        fake_path = os.path.join(fake_dir, name)
        real_path = os.path.join(real_dir, name)
        if not os.path.exists(real_path):
            continue

        fake_img = imageio.imread(fake_path).astype(np.float32)
        real_img = imageio.imread(real_path).astype(np.float32)

        if fake_img.shape != real_img.shape:
            continue

        ssim_scores.append(ssim(fake_img, real_img, channel_axis=channel_axis, data_range=data_range))
        psnr_scores.append(psnr(real_img, fake_img, data_range=data_range))

    return {
        "eval/n_images": int(len(ssim_scores)),
        "eval/ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else float("nan"),
        "eval/ssim_std": float(np.std(ssim_scores)) if ssim_scores else float("nan"),
        "eval/psnr_mean": float(np.mean(psnr_scores)) if psnr_scores else float("nan"),
        "eval/psnr_std": float(np.std(psnr_scores)) if psnr_scores else float("nan"),
    }

def log_sample_images(results_dir, epoch, num_samples=5):
    fake_dir = os.path.join(results_dir, "fake_B")
    real_dir = os.path.join(results_dir, "real_B")

    names = sorted(os.listdir(fake_dir))[:num_samples]

    table = wandb.Table(columns=["filename", "real_B", "fake_B"])

    for name in names:
        real_path = os.path.join(real_dir, name)
        fake_path = os.path.join(fake_dir, name)

        if not os.path.exists(real_path):
            continue

        real = Image.open(real_path).convert("RGB")
        fake = Image.open(fake_path).convert("RGB")

        table.add_data(
            name,
            wandb.Image(real),
            wandb.Image(fake),
        )

    wandb.log({f"eval_samples_epoch_{epoch}": table}, step=epoch)

def run_test_py(repo_root, name, dataroot, epoch, results_dir, extra_args=None):
    cmd = [
        "python", "test.py",
        "--name", name,
        "--dataroot", dataroot,
        "--model", "cycle_gan",
        "--dataset_mode", "hsi_unaligned",
        "--epoch", str(epoch),
        "--results_dir", str(results_dir),
        "--num_test", "inf",
        "--no_dropout",
        "--preprocess", "resize_and_crop",
        "--load_size", "286",
        # crop size is 256 usually, 224 for batch size 4
        "--crop_size", "224",
    ]
    if extra_args:
        cmd += extra_args

    subprocess.run(cmd, cwd=repo_root, check=True)

# TODO: Check naming of this is right

if __name__ == "__main__":
    repo_root = "/local/scratch-3/tb789/projects/virtual-staining-gan-cut"
    experiment_name = "hsi_to_rgb_cyclegan_run2"
    dataroot = "./datasets/histology/"
    results_root = Path(repo_root) / "results" / experiment_name
    # max epoch 100
    epochs = list(range(5, 101, 5))
    wandb_project = "hyperspectral_image_reconstruction"
    wandb_run_id = None  # set to your training run id if you want metrics on SAME run
    # --------------------

    wandb.init(
        project=wandb_project,
        name=f"{experiment_name}_eval_{Path(dataroot).name}",
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
    )

    for ep in epochs:
        # 1: generate outputs (skip if already exists)
        out_images_dir = results_root / f"test_{ep}" / "images"
        if not out_images_dir.exists():
            run_test_py(repo_root, experiment_name, dataroot, ep, results_dir=str(results_root.parent))
        else:
            print(f"Found existing outputs for epoch {ep}, skipping test.py")

        # 2: evaluate and log
        metrics = evaluate_epoch(str(out_images_dir))
        wandb.log(metrics, step=ep)
        print(ep, metrics)

        # 3: visualise and log
        metrics = evaluate_epoch(results_dir)
        wandb.log(metrics, step=epoch)
        log_sample_images(results_dir, epoch)

    wandb.finish()