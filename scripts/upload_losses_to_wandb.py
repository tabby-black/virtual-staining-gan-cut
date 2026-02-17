import wandb
import re


project_name = "hyperspectral_image_reconstruction"
run_name = "hsi_to_rgb_cyclegan_run2"
loss_file_path = "checkpoints/hsi_to_rgb_run2/loss_log.txt"

wandb.init(project=project_name, name=run_name)

pattern = re.compile(r'([A-Za-z_]+): ([0-9.+-eE]+)')

with open(loss_file_path, 'r') as f:
    for line in f:
        if "(epoch:" not in line:
            continue

        # extract epoch
        epoch_match = re.search(r'epoch:\s*(\d+)', line)
        iter_match = re.search(r'iters:\s*(\d+)', line)

        if epoch_match and iter_match:
            epoch = int(epoch_match.group(1))
            iters = int(iter_match.group(1))

            losses = dict(pattern.findall(line))
            losses = {k: float(v) for k, v in losses.items()}

            # Use combined step index
            step = epoch * 100000 + iters

            wandb.log(losses, step=step)

print("Upload complete.")
wandb.finish()