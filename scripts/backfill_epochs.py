import re
from pathlib import Path
import wandb


PROJECT = "hyperspectral_image_reconstruction"
RUN_NAME = "rep2a_backfilled"
LOSS_LOG = "checkpoints/hsi_to_rgb_cyclegan_rep2a/loss_log.txt"

# Parse: (epoch: 1, iters: 700, time: ..., data: ...) D_A: 0.2 G_A: 0.3 ...
epoch_pat = re.compile(r"epoch:\s*(\d+)")
iters_pat = re.compile(r"iters:\s*(\d+)")
kv_pat = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")

wandb.init(project=PROJECT, name=RUN_NAME)

path = Path(LOSS_LOG)
text = path.read_text().splitlines()

# CycleGAN loss logs are sometimes split across 2 lines; merge entries starting with "(epoch:"
entries = []
buf = ""
for line in text:
    line = line.strip()
    if not line:
        continue
    if "(epoch:" in line:
        if buf:
            entries.append(buf)
        buf = line
    else:
        buf += " " + line
if buf:
    entries.append(buf)

logged = 0
for s in entries:
    em = epoch_pat.search(s)
    im = iters_pat.search(s)
    if not em or not im:
        continue

    epoch = int(em.group(1))
    iters = int(im.group(1))

    metrics = {"epoch": epoch, "iters": iters}
    for k, v, *_ in kv_pat.findall(s):
        if k in {"epoch", "iters", "time", "data"}:
            continue
        metrics[k] = float(v)

    # Monotonic step for W&B (safe even if iters resets each epoch)
    step = epoch * 100000 + iters

    wandb.log(metrics, step=step)
    logged += 1

print(f"Logged {logged} points from {LOSS_LOG}")
wandb.finish()