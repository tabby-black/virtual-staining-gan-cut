import re
import wandb
from pathlib import Path


project_name = "hyperspectral_image_reconstruction"
run_name = "hsi_to_rgb_cyclegan_run2.1"
loss_file_path = "checkpoints/hsi_to_rgb_cyclegan_run2.1/loss_log.txt"
entity = None

wandb.init(project=project_name, entity=entity, name=run_name)

# Matches: key: value   where value can be 1, 1.0, -0.3, 2e-4, etc.
# and allows trailing punctuation like commas/parentheses which we strip.
kv_pattern = re.compile(r'([A-Za-z][A-Za-z0-9_]*)\s*:\s*([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)[,)]?')

epoch_pattern = re.compile(r'epoch:\s*(\d+)')
iters_pattern = re.compile(r'iters:\s*(\d+)')

def parse_entry(lines: str):
    """Parse one combined log entry (header + maybe next line)."""
    if "epoch:" not in lines or "iters:" not in lines:
        return None

    em = epoch_pattern.search(lines)
    im = iters_pattern.search(lines)
    if not em or not im:
        return None

    epoch = int(em.group(1))
    iters = int(im.group(1))

    losses = {}
    for m in kv_pattern.finditer(lines):
        key = m.group(1)
        val = m.group(2)  # clean numeric part only
        # filter out header fields we don't want as "losses"
        if key in {"epoch", "iters", "time", "data"}:
            continue
        losses[key] = float(val)

    return epoch, iters, losses

path = Path(loss_file_path)
if not path.exists():
    raise FileNotFoundError(f"Could not find {loss_file_path}")

# CycleGAN often writes a header line then a loss line; sometimes all on one line.
# merge consecutive lines into "entries" starting at "(epoch:".
entries = []
buffer = ""

with path.open("r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if "(epoch:" in line:
            # flush previous buffer
            if buffer:
                entries.append(buffer)
            buffer = line
        else:
            # append continuation line(s)
            buffer += " " + line

# flush last
if buffer:
    entries.append(buffer)

logged = 0
skipped = 0

for combined in entries:
    parsed = parse_entry(combined)
    if not parsed:
        skipped += 1
        continue

    epoch, iters, losses = parsed
    if not losses:
        skipped += 1
        continue

    # monotonically increasing step even if iters resets each epoch
    step = epoch * 100000 + iters
    wandb.log(losses, step=step)
    logged += 1

print(f"Done. Logged {logged} entries, skipped {skipped}.")
wandb.finish()