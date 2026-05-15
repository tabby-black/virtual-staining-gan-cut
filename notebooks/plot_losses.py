# jupyter notebook transferred to python file for submission

# notebook to generate loss plots for 3.3.2 Loss Functions


import numpy as np
import matplotlib.pyplot as plt


# plot generator and discriminator losses
# avoid log(0) by clipping values
x = np.linspace(1e-5, 1 - 1e-5, 500)

# LSGAN-based GAN losses
generator_loss = (x - 1)**2
discriminator_loss = x**2

plt.figure(figsize=(6, 4))

plt.plot(x, generator_loss, label="Generator Loss")
plt.plot(x, discriminator_loss, label="Discriminator Loss (fake)")

plt.xlabel("Discriminator Output")
plt.ylabel("Adversarial Loss")
plt.title("Adversarial Loss vs Discriminator Output")

plt.legend()
plt.grid()

plt.show()


# plot cycle-consistency loss
# reconstruction error range
e = np.linspace(-1, 1, 500)

# L1 loss
cycle_loss = np.abs(e)

plt.figure(figsize=(6, 4))

plt.plot(e, cycle_loss, label="Cycle Loss (L1)", linewidth=2)

plt.xlabel("Reconstruction error")
plt.ylabel("Cycle Consistency Loss")
plt.title("Cycle Consistency Loss vs Reconstruction Error")

plt.grid()
#plt.legend()

plt.show()


# plot bar graph showing how losses contribute to overall objective

# loss components and weights
losses = ["Adversarial", "Cycle Consistency", "Identity"]
weights = [1, 10, 0]

plt.figure(figsize=(6, 4))

plt.bar(losses, weights)

plt.ylabel("Loss Weight")
plt.title("CycleGAN Loss Weighting")

plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# plot visualisation of PatchNCE effects

# similarity range
s = np.linspace(0, 1, 500)

# conceptual loss curves
# encourage high similarity
positive_loss = -np.log(s + 1e-5)   
# discourage high similarity
negative_loss = -np.log(1 - s + 1e-5)    

plt.figure(figsize=(6, 4))

plt.plot(s, positive_loss, label="Positive pairs", linewidth=2)
plt.plot(s, negative_loss, label="Negative pairs", linewidth=2)

plt.xlabel("Feature similarity")
plt.ylabel("PatchNCE Loss")
plt.title("PatchNCE Loss vs Feature Similarity")

plt.ylim(0, 5)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()