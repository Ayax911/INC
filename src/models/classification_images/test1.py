from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

image_path= "/home/akira/snap/steam/preproccesed_julian/norm_neg1_1/cmmd/cmmd_1814.tiff"

image = Image.open(image_path)
image_array = np.array(image)
print(image_array.shape)

plt.imshow(image_array, cmap="gray", vmin=-1, vmax=1)
plt.xlabel(f"Min: {image_array.min()}, Max: {image_array.max()}")
save_path = os.path.abspath(f"debug_batch_test.png")
plt.savefig(save_path)