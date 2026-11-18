
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. FIND THE NIfTI FILE AUTOMATICALLY
# ==========================================

root_folder = Path(
    r"C:\Users\Lenovo\Desktop\minor_project\archive (8)\ABIDE"
)

file_name = "50009.nii"

print("Searching for NIfTI file...")

files = list(root_folder.rglob(file_name))

if not files:
    raise FileNotFoundError(
        f"{file_name} was not found inside:\n{root_folder}\n"
        "Check the folder location and filename."
    )

file_path = files[0]

print("\nFile found successfully!")
print("File path:", file_path)

# ==========================================
# 2. LOAD THE NIfTI IMAGE
# ==========================================

print("\nLoading image...")

img = nib.load(str(file_path))
data = img.get_fdata()

print("\n========== IMAGE INFORMATION ==========")
print("Data shape:", data.shape)
print("Number of dimensions:", data.ndim)
print("Voxel size:", img.header.get_zooms())
print("Data type:", data.dtype)
print("Minimum intensity:", np.nanmin(data))
print("Maximum intensity:", np.nanmax(data))
print("=======================================\n")

# ==========================================
# 3. CHECK IMAGE DATA
# ==========================================

if data.size == 0:
    raise ValueError("The NIfTI file contains no image data.")

if not np.isfinite(data).any():
    raise ValueError("The image contains no finite intensity values.")

# ==========================================
# 4. DISPLAY THE IMAGE
# ==========================================

if data.ndim == 2:
    # Two-dimensional image
    plt.figure(figsize=(8, 8))
    plt.imshow(data.T, cmap="gray", origin="lower")
    plt.title("NIfTI Image - 2D")

elif data.ndim == 3:
    # Three-dimensional volume
    middle_slice = data.shape[2] // 2

    plt.figure(figsize=(8, 8))
    plt.imshow(
        data[:, :, middle_slice].T,
        cmap="gray",
        origin="lower"
    )
    plt.title(f"Middle Slice - {middle_slice}")

elif data.ndim == 4:
    # Four-dimensional image
    middle_volume = data.shape[3] // 2
    middle_slice = data.shape[2] // 2

    plt.figure(figsize=(8, 8))
    plt.imshow(
        data[:, :, middle_slice, middle_volume].T,
        cmap="gray",
        origin="lower"
    )
    plt.title(
        f"4D Image - Volume {middle_volume}, "
        f"Slice {middle_slice}"
    )

else:
    raise ValueError(
        f"Unsupported image dimensions: {data.ndim}. "
        f"Image shape: {data.shape}"
    )

plt.axis("off")
plt.tight_layout()
plt.show()

# ==========================================
# 5. DISPLAY THREE VIEWS FOR 3D DATA
# ==========================================

if data.ndim == 3:
    x, y, z = (n // 2 for n in data.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(data[x, :, :].T, cmap="gray", origin="lower")
    axes[0].set_title("Sagittal View")

    axes[1].imshow(data[:, y, :].T, cmap="gray", origin="lower")
    axes[1].set_title("Coronal View")

    axes[2].imshow(data[:, :, z].T, cmap="gray", origin="lower")
    axes[2].set_title("Axial View")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

print("Processing completed successfully!")
