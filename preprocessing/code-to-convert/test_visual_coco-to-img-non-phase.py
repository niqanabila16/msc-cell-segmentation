import json
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

def decode_and_combine_masks(annotations, height, width):
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        segm = ann["segmentation"]
        decoded_mask = mask_utils.decode(segm)

        if decoded_mask.ndim == 3:
            decoded_mask = decoded_mask.squeeze(axis=2)

        combined_mask = np.maximum(combined_mask, decoded_mask.astype(np.uint8))
    return combined_mask

def visualize_single_sample(json_path, image_path, mask_path, save_path=None):
    # LOAD JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = Path(image_path).name

    # Cari image info
    try:
        img_info = next(img for img in data["images"] if img["file_name"] == file_name)
    except StopIteration:
        raise ValueError(f"Gambar {file_name} tidak ditemukan di dalam file JSON!")

    image_id = img_info["id"]
    height = img_info["height"]
    width = img_info["width"]

    # Cari anotasi terkait
    img_annotations = [ann for ann in data["annotations"] if ann["image_id"] == image_id]

    # DECODE RLE
    decoded_mask = decode_and_combine_masks(img_annotations, height, width)

    # LOAD ORIGINAL IMAGE & MASK
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gagal membaca image:\n{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if original_mask is None:
        raise FileNotFoundError(f"Gagal membaca mask:\n{mask_path}")

    # PRINT SAMPLE JSON
    sample_json = {
        "images": [img_info],
        "annotations": [img_annotations[0]] if img_annotations else []
    }
    print("\n===== SAMPLE JSON (1 Anotasi) =====\n")
    print(json.dumps(sample_json, indent=2)[:1500])
    print("\n===================================\n")

    # VISUALIZATION
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title("Citra Asli (.jpg)")
    axes[0].axis("off")

    axes[1].imshow(original_mask, cmap="gray")
    axes[1].set_title("Mask Asli (.png)")
    axes[1].axis("off")

    axes[2].imshow(decoded_mask, cmap="gray")
    axes[2].set_title("Mask Hasil Decode RLE")
    axes[2].axis("off")

    axes[3].imshow(image)
    axes[3].imshow(decoded_mask, cmap="viridis", alpha=0.5)
    axes[3].set_title("Overlay (Gambar + RLE)")
    axes[3].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Hasil visualisasi disimpan di: {save_path}")

    plt.show()

if __name__ == "__main__":
    base_dir = r"D:\Eksperimen TA\msc-cell-segmentation\preprocessing\images-original-converted-non-phase"
    
    json_path = os.path.join(base_dir, "annotations.json")
    
    nama_gambar_tes = "Aufnahme-01_0,5Mio_nach_24h.jpg"
    nama_mask_tes = "Aufnahme-01_0,5Mio_nach_24h_mask.png"

    image_path = os.path.join(base_dir, "images", nama_gambar_tes)
    mask_path = os.path.join(base_dir, "masks", nama_mask_tes)
    save_path = os.path.join(base_dir, "rle_validation_result.png")

    visualize_single_sample(json_path, image_path, mask_path, save_path)