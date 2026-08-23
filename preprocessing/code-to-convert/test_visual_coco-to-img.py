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

        combined_mask = np.maximum(
            combined_mask,
            decoded_mask.astype(np.uint8)
        )

    return combined_mask


def visualize_single_sample(
    json_path,
    image_path,
    mask_path,
    save_path=None
):

    # =========================
    # LOAD JSON
    # =========================

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_name = Path(image_path).name

    # Cari image info berdasarkan nama file
    img_info = next(
        img for img in data["images"]
        if img["file_name"] == file_name
    )

    image_id = img_info["id"]
    height = img_info["height"]
    width = img_info["width"]

    # Cari anotasi gambar tersebut
    img_annotations = [
        ann for ann in data["annotations"]
        if ann["image_id"] == image_id
    ]

    # =========================
    # DECODE RLE
    # =========================

    decoded_mask = decode_and_combine_masks(
        img_annotations,
        height,
        width
    )

    # =========================
    # LOAD ORIGINAL IMAGE
    # =========================

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Gagal membaca image:\n{image_path}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # =========================
    # LOAD ORIGINAL PNG MASK
    # =========================

    original_mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if original_mask is None:
        raise FileNotFoundError(
            f"Gagal membaca mask:\n{mask_path}"
        )

    # =========================
    # PRINT SAMPLE JSON
    # =========================

    sample_json = {
        "images": [img_info],
        "annotations": [img_annotations[0]]
    }

    print("\n===== SAMPLE JSON =====\n")

    print(
        json.dumps(
            sample_json,
            indent=2
        )[:1500]
    )

    print("\n=======================\n")

    # =========================
    # VISUALIZATION
    # =========================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Original PNG mask
    axes[1].imshow(original_mask, cmap="gray")
    axes[1].set_title("Original PNG Mask")
    axes[1].axis("off")

    # Decoded RLE mask
    axes[2].imshow(decoded_mask, cmap="gray")
    axes[2].set_title("Decoded RLE Mask")
    axes[2].axis("off")

    # Overlay
    axes[3].imshow(image)
    axes[3].imshow(
        decoded_mask,
        cmap="viridis",
        alpha=0.5
    )

    axes[3].set_title("Overlay Result")
    axes[3].axis("off")

    plt.tight_layout()

    # Save
    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


if __name__ == "__main__":

    json_path = r"""
C:\Users\Niqa Nabila\Documents\Kuliah\Semester 8\Tugas Akhir\msc-cell-segmentation\preprocessing\images-original-converted\converted-images\step2_padded\test\annotations.json
""".strip()

    image_path = r"""
C:\Users\Niqa Nabila\Documents\Kuliah\Semester 8\Tugas Akhir\msc-cell-segmentation\preprocessing\images-original-converted\converted-images\step2_padded\test\images\218-4_part_1_tile_3.png
""".strip()

    mask_path = r"""
C:\Users\Niqa Nabila\Documents\Kuliah\Semester 8\Tugas Akhir\msc-cell-segmentation\preprocessing\images-original-converted\converted-images\step2_padded\test\masks\218-4_part_1_tile_3.png
""".strip()

    save_path = r"""
C:\Users\Niqa Nabila\Documents\Kuliah\Semester 8\Tugas Akhir\msc-cell-segmentation\preprocessing\images-original-converted\converted-images\step2_padded\test\rle_validation_result.png
""".strip()

    visualize_single_sample(
        json_path=json_path,
        image_path=image_path,
        mask_path=mask_path,
        save_path=save_path
    )