import os
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils


def _encode_binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """
    Encode binary mask (H, W) -> COCO RLE dict.
    pycocotools expects Fortran/column-major order.
    """
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(binary_mask)

    # pycocotools returns counts as bytes; JSON needs string
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")

    return rle


def convert_to_coco_rle_padded(konversi_path, splits=("train", "test")):
    konversi_path = Path(konversi_path)

    for split in splits:
        step2_dir = konversi_path / "step2_padded" / split
        img_dir = step2_dir / "images"
        mask_dir = step2_dir / "masks"
        out_path = step2_dir / "annotations.json"

        coco_format = {
            "info": {
                "description": f"Pixel Perfect RLE (PADDED) - {split}"
            },
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "cell"}
            ]
        }

        image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
        annotation_id = 1
        image_id = 1

        print(f"\nProcessing {split} (PADDED)...")

        for filename in tqdm(image_files, desc=f"{split}"):
            img_path = img_dir / filename
            mask_path = mask_dir / filename

            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Skip (missing or unreadable): {filename}")
                continue

            height, width = mask.shape

            coco_format["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height
            })

            # Binerisasi mask: foreground = 1, background = 0
            binary_mask = (mask > 0).astype(np.uint8)

            # Pisahkan komponen yang saling terhubung menjadi instance terpisah
            num_labels, labels = cv2.connectedComponents(binary_mask)

            for label_id in range(1, num_labels):
                instance_mask = (labels == label_id).astype(np.uint8)

                if instance_mask.sum() == 0:
                    continue

                # Encode instance mask ke COCO RLE
                rle = _encode_binary_mask_to_rle(instance_mask)

                area = int(maskUtils.area(rle))
                bbox = maskUtils.toBbox(rle).tolist()

                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })

                annotation_id += 1

            image_id += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco_format, f, ensure_ascii=False, indent=2)

        print(f"Saved: {out_path}")
        print(f"Total images: {len(coco_format['images'])}")
        print(f"Total annotations: {len(coco_format['annotations'])}")


if __name__ == "__main__":
    base_path = r"C:\Users\Niqa Nabila\Documents\Kuliah\Semester 8\Tugas Akhir\msc-cell-segmentation\preprocessing\images-original-converted"
    konversi_path = os.path.join(base_path, "converted-images")

    convert_to_coco_rle_padded(
        konversi_path=konversi_path,
        splits=("train", "test")
    )