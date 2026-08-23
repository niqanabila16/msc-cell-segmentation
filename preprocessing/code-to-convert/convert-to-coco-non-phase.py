import os
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask as maskUtils

def _encode_binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    # Mengubah mask biner menjadi format COCO RLE
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = maskUtils.encode(binary_mask)

    # Memastikan format string pada JSON
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")

    return rle

def convert_to_coco_rle_padded(base_dir):
    base_path = Path(base_dir)
    img_dir = base_path / "images"
    mask_dir = base_path / "masks"
    out_path = base_path / "annotations.json"

    coco_format = {
        "info": {
            "description": "Dataset LiveCell Padded & Cropped (1024x1024)"
        },
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "cell"}
        ]
    }

    # Filter ekstensi yang valid (termasuk .jpg)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif')
    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)])
    
    annotation_id = 1
    image_id = 1

    print("\nMembuat file annotations.json (COCO RLE)...")

    for filename in tqdm(image_files):
        img_path = img_dir / filename
        
        # Menyesuaikan nama file mask (contoh: gambar.jpg -> gambar_mask.png)
        base_name, _ = os.path.splitext(filename)
        mask_filename = base_name + "_mask.png"
        mask_path = mask_dir / mask_filename

        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"\nMelewati {filename} - Tidak terbaca atau mask tidak ditemukan.")
            continue

        height, width = mask.shape

        # Mendaftarkan informasi gambar
        coco_format["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        # Binerisasi mask: cell = 1, background = 0
        binary_mask = (mask > 0).astype(np.uint8)

        # Memisahkan komponen yang saling terhubung (instance segmentation)
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

    # Menyimpan file JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco_format, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Selesai! File tersimpan di: {out_path}")
    print(f"Total gambar: {len(coco_format['images'])}")
    print(f"Total anotasi instance sel: {len(coco_format['annotations'])}")

if __name__ == "__main__":
    # Path utama tempat folder images dan masks berada
    dataset_dir = r"D:\Eksperimen TA\msc-cell-segmentation\preprocessing\images-original-converted-non-phase-lazy"
    convert_to_coco_rle_padded(dataset_dir)