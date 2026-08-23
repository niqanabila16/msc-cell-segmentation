# RGB Original (1-channel) to RGB 3-channel

import os
import cv2
import albumentations as A
from tqdm import tqdm

def convert_to_rgb(base_path, konversi_path, splits=['train', 'test']):
    # Inisialisasi transformasi Albumentations
    transform = A.Compose([A.ToRGB(p=1.0)])
    
    for split in splits:
        input_dir = os.path.join(base_path, split, 'images')
        
        # Output diletakkan di dalam folder 'konversi'
        output_dir = os.path.join(konversi_path, 'step1_rgb', split, 'images')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Mulai konversi RGB untuk split: {split}")
        for filename in tqdm(os.listdir(input_dir)):
            if filename.endswith('.png'):
                img_path = os.path.join(input_dir, filename)
                
                # Baca 1-channel (Grayscale)
                img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Ubah ke 3-channel menggunakan Albumentations
                img_rgb = transform(image=img_gray)['image']
                
                # Simpan hasil
                cv2.imwrite(os.path.join(output_dir, filename), img_rgb)

# Eksekusi Tahap 1
base_dataset_path = r"C:\Users\Niqa Nabila\Documents\Kuliah\Semester 8\Tugas Akhir\msc-cell-segmentation\preprocessing\images-original-converted"
konversi_dir = os.path.join(base_dataset_path, "converted-images")

convert_to_rgb(base_dataset_path, konversi_dir)
