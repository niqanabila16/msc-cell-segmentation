# Program Python untuk menyesuaikan dimensi citra (Padding & Cropping)
# Diadaptasi untuk dataset LiveCell (1280x960 -> 1024x1024)

import os
import cv2
from tqdm import tqdm

def adjust_dimensions_center(img_input_dir, mask_input_dir, output_base_dir, target_size=1024):
    # Menentukan path direktori output
    img_output_dir = os.path.join(output_base_dir, 'images')
    mask_output_dir = os.path.join(output_base_dir, 'masks')

    # Membuat folder output jika belum ada
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    print(f"\nMemproses penyesuaian dimensi ke {target_size}x{target_size}...")

    # 1. Mendefinisikan format gambar yang diizinkan (memasukkan .jpg)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif')
    
    # 2. Filter file agar hanya memproses gambar, mengabaikan file seperti .json
    image_files = [f for f in sorted(os.listdir(img_input_dir)) if f.lower().endswith(valid_extensions)]

    # Pengecekan keamanan
    if len(image_files) == 0:
        print("⚠️ PERINGATAN: Tidak ada file gambar yang ditemukan di direktori input!")
        return

    # 3. Proses iterasi menggunakan tqdm
    for filename in tqdm(image_files):
        # Membentuk path ke file gambar (.jpg)
        img_path = os.path.join(img_input_dir, filename)
        
        # Memisahkan nama file dari ekstensinya (Misal: "Aufnahme.jpg" menjadi "Aufnahme" dan ".jpg")
        base_name, original_ext = os.path.splitext(filename)

        # Membentuk path ke file mask sesuai contoh Anda (Menambahkan _mask.png)
        mask_filename = base_name + "_mask.png"
        mask_path = os.path.join(mask_input_dir, mask_filename)

        # Membaca citra dan mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"\nMelewati {filename} - Gambar atau Mask tidak ditemukan/rusak.")
            continue

        # Mendapatkan dimensi saat ini (Tinggi: 960, Lebar: 1280)
        h, w = image.shape[:2]

        # ==========================================
        # TAHAP 1: PADDING (Hanya jika dimensi < target)
        # ==========================================
        pad_top = (target_size - h) // 2 if h < target_size else 0
        pad_bottom = (target_size - h) - pad_top if h < target_size else 0
        
        pad_left = (target_size - w) // 2 if w < target_size else 0
        pad_right = (target_size - w) - pad_left if w < target_size else 0

        borderType = cv2.BORDER_REFLECT_101
        
        padded_img = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, borderType)
        padded_mask = cv2.copyMakeBorder(mask, pad_top, pad_bottom, pad_left, pad_right, borderType)

        # ==========================================
        # TAHAP 2: CROPPING (Hanya jika dimensi > target)
        # ==========================================
        new_h, new_w = padded_img.shape[:2]

        start_y = (new_h - target_size) // 2 if new_h > target_size else 0
        start_x = (new_w - target_size) // 2 if new_w > target_size else 0

        final_img = padded_img[start_y:start_y+target_size, start_x:start_x+target_size]
        final_mask = padded_mask[start_y:start_y+target_size, start_x:start_x+target_size]

        # Menyimpan hasil akhir (gambar disimpan sebagai .jpg, mask disimpan sebagai .png)
        cv2.imwrite(os.path.join(img_output_dir, filename), final_img)
        cv2.imwrite(os.path.join(mask_output_dir, mask_filename), final_mask)

    print("\n✅ Penyesuaian dimensi selesai.")

# ================================
# Eksekusi Tahap Konversi
# ================================

img_dir = r"D:\Semester 7\00_data\00_data\livecell\LC_GENERALPREP_LAZY\04cur_data\gabungan"
mask_dir = r"D:\Semester 7\00_data\00_data\livecell\LC_GENERALPREP_LAZY\04cur_data\gabungan\masks"

output_dir = r"D:\Eksperimen TA\msc-cell-segmentation\preprocessing\images-original-converted-non-phase-lazy"

adjust_dimensions_center(img_dir, mask_dir, output_dir, target_size=1024)