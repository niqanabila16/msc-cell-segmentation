# Program Python untuk menjelaskan metode cv2.copyMakeBorder()
# Diadaptasi untuk pemrosesan batch pada dataset citra

import os           # Mengimpor modul os untuk berinteraksi dengan sistem operasi (membuat direktori, mengatur path file)
import cv2          # Mengimpor pustaka OpenCV untuk pemrosesan citra digital
from tqdm import tqdm # Mengimpor modul tqdm untuk menampilkan progress bar (visualisasi proses) di terminal

def add_padding(base_path, konversi_path, splits=('train', 'test'), pad_size=12):
    # Melakukan iterasi untuk setiap subset data yang ditentukan (secara default memproses folder 'train' lalu 'test')
    for split in splits:
        # Menentukan path direktori tempat citra RGB (hasil tahap sebelumnya) dan mask asli berada
        img_input_dir = os.path.join(konversi_path, 'step1_rgb', split, 'images')
        mask_input_dir = os.path.join(base_path, split, 'masks')

        # Menentukan path direktori output tempat citra dan mask yang sudah diberi padding akan disimpan
        img_output_dir = os.path.join(konversi_path, 'step2_padded', split, 'images')
        mask_output_dir = os.path.join(konversi_path, 'step2_padded', split, 'masks')

        # Membuat folder output di sistem komputer. exist_ok=True mencegah program error jika folder tersebut sudah ada
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(mask_output_dir, exist_ok=True)

        # Mencetak pesan ke terminal untuk memberi tahu pengguna subset mana yang sedang diproses padding-nya
        print(f"\n Padding split: {split}")

        # Mengambil daftar file di direktori, mengurutkannya, lalu mengiterasinya satu per satu. tqdm membungkus proses ini untuk menampilkan progress bar
        for filename in tqdm(sorted(os.listdir(img_input_dir))):
            # Validasi: memastikan hanya file berekstensi .png yang diproses. Jika bukan, lewati iterasi ini (continue)
            if not filename.endswith('.png'):
                continue

            # Membentuk path (jalur lengkap) ke file citra tunggal yang sedang diproses
            img_path = os.path.join(img_input_dir, filename)
            # Membentuk path ke file mask pasangannya. Mengganti ekstensi ".png" menjadi "_mask.png" sesuai asumsi format penamaan dataset
            mask_path = os.path.join(mask_input_dir, filename.replace(".png", "_mask.png"))

            # Membaca citra RGB ke dalam memori komputer menggunakan mode default OpenCV
            image = cv2.imread(img_path)
            # Membaca mask (ground truth) ke dalam memori dengan format grayscale (1-channel/hitam-putih)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Pengecekan keamanan: jika gambar atau mask gagal terbaca (misal karena file rusak atau path salah), lewati ke file berikutnya
            if image is None or mask is None:
                continue

            # Mendefinisikan ukuran penambahan tepi (border) untuk keempat sisi, diatur sesuai parameter pad_size (default 12 piksel)
            top = pad_size
            bottom = pad_size
            left = pad_size
            right = pad_size
            
            # Mendefinisikan jenis tipe border. BORDER_REFLECT_101 mencerminkan piksel citra ke arah luar tanpa menduplikasi piksel yang berada persis di garis batas
            borderType = cv2.BORDER_REFLECT_101

            # Menggunakan metode cv2.copyMakeBorder() untuk mengaplikasikan padding reflektif pada matriks citra
            padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType)
            
            # Menggunakan metode cv2.copyMakeBorder() secara identik pada mask agar transformasi spasialnya seimbang (koordinat piksel tetap selaras)
            padded_mask = cv2.copyMakeBorder(mask, top, bottom, left, right, borderType)

            # Menyimpan citra hasil padding dari memori ke direktori penyimpanan output di hardisk
            cv2.imwrite(os.path.join(img_output_dir, filename), padded_image)
            # Menyimpan mask hasil padding ke direktori penyimpanan output dengan nama file yang sama
            cv2.imwrite(os.path.join(mask_output_dir, filename), padded_mask)

    # Mencetak pesan konfirmasi ke terminal dengan tanda centang setelah seluruh proses iterasi dari kedua split selesai dilakukan
    print("\n✅ Padding selesai.")

# ================================
# Eksekusi Tahap 2
# ================================

# Menetapkan variabel path direktori akar tempat dataset asli disimpan
base_dataset_path = r"C:\Users\Niqa Nabila\Documents\Kuliah\Semester 8\Tugas Akhir\msc-cell-segmentation\preprocessing\images-original-converted"

# Menetapkan variabel path untuk direktori kerja tahapan konversi
konversi_dir = os.path.join(base_dataset_path, "converted-images")

# Memanggil fungsi yang telah didefinisikan di atas dan memberikan argumen pad_size=12
# (Contoh: Menambah 12px di atas, bawah, kiri, dan kanan untuk mengubah ukuran 1000x1000 menjadi 1024x1024)
add_padding(base_dataset_path, konversi_dir, pad_size=12)