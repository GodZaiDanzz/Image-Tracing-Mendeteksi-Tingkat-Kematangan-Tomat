import os
from PIL import Image, ImageTk

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# ============================================================
# DETEKSI TINGKAT KEMATANGAN TOMAT MENGGUNAKAN HSV + OPENCV
# ============================================================
# Kategori:
# - Mentah      : Hijau dominan
# - Setengah Matang : Kuning / Orange dominan
# - Matang      : Merah dominan
# ============================================================

def classify_tomato(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Gagal membaca gambar.")
        return

    image = cv2.resize(image, (700, 500))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Hijau (Mentah)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Kuning / Orange (Setengah Matang)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([34, 255, 255])

    # Merah memiliki 2 range pada HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2

    green_pixels = cv2.countNonZero(mask_green)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    red_pixels = cv2.countNonZero(mask_red)
    if red_pixels > yellow_pixels and red_pixels > green_pixels:
        maturity = "MATANG"
        color = (0, 0, 255)

    elif yellow_pixels > green_pixels:
        maturity = "SETENGAH MATANG"
        color = (0, 255, 255)

    else:
        maturity = "MENTAH"
        color = (0, 255, 0)

    combined_mask = mask_green + mask_yellow + mask_red

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        combined_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 1000:
            x, y, w, h = cv2.boundingRect(largest_contour)

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                color,
                3
            )

            cv2.putText(
                image,
                maturity,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )

            # Periksa area busuk pada ROI tomat
            roi_hsv = hsv[y:y + h, x:x + w]
            rotten_ratio, rotten_mask = detect_rotten(roi_hsv)

            if rotten_ratio > 0.05:
                cv2.putText(
                    image,
                    f"BUSUK ({rotten_ratio*100:.1f}%)",
                    (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                cv2.imshow("Mask Busuk", rotten_mask)

    print("===== HASIL ANALISIS =====")
    print(f"Hijau   : {green_pixels} pixel")
    print(f"Kuning  : {yellow_pixels} pixel")
    print(f"Merah   : {red_pixels} pixel")
    print(f"Kondisi : {maturity}")

    # Tampilkan hasil
    cv2.imshow("Deteksi Kematangan Tomat", image)
    cv2.imshow("Mask Gabungan", combined_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return maturity


def detect_rotten(roi_hsv, min_ratio=0.05):
    """Deteksi area busuk sederhana pada ROI HSV.

    Mengembalikan (ratio, mask) di mana ratio adalah persentase pixel yang
    cocok sebagai busuk terhadap total ROI.
    """
    if roi_hsv is None or roi_hsv.size == 0:
        return 0.0, np.zeros((1, 1), dtype=np.uint8)

    # Fokus pada warna coklat (brown) dengan saturasi/val tertentu,
    # lalu singkirkan area ber-S rendah dan V sangat rendah (kemungkinan bayangan).
    lower_brown = np.array([5, 80, 40])
    upper_brown = np.array([30, 255, 200])
    mask_brown = cv2.inRange(roi_hsv, lower_brown, upper_brown)

    # Definisi bayangan/gelap: saturasi rendah OR value sangat rendah
    mask_shadow = cv2.inRange(roi_hsv, np.array([0, 0, 0]), np.array([180, 60, 40]))

    # Hapus area yang kemungkinan bayangan dari mask coklat
    mask_brown = cv2.bitwise_and(mask_brown, cv2.bitwise_not(mask_shadow))

    # Bersihkan noise kecil
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Hilangkan komponen terlalu kecil
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask_clean)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 100:  # threshold area kecil
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)

    rotten_pixels = cv2.countNonZero(cleaned)
    total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]

    ratio = (rotten_pixels / total_pixels) if total_pixels > 0 else 0.0

    return ratio, cleaned


def pilih_gambar():
    root = tk.Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Pilih file gambar tomat",
        filetypes=[
            ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return image_path


status_var = None
file_var = None
preview_label = None
current_image_path = None


def reset_status():
    if file_var is not None:
        file_var.set("Belum ada file dipilih")
    if status_var is not None:
        status_var.set("Belum ada hasil deteksi")
    if preview_label is not None:
        preview_label.config(image="")
        preview_label.image = None


def choose_and_detect():
    global current_image_path

    image_path = pilih_gambar()

    if image_path:
        current_image_path = image_path
        maturity = classify_tomato(image_path)
        file_name = os.path.basename(image_path)

        if file_var is not None:
            file_var.set(f"File: {file_name}")
        if status_var is not None:
            status_var.set(f"Hasil terakhir: {maturity}")

        if preview_label is not None:
            image = Image.open(image_path)
            image = image.resize((120, 90))
            photo = ImageTk.PhotoImage(image)
            preview_label.config(image=photo)
            preview_label.image = photo
    else:
        messagebox.showinfo("Batal", "Tidak ada file yang dipilih.")


def run_detection_loop():
    global status_var, file_var, preview_label

    root = tk.Tk()
    root.title("Deteksi Kematangan Tomat")
    root.geometry("420x300")
    root.resizable(False, False)

    status_var = tk.StringVar(value="Belum ada hasil deteksi")
    file_var = tk.StringVar(value="Belum ada file dipilih")

    title_label = tk.Label(
        root,
        text="Deteksi Kematangan Tomat",
        font=("Arial", 14, "bold")
    )
    title_label.pack(pady=(16, 8))

    preview_label = tk.Label(root)
    preview_label.pack(pady=(0, 6))

    file_label = tk.Label(
        root,
        textvariable=file_var,
        wraplength=380,
        justify="center"
    )
    file_label.pack(pady=(0, 4))

    status_label = tk.Label(
        root,
        textvariable=status_var,
        wraplength=380,
        justify="center"
    )
    status_label.pack(pady=(0, 8))

    button_frame = tk.Frame(root)
    button_frame.pack(pady=6)

    select_button = tk.Button(
        button_frame,
        text="Pilih Gambar Tomat",
        width=20,
        height=2,
        command=choose_and_detect,
    )
    select_button.pack(side=tk.LEFT, padx=6)

    reset_button = tk.Button(
        button_frame,
        text="Reset",
        width=10,
        height=2,
        command=reset_status,
    )
    reset_button.pack(side=tk.LEFT, padx=6)

    exit_button = tk.Button(
        root,
        text="Keluar",
        width=16,
        height=1,
        command=root.destroy,
    )
    exit_button.pack(pady=6)

    root.mainloop()


if __name__ == "__main__":
    run_detection_loop()