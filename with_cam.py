import cv2
import numpy as np

# ============================================================
# DETEKSI KEMATANGAN TOMAT REAL-TIME MENGGUNAKAN KAMERA
# ============================================================
# Tombol:
# - Q : Keluar
# ============================================================

def detect_tomato_maturity():
    # Buka kamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka.")
        return

    print("Kamera berhasil dibuka.")
    print("Tekan tombol 'Q' untuk keluar.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame.")
            break

        # Resize
        frame = cv2.resize(frame, (800, 600))

        # Blur
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Konversi HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ============================================================
        # RANGE WARNA
        # ============================================================

        # Hijau (Mentah)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        # Kuning / Orange (Setengah Matang)
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([34, 255, 255])

        # Merah (Matang)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # ============================================================
        # MASKING
        # ============================================================

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2

        # Gabungkan semua mask
        combined_mask = mask_green + mask_yellow + mask_red

        # Hilangkan noise
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Cari kontur
        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter objek kecil
            if area < 3000:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Ambil ROI
            roi = hsv[y:y+h, x:x+w]

            # Hitung pixel warna dalam ROI
            green = cv2.countNonZero(
                cv2.inRange(roi, lower_green, upper_green)
            )

            yellow = cv2.countNonZero(
                cv2.inRange(roi, lower_yellow, upper_yellow)
            )

            red = cv2.countNonZero(
                cv2.inRange(roi, lower_red1, upper_red1)
            ) + cv2.countNonZero(
                cv2.inRange(roi, lower_red2, upper_red2)
            )

            # Tentukan kematangan
            if red > yellow and red > green:
                maturity = "MATANG"
                color = (0, 0, 255)

            elif yellow > green:
                maturity = "SETENGAH MATANG"
                color = (0, 255, 255)

            else:
                maturity = "MENTAH"
                color = (0, 255, 0)

            # Gambar bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                3
            )

            # Tampilkan teks
            cv2.putText(
                frame,
                maturity,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                3
            )

            # Tampilkan persentase warna dominan
            total = red + yellow + green
            if total > 0:
                red_percent = (red / total) * 100
                cv2.putText(
                    frame,
                    f"Merah: {red_percent:.1f}%",
                    (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        # Tampilkan hasil
        cv2.imshow("Deteksi Kematangan Tomat", frame)
        cv2.imshow("Mask HSV", combined_mask)

        # Keluar dengan tombol Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan resource
    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    detect_tomato_maturity()