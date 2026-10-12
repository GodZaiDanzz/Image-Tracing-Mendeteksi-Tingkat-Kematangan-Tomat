import cv2
import numpy as np
import os
from datetime import datetime

# ============================================================
# DETEKSI KEMATANGAN TOMAT
# - Real Time Camera
# - Multi Tomato Detection
# - Auto Capture
# - HSV Method
# ============================================================

SAVE_FOLDER = "hasil_deteksi"

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


def save_detection(frame):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(SAVE_FOLDER, filename)
    cv2.imwrite(filepath, frame)
    print(f"[INFO] Foto disimpan: {filepath}")


def detect_rotten(roi_hsv, min_ratio=0.05):
    """Deteksi area busuk sederhana pada ROI HSV.

    Mengembalikan (ratio, mask) di mana ratio adalah persentase pixel yang
    cocok sebagai busuk terhadap total ROI.
    """
    if roi_hsv is None or roi_hsv.size == 0:
        return 0.0, np.zeros((1, 1), dtype=np.uint8)

    lower_brown = np.array([5, 80, 40])
    upper_brown = np.array([30, 255, 200])
    mask_brown = cv2.inRange(roi_hsv, lower_brown, upper_brown)

    h, s, v = cv2.split(roi_hsv)
    mean_v = float(np.mean(v))
    shadow_thresh = max(40, mean_v * 0.6)
    mask_shadow = v < shadow_thresh

    mask_brown = cv2.bitwise_and(mask_brown, cv2.bitwise_not(mask_shadow.astype(np.uint8) * 255))

    try:
        roi_bgr = cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = cv2.cvtColor(cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap).astype(np.uint8)
    lap_thresh = 12
    texture_mask = lap_abs > lap_thresh

    final_mask = cv2.bitwise_and(mask_brown, (texture_mask.astype(np.uint8) * 255))

    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask_clean)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 100:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)

    rotten_pixels = cv2.countNonZero(cleaned)
    total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]

    ratio = (rotten_pixels / total_pixels) if total_pixels > 0 else 0.0
    return ratio, cleaned


def detect_tomato_maturity():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera tidak dapat dibuka.")
        return

    print("Kamera berhasil dibuka.")
    print("Tekan tombol Q untuk keluar.")

    last_capture_time = 0
    capture_interval = 5  # detik

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal membaca frame.")
            break

        frame = cv2.resize(frame, (900, 700))
        output = frame.copy()

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ========================================================
        # RANGE WARNA
        # ========================================================

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([34, 255, 255])

        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # ========================================================
        # MASKING
        # ========================================================

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_red = (
            cv2.inRange(hsv, lower_red1, upper_red1) +
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        combined_mask = mask_green + mask_yellow + mask_red

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

        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        tomato_count = 0
        detected_this_frame = False

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 3000:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Filter agar hanya objek bulat seperti tomat
            if circularity < 0.65:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            aspect_ratio = w / float(h)

            if aspect_ratio < 0.75 or aspect_ratio > 1.35:
                continue

            roi = hsv[y:y + h, x:x + w]

            green = cv2.countNonZero(
                cv2.inRange(roi, lower_green, upper_green)
            )

            yellow = cv2.countNonZero(
                cv2.inRange(roi, lower_yellow, upper_yellow)
            )

            red = (
                cv2.countNonZero(
                    cv2.inRange(roi, lower_red1, upper_red1)
                ) +
                cv2.countNonZero(
                    cv2.inRange(roi, lower_red2, upper_red2)
                )
            )

            total = red + yellow + green

            if total < (w * h * 0.35):
                continue

            tomato_count += 1
            detected_this_frame = True

            # ====================================================
            # KLASIFIKASI
            # ====================================================

            if red > yellow and red > green:
                maturity = "MATANG"
                color = (0, 0, 255)

            elif yellow > green:
                maturity = "SETENGAH"
                color = (0, 255, 255)

            else:
                maturity = "MENTAH"
                color = (0, 255, 0)

            red_percent = (red / total) * 100
            red_percent = (red / total) * 100

            # Deteksi area busuk pada ROI
            rotten_ratio, rotten_mask = detect_rotten(roi)

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                color,
                3
            )

            cv2.putText(
                output,
                f"{maturity}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                output,
                f"Merah: {red_percent:.1f}%",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            if rotten_ratio > 0.05:
                cv2.putText(
                    output,
                    f"BUSUK: {rotten_ratio*100:.1f}%",
                    (x, y + h + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.imshow("Mask Busuk", rotten_mask)

        # ========================================================
        # TAMPILKAN JUMLAH TOMAT
        # ========================================================

        cv2.putText(
            output,
            f"Jumlah Tomat: {tomato_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )

        # ========================================================
        # AUTO CAPTURE
        # ========================================================

        current_time = cv2.getTickCount() / cv2.getTickFrequency()

        if detected_this_frame:
            if current_time - last_capture_time >= capture_interval:
                save_detection(output)
                last_capture_time = current_time

        cv2.imshow("Deteksi Kematangan Tomat", output)
        cv2.imshow("Mask HSV", combined_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_tomato_maturity()