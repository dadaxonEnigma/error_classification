import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
from rapidfuzz import fuzz
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def find_red_region(img_path, show_debug=True):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {img_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours) == 0:
        print("❌ Красная область не найдена – переключаемся на поиск статуса")
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    if show_debug:
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        save_path = os.path.join(os.path.dirname(img_path), "debug_detected_error.jpg")
        cv2.imwrite(save_path, debug_img)
        print(f"✅ Сохранено изображение с рамкой: {save_path}")

    cropped = img[y:y+h, x:x+w]
    return cropped


import cv2
import numpy as np
import os

# -----------------------------------------------
# SUPER RESOLUTION fallback (OpenCV)
# -----------------------------------------------
def upscale_cv2(image):
    """OpenCV Super Resolution: EDSR/FSRCNN fallback."""
    sr = cv2.dnn_superres.DnnSuperResImpl_create()

    model_paths = [
        "EDSR_x2.pb",
        "FSRCNN_x2.pb"
    ]

    for mp in model_paths:
        if os.path.exists(mp):
            sr.readModel(mp)
            if "EDSR" in mp:
                sr.setModel("edsr", 2)
            else:
                sr.setModel("fsrcnn", 2)
            return sr.upsample(image)

    # fallback: обычный upscale ×2
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


# -----------------------------------------------
# HIGH-RES PREPROCESS + DEBUG
# -----------------------------------------------
def high_res_preprocess(img, debug_dir=None, prefix="debug"):
    """
    Улучшение изображения + сохранение ВСЕХ этапов в debug_dir.
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # 0) original
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_step0_original.png"), img)

    # 1) шумоподавление
    den = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_step1_denoise.png"), den)

    # 2) super-resolution x2
    up = upscale_cv2(den)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_step2_upscaled.png"), up)

    # 3) CLAHE
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_step3_clahe.png"), cl)

    # 4) sharpen
    blur = cv2.GaussianBlur(cl, (0, 0), 2)
    sharp = cv2.addWeighted(cl, 1.7, blur, -0.7, 0)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_step4_sharpen.png"), sharp)
    return sharp


def detect_status_smart(img_path, show_debug=False):
    import cv2
    import numpy as np
    import pytesseract
    import os
    from rapidfuzz import fuzz

    STATUS_PATTERNS = [
        "not approved",
        "approved",
        "tasdiqlanmagan",
        "tasdiqlangan",
        "bekor qilindi",
        "bekor qilingan"
    ]

    NEGATION_MARKERS = [
        "not", "no", "ne", "не",
        "magan", "mas", "emas",
        "bekor",
        "rad", "rad etilgan"
    ]

    img = cv2.imread(img_path)
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(
        rgb, lang="eng+rus+uzb",
        config="--psm 6 --oem 3",
        output_type=pytesseract.Output.DICT
    )

    # ===== СБОР ТЕКСТА ПО СТРОКАМ =====
    lines = {}
    for i, text in enumerate(data["text"]):
        if not text or not text.strip():
            continue

        ln = data["line_num"][i]
        if ln not in lines:
            lines[ln] = {"text": "", "boxes": []}

        lines[ln]["text"] += " " + text.strip()
        lines[ln]["boxes"].append((
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i]
        ))

    found = []

    for ln, info in lines.items():
        line_text = info["text"].lower().strip()
        has_neg = any(marker in line_text for marker in NEGATION_MARKERS)

        for pattern in STATUS_PATTERNS:
            if pattern in ("approved", "tasdiqlangan") and has_neg:
                continue

            threshold = 85 if pattern in ("approved", "tasdiqlangan") else 72
            score = fuzz.partial_ratio(pattern, line_text)

            if score >= threshold:
                xs = [b[0] for b in info["boxes"]]
                ys = [b[1] for b in info["boxes"]]
                rights = [b[0] + b[2] for b in info["boxes"]]
                bottoms = [b[1] + b[3] for b in info["boxes"]]

                # ---- НОВЫЙ КОРРЕКТНЫЙ формат box ----
                x = min(xs)
                y = min(ys)
                w = max(rights) - x
                h = max(bottoms) - y

                found.append({
                    "text": pattern,
                    "score": score,
                    "box": (x, y, w, h),
                    "line_text": line_text
                })

    if not found:
        return None

    best = sorted(found, key=lambda x: x["box"][1])[-1]

    # ====== DEBUG РИСОВАНИЕ ======
    if show_debug:
        x, y, w, h = best["box"]
        dbg = img.copy()

        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"{best['text']} (score={best['score']})"
        cv2.putText(dbg, label, (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out = os.path.join(os.path.dirname(img_path), "debug_status_fixed.jpg")
        cv2.imwrite(out, dbg)
        print("📸 Debug saved:", out)

    return {
        "text": best["text"],
        "box": best["box"],   # (x, y, w, h) — правильный формат
        "score": best["score"],
        "line_text": best["line_text"]
    }





def enhance_edges(img):
    """Повышает резкость и восстанавливает контуры букв."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    unsharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    lap = cv2.Laplacian(unsharp, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    sharpened = cv2.addWeighted(unsharp, 1.0, lap, 0.4, 0)
    return sharpened

def super_preprocess(image, img_path=None, show_debug=True):

    base_dir = os.path.dirname(img_path) if img_path else "."

    # 🔹 1. Повышаем резкость
    step1 = enhance_edges(image)
    cv2.imwrite(os.path.join(base_dir, "debug_step1_sharpened.jpg"), step1)


    print(f"🧩 Сохранены все этапы улучшения в: {base_dir}")

    return step1


def extract_error_text(img_path, langs="eng+rus+uzb", show_debug=True):
    red_region = find_red_region(img_path, show_debug=show_debug)
    if red_region is None:
        return "❌ Красная область не найдена"

    debug_path = os.path.join(os.path.dirname(img_path), "debug_hr")
    enhanced = high_res_preprocess(red_region, debug_dir=debug_path)
    pil_img = Image.fromarray(enhanced)

    custom_config = r'--oem 3 --psm 6'  
    text = pytesseract.image_to_string(pil_img, lang=langs, config=custom_config)
    return text.strip()


if __name__ == "__main__":
    path = "../data/raw/all/test35.jpg" 

    print("🔍 Анализ изображения...")

    # 1️⃣ Пытаемся найти красную область (ошибка)
    try:
        text = extract_error_text(path, show_debug=True)
        if text and "❌" not in text and len(text.strip()) > 2:
            print("\n📜 Извлечённый текст ошибки:")
            print(text)
        else:
            # 2️⃣ Если красного баннера нет — ищем статус (Approved / Not approved)
            from ocr_extractor import detect_status_smart
            status = detect_status_smart(path, show_debug=True)
            print("\n📗 Извлечённый статус:")
            print(status)
    except Exception as e:
        print(f"⚠️ Ошибка при обработке: {e}")