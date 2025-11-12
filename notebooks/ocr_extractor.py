import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import re

# Укажи путь к Tesseract, если нужно
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================
# 🔴 1. Поиск красной области — оставляем как есть
# =========================================================
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
    if not contours:
        print("❌ Красная область не найдена.")
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


def extract_status_text(img_path, langs="eng+rus+uzb", show_debug=True):
    """
    Извлекает статус (Approved / Not approved / Tasdiqlangan / Tasdiqlanmagan / Bekor Qilingan / Одобрено)
    с поддержкой зелёных и жёлтых текстов на белом фоне.
    """
    import re

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {img_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 🎨 Диапазоны цвета
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])

    # Маски
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_green, mask_yellow)

    # Изолируем цветной текст
    result = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Усиливаем контраст и инвертируем
    gray = cv2.convertScaleAbs(gray, alpha=3.0, beta=0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if show_debug:
        debug_path = os.path.join(os.path.dirname(img_path), "debug_status_mask.jpg")
        cv2.imwrite(debug_path, thresh)
        print(f"🧩 Отладочная маска сохранена: {debug_path}")

    # OCR по цветным надписям
    data = pytesseract.image_to_data(
        thresh, lang=langs, config="--psm 6", output_type=pytesseract.Output.DICT
    )

    found_status = None
    best_conf = 0

    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue

        conf_raw = data["conf"][i]
        try:
            conf = int(float(conf_raw))
        except (ValueError, TypeError):
            conf = 0

        clean_word = re.sub(r'[^A-Za-zА-Яа-я]', '', word).lower()

        if conf < 40:
            continue

        if clean_word in ["approved", "tasdiqlangan", "одобрено"]:
            found_status = "Approved"
            best_conf = conf
        elif clean_word in ["notapproved", "radetilgan", "неодобрено"]:
            found_status = "Not approved"
            best_conf = conf

    if found_status:
        print(f"✅ Найден статус: {found_status} (уверенность {best_conf}%)")
        return found_status
    else:
        print("❌ Статус не найден.")
        return "❌ Статус не найден"




# =========================================================
# 🧠 2. Улучшение изображения после обрезки
# =========================================================

def enhance_edges(img):
    """Повышает резкость и восстанавливает контуры букв."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
    unsharp = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)
    lap = cv2.Laplacian(unsharp, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    sharpened = cv2.addWeighted(unsharp, 1.0, lap, 0.4, 0)
    return sharpened


def normalize_lighting(gray):
    """Выравнивает освещённость и повышает локальный контраст."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    norm = clahe.apply(tophat)
    return norm


def adaptive_thicken(gray):
    """Делает тонкий текст чуть жирнее, не искажая форму."""
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.Canny(gray, 30, 100)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    combined = cv2.bitwise_or(gray, dilated)
    return combined


def super_preprocess(image, img_path=None, show_debug=True):
    """
    Комбинированное улучшение качества изображения перед OCR.
    Сохраняет все промежуточные этапы.
    """
    base_dir = os.path.dirname(img_path) if img_path else "."

    # 🔹 1. Повышаем резкость
    step1 = enhance_edges(image)
    cv2.imwrite(os.path.join(base_dir, "debug_step1_sharpened.jpg"), step1)


    print(f"🧩 Сохранены все этапы улучшения в: {base_dir}")

    return step1


# =========================================================
# 🔤 3. Извлечение текста
# =========================================================
def extract_error_text(img_path, langs="eng+rus+uzb", show_debug=True):
    red_region = find_red_region(img_path, show_debug=show_debug)
    if red_region is None:
        return "❌ Красная область не найдена"

    processed = super_preprocess(red_region, img_path, show_debug)
    pil_img = Image.fromarray(processed)

    custom_config = r'--oem 3 --psm 6'  # 6 — режим многострочного текста
    text = pytesseract.image_to_string(pil_img, lang=langs, config=custom_config)
    return text.strip()


if __name__ == "__main__":
    path = "../data/raw/test1_8.jpg"  # Укажи путь к изображению

    print("🔍 Анализ изображения...")

    # 1️⃣ Пытаемся найти красную область (ошибка)
    try:
        text = extract_error_text(path, show_debug=True)
        if text and "❌" not in text and len(text.strip()) > 2:
            print("\n📜 Извлечённый текст ошибки:")
            print(text)
        else:
            # 2️⃣ Если красного баннера нет — ищем статус (Approved / Not approved)
            from ocr_extractor import extract_status_text
            status = extract_status_text(path, show_debug=True)
            print("\n📗 Извлечённый статус:")
            print(status)
    except Exception as e:
        print(f"⚠️ Ошибка при обработке: {e}")