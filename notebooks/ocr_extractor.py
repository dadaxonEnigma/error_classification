import cv2
import numpy as np
from PIL import Image
import pytesseract
import os

# путь до tesseract, если нужно
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def find_red_region(img_path, show_debug=True):
    """
    Находит красную область на изображении и возвращает её обрезанную версию.
    Если show_debug=True — показывает и сохраняет визуализацию.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {img_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Красный цвет имеет два диапазона оттенков в HSV
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

    # Берём самый большой контур — обычно это плашка ошибки
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # рисуем рамку для визуализации
    if show_debug:
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        save_path = os.path.join(os.path.dirname(img_path), "debug_detected_error.jpg")
        cv2.imwrite(save_path, debug_img)
        print(f"✅ Сохранено изображение с рамкой: {save_path}")

        # Если хочешь показать окно — можно раскомментировать (вне Jupyter)
        # cv2.imshow("Detected Error Region", debug_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    cropped = img[y:y+h, x:x+w]
    return cropped


def preprocess_for_ocr(image):
    """Улучшаем изображение перед OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_error_text(img_path, langs="eng+rus+uzb", show_debug=True):
    """
    Ищет красную плашку, показывает где она, и извлекает из неё текст ошибки.
    """
    red_region = find_red_region(img_path, show_debug=show_debug)
    if red_region is None:
        return "❌ Красная область не найдена"

    processed = preprocess_for_ocr(red_region)
    pil_img = Image.fromarray(processed)
    text = pytesseract.image_to_string(pil_img, lang=langs)
    return text.strip()


if __name__ == "__main__":
    path = "../data/raw/test13.jpg"  # укажи свой файл
    text = extract_error_text(path)
    print("📜 Извлечённый текст ошибки:")
    print(text)
