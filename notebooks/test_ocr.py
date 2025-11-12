import os
import re
import json
from rapidfuzz import fuzz
from ocr_extractor import extract_error_text, extract_status_text

# =========================================================
# 🔹 Шаг 1. Список допустимых (правильных) текстов ошибок
# =========================================================
ERROR_TEMPLATES = [
    "The organization's address is not completely entered",
    "Network error",
    "Query did not return a unique result: 2 results was returned",
    "User address is mandatory",
    "Coal is already included in this number",
    "Wrong verification code",
    "No matching applications found",
    "Failed to parse multipart servlet request",
    "Invalid date",
]

# =========================================================
# 🔹 Вспомогательные функции
# =========================================================
def numeric_sort_key(filename):
    nums = re.findall(r'\d+', filename)
    return int(nums[0]) if nums else float('inf')

def clean_text(text: str) -> str:
    """Очистка текста от мусора и приведение к нижнему регистру"""
    text = re.sub(r'[^A-Za-zА-Яа-я0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def correct_text(ocr_text: str, threshold=70):
    """
    Возвращает наиболее подходящую фразу из ERROR_TEMPLATES
    или 'Unknown', если совпадение слишком слабое.
    """
    cleaned = clean_text(ocr_text)
    if not cleaned:
        return "Unknown"

    best_match = None
    best_score = 0

    for template in ERROR_TEMPLATES:
        score = fuzz.partial_ratio(cleaned, template.lower())
        if score > best_score:
            best_score = score
            best_match = template

    if best_score >= threshold:
        return best_match
    else:
        return "Unknown"

# =========================================================
# 🚀 Шаг 2. Основная обработка всех изображений
# =========================================================
def batch_ocr(input_folder="../data/raw/", output_json="../data/test_corrected.json"):
    results = {}

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"❌ Папка {input_folder} не найдена")

    # ✅ Числовая сортировка
    images = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=numeric_sort_key
    )

    if not images:
        print("⚠️ Нет изображений для обработки.")
        return

    print(f"🔍 Найдено {len(images)} изображений. Начинаю обработку...\n")

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        try:
            # 🔹 Сначала ищем ошибку (красный баннер)
            text = extract_error_text(img_path, show_debug=False)

            # Если ошибка не найдена — ищем статус
            if not text or "❌" in text or len(text.strip()) < 3:
                text = extract_status_text(img_path, show_debug=False)

            # 🔹 Исправляем результат
            corrected_text = correct_text(text)
            results[img_name] = corrected_text

            print(f"✅ {img_name}: {corrected_text}")

        except Exception as e:
            results[img_name] = f"Ошибка: {str(e)}"
            print(f"❌ Ошибка при обработке {img_name}: {e}")

    # 📁 Сохраняем результаты в JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n📄 Результаты сохранены в: {output_json}")

# =========================================================
# ▶️ Точка входа
# =========================================================
if __name__ == "__main__":
    batch_ocr()
