import cv2
import numpy as np
import os

# ============================================================
# 🧠 1. Оценка качества изображения
# ============================================================
def estimate_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    score = (contrast * 0.6 + lap_var * 0.4) / 5
    score = np.clip(score, 0, 100)
    return round(score, 2), contrast, lap_var, brightness

# ============================================================
# 🚀 2. Повышение разрешения и читаемости текста
# ============================================================
def text_enhance(img_path):
    """
    Повышает читаемость текста:
      - апскейл ×2 через ESPCN (OpenCV DNN)
      - контраст, резкость, бинаризация
    """
    from cv2 import dnn_superres
    sr = dnn_superres.DnnSuperResImpl_create()
    model_path = "ESPCN_x2.pb"

    if not os.path.exists(model_path):
        import urllib.request
        print("⬇️ Загружаю модель ESPCN_x2...")
        urllib.request.urlretrieve(
            "https://github.com/Saafke/ESPCN_super_resolution/raw/master/ESPCN_x2.pb",
            model_path
        )

    sr.readModel(model_path)
    sr.setModel("espcn", 2)

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не удалось открыть {img_path}")

    # Апскейл ×2
    upscaled = sr.upsample(img)

    # Переводим в grayscale и усиливаем резкость
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

    # Повышаем контраст и очищаем фон
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(sharp)

    # Адаптивная бинаризация для читаемости текста
    binary = cv2.adaptiveThreshold(
        contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 35, 15
    )

    save_path = os.path.splitext(img_path)[0] + "_text_enhanced.jpg"
    cv2.imwrite(save_path, binary)
    print(f"✅ Улучшенное изображение сохранено: {save_path}")
    return binary

# ============================================================
# ⚙️ 3. Автоматическая логика выбора
# ============================================================
def auto_enhance(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не удалось открыть файл: {img_path}")

    score, contrast, lap, bright = estimate_quality(img)
    print(f"📊 Качество: {score}/100 | Контраст={contrast:.1f} | Резкость={lap:.1f} | Яркость={bright:.1f}")

    if score < 60:
        print("🧩 Фото с низким качеством — применяю улучшение текста")
        enhanced = text_enhance(img_path)
    else:
        print("🧩 Фото достаточно чёткое — лёгкое улучшение")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

    save_path = os.path.splitext(img_path)[0] + "_final_text.jpg"
    cv2.imwrite(save_path, enhanced)
    print(f"✅ Финальный результат сохранён: {save_path}")
    return enhanced

# ============================================================
# 🧪 Тест
# ============================================================
if __name__ == "__main__":
    path = "../data/raw/test18.jpg"
    enhanced = auto_enhance(path)
    print("✅ Улучшение завершено успешно.")
