import cv2
import numpy as np
from PIL import Image
import os

try:
    from skimage import exposure, restoration, filters, util, color
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ scikit-image не установлен — улучшение ограничено OpenCV.")

try:
    from realesrgan import RealESRGAN
    USE_AI = True
except ImportError:
    USE_AI = False
    print("⚠️ Real-ESRGAN не установлен. AI-апскейл будет пропущен.")


# ============================================================
# 🧠 1. Оценка качества изображения (fallback, без imquality)
# ============================================================
def estimate_quality(img):
    """
    Оценивает качество изображения по контрасту и резкости.
    Возвращает "оценку качества" 0–100.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)

    # Простейшая эвристика
    score = (contrast * 0.6 + lap_var * 0.4) / 5
    score = np.clip(score, 0, 100)
    return round(score, 2), contrast, lap_var, brightness


# ============================================================
# 🧩 2. Классическое улучшение (OpenCV + skimage)
# ============================================================
def classical_enhance(img, img_path=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blur)

    if SKIMAGE_AVAILABLE:
        denoised = restoration.denoise_wavelet(clahe_img, rescale_sigma=True)
        sharp = filters.unsharp_mask(denoised, radius=1.0, amount=1.7)
        final = util.img_as_ubyte(sharp)
    else:
        final = cv2.addWeighted(clahe_img, 1.5, cv2.GaussianBlur(clahe_img, (0, 0), 3), -0.5, 0)

    if img_path:
        save_path = os.path.splitext(img_path)[0] + "_enhanced_classic.jpg"
        cv2.imwrite(save_path, final)
        print(f"✅ Классическое улучшение сохранено: {save_path}")

    return final


# ============================================================
# 🚀 3. AI-суперрезолюшен (если Real-ESRGAN установлен)
# ============================================================
def ai_upscale(img_path):
    if not USE_AI:
        print("⚠️ Real-ESRGAN не доступен.")
        return None

    try:
        model = RealESRGAN.from_pretrained('RealESRGAN_x4plus')
        image = Image.open(img_path).convert('RGB')
        sr_image = model.predict(image)
        save_path = os.path.splitext(img_path)[0] + "_upscaled.png"
        sr_image.save(save_path)
        print(f"🚀 AI-апскейл сохранён: {save_path}")
        return cv2.imread(save_path)
    except Exception as e:
        print(f"❌ Ошибка Real-ESRGAN: {e}")
        return None


# ============================================================
# ⚙️ 4. Основная функция — автоулучшение
# ============================================================
def auto_enhance(img_path, save_debug=True):
    """
    Автоматически выбирает стратегию улучшения качества фото.
    Возвращает улучшенное изображение.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не удалось открыть файл: {img_path}")

    score, contrast, lap, bright = estimate_quality(img)
    print(f"📊 Качество: {score}/100 | Контраст={contrast:.1f} | Резкость={lap:.1f} | Яркость={bright:.1f}")

    # Логика выбора
    if score > 70:
        method = "light"
    elif 40 < score <= 70 and USE_AI:
        method = "ai"
    else:
        method = "heavy"

    print(f"🧩 Выбран метод улучшения: {method.upper()}")

    # Применяем
    if method == "ai":
        enhanced = ai_upscale(img_path)
        if enhanced is None:
            enhanced = classical_enhance(img, img_path)
    elif method == "heavy":
        enhanced = classical_enhance(img, img_path)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    else:
        enhanced = classical_enhance(img, img_path)

    # Сохраняем результат
    if save_debug:
        save_path = os.path.splitext(img_path)[0] + "_enhanced_final.jpg"
        cv2.imwrite(save_path, enhanced)
        print(f"✅ Финальное улучшение сохранено: {save_path}")

    return enhanced


# ============================================================
# 🧪 Тест
# ============================================================
if __name__ == "__main__":
    path = "../data/raw/test18.jpg"  # путь к твоему изображению
    enhanced = auto_enhance(path)
    print("✅ Улучшение завершено успешно.")
