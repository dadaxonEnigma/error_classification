import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

from ocr_extractor import extract_error_text, detect_status_smart, find_red_region
from test_ocr import correct_text

st.set_page_config(page_title="OCR Status Detector", layout="centered")

st.title("📄 OCR Error / Status Detector")
st.write("Загрузите одно или несколько изображений. Приложение определит ошибку/статус и покажет найденные элементы.")

uploaded_files = st.file_uploader(
    "Выберите изображения",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Создаём папки
os.makedirs("debug", exist_ok=True)
os.makedirs("result", exist_ok=True)

def resize_image(img_cv, max_width=1000):
    """Уменьшает изображение, сохраняя пропорции."""
    h, w = img_cv.shape[:2]
    if w <= max_width:
        return img_cv
    scale = max_width / w
    return cv2.resize(img_cv, (int(w * scale), int(h * scale)))


def draw_box(image, box, color=(0, 255, 0), thickness=3):
    """Рисует прямоугольник по box=(x,y,w,h)."""
    x, y, w, h = box
    img = image.copy()
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img


if uploaded_files:
    for uploaded_file in uploaded_files:

        with st.container():
            st.markdown(
                """
                <div style="background-color:#f7f7f9;padding:18px;border-radius:14px;
                border:1px solid #ddd;margin-bottom:20px;">
                """,
                unsafe_allow_html=True
            )

            st.subheader(f"📌 Файл: {uploaded_file.name}")

            # --- Load image ---
            img_bytes = uploaded_file.read()
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_cv_original = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            # Уменьшаем изображение
            img_cv = resize_image(img_cv_original)

            # Сохраняем уменьшенное изображение
            base_name = os.path.splitext(uploaded_file.name)[0]
            original_out = f"result/{base_name}_original.jpg"
            cv2.imwrite(original_out, img_cv)

            img_display = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

            tmp_path = f"temp_{uploaded_file.name}.jpg"
            cv2.imwrite(tmp_path, img_cv)

            st.image(img_display, caption="Исходное изображение (уменьшенное)", use_column_width=True)

            final = "Unknown"
            found_text = None

            # 🔶 Данные для визуализации
            red_box = None
            status_box = None

            # 1️⃣ — Пытаемся найти красную область
            try:
                red_region = find_red_region(tmp_path, show_debug=False)
                if red_region is not None:
                    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

                    lower_red1 = np.array([0, 100, 100])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([160, 100, 100])
                    upper_red2 = np.array([179, 255, 255])

                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    mask = cv2.bitwise_or(mask1, mask2)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(contour)
                        red_box = (x, y, w, h)

                text_err = extract_error_text(tmp_path, show_debug=False)
            except:
                text_err = None

            # Если ошибка существует
            if text_err and "❌" not in text_err and len(text_err.strip()) > 2:
                found_text = text_err
                final = correct_text(text_err)
            else:
                # 2️⃣ — Ищем статус
                status = detect_status_smart(tmp_path, show_debug=False)

                if status:
                    found_text = status["line_text"]
                    final = status["text"]
                    status_box = status["box"]

            # 🎯 Итоговый результат
            st.markdown("### 🎯 Итоговый результат")
            st.success(final)

            # Найденный текст
            if found_text:
                st.markdown("### 🔠 Найденный текст")
                st.info(found_text)

            # 🟥 Красная область
            if red_box:
                st.markdown("### 🟥 Красная область (ошибка)")

                red_img = draw_box(img_cv, red_box, color=(255, 0, 0))
                st.image(red_img, use_column_width=True)

                result_path = f"result/{base_name}_red.jpg"
                cv2.imwrite(result_path, red_img)

            # 🟩 Статус
            if status_box:
                st.markdown("### 🟩 Статус")

                status_img = draw_box(img_cv, status_box, color=(0, 255, 0))
                st.image(status_img, use_column_width=True)

                result_path = f"result/{base_name}_status.jpg"
                cv2.imwrite(result_path, status_img)

            st.markdown("</div>", unsafe_allow_html=True)
