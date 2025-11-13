import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

from ocr_extractor import extract_error_text, detect_status_smart, find_red_region
from test_ocr import correct_text
from response_base import get_response_text  # текстовые ответы


st.set_page_config(page_title="OCR Status Detector", layout="centered")

st.title("📄 OCR Error / Status Detector")
st.write("Загрузите одно или несколько изображений. Система определит ошибку или статус и даст текстовый ответ.")

uploaded_files = st.file_uploader(
    "Выберите изображения",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# === создаём только нужную папку ===
os.makedirs("result", exist_ok=True)


def resize_image(img_cv, max_width=900):
    """Уменьшает изображение, сохраняя пропорции."""
    h, w = img_cv.shape[:2]
    if w <= max_width:
        return img_cv
    scale = max_width / w
    return cv2.resize(img_cv, (int(w * scale), int(h * scale)))


def draw_box(image, box, color, thickness=3):
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
                <div style="background-color:#f7f7f9;padding:16px;border-radius:14px;
                border:1px solid #ddd;margin-bottom:20px;">
                """,
                unsafe_allow_html=True
            )

            st.subheader(f"📌 Файл: {uploaded_file.name}")

            # ========== загрузка ==========
            img_bytes = uploaded_file.read()
            img_np = np.frombuffer(img_bytes, np.uint8)
            img_cv_orig = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            # уменьшаем
            img_cv = resize_image(img_cv_orig)

            # временный файл
            tmp_path = f"temp_{uploaded_file.name}.jpg"
            cv2.imwrite(tmp_path, img_cv)

            final = "Unknown"
            found_text = None
            red_box = None
            status_box = None

            # ========== 1) Красная ошибка ==========
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
                        cont = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(cont)
                        red_box = (x, y, w, h)

                text_err = extract_error_text(tmp_path, show_debug=False)

            except:
                text_err = None

            # обработка ошибки
            if text_err and "❌" not in text_err and len(text_err.strip()) > 2:
                found_text = text_err
                final = correct_text(text_err)

            else:
                # ========= 2) Ищем статус =========
                status = detect_status_smart(tmp_path, show_debug=False)

                if status:
                    status_box = status["box"]
                    found_text = status["line_text"]
                    final = status["text"]

            # ========== Итоговый результат ==========
            st.markdown("### 🎯 Aniqlangan holat")
            st.success(final)

            # найденный текст
            if found_text:
                st.markdown("### 🔠 Topilgan matn")
                st.info(found_text)

            # 📝 текстовый ответ из базы
            response_text = get_response_text(final)
            st.markdown("### 📝 Tavsiya / Yechim")
            st.write(response_text)

            # ========== визуализация ==========
            # Красная область
            if red_box:
                st.markdown("### 🟥 Xato joyi")
                red_img = draw_box(img_cv, red_box, (255, 0, 0))
                st.image(red_img, use_column_width=True)
                cv2.imwrite(f"result/{uploaded_file.name}_red.jpg", red_img)

            # Статус область
            if status_box:
                st.markdown("### 🟩 Status joyi")
                status_img = draw_box(img_cv, status_box, (0, 255, 0))
                st.image(status_img, use_column_width=True)
                cv2.imwrite(f"result/{uploaded_file.name}_status.jpg", status_img)

            st.markdown("</div>", unsafe_allow_html=True)
