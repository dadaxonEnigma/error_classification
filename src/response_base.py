RESPONSE_BASE = {
    "network error":
        "Assalomu alaykum! Internet ulanishida muammo aniqlandi. "
        "Iltimos, internetingizni qayta tekshirib ko‘ring.",

    "coal is already included in this phone number":
        "Assalomu alaykum! Ushbu raqamga ko‘mir uchun ariza berilgan.",

    "not approved":
        "Assalomu alaykum! Sizning arizangiz hozircha tasdiqlanmadi. "
        "2–3 kun ichida tasdiqlash amalga oshmasa, mahalla raisiga murojaat qilishingiz tavsiya etiladi.",

    "approved":
        "Assalomu aleykum! Arizangiz tasdiqlangan. Ko‘mir kelganda olib ketishingiz mumkin.",
    
    "user address is mandatory":
        "Assalomu aleykum! Assalomu alaykum! Sozlamalarga kirib, shaxsiy ma’lumotlaringizni to‘g‘rilang.",

    "wrong verification code":
        "Assalomu alaykum! Siz kiritgan tasdiqlash kodi noto‘g‘ri. Iltimos, SMSda yuborilgan kodni qayta kiriting.",


    # fallback
    "unknown":
        "Kechirasiz, ushbu rasm bo‘yicha aniq xulosa chiqarib bo‘lmadi. "
        "Iltimos, rasmni aniqroq va to‘liq yuboring."
}



def get_response_text(result_text: str) -> str:
    result_text = result_text.strip().lower()
    
    # прямое совпадение
    if result_text in RESPONSE_BASE:
        return RESPONSE_BASE[result_text]

    # fallback – ищем похожие ключи
    from rapidfuzz import fuzz

    best_key = None
    best_score = 0

    for key in RESPONSE_BASE.keys():
        score = fuzz.partial_ratio(result_text, key)
        if score > best_score:
            best_score = score
            best_key = key

    if best_score > 70:
        return RESPONSE_BASE[best_key]

    return RESPONSE_BASE["Unknown"]
