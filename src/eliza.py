"""B · Klasik NLP: ELIZA tarzı kural tabanlı sohbet cevapları.

ELIZA (Weizenbaum, 1966) gibi çalışır: mesaj bir regex kalıbına uyuyorsa, kalıptan yakalanan
parçayı cevap şablonuna yerleştirip döndürür ("I am X" → "Why are you X?").

Router'da iki yerde kullanılır:
  1. Model çalışmadan ÖNCE: mesajın TAMAMI bir sohbet kalıbına uyuyorsa (ör. "merhaba", "thanks")
     cevap doğrudan buradan döner ve mesaj hiçbir ekibe gönderilmez.
  2. Model "small_talk" niyeti tahmin ederse: reply_for_intent() ile kural tabanlı cevap verilir.

"merhaba, bulaşık makinem bozuldu" gibi selam + gerçek sorun içeren mesajlar 1. adımda
YAKALANMAZ, çünkü kalıplar mesajın tamamına uymak zorunda (fullmatch).
"""
import random
import re

# (kalıp, Türkçe cevaplar, İngilizce cevaplar). Kalıplar normalize edilmiş (küçük harf) metne uygulanır.
# {0} → kalıptaki ilk yakalanan grup (ELIZA'nın "yansıtma" özelliği)
RULES = [
    (r"(merhaba|merhabalar|selam|selamlar|iyi günler|günaydın|iyi akşamlar|hey)( (efendim|arkadaşlar))?",
     ["Merhaba! Size nasıl yardımcı olabilirim? Ürününüzle ilgili sorunu kısaca yazabilirsiniz."], None),
    (r"(hi|hello|hey there|hey|good (morning|afternoon|evening))( there)?",
     None, ["Hello! How can I help you today? Please describe the problem with your appliance."]),
    (r"(nasılsın|nasılsınız|naber|ne haber)",
     ["İyiyim, teşekkürler! Sizin için ne yapabilirim?"], None),
    (r"(merhaba|selam) (nasılsın|nasılsınız)",
     ["Merhaba, iyiyim teşekkürler! Ürününüzle ilgili nasıl yardımcı olabilirim?"], None),
    (r"(how are you|how's it going|how are you doing)( today)?",
     None, ["I'm doing well, thanks for asking! What can I do for you?"]),
    (r"(teşekkürler|teşekkür ederim|çok teşekkürler|sağol|sağolun|eyvallah)( çok)?",
     ["Rica ederim! Başka bir konuda yardımcı olabilir miyim?"], None),
    (r"(thanks|thank you|thank you so much|thanks a lot|cheers)",
     None, ["You're welcome! Is there anything else I can help with?"]),
    (r"(görüşürüz|hoşça kal|hoşçakal|iyi günler dilerim|bay bay)",
     ["Görüşmek üzere, iyi günler dileriz!"], None),
    (r"(bye|goodbye|see you|see you later|have a nice day)",
     None, ["Goodbye, have a great day!"]),
    (r"(sen kimsin|bot musun|robot musun|insan mısın)",
     ["Ben otomatik destek asistanıyım. Mesajınızı doğru ekibe yönlendiriyorum."], None),
    (r"(who are you|are you a bot|are you a robot|are you human)",
     None, ["I'm an automated support assistant. I route your message to the right team."]),
    # Klasik ELIZA yansıtması: duyguyu geri yansıt
    (r"(?:çok )?(sinirliyim|kızgınım|üzgünüm|bıktım|mutsuzum)",
     ["{0} olduğunuzu duyduğuma üzüldüm. Sorunu anlatırsanız hemen ilgili ekibe iletebilirim."], None),
    (r"i(?: am|'m) (?:so |very |really )?(angry|upset|frustrated|annoyed|sad|tired of this)",
     None, ["I'm sorry you feel {0}. Tell me what's wrong and I'll route it to the right team."]),
]
_COMPILED = [(re.compile(p), tr, en) for p, tr, en in RULES]

# ELIZA'nın "reflection" adımı: kullanıcının kelimesini cevaba uygun biçime çevirir
# ("sinirliyim" → "sinirli": "Sinirli olduğunuzu duyduğuma üzüldüm")
REFLECTIONS = {"sinirliyim": "sinirli", "kızgınım": "kızgın", "üzgünüm": "üzgün",
               "bıktım": "bıkmış", "mutsuzum": "mutsuz"}

# Model small_talk niyeti tahmin ettiğinde verilen cevaplar (CLINC150 small_talk alanı)
SMALL_TALK_INTENTS = {
    "greeting", "goodbye", "thank_you", "how_old_are_you", "what_is_your_name", "who_made_you",
    "meaning_of_life", "fun_fact", "tell_joke", "where_are_you_from", "are_you_a_bot",
    "what_are_your_hobbies", "do_you_have_pets", "who_do_you_work_for", "what_can_i_ask_you",
}
_INTENT_REPLY = {
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Goodbye, have a great day!",
    "thank_you": "You're welcome!",
    "are_you_a_bot": "Yes, I'm an automated support assistant.",
    "what_is_your_name": "I'm the support assistant of the appliance service center.",
    "who_made_you": "I was built as a course project for SWE015.",
    "what_can_i_ask_you": "You can describe any problem with your appliance, warranty or appointment.",
}
_DEFAULT_SMALL_TALK = "I'm here to help with your appliance questions. What seems to be the problem?"


def eliza_reply(text_norm: str) -> str | None:
    """Mesajın TAMAMI bir sohbet kalıbına uyuyorsa cevap döndürür, uymuyorsa None.
    text_norm: normalize.normalize_text() çıktısı (küçük harf, noktalamasız)."""
    t = text_norm.strip()
    for pattern, tr, en in _COMPILED:
        m = pattern.fullmatch(t)
        if m:
            replies = tr or en
            reply = random.choice(replies)
            groups = [REFLECTIONS.get(g, g) for g in m.groups() if g]
            if "{0}" in reply and groups:
                reply = reply.format(*groups)
                reply = reply[0].upper() + reply[1:]
            return reply
    return None


def reply_for_intent(intent: str) -> str:
    """Model small_talk niyeti bulduğunda kural tabanlı cevap."""
    return _INTENT_REPLY.get(intent, _DEFAULT_SMALL_TALK)


if __name__ == "__main__":
    from src.normalize import normalize_text
    for s in ["Merhaba!", "selam nasılsın", "Thanks a lot!!", "I'm so frustrated",
              "çok sinirliyim", "merhaba, bulaşık makinem bozuldu", "hello there"]:
        print(f"{s!r:40s} → {eliza_reply(normalize_text(s))}")
