"""Yönlendirme mantığı: mesaj → normalize → (ELIZA) → model → iki eşikli karar → ekip.

Uygulama (app.py) ve uçtan uca değerlendirme (eval_app.py) bu dosyayı kullanır. Karar kuralı
değerlendirmeyle AYNIDIR: evaluate.decide() + modelin thresholds.json dosyası.

Akış:
  1. normalize()      → model kodu çıkarma, küçük harf, yazım düzeltme (B)
  2. eliza_reply()    → mesajın tamamı selam/teşekkür/veda kalıbına uyuyorsa kural tabanlı cevap (B), ekip yok
  3. Karar modeli     → DistilmBERT INT8 (ONNX Runtime) niyeti ve güveni tahmin eder (C + D)
  4. decide()         → "oos" ya da güven < eşik  →  "human agent"
  5. Niyet small_talk → kural tabanlı cevap (ELIZA), değilse niyet → ekip
Diğer modeller aynı mesaj için yan yana gösterilir. Kararı sadece DECISION_MODEL verir.
"""
import time

from src.eliza import SMALL_TALK_INTENTS, eliza_reply, reply_for_intent
from src.evaluate import CUSTOM_INTENTS, OOS, decide
from src.normalize import normalize

DECISION_MODEL = "student_int8"   # R5: uygulamada kararı quantize Transformer verir
HUMAN_AGENT = "human agent"
RULE_REPLY = "rule-based reply (ELIZA)"

# Beyaz eşya niyetleri (bizim veri setimiz) → ekip
APPLIANCE_TEAMS = {
    "dishwasher_issue": "Technical Service – Dishwashers",
    "washing_machine_issue": "Technical Service – Washing Machines",
    "refrigerator_issue": "Technical Service – Refrigerators",
    "warranty_inquiry": "Warranty & After-Sales",
    "technician_appointment": "Appointment Scheduling",
}

# CLINC150'nin 10 alanı (her birinde 15 niyet) → ekip. Doğrulama: python -m src.router
CLINC_DOMAINS = {
    "Banking": ["transfer", "transactions", "balance", "freeze_account", "pay_bill", "bill_balance", "bill_due",
                "interest_rate", "routing", "min_payment", "order_checks", "pin_change", "report_fraud",
                "account_blocked", "spending_history"],
    "Credit Cards": ["credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
                     "application_status", "card_declined", "international_fees", "apr", "redeem_rewards",
                     "credit_limit_change", "damaged_card", "replacement_card_duration", "improve_credit_score",
                     "expiration_date"],
    "Kitchen & Dining": ["recipe", "restaurant_reviews", "calories", "nutrition_info", "restaurant_suggestion",
                         "ingredients_list", "ingredient_substitution", "cook_time", "food_last", "meal_suggestion",
                         "restaurant_reservation", "confirm_reservation", "how_busy", "cancel_reservation",
                         "accept_reservations"],
    "Home": ["shopping_list", "shopping_list_update", "next_song", "play_music", "update_playlist", "todo_list",
             "todo_list_update", "calendar", "calendar_update", "what_song", "order", "order_status", "reminder",
             "reminder_update", "smart_home"],
    "Auto & Commute": ["traffic", "directions", "gas", "gas_type", "distance", "current_location", "mpg",
                       "oil_change_when", "oil_change_how", "jump_start", "uber", "schedule_maintenance",
                       "last_maintenance", "tire_pressure", "tire_change"],
    "Travel": ["book_flight", "book_hotel", "car_rental", "travel_suggestion", "travel_alert",
               "travel_notification", "carry_on", "timezone", "vaccines", "translate", "flight_status",
               "international_visa", "lost_luggage", "plug_type", "exchange_rate"],
    "Utility": ["time", "alarm", "share_location", "find_phone", "weather", "text", "spelling", "make_call",
                "timer", "date", "calculator", "measurement_conversion", "flip_coin", "roll_dice", "definition"],
    "Work & HR": ["direct_deposit", "pto_request", "taxes", "payday", "w2", "pto_balance", "pto_request_status",
                  "next_holiday", "insurance", "insurance_change", "schedule_meeting", "pto_used",
                  "meeting_schedule", "rollover_401k", "income"],
    "Small Talk": sorted(SMALL_TALK_INTENTS),
    "Assistant Settings": ["change_ai_name", "change_user_name", "cancel", "user_name", "reset_settings",
                           "whisper_mode", "repeat", "no", "yes", "maybe", "change_language", "change_accent",
                           "change_volume", "change_speed", "sync_device"],
}
TEAM_OF = {**{i: f"{d} Team" for d, intents in CLINC_DOMAINS.items() for i in intents}, **APPLIANCE_TEAMS}


def team_for(decision: str) -> str:
    """Karar verilmiş niyet ("oos" olabilir) → ekip / yanıt türü."""
    if decision == OOS:
        return HUMAN_AGENT
    if decision in SMALL_TALK_INTENTS:
        return RULE_REPLY
    return TEAM_OF[decision]


def route(text: str, predictors: dict) -> dict:
    """predictors: {ad: predictor}. predictor.thresholds ve predictor.predict([metin]) gerekir."""
    norm = normalize(text)
    eliza = eliza_reply(norm.text)
    rows = {}
    for name, pred in predictors.items():
        t0 = time.perf_counter()
        intent, conf = pred.predict([norm.text])
        ms = (time.perf_counter() - t0) * 1000
        decision = decide(intent, conf, pred.thresholds)[0]
        rows[name] = {"intent": intent[0], "confidence": float(conf[0]), "decision": decision,
                      "team": team_for(decision), "ms": ms}

    main = rows.get(DECISION_MODEL)
    if eliza is not None:                    # 2. adım: kural tabanlı cevap, ekip yok
        team, reply, reason = RULE_REPLY, eliza, "ELIZA rule matched the whole message"
    elif main["decision"] == OOS:            # 4. adım
        team, reply = HUMAN_AGENT, None
        th = predictors[DECISION_MODEL].thresholds
        thr = th["custom"] if main["intent"] in CUSTOM_INTENTS else th["clinc"]
        reason = ("model predicted out-of-scope" if main["intent"] == OOS
                  else f"low confidence ({main['confidence']:.3f} < threshold {thr:.2f})")
    elif main["decision"] in SMALL_TALK_INTENTS:   # 5. adım
        team, reply, reason = RULE_REPLY, reply_for_intent(main["decision"]), "small-talk intent"
    else:
        team, reply, reason = TEAM_OF[main["decision"]], None, "confident in-scope prediction"
    return {"input": text, "normalized": norm.text, "model_codes": norm.model_codes,
            "corrections": norm.corrections, "eliza": eliza is not None, "team": team, "reply": reply,
            "reason": reason, "decision_model": DECISION_MODEL, "models": rows}


if __name__ == "__main__":
    # Ekip eşlemesinin doğrulaması: her CLINC alanında 15 niyet var mı, veri setindeki 156 niyetin hepsi eşlenmiş mi?
    import pandas as pd

    from src.evaluate import PROCESSED
    intents = set(pd.read_csv(PROCESSED / "train.csv")["intent"]) - {OOS}
    sizes = {d: len(v) for d, v in CLINC_DOMAINS.items()}
    print("Alan başına niyet sayısı:", sizes)
    print("Eşlenmemiş niyetler:", sorted(intents - set(TEAM_OF)))
    print("Veride olmayan eşlemeler:", sorted(set(TEAM_OF) - intents))
    assert all(n == 15 for n in sizes.values()) and intents == set(TEAM_OF), "Ekip eşlemesi hatalı!"
    print(f"Tamam: {len(TEAM_OF)} niyet → {len(set(TEAM_OF.values()))} ekip")
