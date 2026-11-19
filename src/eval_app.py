"""Uygulama düzeyinde (uçtan uca) değerlendirme: router.py'deki akışın TEST setinde ekip bazında doğruluğu.

Çalıştırma:  python -m src.eval_app        (önce calibrate çalışmış olmalı)
Çıktı:       results/app_routing.json

Model metrikleri (benchmark.py) niyet düzeyindedir ve ELIZA ön filtresini içermez. Burada uygulamanın
gerçek davranışı ölçülür:
    doğru hedef = team_for(gerçek niyet)   ("oos" → human agent, small talk → kural cevabı, diğer → ekip)
    tahmin      = ELIZA mesajın tamamına uyduysa kural cevabı, uymadıysa model + iki eşik → team_for(karar)
Her model için aynı akış uygulanır. Uygulamada kararı DECISION_MODEL verir.
"""
import json

import numpy as np

from src.eliza import eliza_reply
from src.evaluate import RESULTS, decide, load_split, split_sha256
from src.predictors import MODEL_DIRS, load_predictor
from src.router import DECISION_MODEL, RULE_REPLY, team_for


def main() -> None:
    test = load_split("test")
    true_team = np.array([team_for(i) for i in test["intent"]], dtype=object)
    eliza_hit = np.array([eliza_reply(t) is not None for t in test["text_norm"]])
    out = {"test_sha256": split_sha256("test"), "decision_model": DECISION_MODEL,
           "eliza_matches": int(eliza_hit.sum()),
           "eliza_matches_true_small_talk": int((eliza_hit & (true_team == RULE_REPLY)).sum()),
           "models": {}}
    for name, path in MODEL_DIRS.items():
        if not path.exists():
            continue
        pred = load_predictor(name, device="cpu")
        p, c = pred.predict(test["text_norm"].tolist())
        model_team = np.array([team_for(d) for d in decide(p, c, pred.thresholds)], dtype=object)
        app_team = np.where(eliza_hit, RULE_REPLY, model_team)
        cust = (test["source"] == "custom").values
        out["models"][name] = {
            "routing_acc_with_eliza": float((app_team == true_team).mean()),
            "routing_acc_model_only": float((model_team == true_team).mean()),
            "routing_acc_appliance": float((app_team[cust] == true_team[cust]).mean()),
        }
        print(name, out["models"][name], flush=True)
    print(f"ELIZA {out['eliza_matches']} test mesajını yakaladı, "
          f"{out['eliza_matches_true_small_talk']} tanesinin gerçek niyeti small talk.")
    (RESULTS / "app_routing.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
