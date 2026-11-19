"""Ek değerlendirme: Türkçe kapsam dışı mesajlar ve desteklenmeyen cihazlar ("zor negatifler").

Çalıştırma:  python -m src.eval_extra_oos
Veri:        data/custom/extra_oos_tr_en.csv (40 cümle)
               general      : 20 genel Türkçe soru (CLINC'in kapsadığı hava durumu, tarif, seyahat gibi
                              konulardan bilerek kaçınıldı)
               other_device : 20 şikâyet (15 TR + 5 EN), markanın desteklemediği cihazlar
                              (televizyon, süpürge, klima, kahve makinesi...)
Çıktı:       results/extra_oos.json

NEDEN: Ana test setindeki 1.000 kapsam dışı cümlenin hepsi İngilizce. Türkçe ilgisiz mesajlar hiç
ölçülmüyordu (doğruluk denetiminin bulduğu sınırlılık). Bu set ana sonuçlar alındıktan SONRA,
sadece ek değerlendirme için yazıldı. Hiçbir eğitimde, eşik seçiminde ya da karar kuralında
kullanılmaz. Ana sonuç tablosu değişmez.
Metrik: human agent'a gönderilme oranı (uygulamadaki iki eşikli karar kuralı + ELIZA ile aynı akış).
"""
import json
from collections import Counter

import numpy as np
import pandas as pd

from src.eliza import eliza_reply
from src.evaluate import RESULTS, ROOT, decide
from src.normalize import normalize_text
from src.predictors import MODEL_DIRS, load_predictor
from src.router import HUMAN_AGENT, RULE_REPLY, team_for

DATA = ROOT / "data" / "custom" / "extra_oos_tr_en.csv"


def main() -> None:
    df = pd.read_csv(DATA)
    texts = [normalize_text(t) for t in df["text"]]
    eliza_hit = np.array([eliza_reply(t) is not None for t in texts])
    out = {"n": len(df), "groups": df["group"].value_counts().to_dict(), "models": {}}
    for name, path in MODEL_DIRS.items():
        if not path.exists():
            continue
        pred = load_predictor(name, device="cpu")
        p, c = pred.predict(texts)
        team = np.array([team_for(d) for d in decide(p, c, pred.thresholds)], dtype=object)
        team = np.where(eliza_hit, RULE_REPLY, team)
        to_human = team == HUMAN_AGENT
        res = {"human_agent_rate": float(to_human.mean())}
        for g in ["general", "other_device"]:
            m = (df["group"] == g).values
            res[f"human_agent_rate_{g}"] = float(to_human[m].mean())
        res["wrongly_routed_to"] = dict(Counter(team[~to_human]).most_common(5))
        out["models"][name] = res
        print(f"{name:13s} human agent: {res['human_agent_rate']:.1%} "
              f"(general {res['human_agent_rate_general']:.1%}, other devices {res['human_agent_rate_other_device']:.1%})"
              f" | yanlış gidenler: {res['wrongly_routed_to']}", flush=True)
    (RESULTS / "extra_oos.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
