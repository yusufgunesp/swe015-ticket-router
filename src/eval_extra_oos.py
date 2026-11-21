"""Ek değerlendirme: Türkçe kapsam dışı mesajlar ve desteklenmeyen cihazlar ("zor negatifler").

Çalıştırma:  python -m src.eval_extra_oos [set_adı] [çıktı_eki]
Setler (data/custom/, her biri 40 cümle: 20 genel soru + 20 desteklenmeyen cihaz şikâyeti, 35 TR + 5 EN):
    extra_oos          : ilk ek set. 170 ek eğitim cümlesi bu setin sonucu görülerek yazıldı,
                         bu yüzden son modeller için bağımsız DEĞİL (sadece sorunu teşhis etmek için).
    extra_oos_holdout  : bağımsız set. Ek eğitim verisinden SONRA yazıldı, eğitimde geçmeyen cihaz ve
                         konular içerir. Hiçbir eğitimde, eşik seçiminde ya da kural ayarında kullanılmadı.
Çıktı:       results/<set_adı><çıktı_eki>.json

NEDEN: Ana test setindeki 1.000 kapsam dışı cümlenin hepsi İngilizce. Türkçe ilgisiz mesajlar
orada ölçülmüyor. Ana sonuç tablosu bu setlerden etkilenmez.
Metrik: human agent'a gönderilme oranı (uygulamadaki iki eşikli karar kuralı + ELIZA ile aynı akış).
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

from src.eliza import eliza_reply
from src.evaluate import RESULTS, ROOT, decide
from src.normalize import normalize_text
from src.predictors import MODEL_DIRS, MODELS, load_predictor
from src.router import HUMAN_AGENT, RULE_REPLY, team_for

SETS = {
    "extra_oos": ROOT / "data" / "custom" / "extra_oos_tr_en.csv",
    "extra_oos_holdout": ROOT / "data" / "custom" / "extra_oos_holdout_tr_en.csv",
}


def main() -> None:
    """Örnek: python -m src.eval_extra_oos extra_oos_holdout
    Ek eğitim verisi öncesi modellerle aynı ölçüm (results/*_v1.json):
           SWE015_MODELS_DIR=models_v1 python -m src.eval_extra_oos extra_oos_holdout _v1"""
    name = sys.argv[1] if len(sys.argv) > 1 else "extra_oos"
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    df = pd.read_csv(SETS[name])
    texts = [normalize_text(t) for t in df["text"]]
    eliza_hit = np.array([eliza_reply(t) is not None for t in texts])
    out = {"set": name, "models_dir": MODELS.name, "n": len(df),
           "groups": df["group"].value_counts().to_dict(), "models": {}}
    for model_name, path in MODEL_DIRS.items():
        if not path.exists():
            continue
        pred = load_predictor(model_name, device="cpu")
        p, c = pred.predict(texts)
        team = np.array([team_for(d) for d in decide(p, c, pred.thresholds)], dtype=object)
        team = np.where(eliza_hit, RULE_REPLY, team)
        to_human = team == HUMAN_AGENT
        res = {"human_agent_rate": float(to_human.mean())}
        for g in ["general", "other_device"]:
            m = (df["group"] == g).values
            res[f"human_agent_rate_{g}"] = float(to_human[m].mean())
        res["wrongly_routed_to"] = dict(Counter(team[~to_human]).most_common(5))
        out["models"][model_name] = res
        print(f"{model_name:13s} human agent: {res['human_agent_rate']:.1%} "
              f"(general {res['human_agent_rate_general']:.1%}, other devices {res['human_agent_rate_other_device']:.1%})"
              f" | yanlış gidenler: {res['wrongly_routed_to']}", flush=True)
    (RESULTS / f"{name}{suffix}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
