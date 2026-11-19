"""SWE015 · Use Case 2 – Customer Support Ticket Router (Gradio arayüzü).

Çalıştırma:  python app.py      → http://127.0.0.1:7860

Akış (src/router.py): mesaj → normalize (regex, model kodu, edit distance) → ELIZA → karar modeli
(DistilmBERT INT8, ONNX Runtime, CPU) → iki eşikli karar → ekip / human agent / kural cevabı.
Diğer tüm modellerin tahmini yan yana tabloda gösterilir. Kararı sadece INT8 model verir (R5).
"""
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr  # noqa: E402

from src.download import ensure_models  # noqa: E402
from src.predictors import MODEL_DIRS, load_predictor  # noqa: E402
from src.router import DECISION_MODEL, HUMAN_AGENT, RULE_REPLY, route  # noqa: E402

# (anahtar, arayüzdeki ad, hassasiyet). Sıra = model merdiveni
MODELS = [
    ("baseline", "tf-idf + Logistic Regression (baseline)", "float64"),
    ("ffn", "fastText avg. + FFN, SGD (neural network)", "float32"),
    ("teacher", "mBERT (Transformer)", "FP32"),
    ("student", "DistilmBERT, distilled", "FP32"),
    ("student_int8", "DistilmBERT INT8, ONNX Runtime (quantized, DECISION)", "INT8"),
    ("teacher_int8", "mBERT INT8, ONNX Runtime", "INT8"),
]

ensure_models()
PREDICTORS = {k: load_predictor(k, device="cpu") for k, _, _ in MODELS if MODEL_DIRS[k].exists()}
assert DECISION_MODEL in PREDICTORS, "Karar modeli (student_int8) bulunamadı"
print("Yüklenen modeller:", list(PREDICTORS))
for _ in range(3):  # ısınma: ilk gerçek mesajın gecikmesi, ilk çalıştırma maliyetini içermesin
    route("warm-up message for the models", PREDICTORS)

EXAMPLES = [
    "Merhaba, bulaşık makinem DW-6534 su boşaltmıyor ve E24 hatası veriyor",
    "camasir makinam sıkma yaparken cok gurultu yapiyor",
    "my refridgerator stopped cooling yesterday, model RF-450X",
    "garanti belgemi kaybettim, yine de ücretsiz tamir olur mu?",
    "can I move my technician visit to friday afternoon?",
    "how do I transfer money to my savings account",
    "what is the capital of mongolia",
    "selam nasılsın",
]


def handle(text: str):
    if not text or not text.strip():
        return "Please type a message.", "", []
    r = route(text, PREDICTORS)
    main = r["models"][DECISION_MODEL]

    if r["team"] == HUMAN_AGENT:
        head = f"## ➜ {HUMAN_AGENT}\n**Reason:** {r['reason']}"
    elif r["team"] == RULE_REPLY:
        head = f"## ➜ {RULE_REPLY}\n> {r['reply']}"
    else:
        head = f"## ➜ {r['team']}"
    model_line = (f"`{main['intent']}` · **confidence:** {main['confidence']:.3f}"
                  f" · **decision model:** DistilmBERT INT8 (ONNX Runtime, CPU) · {main['ms']:.1f} ms")
    if r["eliza"]:
        head += f"\n\n**Model prediction (not used, ELIZA answered):** {model_line}"
    else:
        head += f"\n\n**Predicted intent:** {model_line}"

    details = [f"**Normalized text:** `{r['normalized']}`"]
    if r["model_codes"]:
        details.append(f"**Extracted model code(s):** {', '.join(r['model_codes'])}")
    if r["corrections"]:
        details.append("**Spelling corrections (edit distance):** "
                       + ", ".join(f"`{a}` → `{b}`" for a, b in r["corrections"]))
    if r["eliza"]:
        details.append("**ELIZA:** the whole message matched a small-talk rule, so no team is assigned.")

    table = []
    for key, label, precision in MODELS:
        if key not in r["models"]:
            continue
        m = r["models"][key]
        table.append([label, precision, m["intent"], round(m["confidence"], 3),
                      HUMAN_AGENT if m["team"] == HUMAN_AGENT else m["team"], round(m["ms"], 1)])
    return head, "\n\n".join(details), table


with gr.Blocks() as demo:
    gr.Markdown("# Customer Support Ticket Router\n"
                "SWE015 · Use Case 2 · Turkish / English home-appliance support messages are routed to the "
                "right team. Low-confidence or out-of-scope messages go to **human agent**; greetings and "
                "small talk get a rule-based reply.")
    with gr.Row():
        inp = gr.Textbox(label="Customer message", lines=3, scale=4,
                         placeholder="e.g. bulaşık makinem su boşaltmıyor, model DW-6534")
        btn = gr.Button("Route", variant="primary", scale=1)
    head = gr.Markdown()
    details = gr.Markdown()
    table = gr.Dataframe(headers=["Model", "Precision", "Predicted intent", "Confidence",
                                  "Routed to (after threshold)", "Latency (ms)"],
                         label="All models side by side (only the INT8 model makes the decision)",
                         interactive=False, wrap=True)
    gr.Examples(EXAMPLES, inputs=inp)
    btn.click(handle, inputs=inp, outputs=[head, details, table])
    inp.submit(handle, inputs=inp, outputs=[head, details, table])

if __name__ == "__main__":
    demo.launch()
