"""SWE015 · Use Case 2 – Customer Support Ticket Router (Gradio arayüzü).

Çalıştırma:  python app.py      → http://127.0.0.1:7860

Akış (src/router.py): mesaj → normalize (regex, model kodu, edit distance) → ELIZA → karar modeli
(DistilmBERT INT8, ONNX Runtime, CPU) → iki eşikli karar → ekip / human agent / kural cevabı.
Diğer tüm modellerin tahmini yan yana tabloda gösterilir. Kararı sadece INT8 model verir (R5).
"""
import html
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr  # noqa: E402

from src.download import ensure_models  # noqa: E402
from src.predictors import MODEL_DIRS, load_predictor  # noqa: E402
from src.router import DECISION_MODEL, HUMAN_AGENT, RULE_REPLY, route  # noqa: E402

# (anahtar, tablodaki kısa ad). Sıra = model merdiveni
MODELS = [
    ("student_int8", "★ DistilmBERT INT8 (decides)"),
    ("baseline", "tf-idf + LR"),
    ("ffn", "fastText + FFN"),
    ("teacher", "mBERT"),
    ("student", "DistilmBERT"),
    ("teacher_int8", "mBERT INT8"),
]

ensure_models()
PREDICTORS = {k: load_predictor(k, device="cpu") for k, _ in MODELS if MODEL_DIRS[k].exists()}
assert DECISION_MODEL in PREDICTORS, "Decision model (student_int8) not found in models/"
for _ in range(3):  # ısınma: ilk gerçek mesajın gecikmesi, ilk çalıştırma maliyetini içermesin
    route("warm-up message for the models", PREDICTORS)

EXAMPLES = [
    "bulaşık makinem DW-6534 su boşaltmıyor",
    "camasir makinam sikarken cok ses yapiyor",
    "my fridge stopped cooling, model RF-450X",
    "garanti belgemi kaybettim, ücretsiz tamir olur mu?",
    "what is the capital of mongolia",
    "merhaba",
]

CSS = """
.wrap {max-width: 860px; margin: 0 auto;}
.card {border-radius: 14px; padding: 22px 24px; margin-top: 4px;}
.card .label {font-size: 13px; letter-spacing: .06em; text-transform: uppercase; opacity: .7;}
.card .team {font-size: 28px; font-weight: 700; margin: 4px 0 10px;}
.card .sub {font-size: 15px; opacity: .85;}
.card.team-ok {background: rgba(16,185,129,.12); border: 1px solid rgba(16,185,129,.45);}
.card.human {background: rgba(245,158,11,.13); border: 1px solid rgba(245,158,11,.5);}
.card.eliza {background: rgba(99,102,241,.12); border: 1px solid rgba(99,102,241,.45);}
.bar {height: 8px; border-radius: 4px; background: rgba(127,127,127,.25); margin: 8px 0 4px; overflow: hidden;}
.bar > div {height: 100%; border-radius: 4px; background: currentColor; opacity: .7;}
.reply {margin-top: 10px; padding: 12px 14px; border-radius: 10px; background: rgba(127,127,127,.12);
        font-size: 16px;}
.muted {font-size: 13px; opacity: .65;}
.wrap table, .wrap table * {font-family: inherit !important;}
"""


def result_card(r: dict) -> str:
    main = r["models"][DECISION_MODEL]
    conf = main["confidence"]
    bar = (f'<div class="bar"><div style="width:{conf * 100:.0f}%"></div></div>'
           f'<div class="muted">confidence {conf:.2f} · predicted intent <code>{html.escape(main["intent"])}</code>'
           f' · {main["ms"]:.0f} ms</div>')
    if r["team"] == RULE_REPLY:
        return (f'<div class="card eliza"><div class="label">Automatic reply</div>'
                f'<div class="team">💬 Small talk</div>'
                f'<div class="reply">{html.escape(r["reply"])}</div></div>')
    if r["team"] == HUMAN_AGENT:
        why = ("The message is outside the topics this system handles."
               if main["intent"] == "oos" else "The model is not confident enough to choose a team.")
        return (f'<div class="card human"><div class="label">Routed to</div>'
                f'<div class="team">🙋 Human agent</div><div class="sub">{why}</div>{bar}</div>')
    return (f'<div class="card team-ok"><div class="label">Routed to</div>'
            f'<div class="team">✅ {html.escape(r["team"])}</div>{bar}</div>')


def details(r: dict) -> str:
    rows = [f"**Normalized text:** `{r['normalized']}`"]
    rows.append("**Model code:** " + (", ".join(r["model_codes"]) if r["model_codes"] else "—"))
    rows.append("**Spelling corrections:** " + (", ".join(f"{a} → {b}" for a, b in r["corrections"])
                                                 if r["corrections"] else "—"))
    rows.append(f"**Decision reason:** {r['reason']}")
    return "  \n".join(rows)


def handle(text: str):
    if not text or not text.strip():
        return "", "", []
    r = route(text, PREDICTORS)
    table = []
    for key, label in MODELS:
        if key in r["models"]:
            m = r["models"][key]
            # ELIZA her modelden ÖNCE çalışır: eşleştiyse her model için sonuç kural cevabıdır
            # (eval_app.py da aynı şekilde sayar). Modelin kendi tahmini yine gösterilir.
            routed = RULE_REPLY if r["eliza"] else ("human agent" if m["team"] == HUMAN_AGENT else m["team"])
            table.append([label, m["intent"], round(m["confidence"], 2), routed, round(m["ms"], 1)])
    return result_card(r), details(r), table


with gr.Blocks(title="Support Ticket Router") as demo:
    with gr.Column(elem_classes="wrap"):
        gr.Markdown("# Support Ticket Router\nType a customer message in Turkish or English. "
                    "It is routed to the right team, or to a human agent.")
        inp = gr.Textbox(show_label=False, lines=1, placeholder="e.g. bulaşık makinem su boşaltmıyor, model DW-6534")
        with gr.Row():
            btn = gr.Button("Route message", variant="primary")
            gr.ClearButton(inp, value="Clear")
        gr.Examples(EXAMPLES, inputs=inp, label="Try an example")
        card = gr.HTML()
        table = gr.Dataframe(headers=["Model", "Predicted intent", "Confidence", "Routed to", "ms"],
                             label="All models side by side (★ makes the decision)",
                             interactive=False, wrap=True,
                             column_widths=["27%", "22%", "11%", "32%", "8%"])
        with gr.Accordion("How the message was processed", open=False):
            info = gr.Markdown()
        gr.Markdown('<div class="muted">Decision model: DistilmBERT distilled from mBERT, INT8 (ONNX Runtime), '
                    'running on CPU. SWE015 · Use Case 2.</div>')
    btn.click(handle, inputs=inp, outputs=[card, info, table])
    inp.submit(handle, inputs=inp, outputs=[card, info, table])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
