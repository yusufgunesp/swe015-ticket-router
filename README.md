# Customer Support Ticket Router (SWE015 · Use Case 2)

A home-appliance brand receives support messages in **Turkish and English**. This application normalizes each
message, predicts its intent with a confidence score and routes it to the right team. Low-confidence or
out-of-scope messages go to **human agent**, and greetings or small talk get a rule-based (ELIZA-style) reply.
The decision is made by a **distilled and INT8-quantized multilingual Transformer** running on CPU. The
predictions of all models in the ladder are shown side by side.

## Run it (single command)

Requirements: Python 3.11, about 3 GB free disk space, internet access for the first run.

```bash
git clone https://github.com/yusufgunesp/swe015-ticket-router.git
cd swe015-ticket-router
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860.

On the first run, `app.py` downloads the trained model weights (~1.6 GB) from the Hugging Face Hub
([`yusufinfo/swe015-ticket-router-models`](https://huggingface.co/yusufinfo/swe015-ticket-router-models))
into `models/`. Model weights are not stored in this repository.

## Model ladder

| # | Model | Area | Role |
|---|---|---|---|
| 1 | tf-idf (word 1-2 + char 2-5 grams) + Logistic Regression | B · Classical NLP | Classical baseline |
| 2 | fastText (trained on our training set) averaged embeddings → feedforward network, trained with `torch.optim.SGD` + backpropagation | A · Neural networks | Neural network |
| 3 | `bert-base-multilingual-cased` fine-tuned with a classification head | C · Transformers | Transformer (teacher) |
| 4 | `distilbert-base-multilingual-cased`, knowledge distillation from model 3 | D · Efficient models | Distilled student |
| 5 | Model 4 exported to ONNX and dynamically quantized to INT8 (ONNX Runtime) | D · Efficient models | **Quantized Transformer, used by the app** |
| + | Model 3 quantized the same way | D | Extra comparison row |

Classical NLP in the pipeline:
- regex normalization and model-code extraction (`DW-6534`, `WM-1200`, `RF-450X`)
- minimum edit distance correction of misspelled product names (`bulasik makinsi` → `bulaşık makinesi`)
- ELIZA-style replies for small talk

All models are evaluated on the same held-out test set (5,560 sentences). Results, model sizes and CPU latency
are in [`results/results.md`](results/results.md) (produced by `src/benchmark.py`; hardware in
[`results/hardware.json`](results/hardware.json)).

## Data

- **CLINC150** (`clinc/clinc_oos`, "plus" configuration, official train/validation/test splits): 150 intents
  plus an out-of-scope class, English. Larson et al., 2019, *An Evaluation Dataset for Intent Classification
  and Out-of-Scope Prediction*, EMNLP-IJCNLP.
- **300 home-appliance support messages** (`data/custom/appliance_tr_en.csv`): 5 intents × (40 Turkish + 20
  English). Written with AI assistance (see "AI usage") and reviewed. Split 180 / 60 / 60, stratified by intent
  and language. See [`data/custom/NASIL_YAZILIR.md`](data/custom/NASIL_YAZILIR.md).
- **170 extra training-only sentences** (`data/custom/train_extra_v3.csv`): 90 Turkish and 30 English
  out-of-scope messages (general questions, devices the brand does not support) and 50 English appliance
  messages. Added only to the training split; validation and test splits are unchanged.
- **40 extra evaluation sentences** (`data/custom/extra_oos_tr_en.csv`, `src/eval_extra_oos.py`): Turkish
  out-of-scope questions and unsupported-device complaints, used only for evaluation
  ([`results/extra_oos.json`](results/extra_oos.json)). The extra training sentences were written after seeing
  the first result on this set, so a second, **independent hold-out set** of 40 sentences with unseen devices
  and topics (`data/custom/extra_oos_holdout_tr_en.csv`) was written afterwards and never used for any tuning
  (`python -m src.eval_extra_oos extra_oos_holdout` → [`results/extra_oos_holdout.json`](results/extra_oos_holdout.json)).
- Pretrained models from the Hugging Face Hub: `bert-base-multilingual-cased`, `distilbert-base-multilingual-cased`.

## Reproduce everything (optional, GPU recommended for models 3–4)

```bash
python -m src.data               # download CLINC150, build and lock the splits
python -m src.train_baseline     # model 1
python -m src.train_ffn          # model 2
python -m src.train_teacher      # model 3 (GPU)
python -m src.distill_student    # model 4 (GPU)
python -m src.quantize           # model 5 (+ INT8 teacher)
python -m src.calibrate          # choose thresholds on validation, evaluate on test
python -m src.benchmark          # results table: metrics, size (MB), latency (ms)
python -m src.eval_app           # end-to-end routing accuracy (ELIZA + model + thresholds → team)
python -m src.eval_extra_oos     # extra Turkish / unsupported-device out-of-scope evaluation
python -m src.quant_experiments  # quantization method comparison (validation set only)
```

Training scripts only read the train and validation splits. The test set is read only by `src/calibrate.py`,
`src/benchmark.py` and `src/eval_app.py`.

## Project structure

```
app.py                  Gradio interface (single run command)
src/normalize.py        regex normalization, model-code extraction, edit-distance correction
src/eliza.py            ELIZA-style small-talk rules
src/router.py           message → normalize → ELIZA → INT8 model → thresholds → team
src/evaluate.py         shared metrics and decision rule (two confidence thresholds)
src/predictors.py       loads every model with the same predict() interface
src/train_*.py, src/distill_student.py, src/quantize.py   model ladder
src/calibrate.py, src/benchmark.py, src/eval_app.py      evaluation
results/                metrics, results table, error examples
```

## AI usage

AI coding assistant: Claude (Claude Code). It was used for writing and debugging the code, writing the 300
custom support messages (with the instructor's permission) and drafting the documentation. Every number in
the results was produced by running the code in this repository.
