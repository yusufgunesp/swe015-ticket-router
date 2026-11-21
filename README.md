# Customer Support Ticket Router (SWE015 · Use Case 2)

A home-appliance brand receives support messages in **Turkish and English**. This application normalizes each
message, predicts its intent with a confidence score and routes it to the right team. Low-confidence or
out-of-scope messages go to **human agent**, and greetings or small talk get a rule-based (ELIZA-style) reply.
The decision is made by a **distilled and INT8-quantized multilingual Transformer** running on CPU. The
predictions of all models in the ladder are shown side by side.

## Run it (single command)

Requirements: Python 3.11, about 4 GB free disk space, internet access for the first run.

Windows:

```bat
git clone https://github.com/yusufgunesp/swe015-ticket-router.git
cd swe015-ticket-router
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

Linux / macOS:

```bash
git clone https://github.com/yusufgunesp/swe015-ticket-router.git
cd swe015-ticket-router
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

(Activating the environment first and running `pip install -r requirements.txt` and `python app.py` works too.
If several Python versions are installed on Windows, create the environment with `py -3.11 -m venv .venv`.)

Then open http://127.0.0.1:7860.

**Installing the requirements takes several minutes** (PyTorch is large).
**First start takes about 5–10 minutes:** `app.py` downloads the trained model weights (~1.6 GB) from the Hugging
Face Hub ([`yusufinfo/swe015-ticket-router-models`](https://huggingface.co/yusufinfo/swe015-ticket-router-models),
public, no login needed) into `models/` and loads six models. Wait until the console prints
`Running on local URL`. A warning about `HF_TOKEN` / unauthenticated requests is harmless. Later starts skip
the download. Model weights are not stored in this repository.

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

Decision rule (same for every model, in evaluation and in the app): if the model predicts out-of-scope, or its
confidence is below a threshold, the message goes to **human agent**. There are two thresholds, one for the
CLINC150 intents and one for the appliance intents, both chosen on the validation set (minimum 0.3).

Architecture diagram: [`docs/architecture.svg`](docs/architecture.svg).

## Results

All models are evaluated on the same held-out test set (5,560 sentences: 4,500 in-scope CLINC150, 1,000
out-of-scope, 60 appliance messages). Table from [`results/results.md`](results/results.md), produced by
`src/benchmark.py`:

| Model | Precision | Accuracy | Out-of-scope recall | In-scope acc. | Appliance acc. (TR / EN) | Model size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| tf-idf + Logistic Regression | float64 | 87.0% | 89.8% | 86.3% | 81.7% (87.5% / 70.0%) | 69.6 | 2.4 |
| fastText avg. + FFN (SGD) | float32 (FFN) + float32 (fastText) | 82.2% | 87.6% | 81.0% | 40.0% (42.5% / 35.0%) | 43.4 | 0.2 |
| mBERT (bert-base-multilingual-cased) | float32 | 88.3% | 90.4% | 87.9% | 90.0% (95.0% / 80.0%) | 681.7 | 31.7 |
| DistilmBERT, distilled from mBERT | float32 | 89.7% | 81.8% | 91.4% | 86.7% (90.0% / 80.0%) | 519.5 | 16.3 |
| DistilmBERT INT8 (ONNX Runtime) | INT8 (uint8 weights 99.95%, dynamic activations) | 87.2% | 90.6% | 86.5% | 86.7% (90.0% / 80.0%) | 132.4 | 5.7 |
| mBERT INT8 (ONNX Runtime) | INT8 (uint8 weights 99.93%, dynamic activations) | 88.5% | 91.1% | 88.0% | 90.0% (95.0% / 80.0%) | 173.3 | 11.0 |

- **Accuracy**: share of all test sentences where the final decision (intent, or out-of-scope after the
  confidence thresholds) matches the label. **Out-of-scope recall**: share of out-of-scope sentences sent to
  human agent. **In-scope acc.**: the 4,560 in-scope sentences (4,500 CLINC150 + 60 appliance); a sentence sent
  to human agent counts as wrong. **Appliance acc.**: the 60 appliance test sentences (95% Wilson interval in
  `results/results.csv`).
- **Latency**: one message at a time (batch 1), text normalization included, 10 warm-up runs then the average
  of 200 runs. Hardware: Intel Core i5-10300H CPU, 15.9 GB RAM, Windows 11, CPU only, 4 threads
  ([`results/hardware.json`](results/hardware.json)).
- **Model size**: all files needed for inference (weights + tokenizer / vectorizer; small JSON config files
  excluded), 1 MB = 2^20 bytes.
- Other result files: [`results/app_routing.json`](results/app_routing.json) (team-level routing of the whole
  app), [`results/extra_oos_holdout.json`](results/extra_oos_holdout.json) (Turkish out-of-scope hold-out),
  [`results/quantization_experiments.csv`](results/quantization_experiments.csv) (quantization methods
  compared on the validation set), `results/errors_<model>.csv` (every wrong test prediction).

## Data

- **CLINC150** (`clinc/clinc_oos`, "plus" configuration, official train/validation/test splits): 150 intents
  plus an out-of-scope class, English. Larson et al., 2019, *An Evaluation Dataset for Intent Classification
  and Out-of-Scope Prediction*, EMNLP-IJCNLP.
- **300 home-appliance support messages** (`data/custom/appliance_tr_en.csv`): 5 intents × (40 Turkish + 20
  English), split 180 / 60 / 60, stratified by intent and language.
- **170 extra training-only messages** (`data/custom/train_extra_v3.csv`): 90 Turkish and 30 English
  out-of-scope messages (general questions, devices the brand does not support) and 50 English appliance
  messages. Validation and test splits are unchanged.
- **2 × 40 extra evaluation messages** (`data/custom/extra_oos_tr_en.csv`, `data/custom/extra_oos_holdout_tr_en.csv`):
  Turkish out-of-scope questions and unsupported-device complaints, never used for training. Only the hold-out
  set is independent of the extra training data.
- The custom messages were written with AI assistance, reviewed and edited by me; details and limitations in
  [`data/custom/README.md`](data/custom/README.md).
- Pretrained models from the Hugging Face Hub: `bert-base-multilingual-cased` (Devlin et al., 2019, *BERT*,
  NAACL) and `distilbert-base-multilingual-cased` (Sanh et al., 2019, *DistilBERT*, arXiv:1910.01108).
  fastText vectors are trained on our own training split with gensim (Bojanowski et al., 2017, *Enriching Word
  Vectors with Subword Information*, TACL). Knowledge distillation follows Hinton et al., 2015 (arXiv:1503.02531).

## Reproduce everything (optional; models 3–4 need an NVIDIA GPU with CUDA)

```bash
python -m src.data               # download CLINC150, build and lock the splits (SHA-256)
python -m src.train_baseline     # model 1
python -m src.train_ffn          # model 2
python -m src.train_teacher      # model 3 (GPU)
python -m src.distill_student    # model 4 (GPU)
python -m src.quantize           # model 5 (+ INT8 teacher)
python -m src.calibrate          # choose thresholds on validation, evaluate on test
python -m src.benchmark          # results table: metrics, size (MB), latency (ms)
python -m src.eval_app           # end-to-end routing accuracy (ELIZA + model + thresholds → team)
python -m src.eval_extra_oos extra_oos_holdout   # Turkish out-of-scope hold-out evaluation
python -m src.quant_experiments  # quantization method comparison (validation set only)
```

Training scripts only read the train and validation splits. The test set is read only by `src/calibrate.py`,
`src/benchmark.py` and `src/eval_app.py`.

## Project structure

```
app.py                     Gradio interface (single run command)
src/normalize.py           regex normalization, model-code extraction, edit-distance correction
src/eliza.py               ELIZA-style small-talk rules
src/router.py              message → normalize → ELIZA → INT8 model → thresholds → team
src/evaluate.py            shared metrics and decision rule (two confidence thresholds)
src/predictors.py          loads every model with the same predict() interface
src/hf_model.py            shared Transformer training (fine-tuning, distillation) and prediction code
src/download.py            downloads the trained weights from the Hugging Face Hub
src/data.py                builds the train / validation / test splits
src/resources/known_words.txt   correct English words for the spelling corrector (from the training split)
src/train_baseline.py, src/train_ffn.py, src/train_teacher.py, src/distill_student.py, src/quantize.py
                           model ladder
src/calibrate.py, src/benchmark.py, src/eval_app.py, src/eval_extra_oos.py, src/quant_experiments.py
                           evaluation
data/custom/               our Turkish / English messages
docs/architecture.svg      architecture diagram
results/                   metrics, results table, error examples
```

## AI usage

AI coding assistant: Claude (Claude Code). It was used for:
- writing and debugging the code (data pipeline, models, evaluation, app);
- drafting the custom messages (300 appliance messages, 170 extra training messages, 80 extra evaluation
  messages), which I reviewed and edited;
- review agents that checked the requirements, test-set leakage and the README setup;
- drafting the README, the architecture diagram and the slides.

Every number in the results was produced by running the code in this repository.
