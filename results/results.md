| Model | Precision | Accuracy | Out-of-scope recall | In-scope acc. | Appliance acc. (TR / EN) | Model size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| tf-idf + Logistic Regression | float64 | 87.0% | 89.8% | 86.3% | 81.7% (87.5% / 70.0%) | 69.6 | 2.4 |
| fastText avg. + FFN (SGD) | float32 (FFN) + float32 (fastText) | 82.2% | 87.6% | 81.0% | 40.0% (42.5% / 35.0%) | 43.4 | 0.2 |
| mBERT (bert-base-multilingual-cased) | float32 | 88.3% | 90.4% | 87.9% | 90.0% (95.0% / 80.0%) | 681.7 | 31.7 |
| DistilmBERT, distilled from mBERT | float32 | 89.7% | 81.8% | 91.4% | 86.7% (90.0% / 80.0%) | 519.5 | 16.3 |
| DistilmBERT INT8 (ONNX Runtime) | INT8 (uint8 weights 99.95%, dynamic activations) | 87.2% | 90.6% | 86.5% | 86.7% (90.0% / 80.0%) | 132.4 | 5.7 |
| mBERT INT8 (ONNX Runtime) | INT8 (uint8 weights 99.93%, dynamic activations) | 88.5% | 91.1% | 88.0% | 90.0% (95.0% / 80.0%) | 173.3 | 11.0 |

Hardware: Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz, 15.9 GB RAM, Windows 11 (build 10.0.26300); CPU only, batch size 1, latency averaged over 200 runs (60 appliance TR/EN + 140 CLINC messages) after 10 warm-up runs, CPU threads: PyTorch 4, ONNX Runtime 4 (onnxruntime 1.32.0). Test set: 5560 sentences (sha256 7e8db122c096).

In-scope acc. without threshold (top-1 intent only): baseline 91.6%; ffn 87.6%; teacher 94.2%; student 93.5%; student_int8 93.1%; teacher_int8 94.0%

Spelling-correction ablation (inference only; models were trained on corrected text). Correction changes 6 of 60 appliance test sentences and 0 of 5500 CLINC test sentences. Appliance acc. with → without correction: baseline 81.7% → 75.0%; ffn 40.0% → 40.0%; teacher 90.0% → 85.0%; student 86.7% → 80.0%; student_int8 86.7% → 80.0%; teacher_int8 90.0% → 85.0%