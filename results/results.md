| Model | Precision | Accuracy | Out-of-scope recall | In-scope acc. | Appliance acc. (TR / EN) | Model size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| tf-idf + Logistic Regression | float64 | 87.4% | 88.8% | 87.1% | 81.7% (92.5% / 60.0%) | 67.4 | 2.0 |
| fastText avg. + FFN (SGD) | float32 (FFN) + float32 (fastText) | 82.5% | 89.2% | 81.0% | 41.7% (50.0% / 25.0%) | 43.2 | 0.2 |
| mBERT (bert-base-multilingual-cased) | float32 | 89.9% | 86.4% | 90.7% | 81.7% (92.5% / 60.0%) | 681.7 | 30.8 |
| DistilmBERT, distilled from mBERT | float32 | 89.1% | 85.1% | 90.0% | 80.0% (92.5% / 55.0%) | 519.5 | 16.1 |
| DistilmBERT INT8 (ONNX Runtime) | INT8 (uint8 weights 99.95%, dynamic activations) | 88.2% | 88.4% | 88.2% | 73.3% (87.5% / 45.0%) | 132.4 | 5.9 |
| mBERT INT8 (ONNX Runtime) | INT8 (uint8 weights 99.93%, dynamic activations) | 90.1% | 85.3% | 91.2% | 83.3% (95.0% / 60.0%) | 173.3 | 10.9 |

Hardware: Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz, 15.9 GB RAM, Windows 11 (build 10.0.26300); CPU only, batch size 1, latency averaged over 200 runs (60 appliance TR/EN + 140 CLINC messages) after 10 warm-up runs, threads: PyTorch 4, ONNX Runtime 1.32.0 4. Test set: 5560 sentences (sha256 7e8db122c096).

In-scope acc. without threshold (top-1 intent only): baseline 91.9%; ffn 88.6%; teacher 94.5%; student 93.6%; student_int8 93.1%; teacher_int8 94.5%

Spelling-correction ablation (inference only; models were trained on corrected text). Correction changes 6 of 60 appliance test sentences and 0 of 5500 CLINC test sentences. Appliance acc. with → without correction: baseline 81.7% → 75.0%; ffn 41.7% → 40.0%; teacher 81.7% → 73.3%; student 80.0% → 73.3%; student_int8 73.3% → 68.3%; teacher_int8 83.3% → 75.0%