# Custom data (home-appliance support messages)

All sentences in this folder were written with AI assistance (Claude), reviewed and edited by me. They are not real
customer messages. CLINC150 (the main dataset) is downloaded by `python -m src.data` and is not stored here.

| File | Rows | Used for |
|---|---|---|
| `appliance_tr_en.csv` | 300 | train / validation / test (180 / 60 / 60) |
| `train_extra_v3.csv` | 170 | training only |
| `extra_oos_tr_en.csv` | 40 | extra evaluation only (not independent, see below) |
| `extra_oos_holdout_tr_en.csv` | 40 | extra evaluation only (independent hold-out) |

## `appliance_tr_en.csv`

5 intents × (40 Turkish + 20 English) = 300 messages. Columns: `text, intent, lang`.
`src/data.py` splits them 60 / 20 / 20, stratified by intent and language, with a fixed seed.

| intent | Topic | Team in the app |
|---|---|---|
| `dishwasher_issue` | dishwasher faults (not draining, not washing, error codes) | Technical Service – Dishwashers |
| `washing_machine_issue` | washing machine faults (not spinning, noise, door locked) | Technical Service – Washing Machines |
| `refrigerator_issue` | refrigerator faults (not cooling, ice build-up, leaking) | Technical Service – Refrigerators |
| `warranty_inquiry` | warranty period, coverage, documents | Warranty & After-Sales |
| `technician_appointment` | booking, changing or cancelling a service visit | Appointment Scheduling |

Variety on purpose: some messages contain a model code (`DW-6534`, `WM-1200`, `RF-450X`, extracted by regex),
some have misspelled product names (`bulasik makinsi`, `dishwaser`, `refridgerator`, corrected by minimum edit
distance), and some Turkish messages are written without Turkish characters (`bulasik makinem calismiyor`).

Limitation: some Turkish and English messages are translations or close paraphrases of each other. In a
manual check, 11 of the 60 test sentences had a strong and 5 a partial counterpart among the 180 training
sentences of this file.
The split was not changed after this was found, because that would have been a change made by looking at the
test set.

## `train_extra_v3.csv` (training split only)

170 messages added only to the training split; validation and test are unchanged (same SHA-256 fingerprints).
Columns: `text, intent, lang, group`.

- 60 Turkish general out-of-scope questions (`oos`, group `general`)
- 30 Turkish + 30 English complaints about devices the brand does not support (`oos`, group `other_device`)
- 50 English appliance messages, 10 per intent (group `appliance`)

Why: every Turkish training sentence used to be an appliance message, so the Transformers learned
"Turkish = appliance" and sent unrelated Turkish messages to the appliance teams (found with
`extra_oos_tr_en.csv`). The English appliance messages were added because English appliance accuracy was low in
the first models; they were written after those first results (and their error files) had been seen. They were
checked against the validation and test sets for copies.

## Extra evaluation sets (never used for training)

Both sets have 40 messages: 20 general questions + 20 unsupported-device complaints (35 Turkish, 5 English).
Columns: `text, group, lang`. `python -m src.eval_extra_oos <set>` reports how many of them the app sends to
human agent (`results/<set>.json`).

- `extra_oos_tr_en.csv`: the first set. The 170 extra training messages were written after seeing the result on
  this set and cover the same kinds of devices, so it is **not** an independent test for the final models.
- `extra_oos_holdout_tr_en.csv`: written after the final training data, with devices and topics that do not
  appear in training. Not used for any training, threshold or rule decision.

`results/*_v1.json` are the same evaluations run with the models trained before `train_extra_v3.csv` was added
(same code, `SWE015_MODELS_DIR` pointing to those models). Those older weights are not published.
