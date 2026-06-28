# Reproduction Steps

This document describes the recommended workflow for reproducing the SmartADAPT-Net training, replay, and manuscript-output pipeline.

The repository contains code only. Raw participant data are not included.

---

## 1. Prepare the repository

Clone or download the repository, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

A Python environment with TensorFlow, NumPy, pandas, scikit-learn, matplotlib, and Core ML Tools is recommended.

---

## 2. Prepare the Phase 1 data

Organise Phase 1 data as one folder per ADL, with one CSV file per participant:

```text
Phase 1 Data/
├── Brushing Teeth/
├── Mopping_Hoovering/
├── Shelving Items/
├── Walking/
├── Washing Dishes/
└── Washing Face/
```

Each CSV should contain the required CoreMotion columns listed in `docs/data_dictionary.md`.

---

## 3. Run Code 1: Phase 1 training and Core ML export

Code 1 trains the participant-independent CNN-BiLSTM model, estimates sensor normalisation statistics, calibrates the motion-complexity distribution, trains the complexity-aware correction head, saves replay-ready Keras models, and optionally exports the model to Core ML.

Example:

```bash
python scripts/code01_phase1_train_export_coreml.py \
  --data-root "/path/to/Phase 1 Data" \
  --outdir "outputs/code01_phase1" \
  --strict-columns \
  --export-coreml
```

Expected outputs include:

```text
outputs/code01_phase1/
├── models/
├── metadata/
├── tables/
├── figures/
├── arrays/
└── logs/
```

The most important outputs for later stages are:

```text
models/smartadapt_phase1_replay_interface.keras
metadata/phase1_training_metadata.json
metadata/code02_replay_contract.json
```

---

## 4. Prepare the Phase 2 data

Organise Phase 2 data as one folder per participant:

```text
Phase 2 Data_F/
├── P201/
│   ├── manifest.csv
│   ├── P201_D01_S001.csv
│   └── ...
├── P202/
└── ...
```

Each participant folder must contain a `manifest.csv` file describing the sessions.

---

## 5. Run Code 2: chronological deployment replay

Code 2 loads the replay-ready model from Code 1 and reconstructs Phase 2 deployment decisions in chronological participant order.

Example:

```bash
python scripts/code02_phase2_chronological_replay.py \
  --phase2-root "/path/to/Phase 2 Data_F" \
  --code1-outdir "outputs/code01_phase1" \
  --outdir "outputs/code02_replay"
```

Code 2 evaluates each session before updating prototype memory. This preserves deployment chronology and prevents future labels from influencing earlier predictions.

Expected outputs include:

```text
outputs/code02_replay/
├── canonical_code1_outputs/
│   ├── phase2_replay_session_level_CODE1_canonical.csv
│   └── phase2_replay_window_level_CODE1_canonical_compatible.csv
├── diagnostics/
└── metadata/
```

The main input for Code 3 is:

```text
outputs/code02_replay/canonical_code1_outputs/phase2_replay_window_level_CODE1_canonical_compatible.csv
```

---

## 6. Run Code 3: manuscript and supplementary outputs

Code 3 generates manuscript-ready and supplementary figures and tables from the canonical Phase 2 replay CSV.

Example:

```bash
python scripts/code03_generate_phase2_manuscript_outputs.py \
  --input "outputs/code02_replay/canonical_code1_outputs/phase2_replay_window_level_CODE1_canonical_compatible.csv" \
  --outdir "outputs/code03_manuscript_outputs" \
  --strict \
  --dpi 600 \
  --n-boot 2000
```

Expected outputs include:

```text
outputs/code03_manuscript_outputs/
├── figures_main/
├── figures_supplementary/
├── tables_main/
├── tables_supplementary/
└── manifest/
```

---

## 7. Optional: watchOS reference implementation

The `watchos_reference/` folder contains Swift reference files showing how the deployed logic can be implemented on Apple Watch.

The watchOS reference implementation demonstrates:

```text
CoreMotion recording
Core ML inference
motion-complexity scoring
complexity-aware modulation
prototype memory
evidence-adaptive fusion
post-confirmation prototype update
```

This folder is intended as an implementation reference, not a complete production app.

---

## 8. Recommended execution order

```text
Code 1 → Code 2 → Code 3
```

Code 1 trains and exports the model.  
Code 2 replays Phase 2 deployment chronologically.  
Code 3 generates the final manuscript and supplementary outputs.

---

## 9. Data privacy

Do not commit raw participant data, trained models containing sensitive information, or local private file paths to GitHub unless sharing is explicitly permitted by the study ethics and data-management plan.

The repository should include code, documentation, configuration templates, and optional synthetic/example data only.
