# SmartADAPT-Net

Reference implementation for **SmartADAPT-Net**, an adaptive smartwatch-based activity recognition framework for activities of daily living (ADLs).

SmartADAPT-Net combines:

- a frozen participant-independent CNN-BiLSTM population model;
- complexity-aware modulation of population-level logits;
- confirmation-driven participant-specific prototype memory;
- evidence-adaptive fusion between population and prototype evidence;
- chronological replay analysis for free-living smartwatch deployment.

This repository contains the Python training/replay/analysis code, synthetic example data, documentation, configuration templates, and a watchOS reference implementation showing how the adaptive logic can be implemented on Apple Watch.

---

## Repository structure

```text
SmartADAPT-Net-Code/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── README.md
│   ├── code01_phase1_train_export_coreml.py
│   ├── code02_phase2_chronological_replay.py
│   └── code03_generate_phase2_manuscript_outputs.py
├── configs/
│   └── example_config.yaml
├── docs/
│   ├── data_dictionary.md
│   └── reproduction_steps.md
├── example_data/
│   ├── README.md
│   ├── phase1_example/
│   └── phase2_example/
└── watchos_reference/
    ├── README.md
    ├── MotionRecorder.swift
    ├── SmartADAPTInferenceEngine.swift
    ├── ComplexityScorer.swift
    ├── PrototypeMemory.swift
    ├── FusionEngine.swift
    └── SessionController.swift
```

---

## Main workflow

The repository is organised around three Python scripts.

### Code 1: Phase 1 training and Core ML export

`code01_phase1_train_export_coreml.py`

Trains the Phase 1 participant-independent CNN-BiLSTM population model, estimates sensor-normalisation statistics, calibrates the motion-complexity distribution, trains the complexity-aware correction head, saves replay-ready Keras models, and optionally exports the model to Core ML.

Example:

```bash
python scripts/code01_phase1_train_export_coreml.py \
  --data-root "/path/to/Phase 1 Data" \
  --outdir "outputs/code01_phase1" \
  --strict-columns \
  --export-coreml
```

### Code 2: Phase 2 chronological replay

`code02_phase2_chronological_replay.py`

Uses the replay-ready model and metadata from Code 1 to reconstruct Phase 2 deployment decisions in chronological order. Prototype memory is updated only after the current session is evaluated, preserving deployment causality.

Example:

```bash
python scripts/code02_phase2_chronological_replay.py \
  --phase2-root "/path/to/Phase 2 Data_F" \
  --code1-outdir "outputs/code01_phase1" \
  --outdir "outputs/code02_replay"
```

### Code 3: Manuscript and supplementary outputs

`code03_generate_phase2_manuscript_outputs.py`

Generates manuscript-ready and supplementary tables and high-resolution figures from the canonical Phase 2 replay output produced by Code 2.

Example:

```bash
python scripts/code03_generate_phase2_manuscript_outputs.py \
  --input "outputs/code02_replay/canonical_code1_outputs/phase2_replay_window_level_CODE1_canonical_compatible.csv" \
  --outdir "outputs/code03_manuscript_outputs" \
  --strict \
  --dpi 600 \
  --n-boot 2000
```

Recommended order:

```text
Code 1 → Code 2 → Code 3
```

---

## Installation

Create a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

A separate environment is recommended, especially if Core ML export is required.

---

## Data

Raw participant data are **not included** in this repository.

The expected data structures are documented in:

```text
docs/data_dictionary.md
```

Small synthetic example files are provided in:

```text
example_data/
```

The example files are only for testing data loading and folder structure. They are not real participant data and should not be used to reproduce the manuscript results.

---

## watchOS reference implementation

The `watchos_reference/` folder contains Swift reference files demonstrating the smartwatch-side adaptive logic:

```text
CoreMotion recording
Core ML inference
motion-complexity scoring
complexity-aware modulation
prototype memory
evidence-adaptive fusion
post-confirmation prototype update
```

This is a reference implementation of the deployment logic, not a full production Apple Watch application.

---

## Privacy and ethics

Do not commit raw participant data, identifiable metadata, private file paths, generated output folders, or trained models unless sharing is explicitly permitted by the study ethics and data-management plan.

This repository is intended to share:

- source code;
- documentation;
- configuration templates;
- synthetic example data;
- watchOS reference logic.

---

## Citation

A manuscript describing SmartADAPT-Net is currently in preparation. Citation details can be added here after publication or preprint release.

---

## License

Add your chosen licence here before making the repository public. For academic code, common options include MIT, Apache-2.0, or BSD-3-Clause. Check with your institution before selecting a licence.
