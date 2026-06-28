# Scripts

This folder contains the three main Python scripts used to train, replay, and analyse the SmartADAPT-Net framework.

## Code 1: Phase 1 training and Core ML export

`code01_phase1_train_export_coreml.py`

Trains the participant-independent CNN–BiLSTM population model using labelled Phase 1 simulated-living data. The script also estimates the Phase 1 normalisation statistics, calibrates the motion-complexity distribution, trains the complexity-aware correction head, saves the replay-ready Keras model, and optionally exports the model to Core ML for watchOS deployment.

Example:

```bash
python scripts/code01_phase1_train_export_coreml.py \
  --data-root "/path/to/Phase 1 Data" \
  --outdir "outputs/code01_phase1" \
  --strict-columns \
  --export-coreml
```

## Code 2: Phase 2 chronological deployment replay

`code02_phase2_chronological_replay.py`

Uses the replay-ready model and metadata from Code 1 to reconstruct Phase 2 deployment decisions in chronological order. For each participant, sessions are replayed using only prototype memory accumulated from previous sessions. The script outputs canonical replay files containing Base, Global/CAM, Prototype/CDP, and Final predictions.

Example:

```bash
python scripts/code02_phase2_chronological_replay.py \
  --phase2-root "/path/to/Phase 2 Data_F" \
  --code1-outdir "outputs/code01_phase1" \
  --outdir "outputs/code02_replay"
```

## Code 3: Manuscript and supplementary outputs

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

## Suggested execution order

```text
Code 1 → Code 2 → Code 3
```

Code 1 trains and exports the model. Code 2 reconstructs chronological free-living deployment decisions. Code 3 generates the final manuscript and supplementary results.

## Data privacy note

Raw participant data are not included in this repository. Users should provide their own local Phase 1 and Phase 2 data folders following the structure described in `docs/data_dictionary.md`.
