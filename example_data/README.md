# Example Data

This folder contains small synthetic example files showing the expected Phase 1 and Phase 2 data structures for the SmartADAPT-Net pipeline.

These files are not real participant data and should not be used to reproduce the manuscript results. They are provided only to demonstrate folder organisation, column names, and script input requirements.

## Contents

```text
example_data/
├── phase1_example/
│   ├── Brushing Teeth/
│   ├── Mopping_Hoovering/
│   ├── Shelving Items/
│   ├── Walking/
│   ├── Washing Dishes/
│   └── Washing Face/
└── phase2_example/
    ├── P201/
    │   ├── manifest.csv
    │   ├── P201_D01_S001.csv
    │   └── ...
    └── P202/
        ├── manifest.csv
        └── ...
```

## Phase 1 example

The Phase 1 example data contain six synthetic participants, `P001` to `P006`, with one CSV file per ADL per participant. Each file contains 1,200 rows, corresponding to 24 seconds at 50 Hz. This provides three non-overlapping 8-second windows per file.

## Phase 2 example

The Phase 2 example data contain two synthetic participants, `P201` and `P202`, with 12 sessions per participant. Each session file contains 1,500 rows, corresponding to 30 seconds at 50 Hz: 3 seconds pre-stabilisation, 24 seconds analysis, and 3 seconds post-stabilisation.

Each participant folder contains a `manifest.csv` file with relative session file paths.

## Privacy note

No real participant data, real timestamps, or private file paths are included.
