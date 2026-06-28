# SmartADAPT-Net watchOS reference implementation

This folder provides a compact Swift/watchOS reference implementation of the deployment-time SmartADAPT-Net loop. It is not intended to be a complete production app UI. Instead, it shows the implementation-critical parts needed to reproduce the manuscript logic on a smartwatch:

1. collect CoreMotion samples;
2. construct a 30-second ADL session with a 3-second pre-stabilisation period, a 24-second analysis region, and three non-overlapping 8-second windows;
3. run the frozen Core ML model on each window;
4. compute calibrated motion complexity and complexity-aware logit modulation;
5. compute participant-specific prototype probabilities in the frozen embedding space;
6. fuse global and prototype evidence;
7. update prototype memory after participant confirmation/correction.

## Files

- `MotionRecorder.swift` records device-motion samples from Apple Watch using CoreMotion.
- `SmartADAPTInferenceEngine.swift` calls the Core ML model and produces Base, Global/CAM, and embedding outputs per window.
- `ComplexityScorer.swift` implements the calibrated complexity score and sigmoid gate.
- `PrototypeMemory.swift` implements participant-specific prototype storage and confirmation-driven updates.
- `FusionEngine.swift` contains probability utilities, softmax, averaging, and evidence-adaptive fusion.
- `SessionController.swift` orchestrates one completed ADL session and applies the confirmation update.

## Required Core ML outputs

The reference engine expects the deployed Core ML model to expose at least these output names:

- `base_logits`
- `delta_logits`
- `embedding_l2`

The input name is expected to be:

- `motion_window_raw`

If Xcode/Core ML changes the generated names, update the constants in `SmartADAPTInferenceEngine.swift`.

## Input order

The model input order must match Code 1:

1. `motionUserAccelerationX(G)`
2. `motionUserAccelerationY(G)`
3. `motionUserAccelerationZ(G)`
4. `motionRotationRateX(rad/s)`
5. `motionRotationRateY(rad/s)`
6. `motionRotationRateZ(rad/s)`

## Calibration constants

The values used by `ComplexityCalibration` must be copied from Code 1 output:

`metadata/phase1_training_metadata.json`

Specifically, copy:

- descriptor means
- descriptor standard deviations
- `q15`
- `q85`
- `w_acc`
- `tau`
- `gate_k`

## Prototype constants

The reference implementation uses the manuscript parameters:

- `N_min = 3`
- `lambda = 10`
- `alpha_min = 0.05`
- `alpha_max = 0.30`
- `N_ref = 5`
- `gamma = 0.5`

## Integration note

In a full app, the user interface would call:

1. `MotionRecorder.start()` when the participant begins recording;
2. `MotionRecorder.stop()` when the participant stops;
3. `SessionController.predictSession(samples:)` to obtain the predicted ADL;
4. `SessionController.applyConfirmation(...)` after the participant confirms/corrects the prediction.

Prototype memory should be persisted between sessions using the JSON helpers in `PrototypeMemory.swift` or another watchOS persistence layer.
