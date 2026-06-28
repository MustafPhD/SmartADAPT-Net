#!/usr/bin/env python3
"""
Code 1: Phase 1 SmartADAPT-Net training, replay-interface export, and Core ML export.

This script is the Phase 1 source-of-truth for the SmartADAPT-Net pipeline. It trains
and evaluates the participant-independent CNN--BiLSTM population branch, calibrates the
motion-complexity transform, trains the complexity-aware residual correction head, and
saves a replay-ready model interface for chronological Phase 2 analysis.

Expected input layout
---------------------
Phase 1 Data/
  Brushing Teeth/P101.csv ...
  Mopping_Hoovering/P101.csv ...
  Shelving Items/P101.csv ...
  Walking/P101.csv ...
  Washing Dishes/P101.csv ...
  Washing Face/P101.csv ...

Each CSV may contain all 12 CoreMotion channels. The trained model intentionally uses
six deployment channels only, in the fixed order below:
  user acceleration X/Y/Z, then rotation rate X/Y/Z.

Primary outputs
---------------
models/smartadapt_phase1_replay_interface.keras
models/smartadapt_phase1_encoder.keras
models/smartadapt_phase1_coreml_interface.keras
models/SmartADAPT_Phase1_Global_Model.mlpackage     optional, if coremltools works
metadata/phase1_training_metadata.json
metadata/code02_replay_contract.json
metadata/class_order.json
metadata/input_channel_order.json
tables/*.csv
figures/*.png
arrays/*.csv

Typical run
-----------
python scripts/code01_phase1_train_export_coreml.py \
  --data-root "/Users/hubu/Documents/PhD 7/Data/Phase 1 Data" \
  --outdir "/Users/hubu/Documents/PhD 7/SmartADAPT_code01_phase1_outputs" \
  --strict-columns \
  --export-coreml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # deferred, so metadata/docs can still be inspected without TF
    tf = None
    keras = None


RANDOM_SEED = 42
EPS = 1e-8

ADL_ORDER = [
    "Brushing Teeth",
    "Mopping_Hoovering",
    "Shelving Items",
    "Walking",
    "Washing Dishes",
    "Washing Face",
]

ADL_DISPLAY = {
    "Brushing Teeth": "Brushing Teeth",
    "Mopping_Hoovering": "Mopping/Hoovering",
    "Shelving Items": "Shelving Items",
    "Walking": "Walking",
    "Washing Dishes": "Washing Dishes",
    "Washing Face": "Washing Face",
}

ADL_FOLDER_CANDIDATES = {
    "Brushing Teeth": ["Brushing Teeth", "Brushing_Teeth", "Brush Teeth", "Brushing"],
    "Mopping_Hoovering": ["Mopping_Hoovering", "Mopping Hoovering", "Mopping-Hoovering", "Mopping", "Household Cleaning"],
    "Shelving Items": ["Shelving Items", "Shelving_Items", "Shelving"],
    "Walking": ["Walking", "Walk"],
    "Washing Dishes": ["Washing Dishes", "Washing_Dishes", "Dishwashing", "Wash Dishes"],
    "Washing Face": ["Washing Face", "Washing_Face", "Wash Face"],
}

ACC_COLS = [
    "motionUserAccelerationX(G)",
    "motionUserAccelerationY(G)",
    "motionUserAccelerationZ(G)",
]

GYRO_COLS = [
    "motionRotationRateX(rad/s)",
    "motionRotationRateY(rad/s)",
    "motionRotationRateZ(rad/s)",
]

MODEL_COLS = ACC_COLS + GYRO_COLS

ALL_EXPECTED_COREMOTION_COLS = [
    "motionYaw(rad)",
    "motionRoll(rad)",
    "motionPitch(rad)",
    "motionRotationRateX(rad/s)",
    "motionRotationRateY(rad/s)",
    "motionRotationRateZ(rad/s)",
    "motionUserAccelerationX(G)",
    "motionUserAccelerationY(G)",
    "motionUserAccelerationZ(G)",
    "motionGravityX(G)",
    "motionGravityY(G)",
    "motionGravityZ(G)",
]


@dataclass
class ModelConfig:
    sampling_hz: int = 50
    window_seconds: int = 8
    window_size: int = 400
    n_channels: int = 6
    n_classes: int = 6

    # CNN--BiLSTM architecture. Keep synchronised with Supplementary Tables S1/S2.
    conv_filters: Tuple[int, int, int] = (64, 96, 128)
    conv_kernel_sizes: Tuple[int, int, int] = (7, 5, 3)
    conv_strides: Tuple[int, int, int] = (1, 2, 2)
    conv_dropout: float = 0.20
    lstm_units_per_direction: int = 64
    embedding_dim: int = 128
    embedding_dropout: float = 0.30
    correction_hidden_units: int = 64
    correction_dropout: float = 0.20

    # Fixed complexity-aware modulation parameters.
    w_acc: float = 0.4
    tau: float = 0.5
    gate_k: float = 10.0

    # Fixed prototype/fusion parameters saved for Code 2, not used during Phase 1 training.
    prototype_n_min: int = 3
    prototype_temperature_lambda: float = 10.0
    alpha_min: float = 0.05
    alpha_max: float = 0.30
    evidence_n_ref: int = 5
    fusion_gamma: float = 0.5

    # Training configuration. Keep synchronised with Supplementary Table S2.
    batch_size: int = 64
    base_epochs: int = 120
    correction_epochs: int = 80
    learning_rate_base: float = 1e-3
    learning_rate_correction: float = 5e-4
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

    n_splits: int = 5
    final_validation_fraction_participants: float = 0.20


CONFIG = ModelConfig()


def require_tensorflow() -> Tuple[Any, Any]:
    if tf is None or keras is None:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    return tf, keras


if keras is not None:

    @keras.utils.register_keras_serializable(package="SmartADAPT")
    class SensorZScore(keras.layers.Layer):
        """Fixed channel-wise z-score normalisation embedded in the Keras graph."""

        def __init__(self, mean: Sequence[float], std: Sequence[float], **kwargs):
            super().__init__(**kwargs)
            self.mean = [float(x) for x in mean]
            self.std = [float(x) if abs(float(x)) > EPS else 1.0 for x in std]

        def call(self, inputs):
            mean = tf.constant(self.mean, dtype=tf.float32)[None, None, :]
            std = tf.constant(self.std, dtype=tf.float32)[None, None, :]
            return (inputs - mean) / std

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"mean": self.mean, "std": self.std})
            return cfg


    @keras.utils.register_keras_serializable(package="SmartADAPT")
    class L2Normalize(keras.layers.Layer):
        """Unit-normalise embeddings for prototype-memory operations."""

        def __init__(self, axis: int = -1, **kwargs):
            super().__init__(**kwargs)
            self.axis = axis

        def call(self, inputs):
            return tf.math.l2_normalize(inputs, axis=self.axis, epsilon=EPS)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"axis": self.axis})
            return cfg


    @keras.utils.register_keras_serializable(package="SmartADAPT")
    class ComplexityTransform(keras.layers.Layer):
        """Compute raw complexity, calibrated complexity, or sigmoid gate from raw windows.

        The layer uses the Phase 1 descriptor normalisation constants and percentile
        calibration estimated outside the graph, then embeds them as fixed constants.
        """

        def __init__(
            self,
            descriptor_mean: Sequence[float],
            descriptor_std: Sequence[float],
            q15: float,
            q85: float,
            w_acc: float,
            tau: float,
            gate_k: float,
            mode: str,
            **kwargs,
        ):
            super().__init__(**kwargs)
            if mode not in {"raw", "calibrated", "gate"}:
                raise ValueError("mode must be one of: raw, calibrated, gate")
            self.descriptor_mean = [float(x) for x in descriptor_mean]
            self.descriptor_std = [float(x) if abs(float(x)) > EPS else 1.0 for x in descriptor_std]
            self.q15 = float(q15)
            self.q85 = float(q85 if abs(q85 - q15) > EPS else q15 + 1.0)
            self.w_acc = float(w_acc)
            self.tau = float(tau)
            self.gate_k = float(gate_k)
            self.mode = mode

        @staticmethod
        def _descriptors(v):
            mu_abs = tf.reduce_mean(tf.abs(v), axis=[1, 2])
            mu_range = tf.reduce_mean(tf.reduce_max(v, axis=1) - tf.reduce_min(v, axis=1), axis=1)
            diff2 = v[:, 2:, :] - 2.0 * v[:, 1:-1, :] + v[:, :-2, :]
            mu_diff2 = tf.reduce_mean(tf.abs(diff2), axis=[1, 2])
            return mu_abs, mu_range, mu_diff2

        def call(self, inputs):
            acc = inputs[:, :, 0:3]
            gyro = inputs[:, :, 3:6]
            a_abs, a_rng, a_d2 = self._descriptors(acc)
            g_abs, g_rng, g_d2 = self._descriptors(gyro)
            D = tf.stack([a_abs, a_rng, a_d2, g_abs, g_rng, g_d2], axis=1)
            mean = tf.constant(self.descriptor_mean, dtype=tf.float32)[None, :]
            std = tf.constant(self.descriptor_std, dtype=tf.float32)[None, :]
            Dz = (D - mean) / std
            c_acc = tf.reduce_mean(Dz[:, 0:3], axis=1)
            c_gyro = tf.reduce_mean(Dz[:, 3:6], axis=1)
            C = self.w_acc * c_acc + (1.0 - self.w_acc) * c_gyro
            if self.mode == "raw":
                return tf.expand_dims(C, axis=-1)
            Cbar = tf.clip_by_value((C - self.q15) / (self.q85 - self.q15), 0.0, 1.0)
            if self.mode == "calibrated":
                return tf.expand_dims(Cbar, axis=-1)
            gate = tf.sigmoid(self.gate_k * (Cbar - self.tau))
            return tf.expand_dims(gate, axis=-1)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "descriptor_mean": self.descriptor_mean,
                "descriptor_std": self.descriptor_std,
                "q15": self.q15,
                "q85": self.q85,
                "w_acc": self.w_acc,
                "tau": self.tau,
                "gate_k": self.gate_k,
                "mode": self.mode,
            })
            return cfg


else:
    SensorZScore = None
    L2Normalize = None
    ComplexityTransform = None


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_dirs(outdir: Path) -> Dict[str, Path]:
    return {
        "root": ensure_dir(outdir),
        "tables": ensure_dir(outdir / "tables"),
        "figures": ensure_dir(outdir / "figures"),
        "models": ensure_dir(outdir / "models"),
        "metadata": ensure_dir(outdir / "metadata"),
        "arrays": ensure_dir(outdir / "arrays"),
        "logs": ensure_dir(outdir / "logs"),
    }


def label_to_idx() -> Dict[str, int]:
    return {label: i for i, label in enumerate(ADL_ORDER)}


def idx_to_label() -> Dict[int, str]:
    return {i: label for i, label in enumerate(ADL_ORDER)}


def disp(label: str) -> str:
    return ADL_DISPLAY.get(label, label)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(x) for x in value]
    return value


def save_json(obj: Mapping[str, Any] | Sequence[Any], path: Path) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")
    print(f"Saved: {path}")


def find_adl_folder(root: Path, adl: str) -> Path:
    for candidate in ADL_FOLDER_CANDIDATES.get(adl, [adl]):
        path = root / candidate
        if path.exists() and path.is_dir():
            return path
    tried = ", ".join(str(root / x) for x in ADL_FOLDER_CANDIDATES.get(adl, [adl]))
    raise FileNotFoundError(f"Could not find folder for {adl}. Tried: {tried}")


def read_motion_csv(path: Path, strict: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_model_cols = [c for c in MODEL_COLS if c not in df.columns]
    if missing_model_cols:
        raise ValueError(f"{path} missing model columns: {missing_model_cols}")
    if strict:
        missing_all = [c for c in ALL_EXPECTED_COREMOTION_COLS if c not in df.columns]
        if missing_all:
            raise ValueError(f"{path} missing expected CoreMotion columns: {missing_all}")
    out = df[MODEL_COLS].copy()
    for col in MODEL_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.interpolate(limit_direction="both").ffill().bfill().dropna()
    return out


def segment_windows(arr: np.ndarray, window_size: int) -> np.ndarray:
    if len(arr) < window_size:
        return np.empty((0, window_size, arr.shape[1]), dtype=np.float32)
    starts = range(0, len(arr) - window_size + 1, window_size)
    return np.stack([arr[s:s + window_size] for s in starts]).astype(np.float32)


def load_dataset(data_root: Path, strict_columns: bool = True, expected_participants: Optional[int] = 30, cfg: ModelConfig = CONFIG):
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    groups: List[str] = []
    rows: List[Dict[str, Any]] = []
    l2i = label_to_idx()

    for adl in ADL_ORDER:
        folder = find_adl_folder(data_root, adl)
        files = sorted(folder.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {folder}")
        print(f"{adl:18s}: {len(files)} CSV files from {folder}")
        for file_path in files:
            participant = file_path.stem.strip()
            frame = read_motion_csv(file_path, strict=strict_columns)
            arr = frame[MODEL_COLS].to_numpy(np.float32)
            windows = segment_windows(arr, cfg.window_size)
            if len(windows) == 0:
                print(f"WARNING: skipped {file_path}, shorter than {cfg.window_size} samples")
                continue
            Xs.append(windows)
            ys.extend([l2i[adl]] * len(windows))
            groups.extend([participant] * len(windows))
            for window_idx in range(len(windows)):
                rows.append({
                    "participant": participant,
                    "adl": adl,
                    "adl_display": disp(adl),
                    "source_file": str(file_path),
                    "window_idx": int(window_idx),
                    "samples_per_window": int(cfg.window_size),
                })

    if not Xs:
        raise RuntimeError("No usable windows were found. Check the Phase 1 data path and CSV lengths.")

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    groups_arr = np.asarray(groups)
    meta = pd.DataFrame(rows)

    n_participants = len(np.unique(groups_arr))
    if expected_participants is not None and n_participants != expected_participants:
        print(f"WARNING: expected {expected_participants} participants, found {n_participants}.")

    print("\nLoaded Phase 1 windows")
    print("=" * 72)
    print(f"X shape: {X.shape}")
    print(f"Participants: {n_participants}")
    print("Class counts:")
    for idx, adl in idx_to_label().items():
        print(f"  {disp(adl):18s}: {int(np.sum(y == idx))}")
    return X, y, groups_arr, meta


def sensor_norm(X: np.ndarray) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=(0, 1)).astype(np.float32)
    std = X.std(axis=(0, 1)).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def descriptor_matrix(X: np.ndarray) -> np.ndarray:
    acc = X[:, :, 0:3]
    gyro = X[:, :, 3:6]

    def desc(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu_abs = np.mean(np.abs(v), axis=(1, 2))
        mu_range = np.mean(np.max(v, axis=1) - np.min(v, axis=1), axis=1)
        diff2 = v[:, 2:, :] - 2.0 * v[:, 1:-1, :] + v[:, :-2, :]
        mu_diff2 = np.mean(np.abs(diff2), axis=(1, 2))
        return mu_abs, mu_range, mu_diff2

    a_abs, a_rng, a_d2 = desc(acc)
    g_abs, g_rng, g_d2 = desc(gyro)
    return np.vstack([a_abs, a_rng, a_d2, g_abs, g_rng, g_d2]).T.astype(np.float32)


def apply_complexity(X: np.ndarray, calib: Mapping[str, Any], cfg: ModelConfig = CONFIG) -> Dict[str, np.ndarray]:
    D = descriptor_matrix(X)
    dm = np.asarray(calib["descriptor_mean"], dtype=np.float32)
    ds = np.asarray(calib["descriptor_std"], dtype=np.float32)
    Dz = (D - dm) / ds
    c_acc = Dz[:, 0:3].mean(axis=1)
    c_gyro = Dz[:, 3:6].mean(axis=1)
    C = cfg.w_acc * c_acc + (1.0 - cfg.w_acc) * c_gyro
    q15 = float(calib["q15"])
    q85 = float(calib["q85"])
    Cbar = np.clip((C - q15) / (q85 - q15), 0.0, 1.0)
    gate = 1.0 / (1.0 + np.exp(-cfg.gate_k * (Cbar - cfg.tau)))
    return {"descriptors": D, "complexity_raw": C, "complexity_calibrated": Cbar, "complexity_gate": gate}


def fit_complexity(X: np.ndarray, cfg: ModelConfig = CONFIG) -> Dict[str, Any]:
    D = descriptor_matrix(X)
    descriptor_mean = D.mean(axis=0).astype(np.float32)
    descriptor_std = D.std(axis=0).astype(np.float32)
    descriptor_std = np.where(descriptor_std < 1e-6, 1.0, descriptor_std).astype(np.float32)
    Dz = (D - descriptor_mean) / descriptor_std
    c_acc = Dz[:, 0:3].mean(axis=1)
    c_gyro = Dz[:, 3:6].mean(axis=1)
    C = cfg.w_acc * c_acc + (1.0 - cfg.w_acc) * c_gyro
    q15 = float(np.percentile(C, 15))
    q85 = float(np.percentile(C, 85))
    if abs(q85 - q15) < 1e-6:
        q85 = q15 + 1.0
    return {
        "descriptor_names": [
            "acc_mu_abs", "acc_mu_range", "acc_mu_diff2",
            "gyro_mu_abs", "gyro_mu_range", "gyro_mu_diff2",
        ],
        "descriptor_mean": descriptor_mean,
        "descriptor_std": descriptor_std,
        "q15": q15,
        "q85": q85,
        "w_acc": cfg.w_acc,
        "tau": cfg.tau,
        "gate_k": cfg.gate_k,
    }


def build_model(norm: Mapping[str, np.ndarray], calib: Mapping[str, Any], cfg: ModelConfig = CONFIG):
    require_tensorflow()
    inp = keras.Input(shape=(cfg.window_size, cfg.n_channels), name="motion_window_raw")

    x = SensorZScore(norm["mean"], norm["std"], name="sensor_zscore_normalisation")(inp)
    for idx, (filters, kernel, stride) in enumerate(zip(cfg.conv_filters, cfg.conv_kernel_sizes, cfg.conv_strides), start=1):
        x = keras.layers.Conv1D(filters, kernel, strides=stride, padding="same", use_bias=False, name=f"conv{idx}")(x)
        x = keras.layers.BatchNormalization(name=f"bn{idx}")(x)
        x = keras.layers.ReLU(name=f"relu{idx}")(x)
        x = keras.layers.Dropout(cfg.conv_dropout, name=f"conv_dropout{idx}")(x)

    x = keras.layers.Bidirectional(
        keras.layers.LSTM(cfg.lstm_units_per_direction, return_sequences=False),
        name="bilstm",
    )(x)
    x = keras.layers.Dense(cfg.embedding_dim, activation="relu", name="embedding_dense")(x)
    x = keras.layers.Dropout(cfg.embedding_dropout, name="embedding_dropout")(x)
    embedding = keras.layers.Activation("linear", name="embedding")(x)
    embedding_l2 = L2Normalize(name="embedding_l2")(embedding)

    base_logits = keras.layers.Dense(cfg.n_classes, name="base_logits")(embedding)
    base_probs = keras.layers.Softmax(name="base_probs")(base_logits)

    h = keras.layers.Dense(cfg.correction_hidden_units, activation="relu", name="correction_dense1")(embedding)
    h = keras.layers.Dropout(cfg.correction_dropout, name="correction_dropout")(h)
    delta_logits = keras.layers.Dense(cfg.n_classes, name="delta_logits")(h)

    complexity_raw = ComplexityTransform(**_complexity_layer_kwargs(calib, cfg, "raw"), name="complexity_raw")(inp)
    complexity_calibrated = ComplexityTransform(**_complexity_layer_kwargs(calib, cfg, "calibrated"), name="complexity_calibrated")(inp)
    complexity_gate = ComplexityTransform(**_complexity_layer_kwargs(calib, cfg, "gate"), name="complexity_gate")(inp)
    gated_delta = keras.layers.Multiply(name="gated_delta")([delta_logits, complexity_gate])
    global_logits = keras.layers.Add(name="global_logits")([base_logits, gated_delta])
    global_probs = keras.layers.Softmax(name="global_probs")(global_logits)

    full = keras.Model(
        inp,
        [
            base_logits,
            base_probs,
            delta_logits,
            complexity_raw,
            complexity_calibrated,
            complexity_gate,
            global_logits,
            global_probs,
            embedding,
            embedding_l2,
        ],
        name="SmartADAPT_Phase1_Replay_Interface",
    )
    base_train = keras.Model(inp, base_logits, name="phase1_base_train_model")
    global_train = keras.Model(inp, global_logits, name="phase1_global_train_model")
    encoder = keras.Model(inp, [embedding, embedding_l2], name="smartadapt_phase1_encoder")
    coreml_interface = keras.Model(
        inp,
        [base_probs, global_probs, embedding_l2, complexity_calibrated, complexity_gate],
        name="SmartADAPT_Phase1_CoreML_Interface",
    )
    return full, base_train, global_train, encoder, coreml_interface


def _complexity_layer_kwargs(calib: Mapping[str, Any], cfg: ModelConfig, mode: str) -> Dict[str, Any]:
    return {
        "descriptor_mean": np.asarray(calib["descriptor_mean"], np.float32).tolist(),
        "descriptor_std": np.asarray(calib["descriptor_std"], np.float32).tolist(),
        "q15": float(calib["q15"]),
        "q85": float(calib["q85"]),
        "w_acc": float(cfg.w_acc),
        "tau": float(cfg.tau),
        "gate_k": float(cfg.gate_k),
        "mode": mode,
    }


def compile_logits(model, lr: float) -> None:
    require_tensorflow()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def callbacks(path: Path, cfg: ModelConfig = CONFIG):
    require_tensorflow()
    ensure_dir(path)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=cfg.reduce_lr_factor,
            patience=cfg.reduce_lr_patience,
            min_lr=cfg.min_lr,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(path / "training_log.csv"), append=False),
    ]


def freeze_for_correction(full, global_train, cfg: ModelConfig = CONFIG) -> None:
    trainable = {"correction_dense1", "correction_dropout", "delta_logits"}
    for layer in full.layers:
        layer.trainable = layer.name in trainable
    compile_logits(global_train, cfg.learning_rate_correction)


def predict_named(model, X: np.ndarray, batch_size: int) -> Dict[str, np.ndarray]:
    out = model.predict(X, batch_size=batch_size, verbose=0)
    if isinstance(out, dict):
        return out
    if not isinstance(out, (list, tuple)):
        out = [out]
    return {name: np.asarray(value) for name, value in zip(model.output_names, out)}


def overall_row(y_true: np.ndarray, pred: np.ndarray, stage: str, fold: Optional[int] = None) -> Dict[str, Any]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    row: Dict[str, Any] = {
        "Stage": stage,
        "Windows": int(len(y_true)),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, pred)),
        "Macro Precision": float(p),
        "Macro Recall": float(r),
        "Macro F1": float(f1),
    }
    if fold is not None:
        row["Fold"] = int(fold)
    return row


def classwise_table(y_true: np.ndarray, pred: np.ndarray, stage: str) -> pd.DataFrame:
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=np.arange(len(ADL_ORDER)),
        zero_division=0,
    )
    rows: List[Dict[str, Any]] = []
    for idx, adl in enumerate(ADL_ORDER):
        mask = y_true == idx
        rows.append({
            "Stage": stage,
            "ADL": disp(adl),
            "Support": int(support[idx]),
            "Accuracy": float(np.mean(pred[mask] == y_true[mask])) if mask.any() else np.nan,
            "Precision": float(p[idx]),
            "Recall": float(r[idx]),
            "F1-score": float(f1[idx]),
        })
    rows.append({"ADL": "Overall", "Support": int(len(y_true)), **overall_row(y_true, pred, stage)})
    return pd.DataFrame(rows)


def change_table(y_true: np.ndarray, base_pred: np.ndarray, global_pred: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, adl in enumerate(ADL_ORDER):
        mask = y_true == idx
        base_correct = base_pred[mask] == y_true[mask]
        global_correct = global_pred[mask] == y_true[mask]
        changed = base_pred[mask] != global_pred[mask]
        improved = (~base_correct) & global_correct
        degraded = base_correct & (~global_correct)
        rows.append({
            "ADL": disp(adl),
            "Support": int(mask.sum()),
            "Changed": int(changed.sum()),
            "Improved": int(improved.sum()),
            "Degraded": int(degraded.sum()),
            "Net": int(improved.sum() - degraded.sum()),
        })
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def draw_confusion(ax, cm: np.ndarray, panel_label: str, normalised: bool = False, ylabel: bool = True):
    im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=1 if normalised else None)
    labels = [disp(a) for a in ADL_ORDER]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels if ylabel else [])
    ax.set_xlabel("Predicted ADL")
    if ylabel:
        ax.set_ylabel("True ADL")
    ax.text(0.02, 1.04, panel_label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")
    threshold = np.nanmax(cm) * 0.55 if np.nanmax(cm) > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:.2f}" if normalised else str(int(value))
            ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black", fontsize=8)
    return im


def plot_confusions(y_true: np.ndarray, base_pred: np.ndarray, global_pred: np.ndarray, outdir: Path, dpi: int) -> None:
    labels = np.arange(len(ADL_ORDER))
    for normalised in (False, True):
        cm_base = confusion_matrix(y_true, base_pred, labels=labels)
        cm_global = confusion_matrix(y_true, global_pred, labels=labels)
        if normalised:
            cm_base = np.nan_to_num(cm_base / cm_base.sum(axis=1, keepdims=True))
            cm_global = np.nan_to_num(cm_global / cm_global.sum(axis=1, keepdims=True))
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
        draw_confusion(axes[0], cm_base, "A. Backbone", normalised=normalised, ylabel=True)
        im = draw_confusion(axes[1], cm_global, "B. Complexity-aware", normalised=normalised, ylabel=False)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.025)
        cbar.set_label("Row-normalised proportion" if normalised else "Window count")
        stem = "fig_phase1_confusion_matrices_base_global_normalized.png" if normalised else "fig_phase1_confusion_matrices_base_global.png"
        fig.savefig(outdir / stem, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {outdir / stem}")


def fit_fold(X, y, train_idx, val_idx, fold: int, dirs: Mapping[str, Path], cfg: ModelConfig):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    norm = sensor_norm(X_train)
    calib = fit_complexity(X_train, cfg)
    full, base_train, global_train, _, _ = build_model(norm, calib, cfg)

    compile_logits(base_train, cfg.learning_rate_base)
    print(f"Fold {fold}: training Base branch")
    base_train.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.base_epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks(dirs["logs"] / f"fold_{fold:02d}" / "base", cfg),
        verbose=2,
    )

    print(f"Fold {fold}: training complexity-aware correction head")
    freeze_for_correction(full, global_train, cfg)
    global_train.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.correction_epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks(dirs["logs"] / f"fold_{fold:02d}" / "global", cfg),
        verbose=2,
    )

    out = predict_named(full, X_val, cfg.batch_size)
    base_pred = np.argmax(out["base_probs"], axis=1)
    global_pred = np.argmax(out["global_probs"], axis=1)
    rows = [
        overall_row(y_val, base_pred, "Backbone", fold),
        overall_row(y_val, global_pred, "Complexity-aware", fold),
    ]
    return y_val, base_pred, global_pred, rows


def run_cross_validation(X, y, groups, dirs: Mapping[str, Path], cfg: ModelConfig):
    y_true = np.empty_like(y)
    y_base = np.empty_like(y)
    y_global = np.empty_like(y)
    rows: List[Dict[str, Any]] = []
    splitter = GroupKFold(n_splits=cfg.n_splits)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        print("\n" + "=" * 72)
        print(f"Fold {fold}/{cfg.n_splits}")
        print("Train participants:", sorted(np.unique(groups[train_idx])))
        print("Val participants:", sorted(np.unique(groups[val_idx])))
        yt, yb, yg, fold_rows = fit_fold(X, y, train_idx, val_idx, fold, dirs, cfg)
        y_true[val_idx] = yt
        y_base[val_idx] = yb
        y_global[val_idx] = yg
        rows.extend(fold_rows)
    return y_true, y_base, y_global, pd.DataFrame(rows)


def train_final(X, y, groups, dirs: Mapping[str, Path], cfg: ModelConfig):
    unique_participants = np.unique(groups)
    if cfg.final_validation_fraction_participants > 0 and len(unique_participants) >= 5:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.final_validation_fraction_participants,
            random_state=RANDOM_SEED,
        )
        train_idx, val_idx = next(splitter.split(X, y, groups))
        validation = (X[val_idx], y[val_idx])
        print("Final model validation participants:", sorted(np.unique(groups[val_idx])))
    else:
        train_idx = np.arange(len(X))
        validation = None

    norm = sensor_norm(X[train_idx])
    calib = fit_complexity(X[train_idx], cfg)
    full, base_train, global_train, encoder, coreml_interface = build_model(norm, calib, cfg)

    compile_logits(base_train, cfg.learning_rate_base)
    print("Training final Base branch")
    if validation is None:
        base_train.fit(X[train_idx], y[train_idx], epochs=cfg.base_epochs, batch_size=cfg.batch_size, verbose=2)
    else:
        base_train.fit(
            X[train_idx],
            y[train_idx],
            validation_data=validation,
            epochs=cfg.base_epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks(dirs["logs"] / "final_base", cfg),
            verbose=2,
        )

    print("Training final complexity-aware correction head")
    freeze_for_correction(full, global_train, cfg)
    if validation is None:
        global_train.fit(X[train_idx], y[train_idx], epochs=cfg.correction_epochs, batch_size=cfg.batch_size, verbose=2)
    else:
        global_train.fit(
            X[train_idx],
            y[train_idx],
            validation_data=validation,
            epochs=cfg.correction_epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks(dirs["logs"] / "final_global", cfg),
            verbose=2,
        )

    model_paths = {
        "replay_interface": dirs["models"] / "smartadapt_phase1_replay_interface.keras",
        "encoder": dirs["models"] / "smartadapt_phase1_encoder.keras",
        "coreml_interface_keras": dirs["models"] / "smartadapt_phase1_coreml_interface.keras",
    }
    full.save(model_paths["replay_interface"])
    encoder.save(model_paths["encoder"])
    coreml_interface.save(model_paths["coreml_interface_keras"])
    for path in model_paths.values():
        print(f"Saved: {path}")

    complexity_reference = apply_complexity(X[train_idx], calib, cfg)
    pd.DataFrame({
        "complexity_raw": complexity_reference["complexity_raw"],
        "complexity_calibrated": complexity_reference["complexity_calibrated"],
        "complexity_gate": complexity_reference["complexity_gate"],
    }).to_csv(dirs["arrays"] / "phase1_complexity_reference_distribution.csv", index=False)

    metadata = build_metadata(norm, calib, cfg, model_paths, train_participants=np.unique(groups[train_idx]).tolist())
    write_metadata_files(metadata, dirs)
    return full, coreml_interface, metadata


def build_metadata(norm, calib, cfg, model_paths, train_participants: Sequence[str]) -> Dict[str, Any]:
    return {
        "schema_version": "SmartADAPT-Code1-Phase1-v2",
        "random_seed": RANDOM_SEED,
        "class_order": ADL_ORDER,
        "class_to_index": label_to_idx(),
        "adl_display": ADL_DISPLAY,
        "input_channel_order": MODEL_COLS,
        "input_channel_groups": {"accelerometer": ACC_COLS, "gyroscope": GYRO_COLS},
        "input_representation": {
            "shape": [CONFIG.window_size, CONFIG.n_channels],
            "sampling_hz": CONFIG.sampling_hz,
            "window_seconds": CONFIG.window_seconds,
            "units": {
                "accelerometer": "G",
                "gyroscope": "rad/s",
            },
        },
        "model_config": asdict(cfg),
        "sensor_normalisation": {
            "mean": np.asarray(norm["mean"], dtype=float).tolist(),
            "std": np.asarray(norm["std"], dtype=float).tolist(),
        },
        "complexity_calibration": {
            key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
            for key, value in calib.items()
        },
        "prototype_and_fusion_parameters_for_code02": {
            "N_min": cfg.prototype_n_min,
            "lambda": cfg.prototype_temperature_lambda,
            "alpha_min": cfg.alpha_min,
            "alpha_max": cfg.alpha_max,
            "N_ref": cfg.evidence_n_ref,
            "gamma": cfg.fusion_gamma,
        },
        "keras_output_contract": {
            "model_file": str(model_paths["replay_interface"]),
            "input_name": "motion_window_raw",
            "output_names": [
                "base_logits",
                "base_probs",
                "delta_logits",
                "complexity_raw",
                "complexity_calibrated",
                "complexity_gate",
                "global_logits",
                "global_probs",
                "embedding",
                "embedding_l2",
            ],
            "code02_required_outputs": [
                "base_probs",
                "global_probs",
                "embedding_l2",
                "complexity_calibrated",
                "complexity_gate",
            ],
        },
        "coreml_export_contract": {
            "keras_source_file": str(model_paths["coreml_interface_keras"]),
            "intended_coreml_outputs": [
                "base_probs",
                "global_probs",
                "embedding_l2",
                "complexity_calibrated",
                "complexity_gate",
            ],
        },
        "training_participants_used_for_final_model": list(map(str, train_participants)),
        "notes": [
            "Code 2 should load the replay-interface Keras model and this metadata.",
            "The Core ML package is for Xcode/watchOS deployment and is not required for Python replay.",
            "Prototype memory is not trained here; it is reconstructed chronologically in Code 2.",
        ],
    }


def write_metadata_files(metadata: Mapping[str, Any], dirs: Mapping[str, Path]) -> None:
    save_json(metadata, dirs["metadata"] / "phase1_training_metadata.json")
    save_json(metadata["class_order"], dirs["metadata"] / "class_order.json")
    save_json(metadata["input_channel_order"], dirs["metadata"] / "input_channel_order.json")
    save_json(metadata["keras_output_contract"], dirs["metadata"] / "code02_replay_contract.json")
    # Backward-compatible copy for scripts that expect the metadata beside the model.
    shutil.copyfile(dirs["metadata"] / "phase1_training_metadata.json", dirs["models"] / "phase1_training_metadata.json")


def export_coreml(coreml_interface, dirs: Mapping[str, Path], cfg: ModelConfig, target_name: str = "watchOS9") -> Optional[Path]:
    try:
        import coremltools as ct
    except ImportError:
        print("coremltools not installed; skipping Core ML export. Install with: pip install coremltools")
        return None

    target_map = {
        "watchOS8": ct.target.watchOS8,
        "watchOS9": ct.target.watchOS9,
        "watchOS10": ct.target.watchOS10,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }
    target = target_map.get(target_name, ct.target.watchOS9)
    try:
        mlmodel = ct.convert(
            coreml_interface,
            convert_to="mlprogram",
            minimum_deployment_target=target,
            inputs=[ct.TensorType(name="motion_window_raw", shape=(1, cfg.window_size, cfg.n_channels))],
        )
        mlmodel.short_description = "SmartADAPT-Net Phase 1 replay/global interface for watchOS deployment"
        out_path = dirs["models"] / "SmartADAPT_Phase1_Global_Model.mlpackage"
        mlmodel.save(str(out_path))
        print(f"Saved Core ML package: {out_path}")
        return out_path
    except Exception as exc:
        print("Core ML conversion failed:", exc)
        print("The Keras replay interface and metadata were still saved successfully.")
        return None


def save_phase1_outputs(y_true, y_base, y_global, meta, fold_summary, dirs: Mapping[str, Path], dpi: int) -> None:
    fold_summary.to_csv(dirs["tables"] / "table_phase1_fold_summary.csv", index=False)
    pd.DataFrame([
        overall_row(y_true, y_base, "Backbone"),
        overall_row(y_true, y_global, "Complexity-aware"),
    ]).to_csv(dirs["tables"] / "table_phase1_overall_cv_performance.csv", index=False)
    pd.concat([
        classwise_table(y_true, y_base, "Backbone"),
        classwise_table(y_true, y_global, "Complexity-aware"),
    ], ignore_index=True).to_csv(dirs["tables"] / "table_phase1_classwise_cv_performance.csv", index=False)
    change_table(y_true, y_base, y_global).to_csv(
        dirs["tables"] / "table_phase1_complexity_aware_decision_changes.csv",
        index=False,
    )

    pred_df = meta.copy()
    pred_df["true_idx"] = y_true
    pred_df["base_pred_idx"] = y_base
    pred_df["global_pred_idx"] = y_global
    idx_to_lbl = idx_to_label()
    pred_df["true_label"] = [idx_to_lbl[int(i)] for i in y_true]
    pred_df["base_pred_label"] = [idx_to_lbl[int(i)] for i in y_base]
    pred_df["global_pred_label"] = [idx_to_lbl[int(i)] for i in y_global]
    pred_df.to_csv(dirs["arrays"] / "phase1_groupkfold_window_predictions.csv", index=False)
    plot_confusions(y_true, y_base, y_global, dirs["figures"], dpi=dpi)


def main(args) -> None:
    set_seed(args.seed)
    set_plot_style()
    cfg = ModelConfig()
    cfg.batch_size = args.batch_size
    cfg.base_epochs = args.base_epochs
    cfg.correction_epochs = args.correction_epochs
    cfg.learning_rate_base = args.learning_rate_base
    cfg.learning_rate_correction = args.learning_rate_correction
    cfg.n_splits = args.n_splits
    cfg.final_validation_fraction_participants = args.final_validation_fraction_participants

    dirs = make_dirs(Path(args.outdir))
    save_json(asdict(cfg), dirs["root"] / "phase1_code01_config.json")

    X, y, groups, meta = load_dataset(
        Path(args.data_root),
        strict_columns=args.strict_columns,
        expected_participants=args.expected_participants,
        cfg=cfg,
    )
    meta.to_csv(dirs["arrays"] / "phase1_windows_metadata.csv", index=False)

    y_true, y_base, y_global, fold_summary = run_cross_validation(X, y, groups, dirs, cfg)
    save_phase1_outputs(y_true, y_base, y_global, meta, fold_summary, dirs, dpi=args.dpi)

    coreml_path = None
    if not args.skip_final_model:
        _, coreml_interface, metadata = train_final(X, y, groups, dirs, cfg)
        if args.export_coreml:
            coreml_path = export_coreml(coreml_interface, dirs, cfg, args.minimum_coreml_target)
            if coreml_path is not None:
                metadata = dict(metadata)
                metadata["coreml_export_contract"] = dict(metadata["coreml_export_contract"])
                metadata["coreml_export_contract"]["coreml_package"] = str(coreml_path)
                write_metadata_files(metadata, dirs)

    manifest = {
        "script": Path(__file__).name,
        "data_root": args.data_root,
        "outdir": args.outdir,
        "windows": int(len(X)),
        "participants": int(len(np.unique(groups))),
        "adls": ADL_ORDER,
        "model_columns": MODEL_COLS,
        "tables": sorted(p.name for p in dirs["tables"].glob("*.csv")),
        "figures": sorted(p.name for p in dirs["figures"].glob("*.png")),
        "models": sorted(p.name for p in dirs["models"].glob("*")),
        "metadata": sorted(p.name for p in dirs["metadata"].glob("*.json")),
        "coreml_exported": coreml_path is not None,
    }
    save_json(manifest, dirs["root"] / "generation_manifest.json")
    print("\nDone. Outputs saved in:", args.outdir)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Phase 1 SmartADAPT-Net training and replay/Core ML export")
    parser.add_argument("--data-root", default="/Users/hubu/Documents/PhD 7/Data/Phase 1 Data")
    parser.add_argument("--outdir", default="/Users/hubu/Documents/PhD 7/SmartADAPT_code01_phase1_outputs")
    parser.add_argument("--strict-columns", action="store_true")
    parser.add_argument("--expected-participants", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-splits", type=int, default=CONFIG.n_splits)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--base-epochs", type=int, default=CONFIG.base_epochs)
    parser.add_argument("--correction-epochs", type=int, default=CONFIG.correction_epochs)
    parser.add_argument("--learning-rate-base", type=float, default=CONFIG.learning_rate_base)
    parser.add_argument("--learning-rate-correction", type=float, default=CONFIG.learning_rate_correction)
    parser.add_argument("--final-validation-fraction-participants", type=float, default=CONFIG.final_validation_fraction_participants)
    parser.add_argument("--skip-final-model", action="store_true")
    parser.add_argument("--export-coreml", action="store_true")
    parser.add_argument("--minimum-coreml-target", default="watchOS9")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring extra arguments:", " ".join(unknown))
    return args


if __name__ == "__main__":
    main(parse_args())
