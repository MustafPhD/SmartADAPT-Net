#!/usr/bin/env python3
"""
Code 2: Phase 2 chronological deployment-log replay for SmartADAPT-Net.

This script reconstructs the deployment-time decision pipeline from saved Phase 2
session CSVs using the replay-interface model and metadata produced by Code 1.
It does not retrain the model. It replays sessions chronologically within each
participant, reconstructs confirmation-driven prototype memory causally, and saves
a canonical replay file for Code 3 manuscript/supplementary output generation.

Expected Phase 2 layout
-----------------------
Phase 2 Data_F/
  P201/
    manifest.csv
    P201_D01_S001.csv
    P201_D01_S002.csv
    ...
  P202/
    manifest.csv
    ...

Each manifest.csv should contain at least:
  participant, day, session, UserConfirmation_true, PredictedClass, n_rows, file

Each session CSV should contain the CoreMotion columns:
  Timestamp,
  motionRotationRateX(rad/s), motionRotationRateY(rad/s), motionRotationRateZ(rad/s),
  motionUserAccelerationX(G), motionUserAccelerationY(G), motionUserAccelerationZ(G),
  motionGravityX(G), motionGravityY(G), motionGravityZ(G),
  motionYaw(rad), motionRoll(rad), motionPitch(rad),
  PredictedClass, UserConfirmation

Typical run
-----------
python scripts/code02_phase2_chronological_replay.py \
  --phase2-root "/Users/hubu/Documents/Phase 2 Data_F" \
  --code1-outdir "/Users/hubu/Documents/PhD 7/SmartADAPT_code01_phase1_outputs" \
  --outdir "/Users/hubu/Documents/Phase 2 Data_F/code02_replay_outputs"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # Fail later with a clearer message.
    tf = None
    keras = None

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

LABEL_MAP = {
    "Brushing Teeth": "Brushing Teeth",
    "Brush Teeth": "Brushing Teeth",
    "Brushing": "Brushing Teeth",
    "Mopping_Hoovering": "Mopping_Hoovering",
    "Mopping/Hoovering": "Mopping_Hoovering",
    "Mopping Hoovering": "Mopping_Hoovering",
    "Mopping-Hoovering": "Mopping_Hoovering",
    "Mop/Hoover": "Mopping_Hoovering",
    "Mopping": "Mopping_Hoovering",
    "Household Cleaning": "Mopping_Hoovering",
    "Shelving Items": "Shelving Items",
    "Shelving_Items": "Shelving Items",
    "Shelve Items": "Shelving Items",
    "Shelving": "Shelving Items",
    "Walking": "Walking",
    "Walk": "Walking",
    "Washing Dishes": "Washing Dishes",
    "Washing_Dishes": "Washing Dishes",
    "Wash Dishes": "Washing Dishes",
    "Dishwashing": "Washing Dishes",
    "Washing Face": "Washing Face",
    "Washing_Face": "Washing Face",
    "Wash Face": "Washing Face",
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

SESSION_REQUIRED_MOTION_COLS = [
    "motionRotationRateX(rad/s)",
    "motionRotationRateY(rad/s)",
    "motionRotationRateZ(rad/s)",
    "motionUserAccelerationX(G)",
    "motionUserAccelerationY(G)",
    "motionUserAccelerationZ(G)",
]

MANIFEST_MIN_COLS = ["participant", "day", "session", "UserConfirmation_true", "PredictedClass"]


@dataclass
class ReplayConfig:
    sampling_hz: int = 50
    pre_seconds: float = 3.0
    analysis_seconds: float = 24.0
    post_seconds: float = 3.0
    window_seconds: float = 8.0

    N_min: int = 3
    lambda_proto: float = 10.0
    alpha_min: float = 0.05
    alpha_max: float = 0.30
    N_ref: int = 5
    gamma: float = 0.5

    analysis_mode: str = "after_pre"       # after_pre | center | from_start | from_end
    short_session_policy: str = "skip"     # skip | pad
    batch_size: int = 256

    @property
    def window_size(self) -> int:
        return int(round(self.window_seconds * self.sampling_hz))

    @property
    def analysis_size(self) -> int:
        return int(round(self.analysis_seconds * self.sampling_hz))

    @property
    def pre_size(self) -> int:
        return int(round(self.pre_seconds * self.sampling_hz))

    @property
    def expected_n_windows(self) -> int:
        q, r = divmod(self.analysis_size, self.window_size)
        if r != 0:
            raise ValueError("analysis_seconds must be an integer multiple of window_seconds")
        return q


if keras is not None:

    @keras.utils.register_keras_serializable(package="SmartADAPT")
    class SensorZScore(keras.layers.Layer):
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


def require_tensorflow() -> None:
    if tf is None or keras is None:
        raise ImportError("TensorFlow is required for Code 2. Install with: pip install tensorflow")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    print(f"Saved: {path}")


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def label_to_idx(class_order: Sequence[str]) -> Dict[str, int]:
    return {lab: i for i, lab in enumerate(class_order)}


def idx_to_label(class_order: Sequence[str]) -> Dict[int, str]:
    return {i: lab for i, lab in enumerate(class_order)}


def normalise_label(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    if s.lower() == "yes":
        return "Yes"
    return LABEL_MAP.get(s, s)


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def coerce_int(value: Any, fallback: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return fallback


def canonical_session_id(participant: str, day: int, session: int) -> str:
    return f"{participant}_D{day:02d}_S{session:03d}"


def infer_day_session_from_filename(path: Path) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"D(\d+)_S(\d+)", path.stem, flags=re.IGNORECASE)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def resolve_code1_paths(code1_outdir: Optional[Path], model_path: Optional[Path], metadata_path: Optional[Path]) -> Tuple[Path, Path]:
    if code1_outdir is not None:
        if model_path is None:
            model_path = code1_outdir / "models" / "smartadapt_phase1_replay_interface.keras"
        if metadata_path is None:
            p1 = code1_outdir / "metadata" / "phase1_training_metadata.json"
            p2 = code1_outdir / "models" / "phase1_training_metadata.json"
            metadata_path = p1 if p1.exists() else p2
    if model_path is None or metadata_path is None:
        raise ValueError("Provide --code1-outdir, or both --model-path and --metadata-path.")
    return model_path, metadata_path


def load_replay_model(model_path: Path):
    require_tensorflow()
    if not model_path.exists():
        raise FileNotFoundError(f"Replay model not found: {model_path}")
    custom_objects = {
        "SensorZScore": SensorZScore,
        "L2Normalize": L2Normalize,
        "ComplexityTransform": ComplexityTransform,
        "SmartADAPT>SensorZScore": SensorZScore,
        "SmartADAPT>L2Normalize": L2Normalize,
        "SmartADAPT>ComplexityTransform": ComplexityTransform,
    }
    print(f"Loading replay model: {model_path}")
    return keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)


def predict_dict(model, windows: np.ndarray, batch_size: int) -> Dict[str, np.ndarray]:
    out = model.predict(windows, batch_size=batch_size, verbose=0)
    if isinstance(out, dict):
        return {str(k): np.asarray(v) for k, v in out.items()}
    names = list(getattr(model, "output_names", []))
    if not names:
        raise RuntimeError("Model did not expose output_names; cannot map replay outputs.")
    return {name: np.asarray(value) for name, value in zip(names, out)}


def require_model_outputs(out: Mapping[str, np.ndarray]) -> None:
    required = ["base_probs", "global_probs", "embedding_l2", "complexity_calibrated", "complexity_gate"]
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"Replay model is missing required outputs: {missing}. Available: {list(out.keys())}")


def read_motion_session(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Session CSV not found: {path}")
    frame = pd.read_csv(path)
    missing = [c for c in SESSION_REQUIRED_MOTION_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required motion columns: {missing}")
    out = frame.copy()
    for col in MODEL_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out[MODEL_COLS] = out[MODEL_COLS].interpolate(limit_direction="both").ffill().bfill()
    out = out.dropna(subset=MODEL_COLS).reset_index(drop=True)
    return out


def extract_analysis_region(frame: pd.DataFrame, cfg: ReplayConfig) -> Tuple[np.ndarray, int, int, str]:
    n = len(frame)
    need = cfg.analysis_size

    if n >= need:
        if cfg.analysis_mode == "after_pre":
            start = cfg.pre_size
            if start + need > n:
                start = max(0, n - need)
        elif cfg.analysis_mode == "center":
            start = max(0, (n - need) // 2)
        elif cfg.analysis_mode == "from_start":
            start = 0
        elif cfg.analysis_mode == "from_end":
            start = max(0, n - need)
        else:
            raise ValueError("analysis_mode must be one of: after_pre, center, from_start, from_end")
        end = start + need
        arr = frame.loc[start:end - 1, MODEL_COLS].to_numpy(np.float32)
        return arr, int(start), int(end), "ok"

    if cfg.short_session_policy == "pad" and n > 0:
        arr = frame[MODEL_COLS].to_numpy(np.float32)
        pad = np.repeat(arr[-1:, :], repeats=need - n, axis=0)
        return np.concatenate([arr, pad], axis=0), 0, int(n), "padded"

    return np.empty((0, len(MODEL_COLS)), dtype=np.float32), 0, int(n), "too_short"


def split_three_windows(region: np.ndarray, cfg: ReplayConfig) -> np.ndarray:
    if region.shape[0] != cfg.analysis_size:
        raise ValueError(f"Expected analysis region with {cfg.analysis_size} samples, got {region.shape[0]}")
    windows = region.reshape(cfg.expected_n_windows, cfg.window_size, region.shape[1])
    return windows.astype(np.float32)


def softmax_masked(logits: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    probs = np.zeros_like(logits, dtype=np.float64)
    active_idx = np.where(active_mask)[0]
    if len(active_idx) == 0:
        probs[:] = 1.0 / len(logits)
        return probs.astype(np.float32)
    x = logits[active_idx].astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    probs[active_idx] = e / np.sum(e)
    return probs.astype(np.float32)


class PrototypeState:
    def __init__(self, n_classes: int, embedding_dim: int, cfg: ReplayConfig):
        self.n_classes = int(n_classes)
        self.embedding_dim = int(embedding_dim)
        self.cfg = cfg
        self.prototypes = np.zeros((self.n_classes, self.embedding_dim), dtype=np.float32)
        self.counts = np.zeros(self.n_classes, dtype=np.int64)

    @property
    def active_mask(self) -> np.ndarray:
        return self.counts >= int(self.cfg.N_min)

    @property
    def any_active(self) -> bool:
        return bool(np.any(self.active_mask))

    def evidence_ratio(self) -> float:
        return float(np.mean(np.minimum(self.counts.astype(np.float64) / float(self.cfg.N_ref), 1.0)))

    def beta_eff(self) -> float:
        if not self.any_active:
            return 1.0
        return float(1.0 - self.cfg.gamma * self.evidence_ratio())

    def proto_probs_for_embeddings(self, embeddings_l2: np.ndarray) -> np.ndarray:
        active = self.active_mask
        if not np.any(active):
            return np.full((embeddings_l2.shape[0], self.n_classes), 1.0 / self.n_classes, dtype=np.float32)
        sims = embeddings_l2 @ self.prototypes.T
        logits = self.cfg.lambda_proto * sims
        return np.stack([softmax_masked(row, active) for row in logits], axis=0).astype(np.float32)

    def update(self, class_idx: int, session_embedding_l2: np.ndarray, complexity_session: float) -> None:
        z = np.asarray(session_embedding_l2, dtype=np.float32)
        z = z / max(float(np.linalg.norm(z)), EPS)
        alpha = float(self.cfg.alpha_min + (self.cfg.alpha_max - self.cfg.alpha_min) * (1.0 - complexity_session))
        alpha = float(np.clip(alpha, self.cfg.alpha_min, self.cfg.alpha_max))
        p_old = self.prototypes[class_idx]
        p_new = (1.0 - alpha) * p_old + alpha * z
        p_new = p_new / max(float(np.linalg.norm(p_new)), EPS)
        self.prototypes[class_idx] = p_new.astype(np.float32)
        self.counts[class_idx] += 1

    def serialise(self, class_order: Sequence[str]) -> Dict[str, Any]:
        return {
            class_order[i]: {
                "count": int(self.counts[i]),
                "active": bool(self.counts[i] >= self.cfg.N_min),
                "prototype": self.prototypes[i].astype(float).tolist(),
            }
            for i in range(self.n_classes)
        }


def session_embedding_from_windows(embeddings_l2: np.ndarray, final_probs: np.ndarray, confirmed_idx: int) -> np.ndarray:
    weights = final_probs[:, confirmed_idx].astype(np.float64)
    if not np.isfinite(weights).all() or float(weights.sum()) <= EPS:
        weights = np.ones(len(embeddings_l2), dtype=np.float64)
    z = np.average(embeddings_l2.astype(np.float64), axis=0, weights=weights)
    z = z / max(float(np.linalg.norm(z)), EPS)
    return z.astype(np.float32)


def stage_summary(prob: np.ndarray, class_order: Sequence[str]) -> Tuple[str, float]:
    idx = int(np.argmax(prob))
    return class_order[idx], float(prob[idx])


def infer_confirmed_from_session(frame: pd.DataFrame, watch_pred: Optional[str]) -> Optional[str]:
    if "UserConfirmation" not in frame.columns:
        return watch_pred
    vals = [normalise_label(x) for x in frame["UserConfirmation"].dropna().astype(str).tolist()]
    vals = [x for x in vals if x]
    if not vals:
        return watch_pred
    last = vals[-1]
    if last == "Yes":
        return watch_pred
    return last


def infer_watch_pred_from_session(frame: pd.DataFrame) -> Optional[str]:
    if "PredictedClass" not in frame.columns:
        return None
    vals = [normalise_label(x) for x in frame["PredictedClass"].dropna().astype(str).tolist()]
    vals = [x for x in vals if x and x != "Yes"]
    return vals[-1] if vals else None


def resolve_session_file(participant_dir: Path, row: Mapping[str, Any]) -> Path:
    participant = str(row.get("participant", participant_dir.name)).strip()
    day = coerce_int(row.get("day"), -1)
    session = coerce_int(row.get("session"), -1)

    manifest_file = row.get("file", None)
    if manifest_file is not None and not (isinstance(manifest_file, float) and np.isnan(manifest_file)):
        candidate = Path(str(manifest_file))
        if candidate.exists():
            return candidate
        candidate2 = participant_dir / candidate.name
        if candidate2.exists():
            return candidate2

    if day > 0 and session > 0:
        candidate3 = participant_dir / f"{participant}_D{day:02d}_S{session:03d}.csv"
        if candidate3.exists():
            return candidate3

    csvs = sorted(participant_dir.glob("*.csv"))
    csvs = [p for p in csvs if p.name.lower() != "manifest.csv"]
    if day > 0 and session > 0:
        pattern = re.compile(rf"D0*{day}_S0*{session}\b", flags=re.IGNORECASE)
        hits = [p for p in csvs if pattern.search(p.stem)]
        if len(hits) == 1:
            return hits[0]
    raise FileNotFoundError(f"Could not resolve session file for {participant} day={day} session={session} in {participant_dir}")


def read_manifest(participant_dir: Path) -> pd.DataFrame:
    path = participant_dir / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    manifest = pd.read_csv(path)
    missing = [c for c in MANIFEST_MIN_COLS if c not in manifest.columns]
    if missing:
        raise ValueError(f"{path} missing manifest columns: {missing}; available: {list(manifest.columns)}")
    if "participant" not in manifest.columns:
        manifest["participant"] = participant_dir.name
    manifest["participant"] = manifest["participant"].astype(str).fillna(participant_dir.name)
    manifest["day"] = pd.to_numeric(manifest["day"], errors="coerce").astype("Int64")
    manifest["session"] = pd.to_numeric(manifest["session"], errors="coerce").astype("Int64")
    return manifest.sort_values(["participant", "day", "session"], kind="mergesort").reset_index(drop=True)


def discover_participant_dirs(root: Path, participant_glob: str) -> List[Path]:
    dirs = sorted([p for p in root.glob(participant_glob) if p.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No participant folders found under {root} using glob '{participant_glob}'")
    return dirs


def clean_manifest_label(row: Mapping[str, Any], session_frame: Optional[pd.DataFrame] = None) -> Tuple[Optional[str], Optional[str]]:
    watch_pred = normalise_label(row.get("PredictedClass"))
    if (watch_pred is None or watch_pred == "Yes") and session_frame is not None:
        watch_pred = infer_watch_pred_from_session(session_frame)

    confirmed = normalise_label(row.get("UserConfirmation_true"))
    if confirmed == "Yes":
        confirmed = watch_pred
    if confirmed is None and session_frame is not None:
        confirmed = infer_confirmed_from_session(session_frame, watch_pred)
    return watch_pred, confirmed


def replay_one_session(
    participant: str,
    day: int,
    session: int,
    session_file: Path,
    frame: pd.DataFrame,
    confirmed_class: str,
    watch_pred: Optional[str],
    model,
    proto_state: PrototypeState,
    class_order: Sequence[str],
    cfg: ReplayConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    class_to_i = label_to_idx(class_order)
    confirmed_idx = class_to_i[confirmed_class]

    region, start, end, extraction_status = extract_analysis_region(frame, cfg)
    if extraction_status == "too_short":
        return {}, [], extraction_status

    windows = split_three_windows(region, cfg)
    model_out = predict_dict(model, windows, cfg.batch_size)
    require_model_outputs(model_out)

    base_probs_w = np.asarray(model_out["base_probs"], dtype=np.float32)
    global_probs_w = np.asarray(model_out["global_probs"], dtype=np.float32)
    embeddings_l2_w = np.asarray(model_out["embedding_l2"], dtype=np.float32)
    complexity_w = np.asarray(model_out["complexity_calibrated"], dtype=np.float32).reshape(-1)
    gate_w = np.asarray(model_out["complexity_gate"], dtype=np.float32).reshape(-1)

    counts_before = proto_state.counts.copy()
    active_mask_before = proto_state.active_mask.copy()
    beta_eff = proto_state.beta_eff()
    rho = proto_state.evidence_ratio()
    proto_probs_w = proto_state.proto_probs_for_embeddings(embeddings_l2_w)
    final_probs_w = beta_eff * global_probs_w + (1.0 - beta_eff) * proto_probs_w

    base_probs_s = base_probs_w.mean(axis=0)
    global_probs_s = global_probs_w.mean(axis=0)
    proto_probs_s = proto_probs_w.mean(axis=0)
    final_probs_s = final_probs_w.mean(axis=0)

    base_pred, base_conf = stage_summary(base_probs_s, class_order)
    global_pred, global_conf = stage_summary(global_probs_s, class_order)
    if proto_state.any_active:
        proto_pred, proto_conf = stage_summary(proto_probs_s, class_order)
    else:
        proto_pred, proto_conf = "No active prototype", float(1.0 / len(class_order))
    final_pred, final_conf = stage_summary(final_probs_s, class_order)

    complexity_session = float(np.mean(complexity_w))
    gate_session = float(np.mean(gate_w))
    proto_active = bool(proto_state.any_active)

    session_index = int((day - 1) * 1000 + session)
    session_id = canonical_session_id(participant, day, session)
    row = {
        "participant_id": participant,
        "participant": participant,
        "day": int(day),
        "session": int(session),
        "session_index": session_index,
        "session_id": session_id,
        "session_file": str(session_file),
        "n_rows": int(len(frame)),
        "analysis_start_row": int(start),
        "analysis_end_row": int(end),
        "analysis_status": extraction_status,
        "watch_pred": watch_pred,
        "confirmed_class": confirmed_class,
        "base_pred": base_pred,
        "global_pred": global_pred,
        "proto_pred": proto_pred,
        "final_pred": final_pred,
        "base_confidence": base_conf,
        "global_confidence": global_conf,
        "proto_confidence": proto_conf,
        "final_confidence": final_conf,
        "p_base_true": float(base_probs_s[confirmed_idx]),
        "p_global_true": float(global_probs_s[confirmed_idx]),
        "p_proto_true": float(proto_probs_s[confirmed_idx]),
        "p_final_true": float(final_probs_s[confirmed_idx]),
        "base_correct": int(base_pred == confirmed_class),
        "global_correct": int(global_pred == confirmed_class),
        "proto_correct": (float(proto_pred == confirmed_class) if proto_active else np.nan),
        "final_correct": int(final_pred == confirmed_class),
        "complexity_session": complexity_session,
        "complexity_gate_session": gate_session,
        "proto_active": proto_active,
        "n_active_prototypes_before": int(active_mask_before.sum()),
        "active_prototype_classes_before": ";".join([class_order[i] for i, flag in enumerate(active_mask_before) if flag]),
        "rho_personal_evidence_before": rho,
        "beta_eff_before": beta_eff,
        "cum_confirmations_before_adl": int(counts_before[confirmed_idx]),
        "total_confirmations_before": int(counts_before.sum()),
    }

    for i, lab in enumerate(class_order):
        suffix = lab.replace(" ", "_").replace("/", "_")
        row[f"n_before_{suffix}"] = int(counts_before[i])
        row[f"active_before_{suffix}"] = bool(active_mask_before[i])
        row[f"p_base_{suffix}"] = float(base_probs_s[i])
        row[f"p_global_{suffix}"] = float(global_probs_s[i])
        row[f"p_proto_{suffix}"] = float(proto_probs_s[i])
        row[f"p_final_{suffix}"] = float(final_probs_s[i])

    window_rows: List[Dict[str, Any]] = []
    for wi in range(windows.shape[0]):
        b_pred, b_conf = stage_summary(base_probs_w[wi], class_order)
        g_pred, g_conf = stage_summary(global_probs_w[wi], class_order)
        if proto_active:
            p_pred, p_conf = stage_summary(proto_probs_w[wi], class_order)
        else:
            p_pred, p_conf = "No active prototype", float(1.0 / len(class_order))
        f_pred, f_conf = stage_summary(final_probs_w[wi], class_order)
        wr = dict(row)
        wr.update({
            "window_idx": int(wi),
            "window_start_row": int(start + wi * cfg.window_size),
            "window_end_row": int(start + (wi + 1) * cfg.window_size),
            "window_base_pred": b_pred,
            "window_global_pred": g_pred,
            "window_proto_pred": p_pred,
            "window_final_pred": f_pred,
            "window_base_confidence": b_conf,
            "window_global_confidence": g_conf,
            "window_proto_confidence": p_conf,
            "window_final_confidence": f_conf,
            "window_p_base_true": float(base_probs_w[wi, confirmed_idx]),
            "window_p_global_true": float(global_probs_w[wi, confirmed_idx]),
            "window_p_proto_true": float(proto_probs_w[wi, confirmed_idx]),
            "window_p_final_true": float(final_probs_w[wi, confirmed_idx]),
            "window_complexity": float(complexity_w[wi]),
            "window_complexity_gate": float(gate_w[wi]),
        })
        window_rows.append(wr)

    z_session = session_embedding_from_windows(embeddings_l2_w, final_probs_w, confirmed_idx)
    proto_state.update(confirmed_idx, z_session, complexity_session)
    row["n_after_confirmed_class"] = int(proto_state.counts[confirmed_idx])
    row["total_confirmations_after"] = int(proto_state.counts.sum())
    for wr in window_rows:
        wr["n_after_confirmed_class"] = row["n_after_confirmed_class"]
        wr["total_confirmations_after"] = row["total_confirmations_after"]

    return row, window_rows, extraction_status


def replay_participant(
    participant_dir: Path,
    model,
    class_order: Sequence[str],
    cfg: ReplayConfig,
    exclude_inconsistent: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    manifest = read_manifest(participant_dir)
    participant = str(manifest["participant"].iloc[0]) if len(manifest) else participant_dir.name
    embedding_dim: Optional[int] = None
    state: Optional[PrototypeState] = None
    session_rows: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    if exclude_inconsistent and "inconsistent" in manifest.columns:
        manifest = manifest[~manifest["inconsistent"].map(boolish)].copy()

    for _, mrow in manifest.iterrows():
        day = coerce_int(mrow.get("day"), -1)
        session = coerce_int(mrow.get("session"), -1)
        try:
            session_file = resolve_session_file(participant_dir, mrow)
            frame = read_motion_session(session_file)
            watch_pred, confirmed = clean_manifest_label(mrow, frame)
            if confirmed not in class_order:
                raise ValueError(f"Confirmed class is invalid: {confirmed!r}; watch_pred={watch_pred!r}")

            # Lazy state creation after first model call determines embedding dimensionality.
            region, _, _, status0 = extract_analysis_region(frame, cfg)
            if status0 == "too_short":
                raise ValueError(f"Session too short for {cfg.analysis_seconds}s analysis region: {len(frame)} rows")
            windows0 = split_three_windows(region, cfg)
            out0 = predict_dict(model, windows0[:1], cfg.batch_size)
            require_model_outputs(out0)
            if embedding_dim is None:
                embedding_dim = int(np.asarray(out0["embedding_l2"]).shape[-1])
                state = PrototypeState(n_classes=len(class_order), embedding_dim=embedding_dim, cfg=cfg)

            row, wrs, status = replay_one_session(
                participant=participant,
                day=day,
                session=session,
                session_file=session_file,
                frame=frame,
                confirmed_class=confirmed,
                watch_pred=watch_pred,
                model=model,
                proto_state=state,
                class_order=class_order,
                cfg=cfg,
            )
            if row:
                if "improving" in mrow.index:
                    row["manifest_improving"] = boolish(mrow.get("improving"))
                    for wr in wrs:
                        wr["manifest_improving"] = row["manifest_improving"]
                if "inconsistent" in mrow.index:
                    row["manifest_inconsistent"] = boolish(mrow.get("inconsistent"))
                    for wr in wrs:
                        wr["manifest_inconsistent"] = row["manifest_inconsistent"]
                session_rows.append(row)
                window_rows.extend(wrs)
        except Exception as exc:
            skipped.append({
                "participant": participant,
                "participant_folder": str(participant_dir),
                "day": day,
                "session": session,
                "file": str(mrow.get("file", "")),
                "reason": str(exc),
            })

    final_state = state.serialise(class_order) if state is not None else {}
    return session_rows, window_rows, skipped, final_state


def validate_replay_frame(df: pd.DataFrame, class_order: Sequence[str]) -> Dict[str, Any]:
    required = [
        "participant_id", "session_index", "confirmed_class",
        "base_pred", "global_pred", "proto_pred", "final_pred",
        "base_correct", "global_correct", "proto_correct", "final_correct",
        "base_confidence", "global_confidence", "proto_confidence", "final_confidence",
        "p_base_true", "p_global_true", "p_proto_true", "p_final_true",
        "complexity_session", "proto_active",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Internal replay output missing required canonical columns: {missing}")

    unknown = sorted(set(df["confirmed_class"].dropna()) - set(class_order))
    if unknown:
        raise RuntimeError(f"Unknown confirmed labels in replay output: {unknown}")

    summary = {
        "n_sessions": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "n_proto_active": int(df["proto_active"].sum()),
        "base_accuracy": float(df["base_correct"].mean()) if len(df) else float("nan"),
        "global_accuracy": float(df["global_correct"].mean()) if len(df) else float("nan"),
        "prototype_accuracy_active_only": float(df.loc[df["proto_active"], "proto_correct"].mean()) if df["proto_active"].any() else float("nan"),
        "final_accuracy": float(df["final_correct"].mean()) if len(df) else float("nan"),
    }
    return summary


def make_dirs(outdir: Path) -> Dict[str, Path]:
    return {
        "root": ensure_dir(outdir),
        "canonical": ensure_dir(outdir / "canonical_code1_outputs"),
        "diagnostics": ensure_dir(outdir / "diagnostics"),
        "metadata": ensure_dir(outdir / "metadata"),
    }


def overlay_config_from_metadata(cfg: ReplayConfig, metadata: Mapping[str, Any]) -> ReplayConfig:
    params = metadata.get("prototype_and_fusion_parameters_for_code02", {}) or {}
    model_cfg = metadata.get("model_config", {}) or {}
    cfg.sampling_hz = int(model_cfg.get("sampling_hz", cfg.sampling_hz))
    cfg.window_seconds = float(model_cfg.get("window_seconds", cfg.window_seconds))
    cfg.N_min = int(params.get("N_min", cfg.N_min))
    cfg.lambda_proto = float(params.get("lambda", cfg.lambda_proto))
    cfg.alpha_min = float(params.get("alpha_min", cfg.alpha_min))
    cfg.alpha_max = float(params.get("alpha_max", cfg.alpha_max))
    cfg.N_ref = int(params.get("N_ref", cfg.N_ref))
    cfg.gamma = float(params.get("gamma", cfg.gamma))
    return cfg


def run(args: argparse.Namespace) -> None:
    phase2_root = Path(args.phase2_root)
    if not phase2_root.exists():
        raise FileNotFoundError(f"Phase 2 root not found: {phase2_root}")

    code1_outdir = Path(args.code1_outdir) if args.code1_outdir else None
    model_path, metadata_path = resolve_code1_paths(
        code1_outdir=code1_outdir,
        model_path=Path(args.model_path) if args.model_path else None,
        metadata_path=Path(args.metadata_path) if args.metadata_path else None,
    )

    metadata = load_json(metadata_path)
    cfg = ReplayConfig()
    cfg = overlay_config_from_metadata(cfg, metadata)
    cfg.analysis_mode = args.analysis_mode
    cfg.short_session_policy = args.short_session_policy
    cfg.batch_size = int(args.batch_size)

    if args.pre_seconds is not None:
        cfg.pre_seconds = float(args.pre_seconds)
    if args.analysis_seconds is not None:
        cfg.analysis_seconds = float(args.analysis_seconds)
    if args.post_seconds is not None:
        cfg.post_seconds = float(args.post_seconds)

    class_order = metadata.get("class_order", ADL_ORDER)
    input_order = metadata.get("input_channel_order", MODEL_COLS)
    if list(input_order) != MODEL_COLS:
        raise ValueError(
            "Code 2 MODEL_COLS do not match Code 1 input_channel_order.\n"
            f"Code 1: {input_order}\nCode 2: {MODEL_COLS}"
        )

    dirs = make_dirs(Path(args.outdir))
    model = load_replay_model(model_path)
    participant_dirs = discover_participant_dirs(phase2_root, args.participant_glob)

    all_sessions: List[Dict[str, Any]] = []
    all_windows: List[Dict[str, Any]] = []
    all_skipped: List[Dict[str, Any]] = []
    final_states: Dict[str, Any] = {}

    print("\nChronological replay")
    print("=" * 78)
    print(f"Phase 2 root: {phase2_root}")
    print(f"Participants discovered: {len(participant_dirs)}")
    print(f"Analysis mode: {cfg.analysis_mode}; analysis samples: {cfg.analysis_size}; windows/session: {cfg.expected_n_windows}")
    print("=" * 78)

    for pdir in participant_dirs:
        print(f"Replaying {pdir.name} ...")
        sr, wr, sk, state = replay_participant(
            participant_dir=pdir,
            model=model,
            class_order=class_order,
            cfg=cfg,
            exclude_inconsistent=bool(args.exclude_inconsistent),
        )
        all_sessions.extend(sr)
        all_windows.extend(wr)
        all_skipped.extend(sk)
        final_states[pdir.name] = state
        print(f"  sessions kept: {len(sr):4d} | windows: {len(wr):4d} | skipped: {len(sk):3d}")

    session_df = pd.DataFrame(all_sessions)
    window_df = pd.DataFrame(all_windows)
    skipped_df = pd.DataFrame(all_skipped)

    if not session_df.empty:
        session_df = session_df.sort_values(["participant_id", "day", "session"], kind="mergesort").reset_index(drop=True)
    if not window_df.empty:
        window_df = window_df.sort_values(["participant_id", "day", "session", "window_idx"], kind="mergesort").reset_index(drop=True)

    summary = validate_replay_frame(session_df, class_order) if not session_df.empty else {"n_sessions": 0}

    session_csv = dirs["canonical"] / "phase2_replay_session_level_CODE1_canonical.csv"
    window_csv = dirs["canonical"] / "phase2_replay_window_level_CODE1_canonical_compatible.csv"
    session_df.to_csv(session_csv, index=False)
    window_df.to_csv(window_csv, index=False)
    print(f"Saved: {session_csv}")
    print(f"Saved: {window_csv}")

    if not skipped_df.empty:
        skipped_csv = dirs["diagnostics"] / "skipped_sessions.csv"
        skipped_df.to_csv(skipped_csv, index=False)
        print(f"Saved: {skipped_csv}")

    by_participant = (
        session_df.groupby("participant_id", dropna=False)
        .agg(
            sessions=("session_id", "count"),
            base_acc=("base_correct", "mean"),
            global_acc=("global_correct", "mean"),
            final_acc=("final_correct", "mean"),
            proto_active_sessions=("proto_active", "sum"),
        )
        .reset_index()
        if not session_df.empty else pd.DataFrame()
    )
    by_participant.to_csv(dirs["diagnostics"] / "participant_replay_summary.csv", index=False)

    by_adl = (
        session_df.groupby("confirmed_class", dropna=False)
        .agg(
            sessions=("session_id", "count"),
            base_acc=("base_correct", "mean"),
            global_acc=("global_correct", "mean"),
            final_acc=("final_correct", "mean"),
            proto_active_sessions=("proto_active", "sum"),
            mean_complexity=("complexity_session", "mean"),
        )
        .reset_index()
        if not session_df.empty else pd.DataFrame()
    )
    by_adl.to_csv(dirs["diagnostics"] / "adl_replay_summary.csv", index=False)

    save_json(final_states, dirs["metadata"] / "prototype_memory_final_state.json")
    manifest = {
        "schema_version": "SmartADAPT-Code2-Phase2Replay-v1",
        "phase2_root": str(phase2_root),
        "code1_model_path": str(model_path),
        "code1_metadata_path": str(metadata_path),
        "outdir": str(args.outdir),
        "config": asdict(cfg),
        "class_order": class_order,
        "input_channel_order": MODEL_COLS,
        "summary": summary,
        "outputs": {
            "session_level_canonical": str(session_csv),
            "window_level_canonical_compatible": str(window_csv),
            "participant_summary": str(dirs["diagnostics"] / "participant_replay_summary.csv"),
            "adl_summary": str(dirs["diagnostics"] / "adl_replay_summary.csv"),
            "skipped_sessions": str(dirs["diagnostics"] / "skipped_sessions.csv"),
        },
    }
    save_json(manifest, dirs["metadata"] / "code02_generation_manifest.json")

    print("\nReplay summary")
    print("=" * 78)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:32s}: {value:.6f}")
        else:
            print(f"{key:32s}: {value}")
    print("=" * 78)
    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SmartADAPT-Net Phase 2 chronological replay")
    p.add_argument("--phase2-root", default="/Users/hubu/Documents/Phase 2 Data_F")
    p.add_argument("--code1-outdir", default="/Users/hubu/Documents/PhD 7/SmartADAPT_code01_phase1_outputs")
    p.add_argument("--model-path", default=None)
    p.add_argument("--metadata-path", default=None)
    p.add_argument("--outdir", default="/Users/hubu/Documents/Phase 2 Data_F/code02_replay_outputs")
    p.add_argument("--participant-glob", default="P*")
    p.add_argument("--analysis-mode", default="after_pre", choices=["after_pre", "center", "from_start", "from_end"])
    p.add_argument("--short-session-policy", default="skip", choices=["skip", "pad"])
    p.add_argument("--pre-seconds", type=float, default=None)
    p.add_argument("--analysis-seconds", type=float, default=None)
    p.add_argument("--post-seconds", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--exclude-inconsistent", action="store_true")
    args, unknown = p.parse_known_args()
    if unknown:
        print("Ignoring extra arguments:", " ".join(unknown))
    return args


if __name__ == "__main__":
    run(parse_args())
