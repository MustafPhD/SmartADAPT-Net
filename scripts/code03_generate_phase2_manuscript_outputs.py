#!/usr/bin/env python3
"""
generate_phase2_manuscript_outputs.py
=====================================

One-stop Phase 2 output generator for the SmartADAPT-Net manuscript.

This script takes the canonical Phase 2 deployment-log replay CSV, collapses the
window-level file to one row per session, validates the manuscript-matching values,
and generates high-resolution PNG figures plus CSV/LaTeX tables for the main
manuscript and supplementary material.

IMPORTANT
---------
Use the canonical CODE1 manuscript-compatible replay file and the non-*_sess
columns. Do not mix this with CODE0 or *_sess columns.

Default input path, matching Mustafa's local machine:
/Users/hubu/Documents/Phase 2 Data_Edited/_phase2_replay_fold01/canonical_code1_outputs/phase2_replay_window_level_CODE1_canonical_compatible.csv

Example
-------
python generate_phase2_manuscript_outputs.py \
  --input "/Users/hubu/Documents/Phase 2 Data_Edited/_phase2_replay_fold01/canonical_code1_outputs/phase2_replay_window_level_CODE1_canonical_compatible.csv" \
  --outdir "/Users/hubu/Documents/Phase 2 Data_Edited/_phase2_replay_fold01/SmartADAPT_phase2_manuscript_outputs" \
  --strict \
  --dpi 600
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

try:
    from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
except ImportError as exc:
    raise ImportError("This script requires scikit-learn. Install with: pip install scikit-learn") from exc


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_INPUT = (
    "/Users/hubu/Documents/Phase 2 Data_Edited/_phase2_replay_fold01/"
    "canonical_code1_outputs/phase2_replay_window_level_CODE1_canonical_compatible.csv"
)

MODEL_NAME = "SmartADAPT-Net"
RANDOM_SEED = 42

ADL_ORDER = [
    "Brushing Teeth",
    "Mopping_Hoovering",
    "Shelving Items",
    "Walking",
    "Washing Dishes",
    "Washing Face",
]

# Order used for some horizontal bar figures so strongest/high-level ADLs appear at top.
ADL_ORDER_PLOT_TOP = [
    "Walking",
    "Mopping_Hoovering",
    "Washing Dishes",
    "Shelving Items",
    "Brushing Teeth",
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
    "Mop/Hoover": "Mopping_Hoovering",
    "Mopping": "Mopping_Hoovering",
    "Shelving Items": "Shelving Items",
    "Shelve Items": "Shelving Items",
    "Shelving": "Shelving Items",
    "Walking": "Walking",
    "Walk": "Walking",
    "Washing Dishes": "Washing Dishes",
    "Wash Dishes": "Washing Dishes",
    "Dishwashing": "Washing Dishes",
    "Washing Face": "Washing Face",
    "Wash Face": "Washing Face",
}

STAGES = ["Base", "Global", "Prototype", "Final"]
STAGE_PRED_COLS = {
    "Base": "base_pred",
    "Global": "global_pred",
    "Prototype": "proto_pred",
    "Final": "final_pred",
}
STAGE_CORRECT_COLS = {
    "Base": "base_correct",
    "Global": "global_correct",
    "Prototype": "proto_correct",
    "Final": "final_correct",
}
STAGE_TRUE_PROB_COLS = {
    "Base": "p_base_true",
    "Global": "p_global_true",
    "Prototype": "p_proto_true",
    "Final": "p_final_true",
}

OUTCOME_ORDER = ["Stable correct", "Recovered", "Degraded", "Persistent fail"]
RECOVERY_SOURCE_ORDER = [
    "Global branch only (CAM)",
    "Prototype branch only (CDP)",
    "Both branches combined",
    "Unclassified",
]

# Expected manuscript values for strict validation.
EXPECTED = {
    "n_sessions": 2523,
    "n_proto_active": 2490,
    "Base": 0.621,
    "Global": 0.784,
    "Prototype": 0.850,
    "Final": 0.890,
}

TOLERANCES = {
    "n_sessions": 0,
    "n_proto_active": 0,
    "Base": 0.002,
    "Global": 0.002,
    "Prototype": 0.002,
    "Final": 0.002,
}

# Centralised colour palette.
COL = {
    "base": "#6BAED6",
    "cam": "#2CA25F",
    "cdp_pos": "#FD8D3C",
    "cdp_neg": "#DE2D26",
    "global": "#756BB1",
    "prototype": "#2CA25F",
    "final": "#E6550D",
    "stable": "#7DB6E8",
    "recovered": "#9EDCCB",
    "degraded": "#F5C469",
    "persistent": "#EF8589",
    "unclassified": "#BDBDBD",
    "positive": "#2B6CB0",
    "negative": "#D73027",
    "neutral": "#888888",
}

STAGE_COLORS = {
    "Base": "#7F7F7F",
    "Global": "#5B52B8",
    "Prototype": "#2B9A78",
    "Final": "#D66A49",
}

OUTCOME_COLORS = {
    "Stable correct": COL["stable"],
    "Recovered": COL["recovered"],
    "Degraded": COL["degraded"],
    "Persistent fail": COL["persistent"],
}

RECOVERY_SOURCE_COLORS = {
    "Global branch only (CAM)": COL["cam"],
    "Prototype branch only (CDP)": COL["cdp_pos"],
    "Both branches combined": "#2B6CB0",
    "Unclassified": COL["unclassified"],
}


# =============================================================================
# General utilities
# =============================================================================

@dataclass
class OutputDirs:
    root: Path
    figures_main: Path
    figures_supp: Path
    tables_main: Path
    tables_supp: Path
    logs: Path


def make_output_dirs(root: str | Path) -> OutputDirs:
    root = Path(root)
    dirs = OutputDirs(
        root=root,
        figures_main=root / "figures_main_png",
        figures_supp=root / "figures_supplementary_png",
        tables_main=root / "tables_main",
        tables_supp=root / "tables_supplementary",
        logs=root / "logs",
    )
    for d in dirs.__dict__.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, outdir: Path, stem: str, dpi: int = 600, save_pdf: bool = True) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{stem}.png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    if save_pdf:
        pdf = outdir / f"{stem}.pdf"
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {png}")


def _latex_cell(value) -> str:
    """Small LaTeX table cell formatter that does not require pandas Styler/jinja2."""
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex_simple(df: pd.DataFrame, index: bool = False) -> str:
    """Write a lightweight booktabs LaTeX tabular without importing jinja2."""
    tab = df.reset_index() if index else df.copy()
    col_format = "l" * len(tab.columns)
    lines = [
        r"\begin{tabular}{" + col_format + r"}",
        r"\toprule",
        " & ".join(_latex_cell(c) for c in tab.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in tab.iterrows():
        lines.append(" & ".join(_latex_cell(v) for v in row.tolist()) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, outdir: Path, stem: str, index: bool = False, latex: bool = True) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{stem}.csv"
    df.to_csv(csv_path, index=index)
    if latex:
        tex_path = outdir / f"{stem}.tex"
        tex = dataframe_to_latex_simple(df, index=index)
        tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved table: {csv_path}")


def clean_label_series(s: pd.Series) -> pd.Series:
    return s.astype(str).replace(LABEL_MAP)


def require_columns(df: pd.DataFrame, cols: Iterable[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"Missing required columns {context}:\n{missing}\n\nAvailable columns:\n{list(df.columns)}"
        raise ValueError(msg)


def infer_session_keys(df: pd.DataFrame) -> list[str]:
    candidates = [
        ["participant_id", "session_index"],
        ["participant", "session_index"],
        ["participant_id", "session_file"],
        ["participant", "session_file"],
        ["session_file"],
    ]
    for keys in candidates:
        if all(k in df.columns for k in keys):
            return keys
    raise ValueError("Could not infer session keys. Expected participant_id + session_index, or participant + session_file.")


def ci_text(value: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"


def mean_sd_text(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def safe_mean(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").mean())


# =============================================================================
# Loading, cleaning, validation
# =============================================================================

def load_phase2_session_data(input_file: str | Path, strict: bool = False) -> pd.DataFrame:
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    raw = pd.read_parquet(input_file) if input_file.suffix.lower() == ".parquet" else pd.read_csv(input_file)

    print("\nLoaded raw replay file")
    print("=" * 78)
    print(f"File: {input_file}")
    print(f"Raw rows: {len(raw):,}")
    print(f"Raw columns: {len(raw.columns):,}")

    session_keys = infer_session_keys(raw)
    sort_cols = session_keys + (["window_idx"] if "window_idx" in raw.columns else [])
    raw = raw.sort_values(sort_cols)
    df = raw.drop_duplicates(subset=session_keys, keep="first").copy()
    print(f"Session keys: {session_keys}")
    print(f"Session-level rows after de-duplication: {len(df):,}")

    required = [
        "confirmed_class",
        "base_pred", "global_pred", "proto_pred", "final_pred",
        "base_correct", "global_correct", "proto_correct", "final_correct",
        "base_confidence", "global_confidence", "final_confidence",
        "p_base_true", "p_global_true", "p_proto_true", "p_final_true",
        "complexity_session", "proto_active",
    ]
    require_columns(df, required, context="for manuscript-compatible Phase 2 replay")

    sess_cols = [c for c in df.columns if c.endswith("_sess")]
    if sess_cols:
        print("\nSafety note: *_sess columns detected and deliberately ignored for manuscript outputs.")

    if "participant_id" not in df.columns and "participant" in df.columns:
        df["participant_id"] = df["participant"]

    df["confirmed_class"] = clean_label_series(df["confirmed_class"])
    for c in ["base_pred", "global_pred", "proto_pred", "final_pred"]:
        df[c] = clean_label_series(df[c])

    # Convert numerics.
    num_cols = [
        "base_correct", "global_correct", "proto_correct", "final_correct",
        "base_confidence", "global_confidence", "proto_confidence", "final_confidence",
        "p_base_true", "p_global_true", "p_proto_true", "p_final_true",
        "complexity_session", "gain_global_minus_base", "gain_proto_minus_base", "gain_final_minus_base",
    ]
    for c in [x for x in num_cols if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["proto_active"] = df["proto_active"].astype(bool)

    # Recompute correctness from predictions. This protects against stale flags.
    df["base_correct"] = (df["base_pred"] == df["confirmed_class"]).astype(int)
    df["global_correct"] = (df["global_pred"] == df["confirmed_class"]).astype(int)
    df["final_correct"] = (df["final_pred"] == df["confirmed_class"]).astype(int)
    df["proto_correct"] = np.where(
        df["proto_active"],
        (df["proto_pred"] == df["confirmed_class"]).astype(float),
        np.nan,
    )

    unknown = sorted(set(df["confirmed_class"].dropna()) - set(ADL_ORDER))
    if unknown:
        raise ValueError(f"Unknown ADL labels after cleaning: {unknown}")
    df["confirmed_class"] = pd.Categorical(df["confirmed_class"], categories=ADL_ORDER, ordered=True)

    # Outcomes.
    conditions = [
        (df["base_correct"] == 1) & (df["final_correct"] == 1),
        (df["base_correct"] == 0) & (df["final_correct"] == 1),
        (df["base_correct"] == 1) & (df["final_correct"] == 0),
        (df["base_correct"] == 0) & (df["final_correct"] == 0),
    ]
    df["decision_outcome"] = np.select(conditions, OUTCOME_ORDER, default="Unknown")

    # Gains.
    df["gain_global_minus_base"] = df["p_global_true"] - df["p_base_true"]
    df["gain_proto_minus_base"] = df["p_proto_true"] - df["p_base_true"]
    df["gain_final_minus_base"] = df["p_final_true"] - df["p_base_true"]

    # Complexity bands.
    df["complexity_quartile"] = pd.qcut(
        df["complexity_session"].rank(method="first"),
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    )
    df["complexity_band_3"] = pd.qcut(
        df["complexity_session"].rank(method="first"),
        q=3,
        labels=["Low complexity", "Medium complexity", "High complexity"],
    )
    df["baseline_conf_band"] = pd.cut(
        df["base_confidence"],
        bins=[-np.inf, 0.50, 0.70, 0.85, np.inf],
        labels=["Very low\n<0.50", "Low/moderate\n0.50–0.70", "High\n0.70–0.85", "Very high\n>0.85"],
        include_lowest=True,
    )

    # Deployment blocks.
    if "deployment_block" in df.columns:
        df["deployment_block_display"] = df["deployment_block"].astype(str)
    elif "block" in df.columns:
        block_map = {1: "1–5", 2: "6–10", 3: "11–15", 4: "16–20", "1": "1–5", "2": "6–10", "3": "11–15", "4": "16–20"}
        df["deployment_block_display"] = df["block"].map(block_map).fillna(df["block"].astype(str))
    elif "day" in df.columns:
        d = pd.to_numeric(df["day"], errors="coerce")
        df["deployment_block_display"] = pd.cut(
            d,
            bins=[0, 5, 10, 15, 20],
            labels=["1–5", "6–10", "11–15", "16–20"],
            include_lowest=True,
        ).astype(str)

    # Cumulative confirmations before current session within participant and ADL.
    # This is used for CDP learning/exposure figures.
    chronological_cols = ["participant_id"]
    if "session_index" in df.columns:
        chronological_cols.append("session_index")
    elif "day" in df.columns:
        chronological_cols.append("day")
    df = df.sort_values(chronological_cols).copy()
    df["cum_confirmations_before_adl"] = df.groupby(["participant_id", "confirmed_class"], observed=True).cumcount()
    df["confirmation_bin"] = pd.cut(
        df["cum_confirmations_before_adl"],
        bins=[-1, 4, 9, 19, 39, np.inf],
        labels=["0–4", "5–9", "10–19", "20–39", "40+"],
        include_lowest=True,
    )

    # Recovery source categories.
    df["recovery_source"] = classify_recovery_source(df)

    validate_against_expected(df, strict=strict)
    return df


def classify_recovery_source(df: pd.DataFrame) -> pd.Series:
    """Classify recovered sessions by the mechanism that would recover them.

    Uses the existing *_recovered_demo columns when present. Otherwise derives the
    source from Global and Prototype correctness among Base-error/Final-correct sessions.

    Unclassified means the Final decision recovered the Base-stage error, but neither
    Global alone nor Prototype alone clearly explains the correction. This usually
    corresponds to a recovery attributable to final fusion or another non-unique source.
    """
    recovered = (df["base_correct"] == 0) & (df["final_correct"] == 1)
    source = pd.Series("Not recovered", index=df.index, dtype="object")

    demo_cols = {
        "Global branch only (CAM)": "global_only_recovered_demo",
        "Prototype branch only (CDP)": "prototype_only_recovered_demo",
        "Both branches combined": "both_branches_recovered_demo",
        "Unclassified": "fusion_only_recovered_demo",
    }
    if all(c in df.columns for c in demo_cols.values()):
        for label, col in demo_cols.items():
            source.loc[recovered & (pd.to_numeric(df[col], errors="coerce").fillna(0) == 1)] = label
        source.loc[recovered & (source == "Not recovered")] = "Unclassified"
        return source

    # Fallback derivation.
    global_ok = df["global_correct"].fillna(0).astype(int) == 1
    proto_ok = df["proto_correct"].fillna(0).astype(float) == 1
    source.loc[recovered & global_ok & ~proto_ok] = "Global branch only (CAM)"
    source.loc[recovered & ~global_ok & proto_ok] = "Prototype branch only (CDP)"
    source.loc[recovered & global_ok & proto_ok] = "Both branches combined"
    source.loc[recovered & ~global_ok & ~proto_ok] = "Unclassified"
    return source


def validate_against_expected(df: pd.DataFrame, strict: bool = False) -> dict[str, float]:
    summary = {
        "n_sessions": len(df),
        "n_proto_active": int(df["proto_active"].sum()),
        "Base": float(df["base_correct"].mean()),
        "Global": float(df["global_correct"].mean()),
        "Prototype": float(df.loc[df["proto_active"], "proto_correct"].mean()),
        "Final": float(df["final_correct"].mean()),
    }

    print("\nValidation against expected manuscript values")
    print("=" * 78)
    for k, v in summary.items():
        if k.startswith("n_"):
            print(f"{k:18s}: {int(v):,}   expected {EXPECTED[k]:,}")
        else:
            print(f"{k:18s}: {v:.6f} expected ≈ {EXPECTED[k]:.3f}")
    print("=" * 78)

    problems = []
    for k, expected in EXPECTED.items():
        if abs(summary[k] - expected) > TOLERANCES[k]:
            problems.append(f"{k}: found {summary[k]}, expected {expected}")

    if problems:
        msg = "WARNING: manuscript validation mismatch:\n" + "\n".join(problems)
        if strict:
            raise ValueError(msg)
        print(msg)
    else:
        print("Validation passed: this file reproduces the manuscript Phase 2 values.")
    return summary


# =============================================================================
# Bootstrap and statistics
# =============================================================================

def bootstrap_ci(
    df: pd.DataFrame,
    stat_func: Callable[[pd.DataFrame], float],
    cluster_col: str = "participant_id",
    n_boot: int = 2000,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].dropna().unique()
    if len(clusters) == 0:
        return np.nan, np.nan

    vals: list[float] = []
    grouped = {k: g for k, g in df.groupby(cluster_col, observed=True)}
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[c] for c in sampled], ignore_index=True)
        try:
            val = stat_func(boot)
        except Exception:
            val = np.nan
        if pd.notna(val):
            vals.append(float(val))

    if not vals:
        return np.nan, np.nan
    return tuple(np.percentile(vals, [2.5, 97.5]))


def stage_accuracy(df: pd.DataFrame, stage: str) -> float:
    if stage == "Prototype":
        sub = df[df["proto_active"]]
        return float(sub["proto_correct"].mean())
    return float(df[STAGE_CORRECT_COLS[stage]].mean())


def macro_f1_for_stage(df: pd.DataFrame, stage: str) -> float:
    sub = df[df["proto_active"]].copy() if stage == "Prototype" else df.copy()
    y_true = sub["confirmed_class"].astype(str)
    y_pred = sub[STAGE_PRED_COLS[stage]].astype(str)
    return float(f1_score(y_true, y_pred, labels=ADL_ORDER, average="macro", zero_division=0))


def balanced_acc_for_stage(df: pd.DataFrame, stage: str) -> float:
    sub = df[df["proto_active"]].copy() if stage == "Prototype" else df.copy()
    y_true = sub["confirmed_class"].astype(str)
    y_pred = sub[STAGE_PRED_COLS[stage]].astype(str)
    return float(balanced_accuracy_score(y_true, y_pred))


# =============================================================================
# Tables
# =============================================================================

def table_phase2_adl_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for adl in ADL_ORDER:
        sub = df[df["confirmed_class"] == adl]
        rows.append({
            "ADL": ADL_DISPLAY[adl],
            "Sessions": len(sub),
            "Prototype-active sessions": int(sub["proto_active"].sum()),
            "Windows/session": 3.0,
            "Base Acc.": sub["base_correct"].mean(),
            "Global/CAM Acc.": sub["global_correct"].mean(),
            "Final Acc.": sub["final_correct"].mean(),
            "Final p_true": sub["p_final_true"].mean(),
        })
    overall = {
        "ADL": "Overall",
        "Sessions": len(df),
        "Prototype-active sessions": int(df["proto_active"].sum()),
        "Windows/session": 3.0,
        "Base Acc.": df["base_correct"].mean(),
        "Global/CAM Acc.": df["global_correct"].mean(),
        "Final Acc.": df["final_correct"].mean(),
        "Final p_true": np.nan,
    }
    return pd.DataFrame(rows + [overall])


def table_stage_performance(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for stage in ["Base", "Global", "Prototype", "Final"]:
        if stage == "Prototype":
            sub = df[df["proto_active"]]
            acc = float(sub["proto_correct"].mean())
            lo, hi = bootstrap_ci(sub, lambda x: float(x["proto_correct"].mean()), n_boot=n_boot)
            rows.append({
                "Stage": "Prototype",
                "Subset": "Prototype-active only",
                "Sessions": len(sub),
                "Accuracy": acc,
                "95% CI": f"{lo:.3f}–{hi:.3f}",
            })
            matched_base = float(sub["base_correct"].mean())
            lo_b, hi_b = bootstrap_ci(sub, lambda x: float(x["base_correct"].mean()), n_boot=n_boot)
            rows.append({
                "Stage": "Base (matched)",
                "Subset": "Prototype-active only",
                "Sessions": len(sub),
                "Accuracy": matched_base,
                "95% CI": f"{lo_b:.3f}–{hi_b:.3f}",
            })
        else:
            col = STAGE_CORRECT_COLS[stage]
            acc = float(df[col].mean())
            lo, hi = bootstrap_ci(df, lambda x, c=col: float(x[c].mean()), n_boot=n_boot)
            rows.append({
                "Stage": stage if stage != "Global" else "Global/CAM",
                "Subset": "All sessions",
                "Sessions": len(df),
                "Accuracy": acc,
                "95% CI": f"{lo:.3f}–{hi:.3f}",
            })
    return pd.DataFrame(rows)


def table_integrated_adaptive_behaviour(df: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    recovered = (df["base_correct"] == 0) & (df["final_correct"] == 1)
    degraded = (df["base_correct"] == 1) & (df["final_correct"] == 0)
    base_errors = int((df["base_correct"] == 0).sum())
    base_correct = int((df["base_correct"] == 1).sum())

    participant_stage = (
        df.groupby("participant_id", observed=True)
        .agg(
            Base=("base_correct", "mean"),
            Global=("global_correct", "mean"),
            Final=("final_correct", "mean"),
            Sessions=("final_correct", "size"),
        )
        .reset_index()
    )
    participant_stage["Final_minus_Base"] = participant_stage["Final"] - participant_stage["Base"]

    rows = []
    for metric_name, func in [
        ("Session-level accuracy", lambda stage: stage_accuracy(df, stage)),
        ("Session-level macro-F1", lambda stage: macro_f1_for_stage(df, stage)),
        ("Session-level balanced accuracy", lambda stage: balanced_acc_for_stage(df, stage)),
    ]:
        row = {"Analysis": metric_name}
        for stage in ["Base", "Global", "Final"]:
            value = func(stage)
            if metric_name == "Session-level accuracy":
                col = STAGE_CORRECT_COLS[stage]
                lo, hi = bootstrap_ci(df, lambda x, c=col: float(x[c].mean()), n_boot=n_boot)
            elif metric_name == "Session-level macro-F1":
                lo, hi = bootstrap_ci(df, lambda x, s=stage: macro_f1_for_stage(x, s), n_boot=n_boot)
            else:
                lo, hi = bootstrap_ci(df, lambda x, s=stage: balanced_acc_for_stage(x, s), n_boot=n_boot)
            row[stage if stage != "Global" else "Global/CAM"] = ci_text(value, lo, hi)
        rows.append(row)

    rows.append({
        "Analysis": "Participant-level accuracy, mean ± SD",
        "Base": mean_sd_text(participant_stage["Base"].mean(), participant_stage["Base"].std(ddof=1)),
        "Global/CAM": mean_sd_text(participant_stage["Global"].mean(), participant_stage["Global"].std(ddof=1)),
        "Final": mean_sd_text(participant_stage["Final"].mean(), participant_stage["Final"].std(ddof=1)),
    })

    rows.append({
        "Analysis": "Participants improved from Base to Final",
        "Base": "–",
        "Global/CAM": "–",
        "Final": f"{int((participant_stage['Final_minus_Base'] > 0).sum())}/{len(participant_stage)} improved; "
                 f"{int((participant_stage['Final_minus_Base'] == 0).sum())}/{len(participant_stage)} unchanged; "
                 f"{int((participant_stage['Final_minus_Base'] < 0).sum())}/{len(participant_stage)} decreased",
    })

    rows.append({
        "Analysis": "Mean participant-level accuracy gain",
        "Base": "Reference",
        "Global/CAM": "–",
        "Final": mean_sd_text(participant_stage["Final_minus_Base"].mean(), participant_stage["Final_minus_Base"].std(ddof=1)),
    })

    recovery_eff = recovered.sum() / base_errors if base_errors else np.nan
    degr_rate = degraded.sum() / base_correct if base_correct else np.nan
    rows += [
        {"Analysis": "Base errors recovered by Final", "Base": f"{base_errors} Base errors", "Global/CAM": "–", "Final": f"{int(recovered.sum())} recovered"},
        {"Analysis": "Recovery efficiency", "Base": "–", "Global/CAM": "–", "Final": f"{recovery_eff:.3f}"},
        {"Analysis": "Base-correct sessions degraded by Final", "Base": f"{base_correct} Base-correct sessions", "Global/CAM": "–", "Final": f"{int(degraded.sum())} degraded"},
        {"Analysis": "Degradation rate", "Base": "–", "Global/CAM": "–", "Final": f"{degr_rate:.3f}"},
        {"Analysis": "Net recovered decisions", "Base": "–", "Global/CAM": "–", "Final": f"+{int(recovered.sum() - degraded.sum())} sessions"},
    ]
    return pd.DataFrame(rows)


def table_stage_contribution_per_adl(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for adl in ADL_ORDER:
        sub = df[df["confirmed_class"] == adl]
        base = float(sub["base_correct"].mean())
        global_acc = float(sub["global_correct"].mean())
        final = float(sub["final_correct"].mean())
        rows.append({
            "ADL": ADL_DISPLAY[adl],
            "Sessions": len(sub),
            "Base accuracy": base,
            "Global/CAM accuracy": global_acc,
            "Final accuracy": final,
            "CAM gain (Global - Base)": global_acc - base,
            "CDP/final increment (Final - Global)": final - global_acc,
            "Total adaptive gain (Final - Base)": final - base,
        })
    return pd.DataFrame(rows)


def supplementary_tables(df: pd.DataFrame, n_boot: int) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    # Participant-stage accuracies.
    ptab = (
        df.groupby("participant_id", observed=True)
        .agg(
            Sessions=("final_correct", "size"),
            Base=("base_correct", "mean"),
            Global_CAM=("global_correct", "mean"),
            Final=("final_correct", "mean"),
        )
        .reset_index()
    )
    ptab["Final_minus_Base"] = ptab["Final"] - ptab["Base"]
    ptab["Global_minus_Base"] = ptab["Global_CAM"] - ptab["Base"]
    ptab["Final_minus_Global"] = ptab["Final"] - ptab["Global_CAM"]
    tables["supp_table_participant_stage_accuracy"] = ptab

    # Outcome counts by ADL.
    out = pd.crosstab(df["confirmed_class"], df["decision_outcome"]).reindex(index=ADL_ORDER, columns=OUTCOME_ORDER, fill_value=0)
    out.insert(0, "ADL", [ADL_DISPLAY[a] for a in out.index])
    tables["supp_table_outcome_counts_by_adl"] = out.reset_index(drop=True)

    # Recovery source counts.
    rec = df[df["decision_outcome"] == "Recovered"].copy()
    src = pd.crosstab(rec["confirmed_class"], rec["recovery_source"]).reindex(index=ADL_ORDER, columns=RECOVERY_SOURCE_ORDER, fill_value=0)
    src.insert(0, "ADL", [ADL_DISPLAY[a] for a in src.index])
    src.insert(1, "Recovered sessions", src[RECOVERY_SOURCE_ORDER].sum(axis=1).values)
    tables["supp_table_recovery_source_counts_by_adl"] = src.reset_index(drop=True)

    # Per-ADL stage metrics.
    rows = []
    for adl in ADL_ORDER:
        sub = df[df["confirmed_class"] == adl]
        row = {"ADL": ADL_DISPLAY[adl], "Sessions": len(sub)}
        for stage, col in [("Base", "base_correct"), ("Global/CAM", "global_correct"), ("Final", "final_correct")]:
            value = float(sub[col].mean())
            lo, hi = bootstrap_ci(sub, lambda x, c=col: float(x[c].mean()), n_boot=n_boot)
            row[f"{stage} accuracy"] = value
            row[f"{stage} 95% CI"] = f"{lo:.3f}–{hi:.3f}"
        row["CAM gain"] = row["Global/CAM accuracy"] - row["Base accuracy"]
        row["CDP/final increment"] = row["Final accuracy"] - row["Global/CAM accuracy"]
        row["Total adaptive gain"] = row["Final accuracy"] - row["Base accuracy"]
        rows.append(row)
    tables["supp_table_adl_stage_accuracy_ci"] = pd.DataFrame(rows)

    # Confusion pairs by stage.
    pair_rows = []
    for stage in ["Base", "Global", "Final"]:
        pred_col = STAGE_PRED_COLS[stage]
        err = df[df["confirmed_class"].astype(str) != df[pred_col].astype(str)]
        pairs = (
            err.groupby(["confirmed_class", pred_col], observed=True)
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
            .head(20)
        )
        for _, r in pairs.iterrows():
            pair_rows.append({
                "Stage": stage if stage != "Global" else "Global/CAM",
                "Confirmed ADL": ADL_DISPLAY.get(str(r["confirmed_class"]), str(r["confirmed_class"])),
                "Predicted ADL": ADL_DISPLAY.get(str(r[pred_col]), str(r[pred_col])),
                "Count": int(r["Count"]),
            })
    tables["supp_table_top_confusion_pairs_by_stage"] = pd.DataFrame(pair_rows)

    # Prototype activation diagnostics by ADL.
    diag_rows = []
    for adl in ADL_ORDER:
        sub = df[df["confirmed_class"] == adl]
        diag_rows.append({
            "ADL": ADL_DISPLAY[adl],
            "Sessions": len(sub),
            "Prototype-active sessions": int(sub["proto_active"].sum()),
            "Prototype activation rate": float(sub["proto_active"].mean()),
            "Mean cumulative confirmations before session": float(sub["cum_confirmations_before_adl"].mean()),
            "Median cumulative confirmations before session": float(sub["cum_confirmations_before_adl"].median()),
            "Mean complexity_session": float(sub["complexity_session"].mean()),
            "Median complexity_session": float(sub["complexity_session"].median()),
        })
    tables["supp_table_prototype_activation_and_complexity_by_adl"] = pd.DataFrame(diag_rows)

    # CDP bins table.
    bin_rows = []
    for adl in ADL_ORDER:
        for b in ["0–4", "5–9", "10–19", "20–39", "40+"]:
            sub = df[(df["confirmed_class"] == adl) & (df["confirmation_bin"].astype(str) == b)]
            if len(sub) == 0:
                continue
            diff = float((sub["final_correct"] - sub["global_correct"]).mean())
            lo, hi = bootstrap_ci(sub, lambda x: float((x["final_correct"] - x["global_correct"]).mean()), n_boot=n_boot)
            bin_rows.append({
                "ADL": ADL_DISPLAY[adl],
                "Confirmation bin": b,
                "Sessions": len(sub),
                "CDP gain (Final - Global accuracy)": diff,
                "95% CI": f"{lo:.3f}–{hi:.3f}",
            })
    tables["supp_table_cdp_gain_confirmation_bins"] = pd.DataFrame(bin_rows)

    return tables


# =============================================================================
# Figures
# =============================================================================

def _confusion_matrix_panel(ax, df: pd.DataFrame, stage: str, normalize: bool = False, show_ylabel: bool = True):
    sub = df[df["proto_active"]].copy() if stage == "Prototype" else df.copy()
    cm = confusion_matrix(sub["confirmed_class"].astype(str), sub[STAGE_PRED_COLS[stage]].astype(str), labels=ADL_ORDER)
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
        text = np.vectorize(lambda x: f"{x:.2f}")(cm_plot)
        vmax = 1.0
        cbar_label = "Row-normalised proportion"
    else:
        cm_plot = cm
        text = cm.astype(str)
        vmax = None
        cbar_label = "Session count"

    im = ax.imshow(cm_plot, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(ADL_ORDER)))
    ax.set_yticks(np.arange(len(ADL_ORDER)))
    ax.set_xticklabels([ADL_DISPLAY[a] for a in ADL_ORDER], rotation=45, ha="right")
    ax.set_yticklabels([ADL_DISPLAY[a] for a in ADL_ORDER] if show_ylabel else [])
    ax.set_xlabel("Predicted ADL")
    if show_ylabel:
        ax.set_ylabel("Confirmed ADL")
    threshold = (cm_plot.max() if cm_plot.size else 1) * 0.55
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            ax.text(j, i, text[i, j], ha="center", va="center", color="white" if cm_plot[i, j] > threshold else "black", fontsize=7.5)
    return im, cbar_label


def plot_final_confusion_matrix(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im, label = _confusion_matrix_panel(ax, df, "Final", normalize=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    plt.tight_layout()
    save_figure(fig, outdir, "fig05_final_deployed_confusion_matrix_phase2", dpi, save_pdf)


def plot_participantwise_final_accuracy_heatmap(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    heat = (
        df.groupby(["participant_id", "confirmed_class"], observed=True)["final_correct"]
        .mean()
        .unstack()
        .reindex(columns=ADL_ORDER)
    )
    order = heat.mean(axis=1).sort_values(ascending=False).index
    heat = heat.loc[order]

    fig_h = max(5.5, 0.36 * len(heat) + 1.8)
    fig, ax = plt.subplots(figsize=(9.8, fig_h))
    im = ax.imshow(heat.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(ADL_ORDER)))
    ax.set_xticklabels([ADL_DISPLAY[a] for a in ADL_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index.astype(str))
    ax.set_xlabel("ADL")
    ax.set_ylabel("Participant")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.2, color="white" if val >= 0.68 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Final session-level accuracy")
    plt.tight_layout()
    save_figure(fig, outdir, "fig07_participantwise_final_accuracy_by_adl", dpi, save_pdf)


def plot_stage_confusion_matrices(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool, normalize: bool = False) -> None:
    stages = ["Base", "Global", "Final"]
    labels = ["A. Base", "B. Global/CAM", "C. Final"]
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.1))
    ims = []
    cbar_label = "Session count"
    for i, (ax, stage, lab) in enumerate(zip(axes, stages, labels)):
        im, cbar_label = _confusion_matrix_panel(ax, df, stage, normalize=normalize, show_ylabel=(i == 0))
        ims.append(im)
        ax.text(0.02, 1.04, lab, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.16, right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    cbar = fig.colorbar(ims[-1], cax=cbar_ax)
    cbar.set_label(cbar_label)
    stem = "fig08_deployment_log_replay_confusion_matrices_base_global_final"
    if normalize:
        stem = "supp_row_normalised_confusion_matrices_base_global_final"
    save_figure(fig, outdir, stem, dpi, save_pdf)


def plot_stage_contribution_per_adl(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    summary = table_stage_contribution_per_adl(df).set_index("ADL")
    # Convert display ADL back to canonical plotting order.
    plot_labels = [ADL_DISPLAY[a] for a in ADL_ORDER_PLOT_TOP]
    summary = summary.loc[plot_labels]

    base = summary["Base accuracy"].values
    cam = summary["CAM gain (Global - Base)"].values
    cdp = summary["CDP/final increment (Final - Global)"].values
    final = summary["Final accuracy"].values
    y = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    ax.barh(y, base, color=COL["base"], label="Base accuracy")
    ax.barh(y, np.clip(cam, 0, None), left=base, color=COL["cam"], label="CAM gain (Global - Base)")
    # Negative CAM is rare but handled.
    neg_cam = np.clip(cam, None, 0)
    if np.any(neg_cam < 0):
        ax.barh(y, -neg_cam, left=base + neg_cam, color=COL["cdp_neg"], label="CAM negative")

    cdp_pos = np.clip(cdp, 0, None)
    cdp_neg = np.clip(cdp, None, 0)
    ax.barh(y, cdp_pos, left=base + np.clip(cam, 0, None), color=COL["cdp_pos"], label="CDP/final increment positive")
    if np.any(cdp_neg < 0):
        ax.barh(y, -cdp_neg, left=base + cam + cdp_neg, color=COL["cdp_neg"], label="CDP/final increment negative")

    for yi, b, f in zip(y, base, final):
        ax.text(max(0.04, b * 0.55), yi, f"{b:.2f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.text(f + 0.012, yi, f"{f:.2f}", va="center", ha="left", fontsize=10)

    ax.axvline(df["base_correct"].mean(), color="0.65", linestyle="--", linewidth=1.0)
    ax.text(df["base_correct"].mean(), -0.6, f"Overall Base = {df['base_correct'].mean():.2f}", ha="center", va="bottom", color="0.45", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index.tolist())
    ax.set_xlabel("Session-level accuracy")
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.55, -0.24))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, outdir, "fig_stage_contribution_per_adl_cam_cdp_decomposition", dpi, save_pdf)


def plot_temporal_gain_across_blocks(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool, n_boot: int) -> None:
    if "deployment_block_display" not in df.columns:
        print("Skipping temporal gain: no deployment block column found.")
        return
    block_order = ["1-5", "6-10", "11-15", "16-20"]
    # Normalise en-dash variants.
    df = df.copy()
    df["deployment_block_display"] = df["deployment_block_display"].astype(str).str.replace("–", "-", regex=False)
    block_order = [b for b in block_order if b in set(df["deployment_block_display"])]
    if not block_order:
        block_order = sorted(df["deployment_block_display"].dropna().unique().tolist())

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.6), sharey=True)
    axes = axes.ravel()
    for ax, adl in zip(axes, ADL_ORDER):
        sub_adl = df[df["confirmed_class"] == adl]
        means, los, his = [], [], []
        for block in block_order:
            sub = sub_adl[sub_adl["deployment_block_display"] == block]
            if len(sub) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            # Participant-level mean gain then bootstrap participants.
            pg = sub.assign(gain=sub["final_correct"] - sub["base_correct"]).groupby("participant_id", observed=True)["gain"].mean().reset_index()
            mean = float(pg["gain"].mean())
            lo, hi = bootstrap_ci(pg, lambda x: float(x["gain"].mean()), cluster_col="participant_id", n_boot=n_boot)
            means.append(mean); los.append(lo); his.append(hi)
        x = np.arange(len(block_order))
        ax.plot(x, means, marker="o", linewidth=2.2, color=STAGE_COLORS["Final"])
        ax.fill_between(x, los, his, color=STAGE_COLORS["Final"], alpha=0.18, linewidth=0)
        ax.axhline(0, linestyle="--", color="0.4", linewidth=0.8)
        ax.set_title(ADL_DISPLAY[adl], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(block_order)
        ax.set_ylim(-0.05, 0.75)
        ax.grid(True, alpha=0.2)
        if ax in axes[3:]:
            ax.set_xlabel("Deployment block")
        if ax in [axes[0], axes[3]]:
            ax.set_ylabel("Final - Base accuracy gain")
    plt.tight_layout()
    save_figure(fig, outdir, "fig09_final_base_accuracy_gain_across_deployment_blocks", dpi, save_pdf)


def plot_outcome_breakdown_by_adl(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    counts = pd.crosstab(df["confirmed_class"], df["decision_outcome"]).reindex(index=ADL_ORDER, columns=OUTCOME_ORDER, fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(11.6, 5.5))
    y = np.arange(len(ADL_ORDER))
    left = np.zeros(len(ADL_ORDER))
    for outcome in OUTCOME_ORDER:
        vals = props[outcome].values
        ax.barh(y, vals, left=left, label=outcome, color=OUTCOME_COLORS[outcome])
        for i, v in enumerate(vals):
            if v >= 0.075:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=9)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([ADL_DISPLAY[a] for a in ADL_ORDER])
    ax.set_xlabel("Proportion of sessions")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, outdir, "fig10_outcome_breakdown_stable_recovered_degraded_persistent", dpi, save_pdf)


def plot_recovery_source_per_adl(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    rec = df[df["decision_outcome"] == "Recovered"].copy()
    counts = pd.crosstab(rec["confirmed_class"], rec["recovery_source"]).reindex(index=ADL_ORDER_PLOT_TOP, columns=RECOVERY_SOURCE_ORDER, fill_value=0)
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(11.8, 5.3))
    y = np.arange(len(ADL_ORDER_PLOT_TOP))
    left = np.zeros(len(ADL_ORDER_PLOT_TOP))
    for src in RECOVERY_SOURCE_ORDER:
        vals = props[src].values
        ax.barh(y, vals, left=left, label=src, color=RECOVERY_SOURCE_COLORS[src])
        for i, v in enumerate(vals):
            if v >= 0.07:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white" if src != "Unclassified" else "black", fontweight="bold")
        left += vals
    totals = counts.sum(axis=1).values
    for i, n in enumerate(totals):
        ax.text(1.02, i, f"n={int(n)}", va="center", ha="left", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([ADL_DISPLAY[a] for a in ADL_ORDER_PLOT_TOP])
    ax.set_xlabel("Proportion of recovered sessions")
    ax.set_xlim(0, 1.14)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, outdir, "fig_recovery_source_per_adl_cam_cdp_both_unclassified", dpi, save_pdf)


def plot_cdp_gain_confirmation_bins(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool, n_boot: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.8), sharey=True)
    axes = axes.ravel()
    bins = ["0–4", "5–9", "10–19", "20–39", "40+"]
    for ax, adl in zip(axes, ADL_ORDER):
        vals, los, his, ns = [], [], [], []
        for b in bins:
            sub = df[(df["confirmed_class"] == adl) & (df["confirmation_bin"].astype(str) == b)]
            ns.append(len(sub))
            if len(sub) == 0:
                vals.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            value = float((sub["final_correct"] - sub["global_correct"]).mean())
            lo, hi = bootstrap_ci(sub, lambda x: float((x["final_correct"] - x["global_correct"]).mean()), n_boot=n_boot)
            vals.append(value); los.append(lo); his.append(hi)
        x = np.arange(len(bins))
        low_err = np.maximum(0, np.array(vals, dtype=float) - np.array(los, dtype=float))
        high_err = np.maximum(0, np.array(his, dtype=float) - np.array(vals, dtype=float))
        yerr = np.vstack([low_err, high_err])
        ax.errorbar(x, vals, yerr=yerr, marker="o", linewidth=2.0, capsize=3, color="#2B6CB0")
        ax.axhline(0, color="0.5", linestyle="--", linewidth=0.9)
        ax.axhspan(-0.02, 0.02, color="0.85", alpha=0.35, linewidth=0)
        overall = float((df.loc[df["confirmed_class"] == adl, "final_correct"] - df.loc[df["confirmed_class"] == adl, "global_correct"]).mean())
        if overall <= 0.02:
            ax.set_facecolor("#FFF2F2")
        ax.set_title(f"{ADL_DISPLAY[adl]}\n(overall CDP/final increment: {overall:+.3f})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_ylim(-0.20, 0.35)
        ax.grid(True, axis="y", alpha=0.2)
        for xi, n in zip(x, ns):
            ax.text(xi, -0.185, f"n={n}", ha="center", va="bottom", fontsize=7.5, color="0.45")
        if ax in axes[3:]:
            ax.set_xlabel("Cumulative confirmations before session")
        if ax in [axes[0], axes[3]]:
            ax.set_ylabel("CDP/final increment\n(Final - Global accuracy)")
    fig.text(0.5, 0.02, "Grey band: near-zero increment. Error bars: participant-clustered 95% CI.", ha="center", fontsize=9, color="0.35", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, outdir, "fig_cdp_final_increment_confirmation_bins", dpi, save_pdf)


def plot_confirmed_sessions_vs_adaptive_gain(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    rows = []
    for (pid, adl), sub in df.groupby(["participant_id", "confirmed_class"], observed=True):
        if len(sub) == 0:
            continue
        rows.append({
            "participant_id": pid,
            "confirmed_class": adl,
            "n_sessions": len(sub),
            "adaptive_gain": float((sub["final_correct"] - sub["base_correct"]).mean()),
        })
    g = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.0), sharey=True)
    axes = axes.ravel()
    for ax, adl in zip(axes, ADL_ORDER):
        sub = g[g["confirmed_class"] == adl]
        if sub.empty:
            ax.set_visible(False); continue
        colors = np.where(sub["adaptive_gain"] >= 0, COL["positive"], COL["negative"])
        ax.scatter(sub["n_sessions"], sub["adaptive_gain"], s=45, color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.9)
        ax.axvline(3, color="#8E5EA2", linestyle=":", linewidth=1.1)
        ax.text(3.2, 0.43, "N(min)=3", color="#7A3B88", fontsize=8)
        # Show trend line for every panel to avoid selective reporting.
        r = np.nan
        if len(sub) >= 3 and sub["n_sessions"].nunique() > 1:
            x = sub["n_sessions"].to_numpy(dtype=float)
            y = sub["adaptive_gain"].to_numpy(dtype=float)
            r = np.corrcoef(x, y)[0, 1]
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="#7BA6C9", linewidth=1.8, alpha=0.8)
        gained = int((sub["adaptive_gain"] > 0).sum())
        pct = 100 * gained / len(sub)
        r_txt = "NA" if pd.isna(r) else f"{r:+.2f}"
        ax.set_title(f"{ADL_DISPLAY[adl]}\n{pct:.0f}% of participants gained | r={r_txt}", fontsize=10)
        ax.set_ylim(-0.32, 0.56)
        ax.grid(True, alpha=0.2)
        if ax in axes[3:]:
            ax.set_xlabel("Confirmed sessions per ADL")
        if ax in [axes[0], axes[3]]:
            ax.set_ylabel("Full adaptive accuracy gain\n(Final - Base)")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["positive"], markersize=8, label="Final - Base gain ≥ 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["negative"], markersize=8, label="Final - Base gain < 0"),
        Line2D([0], [0], color="#7BA6C9", linewidth=2, label="Linear trend line"),
        Line2D([0], [0], color="#8E5EA2", linestyle=":", linewidth=1.2, label="Prototype activation threshold (N(min)=3)"),
    ]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, outdir, "fig_confirmed_sessions_vs_full_adaptive_gain", dpi, save_pdf)


def plot_probability_gain_ridge(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        print("Skipping ridge plot because scipy is not available.")
        return

    rows = []
    for adl in ADL_ORDER:
        for outcome in OUTCOME_ORDER:
            vals = df.loc[(df["confirmed_class"] == adl) & (df["decision_outcome"] == outcome), "gain_final_minus_base"].dropna().to_numpy()
            rows.append((adl, outcome, vals))

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), sharex=True)
    axes = axes.ravel()
    x_grid = np.linspace(-1, 1, 400)
    for ax, adl in zip(axes, ADL_ORDER):
        for j, outcome in enumerate(OUTCOME_ORDER):
            vals = df.loc[(df["confirmed_class"] == adl) & (df["decision_outcome"] == outcome), "gain_final_minus_base"].dropna().to_numpy()
            y0 = len(OUTCOME_ORDER) - 1 - j
            color = OUTCOME_COLORS[outcome]
            if len(vals) >= 3 and np.std(vals) > 1e-8:
                try:
                    kde = gaussian_kde(vals)
                    dens = kde(x_grid)
                    dens = dens / dens.max() * 0.75
                    ax.fill_between(x_grid, y0, y0 + dens, color=color, alpha=0.6, linewidth=0)
                    ax.plot(x_grid, y0 + dens, color=color, linewidth=0.8)
                except Exception:
                    ax.scatter(vals, np.full_like(vals, y0), s=3, color=color, alpha=0.3)
            elif len(vals) > 0:
                ax.scatter(vals, np.full_like(vals, y0), s=6, color=color, alpha=0.5)
            if len(vals) > 0:
                ax.vlines(np.mean(vals), y0, y0 + 0.55, color="black", linewidth=0.8)
        ax.axvline(0, linestyle="--", color="0.45", linewidth=0.8)
        ax.set_title(ADL_DISPLAY[adl], fontsize=10)
        ax.set_yticks(range(len(OUTCOME_ORDER)))
        ax.set_yticklabels(list(reversed(OUTCOME_ORDER)) if ax in [axes[0], axes[3]] else [])
        ax.set_xlim(-1, 1)
        ax.grid(True, axis="x", alpha=0.15)
        if ax in axes[3:]:
            ax.set_xlabel("True-class probability gain: Final - Base")
    handles = [mpatches.Patch(color=OUTCOME_COLORS[o], label=o, alpha=0.6) for o in OUTCOME_ORDER]
    handles += [Line2D([0], [0], color="black", linewidth=0.8, label="Mean value"), Line2D([0], [0], color="0.45", linestyle="--", linewidth=0.8, label="No gain")]
    fig.legend(handles=handles, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, outdir, "fig11_probability_gain_ridge_final_minus_base", dpi, save_pdf)


def plot_probability_landscape(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    bands = ["Low complexity", "Medium complexity", "High complexity"]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))

    # Panel A: Final-Base probability gain by complexity band and outcome.
    ax = axes[0]
    x = np.arange(len(bands))
    rng = np.random.default_rng(RANDOM_SEED)
    for outcome in OUTCOME_ORDER:
        vals_x, vals_y = [], []
        for i, b in enumerate(bands):
            sub = df[(df["complexity_band_3"].astype(str) == b) & (df["decision_outcome"] == outcome)]
            y = sub["gain_final_minus_base"].dropna().to_numpy()
            if len(y) == 0:
                continue
            jitter = rng.normal(0, 0.055, size=len(y))
            vals_x.extend(np.full(len(y), i) + jitter)
            vals_y.extend(y)
        ax.scatter(vals_x, vals_y, s=8, alpha=0.18, color=OUTCOME_COLORS[outcome], label=outcome, edgecolors="none")
    means = df.groupby("complexity_band_3", observed=True)["gain_final_minus_base"].mean().reindex(bands)
    ax.plot(x, means.values, color="black", marker="o", linewidth=2, label="Mean")
    ax.axhline(0, linestyle="--", color="0.5", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("True-class probability gain: Final - Base")
    ax.set_xlabel("Session-complexity band")
    ax.set_ylim(-1, 1)
    ax.grid(True, axis="y", alpha=0.2)

    # Panel B: mean true-class probability by stage and complexity band.
    ax = axes[1]
    for stage in ["Base", "Global", "Prototype", "Final"]:
        col = STAGE_TRUE_PROB_COLS[stage]
        sub = df[df["proto_active"]].copy() if stage == "Prototype" else df.copy()
        vals = sub.groupby("complexity_band_3", observed=True)[col].mean().reindex(bands)
        ax.plot(x, vals.values, marker="o", linewidth=2, label=(stage if stage != "Global" else "Global/CAM"), color=STAGE_COLORS[stage])
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Mean probability assigned to confirmed class")
    ax.set_xlabel("Session-complexity band")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)

    handles, labels = axes[0].get_legend_handles_labels()
    # Keep only outcome legend in panel A if needed.
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    plt.tight_layout()
    save_figure(fig, outdir, "fig12_probability_landscape_complexity_bands", dpi, save_pdf)


def plot_participant_paired_stage_accuracy(df: pd.DataFrame, outdir: Path, dpi: int, save_pdf: bool) -> None:
    p = df.groupby("participant_id", observed=True).agg(Base=("base_correct", "mean"), Global=("global_correct", "mean"), Final=("final_correct", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    x = np.array([0, 1, 2])
    for _, r in p.iterrows():
        ax.plot(x, [r["Base"], r["Global"], r["Final"]], marker="o", linewidth=1.2, alpha=0.65)
    means = p[["Base", "Global", "Final"]].mean()
    ax.plot(x, means.values, color="black", linewidth=3, marker="o", markersize=8, label="Participant mean")
    ax.set_xticks(x)
    ax.set_xticklabels(["Base", "Global/CAM", "Final"])
    ax.set_ylabel("Participant-level accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    save_figure(fig, outdir, "supp_participant_paired_stage_accuracy", dpi, save_pdf)


# =============================================================================
# Optional Phase 1 vs Phase 2 complexity figure
# =============================================================================

def plot_optional_phase1_vs_free_living_complexity(
    phase1_file: str | None,
    df_phase2: pd.DataFrame,
    outdir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    """Optional: recreate manuscript complexity shift figure if Phase 1 complexity data are supplied.

    The Phase 2 replay CSV alone cannot reproduce the simulated-vs-free-living
    complexity figure because it has no Phase 1 simulated-living complexity values.
    Expected columns for the optional phase1 file: confirmed_class (or ADL) and
    complexity_session (or complexity_score).
    """
    if not phase1_file:
        note = (
            "Figure 6 simulated-vs-free-living complexity was not generated because "
            "no --phase1-complexity-csv was provided. The Phase 2 replay file alone "
            "does not contain simulated-living complexity values.\n"
        )
        (outdir / "SKIPPED_fig06_complexity_shift_note.txt").write_text(note, encoding="utf-8")
        print("Skipped Figure 6 complexity shift: no Phase 1 complexity CSV supplied.")
        return

    p = Path(phase1_file)
    if not p.exists():
        raise FileNotFoundError(f"Phase 1 complexity file not found: {p}")
    phase1 = pd.read_csv(p)
    label_col = "confirmed_class" if "confirmed_class" in phase1.columns else ("ADL" if "ADL" in phase1.columns else None)
    comp_col = "complexity_session" if "complexity_session" in phase1.columns else ("complexity_score" if "complexity_score" in phase1.columns else None)
    if label_col is None or comp_col is None:
        raise ValueError("Phase 1 complexity CSV must contain confirmed_class/ADL and complexity_session/complexity_score columns.")
    phase1 = phase1[[label_col, comp_col]].copy()
    phase1.columns = ["confirmed_class", "complexity_session"]
    phase1["confirmed_class"] = clean_label_series(phase1["confirmed_class"])
    phase1["Environment"] = "Simulated"

    phase2 = df_phase2[["confirmed_class", "complexity_session"]].copy()
    phase2["Environment"] = "Free-living"
    both = pd.concat([phase1, phase2], ignore_index=True)
    both = both[both["confirmed_class"].isin(ADL_ORDER)].copy()

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    positions = np.arange(len(ADL_ORDER))
    width = 0.34
    data_sim = [both[(both["confirmed_class"] == a) & (both["Environment"] == "Simulated")]["complexity_session"].dropna() for a in ADL_ORDER]
    data_free = [both[(both["confirmed_class"] == a) & (both["Environment"] == "Free-living")]["complexity_session"].dropna() for a in ADL_ORDER]
    bp1 = ax.boxplot(data_sim, positions=positions - width/2, widths=0.28, patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_free, positions=positions + width/2, widths=0.28, patch_artist=True, showfliers=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor("#9EDCCB"); patch.set_alpha(0.8)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#F4A582"); patch.set_alpha(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([ADL_DISPLAY[a] for a in ADL_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Motion complexity score C")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Simulated", "Free-living"], frameon=False, ncol=2, loc="upper center")
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    save_figure(fig, outdir, "fig06_motion_complexity_simulated_vs_free_living", dpi, save_pdf)


# =============================================================================
# Main orchestration
# =============================================================================

def generate_all_outputs(args: argparse.Namespace) -> None:
    set_plot_style()
    dirs = make_output_dirs(args.outdir)
    df = load_phase2_session_data(args.input, strict=args.strict)

    # Main tables.
    save_table(table_phase2_adl_summary(df), dirs.tables_main, "table04_phase2_session_summary_by_adl", index=False)
    save_table(table_stage_performance(df, args.n_boot), dirs.tables_main, "table05_stagewise_deployment_replay_performance", index=False)
    save_table(table_integrated_adaptive_behaviour(df, args.n_boot), dirs.tables_main, "table06_integrated_adaptive_decision_behaviour", index=False)
    save_table(table_stage_contribution_per_adl(df), dirs.tables_main, "table_stage_contribution_per_adl_cam_cdp", index=False)

    # Supplementary tables.
    for stem, tab in supplementary_tables(df, args.n_boot).items():
        save_table(tab, dirs.tables_supp, stem, index=False)

    # Main figures.
    plot_final_confusion_matrix(df, dirs.figures_main, args.dpi, args.save_pdf)
    plot_optional_phase1_vs_free_living_complexity(args.phase1_complexity_csv, df, dirs.figures_main, args.dpi, args.save_pdf)
    plot_participantwise_final_accuracy_heatmap(df, dirs.figures_main, args.dpi, args.save_pdf)
    plot_stage_contribution_per_adl(df, dirs.figures_main, args.dpi, args.save_pdf)
    plot_stage_confusion_matrices(df, dirs.figures_main, args.dpi, args.save_pdf, normalize=False)
    plot_temporal_gain_across_blocks(df, dirs.figures_main, args.dpi, args.save_pdf, args.n_boot)
    plot_outcome_breakdown_by_adl(df, dirs.figures_main, args.dpi, args.save_pdf)
    plot_recovery_source_per_adl(df, dirs.figures_main, args.dpi, args.save_pdf)
    plot_cdp_gain_confirmation_bins(df, dirs.figures_main, args.dpi, args.save_pdf, args.n_boot)
    plot_confirmed_sessions_vs_adaptive_gain(df, dirs.figures_main, args.dpi, args.save_pdf)

    # Supplementary figures.
    plot_probability_gain_ridge(df, dirs.figures_supp, args.dpi, args.save_pdf)
    plot_probability_landscape(df, dirs.figures_supp, args.dpi, args.save_pdf)
    plot_stage_confusion_matrices(df, dirs.figures_supp, args.dpi, args.save_pdf, normalize=True)
    plot_participant_paired_stage_accuracy(df, dirs.figures_supp, args.dpi, args.save_pdf)

    # Save clean session-level file.
    session_out = dirs.root / "phase2_session_level_canonical_used_for_outputs.csv"
    df.to_csv(session_out, index=False)
    print(f"Saved session-level canonical file: {session_out}")

    # Manifest / notes.
    manifest = []
    manifest.append("SmartADAPT-Net Phase 2 output generation manifest\n")
    manifest.append(f"Input: {args.input}\n")
    manifest.append(f"Output directory: {dirs.root}\n")
    manifest.append(f"DPI: {args.dpi}\n")
    manifest.append(f"Bootstrap replicates: {args.n_boot}\n")
    manifest.append("\nValidation values:\n")
    manifest.append(f"Sessions: {len(df)}\n")
    manifest.append(f"Prototype-active sessions: {int(df['proto_active'].sum())}\n")
    manifest.append(f"Base accuracy: {df['base_correct'].mean():.6f}\n")
    manifest.append(f"Global/CAM accuracy: {df['global_correct'].mean():.6f}\n")
    manifest.append(f"Prototype accuracy active only: {df.loc[df['proto_active'], 'proto_correct'].mean():.6f}\n")
    manifest.append(f"Final accuracy: {df['final_correct'].mean():.6f}\n")
    manifest.append("\nNote: Unclassified recovery-source sessions are Final-stage recoveries that could not be uniquely assigned to Global/CAM only, Prototype/CDP only, or both branches combined.\n")
    manifest.append("Note: Figure 6 simulated-vs-free-living complexity requires optional Phase 1 complexity data.\n")
    (dirs.logs / "generation_manifest.txt").write_text("".join(manifest), encoding="utf-8")

    print("\nDone.")
    print(f"All outputs saved in: {dirs.root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all Phase 2 SmartADAPT-Net manuscript and supplementary outputs.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to canonical CODE1 Phase 2 replay CSV/parquet file.")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to <input folder>/SmartADAPT_phase2_manuscript_outputs.")
    parser.add_argument("--strict", action="store_true", help="Stop if validation values do not match manuscript values.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution. Default: 600 dpi.")
    parser.add_argument("--n-boot", type=int, default=2000, help="Participant-clustered bootstrap replicates. Default: 2000.")
    parser.add_argument("--save-pdf", action="store_true", help="Also save vector PDF copies of all figures.")
    parser.add_argument("--phase1-complexity-csv", default=None, help="Optional CSV for simulated-living complexity values to generate Fig 6.")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments, usually added by Jupyter/IPython: {unknown}")
    if args.outdir is None:
        args.outdir = str(Path(args.input).resolve().parent / "SmartADAPT_phase2_manuscript_outputs")
    return args


if __name__ == "__main__":
    generate_all_outputs(parse_args())
