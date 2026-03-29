#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q2D Progressive KD (5-fold, strict val/test protocol)

Teacher:
- GSR Time2Graph++ per-question lie score
- teacher penultimate feature

Student:
- GenLie per-question lie prediction on a pre-extracted target modality feature stream

Loss:
1) hard question CE
2) listwise ranking KD (question level)
3) digit-evidence KD (10-class evidence after aggregation)
4) feature alignment (student question embedding -> teacher feature)

Selection:
- split outer-train into inner train/val
- select epoch + agg mode by val only
- evaluate each fold test exactly once
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import logging
import math
import random
import sys
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Reuse modality-agnostic loader/model utilities from baseline script.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eval_video_genlie_5fold import (  # noqa: E402
    TARGET_QIDS,
    GenLieModel,
    _read_ids,
    choose_input_dim,
    compute_norm_stats,
    digit_from_qid,
    load_subject_sample,
    normalize_modality_name,
    sample_triplets,
    vec_to_dim,
)
try:
    import xkd_refine_da  # noqa: F401,E402
except Exception:
    pass


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def make_grad_scaler(device: torch.device, enabled: bool):
    enabled = bool(enabled) and str(device.type) == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return None


def autocast_context(device: torch.device, enabled: bool):
    enabled = bool(enabled) and str(device.type) == "cuda"
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type=device.type, enabled=enabled)
        except TypeError:
            return torch.amp.autocast(enabled=enabled)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def get_device(device: str) -> torch.device:
    d = str(device).strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d.startswith("cuda"):
        if torch.cuda.is_available():
            if d == "cuda":
                return torch.device("cuda")
            try:
                return torch.device(d)
            except Exception:
                return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cpu")


def _parse_csv_list(s: str) -> List[str]:
    out: List[str] = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def better_by_topk(cur: Dict[str, float], best: Optional[Dict[str, float]]) -> bool:
    if best is None:
        return True
    if float(cur["top1"]) != float(best["top1"]):
        return float(cur["top1"]) > float(best["top1"])
    if float(cur["top2"]) != float(best["top2"]):
        return float(cur["top2"]) > float(best["top2"])
    if float(cur["top3"]) != float(best["top3"]):
        return float(cur["top3"]) > float(best["top3"])
    return False


def split_train_val_ids(ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    arr = [str(x) for x in ids]
    if len(arr) <= 2:
        return list(arr), list(arr[:1])
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(arr))
    rng.shuffle(idx)
    n_val = int(round(len(arr) * float(val_ratio)))
    n_val = max(1, min(len(arr) - 1, n_val))
    val_set = set(idx[:n_val].tolist())
    tr: List[str] = []
    va: List[str] = []
    for i, sid in enumerate(arr):
        if i in val_set:
            va.append(sid)
        else:
            tr.append(sid)
    return tr, va


def _sigmoid01(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))


def _cosine01(r: float) -> float:
    r = float(np.clip(float(r), 0.0, 1.0))
    return float(0.5 * (1.0 - math.cos(math.pi * r)))


def _gated_sigmoid(ep: int, start: float, width: float) -> float:
    if float(ep) < float(start):
        return 0.0
    return _sigmoid01((float(ep) - float(start)) / max(1.0, float(width)))


def stage_weights(
    ep: int,
    stage1: int,
    stage2: int,
    digit_ramp: int,
    mode: str = "linear",
    path_order: str = "feature_first",
    rank_width: float = 4.0,
    feat_width: float = 6.0,
    digit_width: float = 8.0,
    feat_max_weight: float = 1.0,
    overlap_ratio: float = 0.0,
    overlap_width: float = 6.0,
) -> Tuple[float, float, float]:
    mode_l = str(mode).lower().strip()
    order_l = str(path_order).lower().strip()
    if mode_l == "step":
        feat_on = float(min(1.0, max(0.0, feat_max_weight)))
        if ep <= int(stage1):
            return 0.0, 0.0, 0.0
        if ep <= int(stage1 + stage2):
            if order_l == "feature_first":
                return 1.0, feat_on, 0.0
            if order_l == "digit_first":
                return 0.0, 0.0, 1.0
            if order_l == "joint":
                return 0.0, feat_on, 1.0
            raise ValueError(f"unknown progressive_path_order={path_order}")
        if order_l == "feature_first":
            return 1.0, feat_on, 1.0
        if order_l == "digit_first":
            return 0.0, feat_on, 1.0
        if order_l == "joint":
            return 0.0, feat_on, 1.0
        raise ValueError(f"unknown progressive_path_order={path_order}")
    if mode_l == "linear":
        if ep <= int(stage1):
            return 0.0, 0.0, 0.0
        if ep <= int(stage1 + stage2):
            r = float(ep - stage1) / float(max(1, stage2))
            if order_l == "feature_first":
                return r, r, 0.0
            if order_l == "digit_first":
                return 0.0, 0.0, r
            if order_l == "joint":
                return 0.0, r, r
            raise ValueError(f"unknown progressive_path_order={path_order}")
        rd = float(ep - stage1 - stage2) / float(max(1, digit_ramp))
        rr = min(1.0, rd)
        if order_l == "feature_first":
            return 1.0, 1.0, rr
        if order_l == "digit_first":
            return 0.0, rr, 1.0
        if order_l == "joint":
            return 0.0, 1.0, 1.0
        raise ValueError(f"unknown progressive_path_order={path_order}")
    if mode_l == "cosine":
        if ep <= int(stage1):
            return 0.0, 0.0, 0.0
        if ep <= int(stage1 + stage2):
            r = float(ep - stage1) / float(max(1, stage2))
            rc = _cosine01(r)
            if order_l == "feature_first":
                return rc, rc, 0.0
            if order_l == "digit_first":
                return 0.0, 0.0, rc
            if order_l == "joint":
                return 0.0, rc, rc
            raise ValueError(f"unknown progressive_path_order={path_order}")
        rd = float(ep - stage1 - stage2) / float(max(1, digit_ramp))
        rc = _cosine01(rd)
        if order_l == "feature_first":
            return 1.0, 1.0, rc
        if order_l == "digit_first":
            return 0.0, rc, 1.0
        if order_l == "joint":
            return 0.0, 1.0, 1.0
        raise ValueError(f"unknown progressive_path_order={path_order}")

    if mode_l == "gated_sigmoid":
        if ep <= int(stage1):
            return 0.0, 0.0, 0.0
        rank_start = float(stage1 + 1)
        second_start = float(stage1 + stage2 + 1)
        wr = _gated_sigmoid(ep, rank_start, rank_width)
        if order_l == "feature_first":
            wf = float(min(1.0, max(0.0, feat_max_weight))) * _gated_sigmoid(ep, rank_start, feat_width)
            wd = _gated_sigmoid(ep, second_start, digit_width)
            return float(wr), float(wf), float(wd)
        if order_l == "digit_first":
            wf = float(min(1.0, max(0.0, feat_max_weight))) * _gated_sigmoid(ep, second_start, feat_width)
            wd = _gated_sigmoid(ep, rank_start, digit_width)
            return 0.0, float(wf), float(wd)
        if order_l == "joint":
            wf = float(min(1.0, max(0.0, feat_max_weight))) * _gated_sigmoid(ep, rank_start, feat_width)
            wd = _gated_sigmoid(ep, rank_start, digit_width)
            return 0.0, float(wf), float(wd)
        raise ValueError(f"unknown progressive_path_order={path_order}")

    if mode_l == "overlap_sigmoid":
        if ep <= int(stage1):
            return 0.0, 0.0, 0.0
        rank_start = float(stage1 + 1)
        second_start = float(stage1 + stage2 + 1)
        overlap = float(np.clip(float(overlap_ratio), 0.0, 1.0))
        wr = _gated_sigmoid(ep, rank_start, rank_width)
        if order_l == "feature_first":
            wf = float(min(1.0, max(0.0, feat_max_weight))) * _gated_sigmoid(ep, rank_start, feat_width)
            wd_pre = _gated_sigmoid(ep, rank_start, overlap_width)
            wd_post = _gated_sigmoid(ep, second_start, digit_width)
            wd = overlap * wd_pre + (1.0 - overlap) * wd_post
            return float(wr), float(wf), float(wd)
        if order_l == "digit_first":
            wd = _gated_sigmoid(ep, rank_start, digit_width)
            wf_pre = float(min(1.0, max(0.0, feat_max_weight))) * _gated_sigmoid(ep, rank_start, overlap_width)
            wf_post = float(min(1.0, max(0.0, feat_max_weight))) * _gated_sigmoid(ep, second_start, feat_width)
            wf = overlap * wf_pre + (1.0 - overlap) * wf_post
            return 0.0, float(wf), float(wd)
        if order_l == "joint":
            wf = float(min(1.0, max(0.0, feat_max_weight))) * _gated_sigmoid(ep, rank_start, feat_width)
            wd = _gated_sigmoid(ep, rank_start, digit_width)
            return 0.0, float(wf), float(wd)
        raise ValueError(f"unknown progressive_path_order={path_order}")

    if ep <= int(stage1):
        return 0.0, 0.0, 0.0
    rank_start = float(stage1 + 1)
    second_start = float(stage1 + stage2 + 1)
    wr = _sigmoid01((float(ep) - rank_start) / max(1.0, float(rank_width)))
    if order_l == "feature_first":
        wf = float(min(1.0, max(0.0, feat_max_weight))) * _sigmoid01((float(ep) - rank_start) / max(1.0, float(feat_width)))
        wd = _sigmoid01((float(ep) - second_start) / max(1.0, float(digit_width)))
        return float(wr), float(wf), float(wd)
    if order_l == "digit_first":
        wf = float(min(1.0, max(0.0, feat_max_weight))) * _sigmoid01((float(ep) - second_start) / max(1.0, float(feat_width)))
        wd = _sigmoid01((float(ep) - rank_start) / max(1.0, float(digit_width)))
        return 0.0, float(wf), float(wd)
    if order_l == "joint":
        wf = float(min(1.0, max(0.0, feat_max_weight))) * _sigmoid01((float(ep) - rank_start) / max(1.0, float(feat_width)))
        wd = _sigmoid01((float(ep) - rank_start) / max(1.0, float(digit_width)))
        return 0.0, float(wf), float(wd)
    raise ValueError(f"unknown progressive_path_order={path_order}")


def kd_conf_threshold(
    ep: int,
    stage1: int,
    stage2: int,
    digit_ramp: int,
    high: float,
    low: float,
) -> float:
    total = float(max(1, int(stage2) + int(digit_ramp)))
    prog = float(np.clip((float(ep) - float(stage1)) / total, 0.0, 1.0))
    return float(high + (low - high) * prog)


def _standardize_np_features(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return x
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < float(eps), 1.0, sd)
    return x / sd


def estimate_linear_cka_similarity(
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int = 4096,
    eps: float = 1e-8,
) -> Optional[float]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        return None
    n = min(int(x.shape[0]), int(y.shape[0]))
    if n < 2:
        return None
    x = x[:n]
    y = y[:n]
    if int(max_rows) > 0 and n > int(max_rows):
        idx = np.linspace(0, n - 1, num=int(max_rows), dtype=np.int64)
        x = x[idx]
        y = y[idx]
    x = _standardize_np_features(x, eps=float(eps))
    y = _standardize_np_features(y, eps=float(eps))
    if x.shape[0] < 2 or y.shape[0] < 2:
        return None
    k = np.matmul(x, x.T)
    l = np.matmul(y, y.T)
    num = float(np.sum(k * l))
    den = math.sqrt(max(float(np.sum(k * k)), float(eps)) * max(float(np.sum(l * l)), float(eps)))
    if den <= float(eps):
        return None
    return float(np.clip(num / den, 0.0, 1.0))


def estimate_modality_gap_from_items(
    items: Sequence[KDSubjectItem],
    max_rows: int = 4096,
) -> Optional[float]:
    if len(items) == 0:
        return None
    xs: List[np.ndarray] = []
    ts: List[np.ndarray] = []
    for it in items:
        if it.x20.size == 0 or it.teacher_feat20.size == 0:
            continue
        xs.append(np.asarray(it.x20, dtype=np.float32))
        ts.append(np.asarray(it.teacher_feat20, dtype=np.float32))
    if len(xs) == 0 or len(ts) == 0:
        return None
    x = np.concatenate(xs, axis=0)
    t = np.concatenate(ts, axis=0)
    sim = estimate_linear_cka_similarity(x, t, max_rows=max_rows)
    if sim is None:
        return None
    return float(np.clip(1.0 - sim, 0.0, 1.0))


def default_modality_gap_prior(student_modality: str) -> float:
    modality = normalize_modality_name(student_modality)
    if modality == "video":
        return 0.70
    if modality == "audio":
        return 0.30
    return 0.50


def _resolve_gap_route_profile(profile: str, student_modality: str) -> str:
    p = str(profile).lower().strip()
    if p in {"", "auto"}:
        modality = normalize_modality_name(student_modality)
        if modality == "video":
            return "video_strong_feature"
        if modality == "audio":
            return "audio_weak_feature"
        return "balanced"
    return p


def apply_gap_route_profile(cfg: argparse.Namespace, student_modality: str) -> str:
    requested = str(getattr(cfg, "gap_route_profile", "auto"))
    resolved = _resolve_gap_route_profile(requested, student_modality)
    presets: Dict[str, Tuple[float, float, float, float, float, float]] = {
        # (no_feature_enter, no_feature_exit, digit_enter, digit_exit, feature_exit, feature_enter)
        "video_strong_feature": (0.30, 0.36, 0.44, 0.50, 0.58, 0.66),
        "audio_weak_feature": (0.40, 0.46, 0.54, 0.60, 0.68, 0.76),
        "balanced": (0.34, 0.40, 0.46, 0.52, 0.56, 0.62),
    }
    calib_presets: Dict[str, Tuple[float, float]] = {
        # calibrated_gap = clip(raw_gap * scale + bias, 0, 1)
        "video_strong_feature": (1.0, 0.0),
        "audio_weak_feature": (0.8, -0.30),
        "balanced": (1.0, 0.0),
    }
    if resolved in presets:
        vals = presets[resolved]
        cfg.gap_route_no_feature_enter = float(vals[0])
        cfg.gap_route_no_feature_exit = float(vals[1])
        cfg.gap_route_digit_enter = float(vals[2])
        cfg.gap_route_digit_exit = float(vals[3])
        cfg.gap_route_feature_exit = float(vals[4])
        cfg.gap_route_feature_enter = float(vals[5])
        scale, bias = calib_presets[resolved]
        if getattr(cfg, "gap_obs_scale", None) is None:
            cfg.gap_obs_scale = float(scale)
        if getattr(cfg, "gap_obs_bias", None) is None:
            cfg.gap_obs_bias = float(bias)
    elif resolved != "manual":
        raise ValueError(f"unknown gap_route_profile={requested!r} resolved={resolved!r}")
    if getattr(cfg, "gap_obs_scale", None) is None:
        cfg.gap_obs_scale = 1.0
    if getattr(cfg, "gap_obs_bias", None) is None:
        cfg.gap_obs_bias = 0.0
    cfg.gap_route_profile_resolved = str(resolved)
    return str(resolved)


def calibrate_gap_observation(raw_gap: float, cfg: argparse.Namespace) -> float:
    scale = float(getattr(cfg, "gap_obs_scale", 1.0))
    bias = float(getattr(cfg, "gap_obs_bias", 0.0))
    g = float(raw_gap) * scale + bias
    return float(np.clip(g, 0.0, 1.0))


def gap_adaptive_feature_scale(gap: float, cfg: argparse.Namespace) -> float:
    # Small gap -> suppress feature KD. Large gap -> fully keep feature KD.
    g = float(np.clip(float(gap), 0.0, 1.0))
    low = float(getattr(cfg, "gap_route_no_feature_exit", 0.40))
    high = float(getattr(cfg, "gap_route_feature_enter", 0.62))
    if high <= low + 1e-8:
        return 1.0 if g >= high else 0.0
    return float(np.clip((g - low) / (high - low), 0.0, 1.0))


def _normalize_hard_route_thresholds(
    no_feature_enter: float,
    no_feature_exit: float,
    digit_enter: float,
    feature_enter: float,
    feature_exit: float,
    digit_exit: float,
) -> Tuple[float, float, float, float, float, float]:
    vals = np.clip(
        np.asarray(
            [no_feature_enter, no_feature_exit, digit_enter, digit_exit, feature_exit, feature_enter],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )
    vals = np.sort(vals)
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5])


def select_hard_route_from_gap(
    gap: float,
    prev_path_order: str,
    hold_epochs: int,
    min_hold_epochs: int,
    no_feature_enter: float,
    no_feature_exit: float,
    digit_enter: float,
    feature_enter: float,
    feature_exit: float,
    digit_exit: float,
) -> Tuple[str, int]:
    prev = str(prev_path_order).lower().strip()
    if prev not in {"feature_first", "joint", "digit_first", "no_feature"}:
        prev = "joint"
    no_feature_enter, no_feature_exit, digit_enter, digit_exit, feature_exit, feature_enter = _normalize_hard_route_thresholds(
        no_feature_enter=no_feature_enter,
        no_feature_exit=no_feature_exit,
        digit_enter=digit_enter,
        feature_enter=feature_enter,
        feature_exit=feature_exit,
        digit_exit=digit_exit,
    )
    hold_epochs = max(0, int(hold_epochs))
    min_hold_epochs = max(0, int(min_hold_epochs))
    if hold_epochs < min_hold_epochs:
        return prev, hold_epochs + 1

    gap = float(np.clip(float(gap), 0.0, 1.0))
    new_state = prev
    if prev == "no_feature":
        if gap >= no_feature_exit:
            if gap >= feature_enter:
                new_state = "feature_first"
            elif gap >= digit_exit:
                new_state = "joint"
            else:
                new_state = "digit_first"
    elif prev == "feature_first":
        if gap < feature_exit:
            if gap <= no_feature_enter:
                new_state = "no_feature"
            elif gap <= digit_enter:
                new_state = "digit_first"
            else:
                new_state = "joint"
    elif prev == "digit_first":
        if gap <= no_feature_enter:
            new_state = "no_feature"
        elif gap >= feature_enter:
            new_state = "feature_first"
        elif gap >= digit_exit:
            new_state = "joint"
    else:
        if gap >= feature_enter:
            new_state = "feature_first"
        elif gap <= no_feature_enter:
            new_state = "no_feature"
        elif gap <= digit_enter:
            new_state = "digit_first"
        else:
            new_state = "joint"

    if new_state != prev:
        return new_state, 1
    return prev, hold_epochs + 1


def init_online_gap_state(cfg: argparse.Namespace, student_modality: str) -> OnlineGapState:
    prior_gap = default_modality_gap_prior(student_modality)
    path_order, hold_epochs = select_hard_route_from_gap(
        gap=prior_gap,
        prev_path_order="joint",
        hold_epochs=int(getattr(cfg, "gap_route_min_hold_epochs", 3)),
        min_hold_epochs=int(getattr(cfg, "gap_route_min_hold_epochs", 3)),
        no_feature_enter=float(getattr(cfg, "gap_route_no_feature_enter", 0.34)),
        no_feature_exit=float(getattr(cfg, "gap_route_no_feature_exit", 0.40)),
        digit_enter=float(getattr(cfg, "gap_route_digit_enter", 0.46)),
        feature_enter=float(getattr(cfg, "gap_route_feature_enter", 0.62)),
        feature_exit=float(getattr(cfg, "gap_route_feature_exit", 0.56)),
        digit_exit=float(getattr(cfg, "gap_route_digit_exit", 0.52)),
    )
    return OnlineGapState(
        ema_gap=float(prior_gap),
        last_gap=None,
        last_gap_raw=None,
        source="modality_prior_bootstrap",
        path_order=str(path_order),
        hold_epochs=int(hold_epochs),
    )


def should_update_online_gap(cfg: argparse.Namespace, epoch: int) -> bool:
    warmup = max(1, int(getattr(cfg, "gap_update_warmup_epochs", 5)))
    interval = max(1, int(getattr(cfg, "gap_update_interval", 2)))
    if int(epoch) < warmup:
        return False
    return ((int(epoch) - warmup) % interval) == 0


def update_online_gap_state(
    cfg: argparse.Namespace,
    state: OnlineGapState,
    student_rows: Sequence[np.ndarray],
    teacher_rows: Sequence[np.ndarray],
) -> OnlineGapState:
    if len(student_rows) == 0 or len(teacher_rows) == 0:
        return state
    x = np.concatenate([np.asarray(v, dtype=np.float32) for v in student_rows], axis=0)
    t = np.concatenate([np.asarray(v, dtype=np.float32) for v in teacher_rows], axis=0)
    sim = estimate_linear_cka_similarity(
        x,
        t,
        max_rows=int(getattr(cfg, "gap_online_max_rows", 4096)),
    )
    min_reliable_similarity = float(getattr(cfg, "gap_online_min_reliable_similarity", 0.02))
    if sim is None or float(sim) < min_reliable_similarity:
        return OnlineGapState(
            ema_gap=float(state.ema_gap),
            last_gap=state.last_gap,
            last_gap_raw=state.last_gap_raw,
            source="ema_hold_unreliable_update",
            path_order=str(state.path_order),
            hold_epochs=int(state.hold_epochs) + 1,
        )

    gap_obs_raw = float(np.clip(1.0 - float(sim), 0.0, 1.0))
    gap_obs = calibrate_gap_observation(gap_obs_raw, cfg)
    momentum = float(np.clip(float(getattr(cfg, "gap_ema_momentum", 0.8)), 0.0, 0.999))
    ema_gap = momentum * float(state.ema_gap) + (1.0 - momentum) * gap_obs
    path_order, hold_epochs = select_hard_route_from_gap(
        gap=ema_gap,
        prev_path_order=str(state.path_order),
        hold_epochs=int(state.hold_epochs),
        min_hold_epochs=int(getattr(cfg, "gap_route_min_hold_epochs", 3)),
        no_feature_enter=float(getattr(cfg, "gap_route_no_feature_enter", 0.34)),
        no_feature_exit=float(getattr(cfg, "gap_route_no_feature_exit", 0.40)),
        digit_enter=float(getattr(cfg, "gap_route_digit_enter", 0.46)),
        feature_enter=float(getattr(cfg, "gap_route_feature_enter", 0.62)),
        feature_exit=float(getattr(cfg, "gap_route_feature_exit", 0.56)),
        digit_exit=float(getattr(cfg, "gap_route_digit_exit", 0.52)),
    )
    return OnlineGapState(
        ema_gap=float(ema_gap),
        last_gap=float(gap_obs),
        last_gap_raw=float(gap_obs_raw),
        source="online_ema_cka",
        path_order=str(path_order),
        hold_epochs=int(hold_epochs),
    )


def _sanitize_nofeat_path_order(path_order: str) -> str:
    p = str(path_order).lower().strip()
    if p in {"feature_first", "joint", "digit_first"}:
        return p
    return "feature_first"


def init_nofeat_route_state(cfg: argparse.Namespace) -> NoFeatRouteState:
    boot = _sanitize_nofeat_path_order(str(getattr(cfg, "nofeat_route_bootstrap", "feature_first")))
    return NoFeatRouteState(
        path_order=str(boot),
        hold_epochs=1,
        ema_rank_err=1.0,
        ema_digit_err=1.0,
        rank_digit_ratio=1.0,
        source="nofeat_bootstrap",
    )


def update_nofeat_route_state(
    cfg: argparse.Namespace,
    state: NoFeatRouteState,
    rank_err: float,
    digit_err: float,
) -> NoFeatRouteState:
    eps = 1e-6
    m = float(np.clip(float(getattr(cfg, "nofeat_route_ema_momentum", 0.8)), 0.0, 0.999))
    rank_obs = max(float(rank_err), eps)
    digit_obs = max(float(digit_err), eps)
    ema_rank = m * float(state.ema_rank_err) + (1.0 - m) * rank_obs
    ema_digit = m * float(state.ema_digit_err) + (1.0 - m) * digit_obs
    ratio = float(ema_rank / max(eps, ema_digit))

    prev = _sanitize_nofeat_path_order(state.path_order)
    hold_epochs = max(0, int(state.hold_epochs))
    min_hold = max(0, int(getattr(cfg, "nofeat_route_min_hold_epochs", 3)))
    if hold_epochs < min_hold:
        return NoFeatRouteState(
            path_order=str(prev),
            hold_epochs=int(hold_epochs + 1),
            ema_rank_err=float(ema_rank),
            ema_digit_err=float(ema_digit),
            rank_digit_ratio=float(ratio),
            source="nofeat_hold",
        )

    switch_up = float(getattr(cfg, "nofeat_route_switch_up", 1.08))
    switch_down = float(getattr(cfg, "nofeat_route_switch_down", 0.92))
    digit_enter = float(getattr(cfg, "nofeat_route_digit_enter", 0.78))
    digit_exit = float(getattr(cfg, "nofeat_route_digit_exit", 0.86))

    new_state = prev
    if prev == "feature_first":
        if ratio < digit_enter:
            new_state = "digit_first"
        elif ratio < switch_down:
            new_state = "joint"
    elif prev == "joint":
        if ratio >= switch_up:
            new_state = "feature_first"
        elif ratio < digit_enter:
            new_state = "digit_first"
    else:  # digit_first
        if ratio >= digit_exit:
            if ratio >= switch_up:
                new_state = "feature_first"
            else:
                new_state = "joint"

    if new_state != prev:
        return NoFeatRouteState(
            path_order=str(new_state),
            hold_epochs=1,
            ema_rank_err=float(ema_rank),
            ema_digit_err=float(ema_digit),
            rank_digit_ratio=float(ratio),
            source="nofeat_switch",
        )
    return NoFeatRouteState(
        path_order=str(prev),
        hold_epochs=int(hold_epochs + 1),
        ema_rank_err=float(ema_rank),
        ema_digit_err=float(ema_digit),
        rank_digit_ratio=float(ratio),
        source="nofeat_keep",
    )


def _grad_cosine_mean(grad_a: Optional[torch.Tensor], grad_b: Optional[torch.Tensor], eps: float = 1e-8) -> Optional[float]:
    if grad_a is None or grad_b is None:
        return None
    if int(grad_a.numel()) == 0 or int(grad_b.numel()) == 0:
        return None
    ga = grad_a.detach().reshape(int(grad_a.shape[0]), -1).float()
    gb = grad_b.detach().reshape(int(grad_b.shape[0]), -1).float()
    ga_n = ga.norm(dim=1)
    gb_n = gb.norm(dim=1)
    valid = (ga_n > float(eps)) & (gb_n > float(eps))
    if int(valid.sum().item()) <= 0:
        return None
    cos = (ga * gb).sum(dim=1) / (ga_n * gb_n + float(eps))
    return float(cos[valid].mean().item())


def init_feature_gate_state(cfg: argparse.Namespace) -> FeatureGateState:
    g0 = float(np.clip(float(getattr(cfg, "feat_gate_init", 1.0)), 0.0, 1.0))
    return FeatureGateState(
        gate=float(g0),
        ema_grad_cos=0.0,
        source="gate_bootstrap",
        updates=0,
    )


def update_feature_gate_state(
    cfg: argparse.Namespace,
    state: FeatureGateState,
    grad_cos: Optional[float],
) -> FeatureGateState:
    if grad_cos is None:
        return FeatureGateState(
            gate=float(state.gate),
            ema_grad_cos=float(state.ema_grad_cos),
            source="gate_hold_no_grad",
            updates=int(state.updates),
        )
    m = float(np.clip(float(getattr(cfg, "feat_gate_ema_momentum", 0.8)), 0.0, 0.999))
    ema = float(m * float(state.ema_grad_cos) + (1.0 - m) * float(grad_cos))
    center = float(getattr(cfg, "feat_gate_center", 0.0))
    scale = float(getattr(cfg, "feat_gate_sigmoid_scale", 8.0))
    g_min = float(np.clip(float(getattr(cfg, "feat_gate_min", 0.0)), 0.0, 1.0))
    g_max = float(np.clip(float(getattr(cfg, "feat_gate_max", 1.0)), 0.0, 1.0))
    if g_max < g_min:
        g_min, g_max = g_max, g_min
    raw = float(1.0 / (1.0 + math.exp(-scale * (ema - center))))
    gate = float(g_min + (g_max - g_min) * raw)
    return FeatureGateState(
        gate=float(np.clip(gate, 0.0, 1.0)),
        ema_grad_cos=float(ema),
        source="grad_conflict_gate",
        updates=int(state.updates) + 1,
    )


def route_feature_scale(path_order: str, cfg: argparse.Namespace) -> float:
    p = str(path_order).lower().strip()
    if p == "feature_first":
        return float(np.clip(float(getattr(cfg, "route_feat_scale_feature_first", 1.0)), 0.0, 1.0))
    if p == "joint":
        return float(np.clip(float(getattr(cfg, "route_feat_scale_joint", 1.0)), 0.0, 1.0))
    if p == "digit_first":
        return float(np.clip(float(getattr(cfg, "route_feat_scale_digit_first", 1.0)), 0.0, 1.0))
    if p == "no_feature":
        return float(np.clip(float(getattr(cfg, "route_feat_scale_no_feature", 0.0)), 0.0, 1.0))
    return 1.0


def standardize_logits(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=dim, keepdim=True)
    sd = x.std(dim=dim, keepdim=True, unbiased=False).clamp(min=float(eps))
    return (x - mu) / sd


@dataclass
class KDSubjectItem:
    sid: str
    pid_idx: int
    y_digit: int
    x20: np.ndarray  # [20, D]
    lie20: np.ndarray  # [20]
    qdigit20: np.ndarray  # [20]
    teacher20: np.ndarray  # [20]
    teacher_feat20: np.ndarray  # [20, Dt]


@dataclass
class OnlineGapState:
    ema_gap: float
    last_gap: Optional[float]
    last_gap_raw: Optional[float]
    source: str
    path_order: str
    hold_epochs: int


@dataclass
class NoFeatRouteState:
    path_order: str
    hold_epochs: int
    ema_rank_err: float
    ema_digit_err: float
    rank_digit_ratio: float
    source: str


@dataclass
class FeatureGateState:
    gate: float
    ema_grad_cos: float
    source: str
    updates: int


class KDSubjectDataset(Dataset):
    def __init__(self, items: Sequence[KDSubjectItem]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        return (
            torch.from_numpy(it.x20.astype(np.float32)),
            torch.from_numpy(it.lie20.astype(np.int64)),
            torch.from_numpy(it.qdigit20.astype(np.int64)),
            torch.from_numpy(it.teacher20.astype(np.float32)),
            torch.from_numpy(it.teacher_feat20.astype(np.float32)),
            torch.tensor(int(it.pid_idx), dtype=torch.long),
            torch.tensor(int(it.y_digit), dtype=torch.long),
            str(it.sid),
        )


def collate_subjects(batch):
    xs, lies, qds, ts, tfeats, pids, yds, sids = zip(*batch)
    return (
        torch.stack(xs, dim=0),
        torch.stack(lies, dim=0),
        torch.stack(qds, dim=0),
        torch.stack(ts, dim=0),
        torch.stack(tfeats, dim=0),
        torch.stack(pids, dim=0),
        torch.stack(yds, dim=0),
        list(sids),
    )


class FeatureAlignHead(nn.Module):
    def __init__(self, s_dim: int, t_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(s_dim), int(hidden)),
            nn.GELU(),
            nn.LayerNorm(int(hidden)),
            nn.Linear(int(hidden), int(t_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float):
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, float(lam))


class Concat20GenLieStudent(nn.Module):
    """
    Concat-style student:
    - input [B,20,D]
    - flatten 20 questions and project jointly
    - emit per-question lie logits [B*20,2] and per-question embedding [B*20,re_dim]
    """

    def __init__(self, feat_dim: int, re_dim: int, hidden: int, drop: float, num_persons: int):
        super().__init__()
        self.re_dim = int(re_dim)
        self.encoder = nn.Sequential(
            nn.Linear(int(feat_dim) * 20, int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(drop)),
            nn.Linear(int(hidden), int(re_dim) * 20),
        )
        self.cls = nn.Sequential(
            nn.Linear(int(re_dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(drop)),
            nn.Linear(int(hidden), 2),
        )
        self.idh = nn.Sequential(
            nn.Linear(int(re_dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(drop)),
            nn.Linear(int(hidden), int(num_persons)),
        )

    def forward(self, x20: torch.Tensor, lam: float = 1.0, return_feat: bool = False):
        bsz = int(x20.shape[0])
        x = x20.reshape(bsz, -1)
        emb20 = self.encoder(x).reshape(bsz * 20, int(self.re_dim))
        logits = self.cls(emb20)
        id_logits = self.idh(grad_reverse(emb20, lam=float(lam)))
        if return_feat:
            return logits, id_logits, emb20
        return logits, id_logits


def forward_student(
    model: nn.Module,
    x20: torch.Tensor,
    device: torch.device,
    student_mode: str,
    lam: float = 1.0,
    return_feat: bool = False,
):
    mode = str(student_mode).lower().strip()
    if mode == "concat20":
        return model(x20.to(device).float(), lam=float(lam), return_feat=bool(return_feat))
    bsz = int(x20.shape[0])
    d = int(x20.shape[-1])
    x_flat = x20.view(bsz * 20, d).to(device).float()
    return model(x_flat, lam=float(lam), return_feat=bool(return_feat))


def aggregate_digit_scores_torch(scores20: torch.Tensor, qdigits20: torch.Tensor, mode: str = "sum") -> torch.Tensor:
    """
    scores20: [B,20], qdigits20: [B,20] in 1..10
    returns [B,10]
    """
    mode_l = str(mode).lower().strip()
    bsz = int(scores20.shape[0])
    out = torch.zeros((bsz, 10), dtype=scores20.dtype, device=scores20.device)
    neg = torch.full_like(scores20, -1e6)

    for d in range(1, 11):
        m = (qdigits20 == int(d)).to(scores20.dtype)  # [B,20]
        if mode_l == "sum":
            out[:, d - 1] = (scores20 * m).sum(dim=1)
        elif mode_l == "mean":
            den = m.sum(dim=1).clamp(min=1.0)
            out[:, d - 1] = (scores20 * m).sum(dim=1) / den
        elif mode_l == "max":
            masked = torch.where(m > 0, scores20, neg)
            out[:, d - 1] = masked.max(dim=1).values
        elif mode_l in ("logsumexp", "lse"):
            masked = torch.where(m > 0, scores20, neg)
            out[:, d - 1] = torch.logsumexp(masked, dim=1)
        else:
            raise ValueError(f"Unknown agg_mode: {mode}")
    return out


def compute_topk_from_logits(digit_logits: torch.Tensor, y_digit: torch.Tensor) -> Tuple[int, int, int, int]:
    rank = torch.argsort(digit_logits, dim=1, descending=True) + 1  # [B,10] in 1..10
    y = y_digit.view(-1, 1)
    h1 = int((rank[:, :1] == y).any(dim=1).sum().item())
    h2 = int((rank[:, :2] == y).any(dim=1).sum().item())
    h3 = int((rank[:, :3] == y).any(dim=1).sum().item())
    n = int(y_digit.shape[0])
    return h1, h2, h3, n


def summarize_mean_std(vals: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(vals), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _binary_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    n = int(y_true.size)
    if n == 0:
        return 0.5
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average-rank implementation for tie-aware ROC-AUC.
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and y_prob[order[j]] == y_prob[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0  # 1-based rank
        ranks[order[i:j]] = avg_rank
        i = j
    sum_ranks_pos = float(ranks[pos_mask].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def compute_binary_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    yt = np.asarray(list(y_true), dtype=np.int64).reshape(-1)
    yp = np.asarray(list(y_prob), dtype=np.float64).reshape(-1)
    if yt.size == 0:
        return {"n": 0.0, "acc": 0.0, "pos_f1": 0.0, "pos_auc": 0.5}
    yhat = (yp >= float(threshold)).astype(np.int64)
    acc = float((yhat == yt).mean())
    tp = int(((yhat == 1) & (yt == 1)).sum())
    fp = int(((yhat == 1) & (yt == 0)).sum())
    fn = int(((yhat == 0) & (yt == 1)).sum())
    prec = float(tp / max(1, tp + fp))
    rec = float(tp / max(1, tp + fn))
    f1 = float((2.0 * prec * rec) / max(1e-12, prec + rec))
    auc = _binary_auc(yt, yp)
    return {"n": float(yt.size), "acc": acc, "pos_f1": f1, "pos_auc": auc}


def estimate_model_complexity(
    model: nn.Module,
    in_dim: int,
    device: torch.device,
    student_mode: str,
) -> Dict[str, float]:
    params = int(sum(p.numel() for p in model.parameters()))
    flops = 0
    hooks: List[Any] = []

    def _linear_hook(mod: nn.Linear, inp, _out):
        nonlocal flops
        if len(inp) == 0 or (not torch.is_tensor(inp[0])):
            return
        x = inp[0]
        if x.ndim == 0:
            return
        batch_items = int(np.prod(list(x.shape[:-1]))) if x.ndim > 1 else int(x.shape[0])
        flops += int(batch_items) * int(mod.in_features) * int(mod.out_features) * 2

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    model_was_train = bool(model.training)
    model.eval()
    with torch.no_grad():
        x20 = torch.zeros((1, 20, int(in_dim)), dtype=torch.float32, device=device)
        _ = forward_student(
            model=model,
            x20=x20,
            device=device,
            student_mode=student_mode,
            lam=0.0,
            return_feat=False,
        )
    if model_was_train:
        model.train()
    for h in hooks:
        h.remove()

    # Add tiny constant for per-sample digit aggregation/logit transform.
    flops += 400
    return {
        "params_k": float(params / 1e3),
        "flops_per_sample": float(flops),
    }


def _import_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_rrg_model_file(teacher_repo_root: Path, teacher_model_file: str) -> Path:
    if str(teacher_model_file).strip():
        p = Path(str(teacher_model_file)).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"teacher_model_file not found: {p}")
        return p
    p = (teacher_repo_root / "rule_residual_gatednet_gsr_code.py").resolve()
    if not p.exists():
        raise FileNotFoundError(f"default RRG model file not found: {p}")
    return p


def _load_teacher_outputs_fold_rrg(
    fold: int,
    teacher_ckpt_root: Path,
    teacher_repo_root: Path,
    teacher_model_file: str,
    data_root: Path,
    subject_ids: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, np.ndarray]], int]:
    ckpt = teacher_ckpt_root / f"fold_{int(fold)}" / "best_ckpt.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {ckpt}")

    eval_mod_path = teacher_repo_root / "eval_transformer_5fold.py"
    if not eval_mod_path.exists():
        raise FileNotFoundError(f"RRG evaluator not found: {eval_mod_path}")
    eval_mod = _import_module_from_file(
        module_name=f"_rrg_eval_fold{int(fold)}",
        file_path=eval_mod_path,
    )

    model_file = _resolve_rrg_model_file(teacher_repo_root, teacher_model_file)
    mdl = eval_mod.import_model_module(str(model_file))

    # Rebuild teacher model with the same kwargs used in teacher training.
    try:
        obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(ckpt), map_location="cpu")
    ckpt_args = obj.get("args", {}) if isinstance(obj, dict) else {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    # Build subject list in fixed order for deterministic cache.
    keep_subjects: List[Any] = []
    keep_ids: List[str] = []
    for sid in subject_ids:
        s = eval_mod.load_subject_from_id(Path(data_root), str(sid))
        if s is not None:
            keep_subjects.append(s)
            keep_ids.append(str(sid))
    if len(keep_subjects) == 0:
        return {}, {}, 0

    ds_candidate_kwargs = {
        "fixed_len": int(ckpt_args.get("fixed_len", 1490)),
        "source_fs": float(ckpt_args.get("source_fs", 256.0)),
        "lowpass_hz": float(ckpt_args.get("lowpass_hz", 2.0)),
        "target_fs": float(ckpt_args.get("target_fs", 32.0)),
        "norm_mode": str(ckpt_args.get("norm_mode", "subject")),
        "use_smooth_channel": bool(int(ckpt_args.get("use_smooth_channel", 1))),
        "data_root": str(data_root),
        "use_personality": bool(int(ckpt_args.get("use_personality", 0))),
        "personality_mode": str(ckpt_args.get("personality_mode", "raw60")),
    }
    ds_sig = inspect.signature(mdl.SubjectDataset.__init__)
    ds_kwargs = {k: v for k, v in ds_candidate_kwargs.items() if k in ds_sig.parameters}
    ds = mdl.SubjectDataset(
        keep_subjects,
        use_augment=False,
        **ds_kwargs,
    )
    ld = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        # Use single-process loader here because the model module is imported dynamically.
        num_workers=0,
    )

    candidate_kwargs = {
        "emb_dim": int(ckpt_args.get("emb_dim", 128)),
        "drop": float(ckpt_args.get("drop", 0.12)),
        "nhead": int(ckpt_args.get("nhead", 4)),
        "n_layers": int(ckpt_args.get("n_layers", 1)),
        "ff_mult": float(ckpt_args.get("ff_mult", 2.0)),
        "enc_base": int(ckpt_args.get("enc_base", 16)),
        "alpha_lie": float(ckpt_args.get("alpha_lie", 0.1)),
        "lie_agg": str(ckpt_args.get("lie_agg", "logsumexp")),
        "alpha_rule": float(ckpt_args.get("alpha_rule", 1.0)),
        "use_ch_raw": int(ckpt_args.get("use_ch_raw", 1)),
        "use_ch_ma": int(ckpt_args.get("use_ch_ma", 1)),
        "use_ch_diff": int(ckpt_args.get("use_ch_diff", 1)),
        "use_rule_amp": int(ckpt_args.get("use_rule_amp", 1)),
        "use_rule_auc": int(ckpt_args.get("use_rule_auc", 1)),
        "use_rule_slope": int(ckpt_args.get("use_rule_slope", 1)),
        "use_rule_lat": int(ckpt_args.get("use_rule_lat", 1)),
        "use_pair_abs": int(ckpt_args.get("use_pair_abs", 1)),
        "use_pair_prod": int(ckpt_args.get("use_pair_prod", 1)),
        "use_pair_rule_aux": int(ckpt_args.get("use_pair_rule_aux", 1)),
        "use_pair_inst_aux": int(ckpt_args.get("use_pair_inst_aux", 1)),
        "use_rule_branch": int(ckpt_args.get("use_rule_branch", 1)),
        "use_lie_branch": int(ckpt_args.get("use_lie_branch", 1)),
        "use_conf_gate": int(ckpt_args.get("use_conf_gate", 1)),
        "use_smooth_channel": int(ckpt_args.get("use_smooth_channel", 1)),
        "hard_gate": int(ckpt_args.get("hard_gate", 0)),
        "use_transformer": int(ckpt_args.get("use_transformer", 1)),
        "use_personality": int(ckpt_args.get("use_personality", 0)),
        "personality_mode": str(ckpt_args.get("personality_mode", "raw60")),
    }
    sig = inspect.signature(mdl.DigitMILNetV2.__init__)
    model_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    model = mdl.DigitMILNetV2(**model_kwargs).to(device).eval()

    state_dict = None
    if isinstance(obj, dict):
        if isinstance(obj.get("model_state", None), dict):
            state_dict = obj["model_state"]
        elif isinstance(obj.get("state_dict", None), dict):
            state_dict = obj["state_dict"]
        elif isinstance(obj.get("model", None), dict):
            state_dict = obj["model"]
    if state_dict is None:
        raise RuntimeError(f"RRG checkpoint missing model weights: {ckpt}")
    model.load_state_dict(state_dict, strict=False)

    fwd_sig = inspect.signature(model.forward)
    supports_personality = ("personality" in fwd_sig.parameters)
    if not hasattr(model, "inst_head"):
        raise RuntimeError("RRG teacher missing `inst_head`; cannot extract question features.")

    hook_buf: Dict[str, torch.Tensor] = {}

    def _capture_inst_penultimate(_mod, inp, _out):
        hook_buf["pen"] = inp[0].detach()

    hook_handle = model.inst_head.register_forward_hook(_capture_inst_penultimate)
    out_s: Dict[str, Dict[int, float]] = {}
    out_f: Dict[str, Dict[int, np.ndarray]] = {}
    feat_dim = 0
    with torch.no_grad():
        for batch in ld:
            x20, dids, m20, y, y_inst, sid, personality = eval_mod.unpack_batch(batch)
            x20 = x20.to(device)
            dids = dids.to(device)
            m20 = m20.to(device)
            if isinstance(personality, torch.Tensor):
                personality = personality.to(device)
            hook_buf.clear()
            _logits10, inst_logits, _extra = eval_mod.model_forward(
                model=model,
                x20=x20,
                dids=dids,
                m20=m20,
                personality=personality,
                supports_personality=supports_personality,
            )
            inst_prob = torch.sigmoid(inst_logits).detach().cpu().numpy().astype(np.float32)  # [B,20]
            pen = hook_buf.get("pen", None)
            if pen is None:
                hook_handle.remove()
                raise RuntimeError("RRG teacher feature hook failed on inst_head.")
            emb = pen.detach().cpu().numpy().astype(np.float32).reshape(inst_prob.shape[0], 20, -1)  # [B,20,Dt]
            feat_dim = int(emb.shape[-1])
            m20_np = m20.detach().cpu().numpy().astype(np.float32)

            for b in range(inst_prob.shape[0]):
                sid_b = str(sid[b])
                for t, qid in enumerate(TARGET_QIDS):
                    if float(m20_np[b, t]) <= 0.5:
                        continue
                    out_s.setdefault(sid_b, {})[int(qid)] = float(inst_prob[b, t])
                    out_f.setdefault(sid_b, {})[int(qid)] = emb[b, t].astype(np.float32)
    hook_handle.remove()
    return out_s, out_f, int(feat_dim)


def _import_teacher_modules(teacher_repo_root: Path):
    if str(teacher_repo_root) not in sys.path:
        sys.path.insert(0, str(teacher_repo_root))
    from gsr_t2g.data import build_segment_table, load_subject_samples, records_to_arrays
    from gsr_t2g.model import Time2GraphPlusForGSR

    return build_segment_table, load_subject_samples, records_to_arrays, Time2GraphPlusForGSR


def _save_teacher_feat_cache_npz(path: Path, feat_map: Dict[str, Dict[int, np.ndarray]]) -> None:
    rows_sid: List[str] = []
    rows_qid: List[int] = []
    rows_feat: List[np.ndarray] = []
    for sid in sorted(feat_map.keys()):
        qmap = feat_map[sid]
        for qid in sorted(qmap.keys()):
            rows_sid.append(str(sid))
            rows_qid.append(int(qid))
            rows_feat.append(np.asarray(qmap[qid], dtype=np.float32))
    if len(rows_feat) == 0:
        return
    feat_arr = np.stack(rows_feat, axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        subject_id=np.asarray(rows_sid, dtype="U"),
        qid=np.asarray(rows_qid, dtype=np.int64),
        feat=feat_arr,
    )


def _load_teacher_feat_cache_npz(path: Path) -> Dict[str, Dict[int, np.ndarray]]:
    obj = np.load(str(path), allow_pickle=False)
    sid_arr = obj["subject_id"].astype(str).tolist()
    qid_arr = obj["qid"].astype(np.int64).tolist()
    feat_arr = obj["feat"].astype(np.float32)
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for sid, qid, feat in zip(sid_arr, qid_arr, feat_arr):
        out.setdefault(str(sid), {})[int(qid)] = np.asarray(feat, dtype=np.float32)
    return out


def load_teacher_outputs_fold(
    fold: int,
    teacher_ckpt_root: Path,
    teacher_repo_root: Path,
    teacher_type: str,
    teacher_model_file: str,
    data_root: Path,
    subject_ids: Sequence[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    teacher_cache_subject_dir: Optional[Path],
    cache_csv: Optional[Path],
    cache_npz: Optional[Path],
    reuse_score_cache: bool = True,
    reuse_feat_cache: bool = True,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, np.ndarray]], int]:
    """
    Return:
      score_map: sid -> {qid(3..22): score_pos}
      feat_map: sid -> {qid(3..22): teacher penultimate feat}
      feat_dim: teacher penultimate feature dim
    """
    score_map: Optional[Dict[str, Dict[int, float]]] = None
    feat_map: Optional[Dict[str, Dict[int, np.ndarray]]] = None

    if reuse_score_cache and cache_csv is not None and cache_csv.exists():
        tmp: Dict[str, Dict[int, float]] = {}
        with cache_csv.open("r", encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            for r in rd:
                sid = str(r.get("subject_id", "")).strip()
                if sid == "":
                    continue
                try:
                    qid = int(float(str(r.get("qid", "")).strip()))
                    sc = float(str(r.get("score_pos", "")).strip())
                except Exception:
                    continue
                tmp.setdefault(sid, {})[qid] = sc
        if len(tmp) > 0:
            score_map = tmp

    if reuse_feat_cache and cache_npz is not None and cache_npz.exists():
        tmp_f = _load_teacher_feat_cache_npz(cache_npz)
        if len(tmp_f) > 0:
            feat_map = tmp_f

    if score_map is not None and feat_map is not None:
        any_sid = next(iter(feat_map.keys()))
        any_qid = next(iter(feat_map[any_sid].keys()))
        feat_dim = int(np.asarray(feat_map[any_sid][any_qid]).shape[-1])
        return score_map, feat_map, feat_dim

    teacher_type_l = str(teacher_type).lower().strip()
    if teacher_type_l == "rrg":
        out_s, out_f, feat_dim = _load_teacher_outputs_fold_rrg(
            fold=fold,
            teacher_ckpt_root=teacher_ckpt_root,
            teacher_repo_root=teacher_repo_root,
            teacher_model_file=teacher_model_file,
            data_root=data_root,
            subject_ids=subject_ids,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        ckpt = teacher_ckpt_root / f"fold_{int(fold)}" / "best_ckpt.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"teacher checkpoint not found: {ckpt}")

        build_segment_table, load_subject_samples, records_to_arrays, Time2GraphPlusForGSR = _import_teacher_modules(teacher_repo_root)

        try:
            obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(ckpt), map_location="cpu")
        model = Time2GraphPlusForGSR.from_ckpt(obj)
        cfg = model.cfg

        cache_dir_txt = str(teacher_cache_subject_dir) if teacher_cache_subject_dir is not None else ""
        samples = load_subject_samples(
            data_root=str(data_root),
            subject_ids=list(subject_ids),
            cfg=cfg,
            cache_dir=cache_dir_txt,
            num_workers=int(max(1, num_workers)),
        )

        keep = [sid for sid in subject_ids if sid in samples]
        if len(keep) == 0:
            return {}, {}, 0

        records, x_raw, p_raw, h_raw = build_segment_table([samples[sid] for sid in keep])
        x, y, meta, p5, h3 = records_to_arrays(records, x_raw, p_raw, h_raw)
        feats, adjs = model.build_graph_inputs(x)

        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(feats.astype(np.float32)),
            torch.from_numpy(adjs.astype(np.float32)),
            torch.from_numpy(p5.astype(np.float32)),
            torch.from_numpy(h3.astype(np.float32)),
        )
        ld = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

        model = model.to(device).eval()
        if not hasattr(model, "cls") or not isinstance(model.cls, nn.Linear):
            raise RuntimeError("Teacher model does not expose linear classifier `cls`; cannot capture penultimate features.")

        hook_buf: Dict[str, torch.Tensor] = {}

        def _capture_penultimate(_mod, inp, _out):
            hook_buf["pen"] = inp[0].detach()

        hook_handle = model.cls.register_forward_hook(_capture_penultimate)

        scores: List[np.ndarray] = []
        feat_rows: List[np.ndarray] = []
        with torch.no_grad():
            for feat, adj, pp, hh in ld:
                feat = feat.to(device).float()
                adj = adj.to(device).float()
                pp = pp.to(device).float()
                hh = hh.to(device).float()
                hook_buf.clear()
                logits = model(feat, adj, pp, hh)
                prob = torch.softmax(logits, dim=1)[:, 1]
                scores.append(prob.detach().cpu().numpy().astype(np.float32))
                pen = hook_buf.get("pen", None)
                if pen is None:
                    hook_handle.remove()
                    raise RuntimeError("Teacher penultimate feature hook failed; no feature captured.")
                feat_rows.append(pen.detach().cpu().numpy().astype(np.float32))
        hook_handle.remove()

        score_all = np.concatenate(scores, axis=0) if len(scores) > 0 else np.zeros((0,), dtype=np.float32)
        feat_all = np.concatenate(feat_rows, axis=0) if len(feat_rows) > 0 else np.zeros((0, 0), dtype=np.float32)
        feat_dim = int(feat_all.shape[1]) if feat_all.ndim == 2 and feat_all.shape[0] > 0 else 0

        sid_arr = list(meta["subject_id"])
        seg_arr = np.asarray(meta["seg_idx"], dtype=np.int64)

        out_s = {}
        out_f = {}
        for i in range(len(sid_arr)):
            sid = str(sid_arr[i])
            seg = int(seg_arr[i])
            if seg < 0 or seg >= len(TARGET_QIDS):
                continue
            qid = int(TARGET_QIDS[seg])
            out_s.setdefault(sid, {})[qid] = float(score_all[i])
            if i < feat_all.shape[0]:
                out_f.setdefault(sid, {})[qid] = feat_all[i].astype(np.float32)

    if cache_csv is not None:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        with cache_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["subject_id", "qid", "score_pos"])
            for sid in sorted(out_s.keys()):
                for qid in sorted(out_s[sid].keys()):
                    w.writerow([sid, int(qid), float(out_s[sid][qid])])

    if cache_npz is not None:
        _save_teacher_feat_cache_npz(cache_npz, out_f)

    return out_s, out_f, int(feat_dim)


def build_kd_subject_items(
    subjects: Sequence[Any],
    in_dim: int,
    mu: np.ndarray,
    sd: np.ndarray,
    pid2idx: Dict[str, int],
    teacher_score_map: Dict[str, Dict[int, float]],
    teacher_feat_map: Dict[str, Dict[int, np.ndarray]],
    teacher_feat_dim: int,
    use_feat_align: bool = True,
) -> List[KDSubjectItem]:
    out: List[KDSubjectItem] = []
    feat_dim_eff = max(1, int(teacher_feat_dim))
    for s in subjects:
        sid = str(s.sid)
        tmap_s = teacher_score_map.get(sid, {})
        tmap_f = teacher_feat_map.get(sid, {})
        if len(tmap_s) == 0 or (bool(use_feat_align) and len(tmap_f) == 0):
            continue

        x20 = np.zeros((20, int(in_dim)), dtype=np.float32)
        lie20 = np.zeros((20,), dtype=np.int64)
        qdigit20 = np.zeros((20,), dtype=np.int64)
        teacher20 = np.zeros((20,), dtype=np.float32)
        teacher_feat20 = np.zeros((20, int(feat_dim_eff)), dtype=np.float32)

        ok = True
        for i, qid in enumerate(TARGET_QIDS):
            v = s.q_feats.get(int(qid))
            if v is None:
                ok = False
                break
            if int(qid) not in tmap_s:
                ok = False
                break
            if bool(use_feat_align) and int(qid) not in tmap_f:
                ok = False
                break
            qd = digit_from_qid(int(qid))
            if qd is None:
                ok = False
                break

            x = vec_to_dim(v, int(in_dim))
            x = (x - mu) / sd
            x20[i] = x.astype(np.float32)
            lie20[i] = int(s.q_lie.get(int(qid), 0))
            qdigit20[i] = int(qd)
            teacher20[i] = float(tmap_s[int(qid)])
            if bool(use_feat_align):
                tfeat = np.asarray(tmap_f[int(qid)], dtype=np.float32).reshape(-1)
                if tfeat.shape[0] != int(feat_dim_eff):
                    ok = False
                    break
                teacher_feat20[i] = tfeat.astype(np.float32)

        if not ok:
            continue

        out.append(
            KDSubjectItem(
                sid=sid,
                # Test subjects are not guaranteed to appear in train pid2idx.
                # Fallback to 0 for non-train IDs (pid index is only used for train ID loss).
                pid_idx=int(pid2idx.get(sid, 0)),
                y_digit=int(s.y_digit),
                x20=x20,
                lie20=lie20,
                qdigit20=qdigit20,
                teacher20=teacher20,
                teacher_feat20=teacher_feat20,
            )
        )
    return out


def evaluate_student(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    agg_mode: str,
    student_mode: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    total_h1 = 0
    total_h2 = 0
    total_h3 = 0
    total_n = 0
    rows: List[Dict[str, Any]] = []
    q_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for x20, lie20, qd20, t20, tfeat20, pid, yd, sids in loader:
            bsz = int(x20.shape[0])
            logits, _ = forward_student(
                model=model,
                x20=x20,
                device=device,
                student_mode=student_mode,
                lam=0.0,
                return_feat=False,
            )
            logits2 = logits.view(bsz, 20, 2)
            s_prob = torch.softmax(logits2, dim=-1)[:, :, 1].clamp(1e-5, 1.0 - 1e-5)
            s_logit = torch.log(s_prob) - torch.log(1.0 - s_prob)

            digit_logits = aggregate_digit_scores_torch(s_logit, qd20.to(device), mode=agg_mode)
            h1, h2, h3, n = compute_topk_from_logits(digit_logits, yd.to(device))
            total_h1 += int(h1)
            total_h2 += int(h2)
            total_h3 += int(h3)
            total_n += int(n)

            rank = (torch.argsort(digit_logits, dim=1, descending=True) + 1).detach().cpu().numpy().astype(np.int64)
            s_prob_np = s_prob.detach().cpu().numpy().astype(np.float32)
            lie_np = lie20.detach().cpu().numpy().astype(np.int64)
            qd_np = qd20.detach().cpu().numpy().astype(np.int64)
            for i in range(bsz):
                rows.append(
                    {
                        "subject_id": str(sids[i]),
                        "y_true": int(yd[i].item()),
                        "y_hat": int(rank[i, 0]),
                        "top3": [int(rank[i, 0]), int(rank[i, 1]), int(rank[i, 2])],
                        "agg_mode": str(agg_mode),
                    }
                )
                for j, qid in enumerate(TARGET_QIDS):
                    q_rows.append(
                        {
                            "subject_id": str(sids[i]),
                            "qid": int(qid),
                            "q_digit": int(qd_np[i, j]),
                            "y_true": int(lie_np[i, j]),
                            "y_prob": float(s_prob_np[i, j]),
                            "y_pred": int(s_prob_np[i, j] >= 0.5),
                        }
                    )

    den = max(1, total_n)
    m = {
        "n": float(total_n),
        "top1": float(total_h1 / den),
        "top2": float(total_h2 / den),
        "top3": float(total_h3 / den),
        "agg_mode": str(agg_mode),
    }
    return m, rows, q_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cross-modal KD: GSR teacher (Time2Graph/RRG) -> GenLie student (5-fold)")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--split_root", type=str, default="./data_split/5fold")
    ap.add_argument("--out_dir", type=str, default="./results/video_genlie_kd_5fold")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--student_modality", type=str, default="video")

    ap.add_argument("--teacher_repo_root", type=str, default="./gsr_time2graphplus")
    ap.add_argument("--teacher_ckpt_root", type=str, required=True, help="dir containing fold_i/best_ckpt.pt")
    ap.add_argument("--teacher_type", type=str, default="time2graph", choices=["time2graph", "rrg"])
    ap.add_argument("--teacher_model_file", type=str, default="", help="used for --teacher_type rrg")
    ap.add_argument("--teacher_cache_subject_dir", type=str, default="")
    ap.add_argument("--teacher_infer_batch_size", type=int, default=256)
    ap.add_argument("--teacher_num_workers", type=int, default=4)
    ap.add_argument("--reuse_teacher_score_cache", type=int, default=1, choices=[0, 1])
    ap.add_argument("--reuse_teacher_feat_cache", type=int, default=1, choices=[0, 1])

    ap.add_argument("--batch_size", type=int, default=16, help="subject batch size")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", type=int, default=0, choices=[0, 1])

    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--re_embed_dim", type=int, default=256)
    ap.add_argument("--dropout_rate", type=float, default=0.3)
    ap.add_argument("--student_mode", type=str, default="per_q", choices=["per_q", "concat20"])
    ap.add_argument("--triplet_margin", type=float, default=1.0)
    ap.add_argument("--id_loss_weight", type=float, default=0.0)
    ap.add_argument("--triplet_loss_weight", type=float, default=0.0)
    ap.add_argument("--id_loss_lambda", type=float, default=1.0)

    ap.add_argument("--input_dim_cap", type=int, default=4096)
    ap.add_argument("--feature_qid_shift", type=int, default=0)
    ap.add_argument("--min_valid_q", type=int, default=20)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--select_on", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--agg_modes", type=str, default="sum,mean,max,logsumexp")
    ap.add_argument("--kd_digit_agg_mode", type=str, default="sum", choices=["sum", "mean", "max", "logsumexp"])

    # Hard losses
    ap.add_argument("--lambda_lie_ce", type=float, default=1.0)
    ap.add_argument("--lambda_digit_ce", type=float, default=0.3)
    ap.add_argument("--lambda_rank_kd", type=float, default=0.8)
    ap.add_argument("--lambda_digit_kd", type=float, default=1.2)
    ap.add_argument("--lambda_feat_align", type=float, default=0.0)

    # progressive stages
    ap.add_argument("--temp_rank", type=float, default=2.0)
    ap.add_argument("--temp_digit", type=float, default=2.0)
    ap.add_argument("--stage1_epochs", type=int, default=8)
    ap.add_argument("--stage2_epochs", type=int, default=12)
    ap.add_argument("--stage3_digit_ramp_epochs", type=int, default=10)
    ap.add_argument("--progressive_mode", type=str, default="linear", choices=["linear", "step", "sigmoid", "cosine", "gated_sigmoid", "overlap_sigmoid"])
    ap.add_argument("--progressive_path_order", type=str, default="feature_first", choices=["feature_first", "digit_first", "joint"])
    ap.add_argument("--progressive_overlap_ratio", type=float, default=0.0)
    ap.add_argument("--progressive_overlap_width", type=float, default=6.0)
    ap.add_argument("--rank_warm_width", type=float, default=4.0)
    ap.add_argument("--feat_warm_width", type=float, default=6.0)
    ap.add_argument("--digit_warm_width", type=float, default=8.0)
    ap.add_argument("--feat_max_weight", type=float, default=1.0)
    ap.add_argument("--kd_conf_mode", type=str, default="none", choices=["none", "hard", "ramp"])
    ap.add_argument("--kd_conf_high", type=float, default=0.35, help="high threshold for ramp start")
    ap.add_argument("--kd_conf_low", type=float, default=0.05, help="low threshold for ramp end / hard mode")
    ap.add_argument("--kd_conf_min_keep_frac", type=float, default=0.15, help="always keep top confidence fraction")
    ap.add_argument("--logitstd_mode", type=str, default="none", choices=["none", "rank", "digit", "both"])
    ap.add_argument("--logitstd_eps", type=float, default=1e-6)
    ap.add_argument("--online_gap_mode", type=str, default="ema_hard", choices=["off", "ema_hard"])
    ap.add_argument("--gap_update_warmup_epochs", type=int, default=5)
    ap.add_argument("--gap_update_interval", type=int, default=2)
    ap.add_argument("--gap_ema_momentum", type=float, default=0.8)
    ap.add_argument("--gap_route_no_feature_enter", type=float, default=0.34)
    ap.add_argument("--gap_route_no_feature_exit", type=float, default=0.40)
    ap.add_argument("--gap_route_digit_enter", type=float, default=0.46)
    ap.add_argument("--gap_route_digit_exit", type=float, default=0.52)
    ap.add_argument("--gap_route_feature_exit", type=float, default=0.56)
    ap.add_argument("--gap_route_feature_enter", type=float, default=0.62)
    ap.add_argument(
        "--gap_route_profile",
        type=str,
        default="auto",
        choices=["auto", "manual", "video_strong_feature", "audio_weak_feature", "balanced"],
    )
    ap.add_argument("--gap_adaptive_feat_weight", type=int, default=1, choices=[0, 1])
    ap.add_argument("--gap_obs_scale", type=float, default=None, help="raw gap calibration scale (None: preset by route profile)")
    ap.add_argument("--gap_obs_bias", type=float, default=None, help="raw gap calibration bias (None: preset by route profile)")
    ap.add_argument("--gap_route_min_hold_epochs", type=int, default=3)
    ap.add_argument("--gap_online_max_rows", type=int, default=4096)
    ap.add_argument("--gap_online_min_reliable_similarity", type=float, default=0.02)
    ap.add_argument("--nofeat_dynamic_route", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--nofeat_route_require_rank_kd",
        type=int,
        default=1,
        choices=[0, 1],
        help="when 1, nofeat dynamic route requires lambda_rank_kd>0; when 0, route can run even if rank kd loss is disabled",
    )
    ap.add_argument(
        "--rank0_force_online_gap_fallback",
        type=int,
        default=1,
        choices=[0, 1],
        help="when 1 and lambda_rank_kd<=0 with online_gap enabled, disable nofeat route and fall back to online-gap routing",
    )
    ap.add_argument("--nofeat_route_bootstrap", type=str, default="feature_first", choices=["feature_first", "joint", "digit_first"])
    ap.add_argument("--nofeat_route_ema_momentum", type=float, default=0.8)
    ap.add_argument("--nofeat_route_min_hold_epochs", type=int, default=3)
    ap.add_argument("--nofeat_route_switch_up", type=float, default=1.08)
    ap.add_argument("--nofeat_route_switch_down", type=float, default=0.92)
    ap.add_argument("--nofeat_route_digit_enter", type=float, default=0.78)
    ap.add_argument("--nofeat_route_digit_exit", type=float, default=0.86)
    ap.add_argument("--feat_gate_mode", type=str, default="grad_conflict", choices=["off", "grad_conflict"])
    ap.add_argument("--feat_gate_init", type=float, default=1.0)
    ap.add_argument("--feat_gate_ema_momentum", type=float, default=0.8)
    ap.add_argument("--feat_gate_sigmoid_scale", type=float, default=8.0)
    ap.add_argument("--feat_gate_center", type=float, default=0.0)
    ap.add_argument("--feat_gate_min", type=float, default=0.0)
    ap.add_argument("--feat_gate_max", type=float, default=1.0)
    ap.add_argument("--feat_gate_update_interval_batches", type=int, default=1)
    ap.add_argument("--route_feat_scale_enable", type=int, default=1, choices=[0, 1])
    ap.add_argument("--route_feat_scale_feature_first", type=float, default=1.0)
    ap.add_argument("--route_feat_scale_joint", type=float, default=1.0)
    ap.add_argument("--route_feat_scale_digit_first", type=float, default=1.0)
    ap.add_argument("--route_feat_scale_no_feature", type=float, default=0.0)

    return ap.parse_args()


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student_modality = normalize_modality_name(getattr(cfg, "student_modality", "video"))
    resolved_gap_profile = apply_gap_route_profile(cfg, student_modality)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(out_dir / "train.log", encoding="utf-8"), logging.StreamHandler()],
    )

    seed_everything(int(cfg.seed))
    device = get_device(str(cfg.device))
    agg_modes = _parse_csv_list(str(cfg.agg_modes))
    if len(agg_modes) == 0:
        agg_modes = ["sum"]
    logging.info("device=%s", device)
    logging.info("student_modality=%s data_root=%s", student_modality, cfg.data_root)
    logging.info("teacher_ckpt_root=%s", cfg.teacher_ckpt_root)
    logging.info("teacher_type=%s teacher_repo_root=%s", str(cfg.teacher_type), str(cfg.teacher_repo_root))
    logging.info("workflow: Q2D progressive KD (question->digit)")
    logging.info("student_mode=%s", str(cfg.student_mode))
    logging.info("model selection on: %s", str(cfg.select_on))
    logging.info(
        "gap route profile: requested=%s resolved=%s adaptive_feat_weight=%s",
        str(getattr(cfg, "gap_route_profile", "auto")),
        str(resolved_gap_profile),
        "on" if bool(int(getattr(cfg, "gap_adaptive_feat_weight", 1))) else "off",
    )
    logging.info(
        "gap calibration: scale=%.3f bias=%.3f",
        float(getattr(cfg, "gap_obs_scale", 1.0)),
        float(getattr(cfg, "gap_obs_bias", 0.0)),
    )
    logging.info("feature alignment enabled: %s (lambda_feat_align=%.4f)", bool(float(cfg.lambda_feat_align) > 0.0), float(cfg.lambda_feat_align))
    logging.info(
        "online gap: mode=%s warmup=%d interval=%d ema=%.2f hard_route=[no_feat_enter=%.2f no_feat_exit=%.2f digit_enter=%.2f digit_exit=%.2f feature_exit=%.2f feature_enter=%.2f] hold=%d",
        str(cfg.online_gap_mode),
        int(cfg.gap_update_warmup_epochs),
        int(cfg.gap_update_interval),
        float(cfg.gap_ema_momentum),
        float(cfg.gap_route_no_feature_enter),
        float(cfg.gap_route_no_feature_exit),
        float(cfg.gap_route_digit_enter),
        float(cfg.gap_route_digit_exit),
        float(cfg.gap_route_feature_exit),
        float(cfg.gap_route_feature_enter),
        int(cfg.gap_route_min_hold_epochs),
    )
    logging.info(
        "no-feature dynamic route: enabled=%s require_rank_kd=%s rank0_fallback_to_online_gap=%s bootstrap=%s ema=%.2f hold=%d switch=[up=%.2f down=%.2f] digit=[enter=%.2f exit=%.2f]",
        "on" if bool(int(getattr(cfg, "nofeat_dynamic_route", 1))) else "off",
        "on" if bool(int(getattr(cfg, "nofeat_route_require_rank_kd", 1))) else "off",
        "on" if bool(int(getattr(cfg, "rank0_force_online_gap_fallback", 1))) else "off",
        str(getattr(cfg, "nofeat_route_bootstrap", "feature_first")),
        float(getattr(cfg, "nofeat_route_ema_momentum", 0.8)),
        int(getattr(cfg, "nofeat_route_min_hold_epochs", 3)),
        float(getattr(cfg, "nofeat_route_switch_up", 1.08)),
        float(getattr(cfg, "nofeat_route_switch_down", 0.92)),
        float(getattr(cfg, "nofeat_route_digit_enter", 0.78)),
        float(getattr(cfg, "nofeat_route_digit_exit", 0.86)),
    )
    logging.info(
        "feature gate: mode=%s init=%.2f ema=%.2f sigmoid=[scale=%.2f center=%.2f] clamp=[%.2f, %.2f] update_interval_batches=%d",
        str(getattr(cfg, "feat_gate_mode", "grad_conflict")),
        float(getattr(cfg, "feat_gate_init", 1.0)),
        float(getattr(cfg, "feat_gate_ema_momentum", 0.8)),
        float(getattr(cfg, "feat_gate_sigmoid_scale", 8.0)),
        float(getattr(cfg, "feat_gate_center", 0.0)),
        float(getattr(cfg, "feat_gate_min", 0.0)),
        float(getattr(cfg, "feat_gate_max", 1.0)),
        int(getattr(cfg, "feat_gate_update_interval_batches", 1)),
    )
    logging.info(
        "route feature scale: enabled=%s [feature_first=%.2f joint=%.2f digit_first=%.2f no_feature=%.2f]",
        "on" if bool(int(getattr(cfg, "route_feat_scale_enable", 1))) else "off",
        float(getattr(cfg, "route_feat_scale_feature_first", 1.0)),
        float(getattr(cfg, "route_feat_scale_joint", 1.0)),
        float(getattr(cfg, "route_feat_scale_digit_first", 1.0)),
        float(getattr(cfg, "route_feat_scale_no_feature", 0.0)),
    )
    logging.info(
        "stages: stage1=%d stage2=%d stage3_ramp=%d",
        int(cfg.stage1_epochs),
        int(cfg.stage2_epochs),
        int(cfg.stage3_digit_ramp_epochs),
    )
    logging.info(
            "progressive: mode=%s order=%s rank_w=%.2f feat_w=%.2f digit_w=%.2f feat_max=%.2f",
            str(cfg.progressive_mode),
            str(cfg.progressive_path_order),
            float(cfg.rank_warm_width),
            float(cfg.feat_warm_width),
        float(cfg.digit_warm_width),
        float(cfg.feat_max_weight),
    )
    logging.info(
        "kd_conf: mode=%s high=%.3f low=%.3f keep_frac=%.2f",
        str(cfg.kd_conf_mode),
        float(cfg.kd_conf_high),
        float(cfg.kd_conf_low),
        float(cfg.kd_conf_min_keep_frac),
    )
    logging.info(
        "logitstd: mode=%s eps=%.1e",
        str(cfg.logitstd_mode),
        float(cfg.logitstd_eps),
    )
    logging.info("selection agg candidates=%s", agg_modes)

    split_root = Path(cfg.split_root)
    all_ids: List[str] = []
    for fold in range(1, 6):
        all_ids.extend(_read_ids(split_root / f"fold_{fold}" / "train_ids.txt"))
        all_ids.extend(_read_ids(split_root / f"fold_{fold}" / "test_ids.txt"))
    all_ids = sorted(set(all_ids))
    if len(all_ids) == 0:
        raise RuntimeError("No subject IDs found in split_root")

    data_root = Path(cfg.data_root)
    student_map: Dict[str, Any] = {}
    for sid in all_ids:
        s = load_subject_sample(
            data_root=data_root,
            sid=sid,
            min_valid_q=int(cfg.min_valid_q),
            feature_qid_shift=int(cfg.feature_qid_shift),
        )
        if s is not None:
            student_map[sid] = s
    logging.info("loaded %s subjects: %d/%d", student_modality, len(student_map), len(all_ids))
    if len(student_map) == 0:
        raise RuntimeError(f"No valid {student_modality} subject loaded")

    teacher_repo_root = Path(cfg.teacher_repo_root)
    teacher_ckpt_root = Path(cfg.teacher_ckpt_root)
    teacher_cache_subject_dir = Path(cfg.teacher_cache_subject_dir) if str(cfg.teacher_cache_subject_dir).strip() else None

    fold_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    detail_q_rows: List[Dict[str, Any]] = []
    complexity_rows: List[Dict[str, Any]] = []
    total_n = 0
    total_h1 = 0
    total_h2 = 0
    total_h3 = 0

    for fold in range(1, 6):
        tr_all = [x for x in _read_ids(split_root / f"fold_{fold}" / "train_ids.txt") if x in student_map]
        te_ids = [x for x in _read_ids(split_root / f"fold_{fold}" / "test_ids.txt") if x in student_map]
        if len(tr_all) == 0 or len(te_ids) == 0:
            logging.warning("fold %d skipped: train=%d test=%d", fold, len(tr_all), len(te_ids))
            continue

        if str(cfg.select_on).lower().strip() == "test":
            tr_ids = list(tr_all)
            va_ids: List[str] = []
            logging.info("fold %d split: train=%d val=%d test=%d (select_on=test)", fold, len(tr_ids), 0, len(te_ids))
        else:
            tr_ids, va_ids = split_train_val_ids(tr_all, val_ratio=float(cfg.val_ratio), seed=int(cfg.seed) + fold)
            if len(tr_ids) == 0 or len(va_ids) == 0:
                logging.warning("fold %d skipped after val split: train=%d val=%d", fold, len(tr_ids), len(va_ids))
                continue
            logging.info("fold %d split: train=%d val=%d test=%d", fold, len(tr_ids), len(va_ids), len(te_ids))

        score_cache_csv = out_dir / f"teacher_scores_fold{fold}.csv"
        feat_cache_npz = out_dir / f"teacher_feats_fold{fold}.npz"
        teacher_scores, teacher_feats, teacher_feat_dim = load_teacher_outputs_fold(
            fold=fold,
            teacher_ckpt_root=teacher_ckpt_root,
            teacher_repo_root=teacher_repo_root,
            teacher_type=str(cfg.teacher_type),
            teacher_model_file=str(cfg.teacher_model_file),
            data_root=data_root,
            subject_ids=tr_all + te_ids,
            device=device,
            batch_size=int(cfg.teacher_infer_batch_size),
            num_workers=int(cfg.teacher_num_workers),
            teacher_cache_subject_dir=teacher_cache_subject_dir,
            cache_csv=score_cache_csv,
            cache_npz=feat_cache_npz,
            reuse_score_cache=bool(int(cfg.reuse_teacher_score_cache)),
            reuse_feat_cache=bool(int(cfg.reuse_teacher_feat_cache)),
        )

        use_feat_align_fold = bool(float(cfg.lambda_feat_align) > 0.0 and int(teacher_feat_dim) > 0)
        use_online_gap_fold = bool(str(cfg.online_gap_mode).lower().strip() != "off" and int(teacher_feat_dim) > 0)
        require_rank_kd = bool(int(getattr(cfg, "nofeat_route_require_rank_kd", 1)) == 1)
        rank_gate_ok = (float(cfg.lambda_rank_kd) > 0.0) if require_rank_kd else True
        use_nofeat_route_fold = bool(
            int(getattr(cfg, "nofeat_dynamic_route", 1)) == 1
            and rank_gate_ok
            and float(cfg.lambda_digit_kd) > 0.0
        )
        # If rank KD loss is disabled but online-gap routing is available, prefer online-gap dynamic routing.
        # This keeps dynamic routing active while avoiding no-feature route collapse on rank-free training.
        rank0_force_online_gap_fallback = bool(int(getattr(cfg, "rank0_force_online_gap_fallback", 1)) == 1)
        if rank0_force_online_gap_fallback and float(cfg.lambda_rank_kd) <= 0.0 and use_online_gap_fold:
            use_nofeat_route_fold = False
        use_auto_feat_gate_fold = bool(
            use_feat_align_fold
            and str(getattr(cfg, "feat_gate_mode", "grad_conflict")).lower().strip() == "grad_conflict"
        )
        if use_nofeat_route_fold and use_online_gap_fold:
            use_online_gap_fold = False
        if float(cfg.lambda_feat_align) > 0.0 and not use_feat_align_fold:
            logging.warning(
                "fold %d: requested feature alignment but teacher_feat_dim=%d; disable feat_align for this fold.",
                fold,
                int(teacher_feat_dim),
            )

        def _has_teacher_for_sid(sid: str) -> bool:
            if sid not in teacher_scores:
                return False
            if (use_feat_align_fold or use_online_gap_fold) and sid not in teacher_feats:
                return False
            return True

        tr_subj = [student_map[sid] for sid in tr_ids if _has_teacher_for_sid(sid)]
        va_subj = [student_map[sid] for sid in va_ids if _has_teacher_for_sid(sid)]
        te_subj = [student_map[sid] for sid in te_ids if _has_teacher_for_sid(sid)]
        need_val = str(cfg.select_on).lower().strip() == "val"
        if len(tr_subj) == 0 or len(te_subj) == 0 or (need_val and len(va_subj) == 0):
            logging.warning(
                "fold %d skipped after teacher alignment: train=%d val=%d test=%d",
                fold,
                len(tr_subj),
                len(va_subj),
                len(te_subj),
            )
            continue

        in_dim = choose_input_dim(tr_subj, int(cfg.input_dim_cap))
        mu, sd = compute_norm_stats(tr_subj, in_dim)
        person_list = sorted(list({s.sid for s in tr_subj}))
        pid2idx = {sid: i for i, sid in enumerate(person_list)}

        need_teacher_feat_items = bool(use_feat_align_fold or use_online_gap_fold)
        feat_dim_for_items = int(teacher_feat_dim) if need_teacher_feat_items else 1
        tr_items = build_kd_subject_items(
            tr_subj,
            in_dim,
            mu,
            sd,
            pid2idx,
            teacher_scores,
            teacher_feats,
            feat_dim_for_items,
            use_feat_align=need_teacher_feat_items,
        )
        va_items = build_kd_subject_items(
            va_subj,
            in_dim,
            mu,
            sd,
            pid2idx,
            teacher_scores,
            teacher_feats,
            feat_dim_for_items,
            use_feat_align=need_teacher_feat_items,
        )
        te_items = build_kd_subject_items(
            te_subj,
            in_dim,
            mu,
            sd,
            pid2idx,
            teacher_scores,
            teacher_feats,
            feat_dim_for_items,
            use_feat_align=need_teacher_feat_items,
        )
        if len(tr_items) == 0 or len(te_items) == 0 or (need_val and len(va_items) == 0):
            logging.warning(
                "fold %d skipped after item build: train=%d val=%d test=%d",
                fold,
                len(tr_items),
                len(va_items),
                len(te_items),
            )
            continue

        online_gap_state = init_online_gap_state(cfg=cfg, student_modality=student_modality)
        logging.info(
            "fold %d online gap init: source=%s ema_gap=%.4f path=%s hold=%d",
            fold,
            online_gap_state.source,
            float(online_gap_state.ema_gap),
            str(online_gap_state.path_order),
            int(online_gap_state.hold_epochs),
        )
        nofeat_route_state: Optional[NoFeatRouteState] = None
        if use_nofeat_route_fold:
            nofeat_route_state = init_nofeat_route_state(cfg)
            logging.info(
                "fold %d nofeat route init: source=%s path=%s hold=%d ratio=%.4f",
                fold,
                str(nofeat_route_state.source),
                str(nofeat_route_state.path_order),
                int(nofeat_route_state.hold_epochs),
                float(nofeat_route_state.rank_digit_ratio),
            )
        feature_gate_state: Optional[FeatureGateState] = None
        if use_auto_feat_gate_fold:
            feature_gate_state = init_feature_gate_state(cfg)
            logging.info(
                "fold %d feature gate init: source=%s gate=%.4f ema_grad_cos=%.4f",
                fold,
                str(feature_gate_state.source),
                float(feature_gate_state.gate),
                float(feature_gate_state.ema_grad_cos),
            )

        tr_ld = DataLoader(KDSubjectDataset(tr_items), batch_size=int(cfg.batch_size), shuffle=True, collate_fn=collate_subjects)
        va_ld = DataLoader(KDSubjectDataset(va_items), batch_size=int(cfg.batch_size), shuffle=False, collate_fn=collate_subjects) if len(va_items) > 0 else None
        te_ld = DataLoader(KDSubjectDataset(te_items), batch_size=int(cfg.batch_size), shuffle=False, collate_fn=collate_subjects)

        lie_labels = np.concatenate([it.lie20 for it in tr_items], axis=0).astype(np.int64)
        cnt = Counter(lie_labels.tolist())
        w_pos = float(max(1, cnt.get(0, 1)) / max(1, cnt.get(1, 1)))

        if str(cfg.student_mode).lower().strip() == "concat20":
            model = Concat20GenLieStudent(
                feat_dim=int(in_dim),
                re_dim=int(cfg.re_embed_dim),
                hidden=int(cfg.hidden_dim),
                drop=float(cfg.dropout_rate),
                num_persons=max(2, len(person_list)),
            ).to(device)
        else:
            model = GenLieModel(
                feat_dim=int(in_dim),
                re_dim=int(cfg.re_embed_dim),
                hidden=int(cfg.hidden_dim),
                drop=float(cfg.dropout_rate),
                num_persons=max(2, len(person_list)),
            ).to(device)
        align_head: Optional[nn.Module] = None
        opt_params = list(model.parameters())
        if use_feat_align_fold:
            align_head = FeatureAlignHead(int(cfg.re_embed_dim), int(feat_dim_for_items), hidden=min(512, int(cfg.hidden_dim))).to(device)
            opt_params.extend(list(align_head.parameters()))

        opt = torch.optim.AdamW(opt_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        ce_q = nn.CrossEntropyLoss(weight=torch.tensor([1.0, w_pos], dtype=torch.float32, device=device))
        ce_id = nn.CrossEntropyLoss()
        ce_digit = nn.CrossEntropyLoss()
        tri_loss_fn = nn.TripletMarginLoss(margin=float(cfg.triplet_margin))
        mse_loss = nn.MSELoss() if use_feat_align_fold else None
        amp_enabled = bool(int(cfg.fp16)) and device.type == "cuda"
        scaler = make_grad_scaler(device, enabled=amp_enabled)

        best_val: Optional[Dict[str, float]] = None
        best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        best_epoch = 0
        best_agg = str(agg_modes[0])

        for ep in range(1, int(cfg.epochs) + 1):
            model.train()
            if align_head is not None:
                align_head.train()
            total_loss = 0.0
            train_n = 0
            if use_nofeat_route_fold and nofeat_route_state is not None:
                temporal_path_order = str(nofeat_route_state.path_order)
            elif use_online_gap_fold:
                temporal_path_order = str(online_gap_state.path_order)
            else:
                temporal_path_order = str(cfg.progressive_path_order)
            stage_path_order = "digit_first" if temporal_path_order == "no_feature" else str(temporal_path_order)
            disable_feature_kd_this_epoch = bool(temporal_path_order == "no_feature")
            online_gap_collect = bool(use_online_gap_fold and should_update_online_gap(cfg, ep))
            gap_student_rows: List[np.ndarray] = []
            gap_teacher_rows: List[np.ndarray] = []
            gap_rows_budget = max(0, int(getattr(cfg, "gap_online_max_rows", 4096)))
            rank_err_sum = 0.0
            digit_err_sum = 0.0
            err_n = 0
            feat_scale = 0.0
            feat_gate_sum = 0.0
            feat_gate_n = 0
            feat_gate_updates_epoch = 0
            feat_grad_cos_sum = 0.0
            feat_grad_cos_n = 0
            iterator = tr_ld if tqdm is None else tqdm(tr_ld, desc=f"fold {fold} ep {ep}/{cfg.epochs}", leave=False)
            for batch_idx, (x20, lie20, qd20, t20, tfeat20, pid, yd, sids) in enumerate(iterator):
                bsz = int(x20.shape[0])
                lie_flat = lie20.view(-1).to(device).long()
                qd20_dev = qd20.to(device).long()
                t20_dev = t20.to(device).float().clamp(1e-4, 1.0 - 1e-4)
                tfeat20_dev = tfeat20.to(device).float()
                yd_dev = yd.to(device).long()

                pid_rep = pid.to(device).long().unsqueeze(1).repeat(1, 20).reshape(-1)
                wr, wf, wd = stage_weights(
                    ep=ep,
                    stage1=int(cfg.stage1_epochs),
                    stage2=int(cfg.stage2_epochs),
                    digit_ramp=int(cfg.stage3_digit_ramp_epochs),
                    mode=str(cfg.progressive_mode),
                    path_order=str(stage_path_order),
                    rank_width=float(cfg.rank_warm_width),
                    feat_width=float(cfg.feat_warm_width),
                    digit_width=float(cfg.digit_warm_width),
                    feat_max_weight=float(cfg.feat_max_weight),
                    overlap_ratio=float(cfg.progressive_overlap_ratio),
                    overlap_width=float(cfg.progressive_overlap_width),
                )
                feat_scale = 1.0
                if use_online_gap_fold and bool(int(getattr(cfg, "gap_adaptive_feat_weight", 1))):
                    feat_scale = gap_adaptive_feature_scale(float(online_gap_state.ema_gap), cfg)
                    wf = float(wf) * float(feat_scale)
                if use_nofeat_route_fold and bool(int(getattr(cfg, "route_feat_scale_enable", 1))):
                    route_feat = route_feature_scale(temporal_path_order, cfg)
                    wf = float(wf) * float(route_feat)
                    feat_scale = float(feat_scale) * float(route_feat)
                if disable_feature_kd_this_epoch:
                    wf = 0.0
                    feat_scale = 0.0
                feat_gate_value = 1.0
                feat_w_eff = float(wf)

                with autocast_context(device, enabled=amp_enabled):
                    logits, id_logits, emb = forward_student(
                        model=model,
                        x20=x20,
                        device=device,
                        student_mode=str(cfg.student_mode),
                        lam=float(cfg.id_loss_lambda),
                        return_feat=True,
                    )
                    logits2 = logits.view(bsz, 20, 2)
                    s20 = torch.softmax(logits2, dim=-1)[:, :, 1].clamp(1e-5, 1.0 - 1e-5)
                    s_logit = torch.log(s20) - torch.log(1.0 - s20)
                    emb20 = emb.view(bsz, 20, -1)

                    # Confidence-based KD gating: start with easy/high-confidence teacher questions.
                    kd_conf_mode = str(cfg.kd_conf_mode).lower().strip()
                    if kd_conf_mode == "none":
                        q_mask = torch.ones_like(t20_dev, dtype=torch.bool)
                        q_w = torch.ones_like(t20_dev, dtype=t20_dev.dtype)
                    else:
                        conf20 = torch.abs(t20_dev - 0.5) * 2.0  # [B,20], range [0,1]
                        if kd_conf_mode == "hard":
                            thr = float(cfg.kd_conf_low)
                        else:
                            thr = kd_conf_threshold(
                                ep=ep,
                                stage1=int(cfg.stage1_epochs),
                                stage2=int(cfg.stage2_epochs),
                                digit_ramp=int(cfg.stage3_digit_ramp_epochs),
                                high=float(cfg.kd_conf_high),
                                low=float(cfg.kd_conf_low),
                            )
                        q_mask = conf20 >= float(thr)

                        keep_frac = float(np.clip(float(cfg.kd_conf_min_keep_frac), 0.0, 1.0))
                        k_keep = max(1, min(20, int(round(20.0 * keep_frac))))
                        topk_idx = torch.topk(conf20, k=k_keep, dim=1, largest=True).indices
                        topk_mask = torch.zeros_like(q_mask)
                        topk_mask.scatter_(1, topk_idx, True)
                        q_mask = q_mask | topk_mask

                        none_rows = (q_mask.sum(dim=1) <= 0)
                        if bool(none_rows.any()):
                            argmax_idx = conf20.argmax(dim=1, keepdim=True)
                            fallback = torch.zeros_like(q_mask)
                            fallback.scatter_(1, argmax_idx, True)
                            q_mask = torch.where(none_rows.unsqueeze(1), fallback, q_mask)
                        q_w = q_mask.to(t20_dev.dtype)

                    # Hard losses
                    l_hard_q = ce_q(logits, lie_flat)
                    s_digit_hard = aggregate_digit_scores_torch(s_logit, qd20_dev, mode=str(cfg.kd_digit_agg_mode))
                    l_digit_ce = ce_digit(s_digit_hard, yd_dev - 1)

                    # ranking KD on 20 questions
                    t_rank = (t20_dev - t20_dev.mean(dim=1, keepdim=True)) / (t20_dev.std(dim=1, keepdim=True) + 1e-6)
                    use_rank_std = str(cfg.logitstd_mode).lower().strip() in ("rank", "both")
                    if use_rank_std:
                        s_rank_base = standardize_logits(s_logit, dim=1, eps=float(cfg.logitstd_eps))
                        t_rank_base = standardize_logits(t_rank, dim=1, eps=float(cfg.logitstd_eps))
                    else:
                        s_rank_base = s_logit
                        t_rank_base = t_rank
                    if kd_conf_mode == "none":
                        s_rank_src = s_rank_base
                        t_rank_src = t_rank_base
                    else:
                        neg = torch.full_like(s_rank_base, torch.finfo(s_rank_base.dtype).min)
                        s_rank_src = torch.where(q_mask, s_rank_base, neg)
                        t_rank_src = torch.where(q_mask, t_rank_base, neg)
                    l_rank = F.kl_div(
                        F.log_softmax(s_rank_src / float(cfg.temp_rank), dim=1),
                        F.softmax(t_rank_src / float(cfg.temp_rank), dim=1),
                        reduction="batchmean",
                    ) * (float(cfg.temp_rank) ** 2)

                    # feature alignment
                    if align_head is not None and mse_loss is not None and not disable_feature_kd_this_epoch:
                        pred_tfeat = align_head(emb20)
                        if kd_conf_mode == "none":
                            l_feat = mse_loss(pred_tfeat, tfeat20_dev)
                        else:
                            diff2 = (pred_tfeat - tfeat20_dev) ** 2
                            den = (q_w.sum() * float(diff2.shape[-1])).clamp(min=1.0)
                            l_feat = (diff2 * q_w.unsqueeze(-1)).sum() / den
                    else:
                        l_feat = torch.zeros((), device=device, dtype=logits.dtype)

                    # digit evidence KD
                    t_logit = torch.log(t20_dev) - torch.log(1.0 - t20_dev)
                    if kd_conf_mode == "none":
                        t_digit_src = t_logit
                        s_digit_src = s_logit
                    else:
                        t_digit_src = t_logit * q_w
                        s_digit_src = s_logit * q_w
                    t_digit = aggregate_digit_scores_torch(t_digit_src, qd20_dev, mode=str(cfg.kd_digit_agg_mode))
                    s_digit = aggregate_digit_scores_torch(s_digit_src, qd20_dev, mode=str(cfg.kd_digit_agg_mode))
                    if str(cfg.logitstd_mode).lower().strip() in ("digit", "both"):
                        t_digit = standardize_logits(t_digit, dim=-1, eps=float(cfg.logitstd_eps))
                        s_digit = standardize_logits(s_digit, dim=-1, eps=float(cfg.logitstd_eps))
                    l_digit_kd = F.kl_div(
                        F.log_softmax(s_digit / float(cfg.temp_digit), dim=-1),
                        F.softmax(t_digit / float(cfg.temp_digit), dim=-1),
                        reduction="batchmean",
                    ) * (float(cfg.temp_digit) ** 2)
                    with torch.no_grad():
                        rank_diff = torch.abs(s_rank_base.detach() - t_rank_base.detach())
                        if kd_conf_mode == "none":
                            rank_err_batch = float(rank_diff.mean().item())
                        else:
                            den = q_w.sum().clamp(min=1.0)
                            rank_err_batch = float((rank_diff * q_w).sum().item() / float(den.item()))
                        digit_err_batch = float(torch.abs(s_digit.detach() - t_digit.detach()).mean().item())

                    if float(cfg.id_loss_weight) > 0.0:
                        l_id = ce_id(id_logits, pid_rep)
                    else:
                        l_id = torch.zeros((), device=device, dtype=logits.dtype)

                    if float(cfg.triplet_loss_weight) > 0.0:
                        a, p, n = sample_triplets(emb, lie_flat, pid_rep)
                        if a.numel() == 0:
                            l_tri = torch.zeros((), device=device, dtype=logits.dtype)
                        else:
                            a = a.to(device)
                            p = p.to(device)
                            n = n.to(device)
                            l_tri = tri_loss_fn(emb[a], emb[p], emb[n])
                    else:
                        l_tri = torch.zeros((), device=device, dtype=logits.dtype)

                    core_loss = (
                        float(cfg.lambda_lie_ce) * l_hard_q
                        + float(cfg.lambda_digit_ce) * l_digit_ce
                        + float(wr) * float(cfg.lambda_rank_kd) * l_rank
                        + float(wd) * float(cfg.lambda_digit_kd) * l_digit_kd
                        + float(cfg.id_loss_weight) * l_id
                        + float(cfg.triplet_loss_weight) * l_tri
                    )
                    feat_grad_cos_batch: Optional[float] = None
                    if (
                        use_auto_feat_gate_fold
                        and feature_gate_state is not None
                        and not disable_feature_kd_this_epoch
                        and bool(l_feat.requires_grad)
                        and float(cfg.lambda_feat_align) > 0.0
                        and float(wf) > 0.0
                    ):
                        update_every = max(1, int(getattr(cfg, "feat_gate_update_interval_batches", 1)))
                        if (int(batch_idx) % update_every) == 0:
                            grad_core = torch.autograd.grad(core_loss, emb, retain_graph=True, allow_unused=True)[0]
                            grad_feat = torch.autograd.grad(l_feat, emb, retain_graph=True, allow_unused=True)[0]
                            feat_grad_cos_batch = _grad_cosine_mean(grad_core, grad_feat)
                            prev_updates = int(feature_gate_state.updates)
                            feature_gate_state = update_feature_gate_state(
                                cfg=cfg,
                                state=feature_gate_state,
                                grad_cos=feat_grad_cos_batch,
                            )
                            if int(feature_gate_state.updates) > prev_updates:
                                feat_gate_updates_epoch += 1
                        feat_gate_value = float(feature_gate_state.gate)
                    feat_w_eff = float(wf) * float(feat_gate_value)
                    loss = core_loss + float(feat_w_eff) * float(cfg.lambda_feat_align) * l_feat

                if scaler is None:
                    loss.backward()
                    opt.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                opt.zero_grad(set_to_none=True)

                n_batch = int(bsz)
                total_loss += float(loss.item()) * n_batch
                train_n += n_batch
                rank_err_sum += float(rank_err_batch) * n_batch
                digit_err_sum += float(digit_err_batch) * n_batch
                err_n += n_batch
                feat_gate_sum += float(feat_gate_value) * n_batch
                feat_gate_n += n_batch
                if feat_grad_cos_batch is not None:
                    feat_grad_cos_sum += float(feat_grad_cos_batch)
                    feat_grad_cos_n += 1

                if online_gap_collect and gap_rows_budget > 0:
                    emb_np = emb20.detach().reshape(-1, emb20.shape[-1]).float().cpu().numpy().astype(np.float32, copy=False)
                    tfeat_np = tfeat20_dev.detach().reshape(-1, tfeat20_dev.shape[-1]).float().cpu().numpy().astype(np.float32, copy=False)
                    take_n = min(int(emb_np.shape[0]), int(tfeat_np.shape[0]), int(gap_rows_budget))
                    if take_n > 0:
                        gap_student_rows.append(emb_np[:take_n].copy())
                        gap_teacher_rows.append(tfeat_np[:take_n].copy())
                        gap_rows_budget -= int(take_n)

            train_loss = float(total_loss / max(1, train_n))
            if online_gap_collect:
                online_gap_state = update_online_gap_state(
                    cfg=cfg,
                    state=online_gap_state,
                    student_rows=gap_student_rows,
                    teacher_rows=gap_teacher_rows,
                )
            nofeat_rank_err = float(rank_err_sum / max(1, err_n))
            nofeat_digit_err = float(digit_err_sum / max(1, err_n))
            if use_nofeat_route_fold and nofeat_route_state is not None:
                nofeat_route_state = update_nofeat_route_state(
                    cfg=cfg,
                    state=nofeat_route_state,
                    rank_err=nofeat_rank_err,
                    digit_err=nofeat_digit_err,
                )
            feat_gate_epoch = float(feat_gate_sum / max(1, feat_gate_n))
            feat_grad_cos_epoch = None if int(feat_grad_cos_n) <= 0 else float(feat_grad_cos_sum / float(feat_grad_cos_n))
            if feature_gate_state is None:
                feat_gate_source = "off"
                feat_gate_ema = None
                feat_gate_updates = 0
            else:
                feat_gate_source = str(feature_gate_state.source)
                feat_gate_ema = float(feature_gate_state.ema_grad_cos)
                feat_gate_updates = int(feat_gate_updates_epoch)

            sel_loader = va_ld if str(cfg.select_on).lower().strip() == "val" else te_ld
            sel_name = "val" if str(cfg.select_on).lower().strip() == "val" else "test"
            best_this_epoch: Optional[Dict[str, float]] = None
            for agg_mode in agg_modes:
                met_sel, _, _ = evaluate_student(
                    model,
                    sel_loader,
                    device,
                    agg_mode=str(agg_mode),
                    student_mode=str(cfg.student_mode),
                )
                if better_by_topk(met_sel, best_this_epoch):
                    best_this_epoch = dict(met_sel)
            assert best_this_epoch is not None
            cur = {
                "epoch": float(ep),
                "train_loss": float(train_loss),
                "top1": float(best_this_epoch["top1"]),
                "top2": float(best_this_epoch["top2"]),
                "top3": float(best_this_epoch["top3"]),
                "agg_mode": str(best_this_epoch["agg_mode"]),
                "in_dim": float(in_dim),
            }
            if use_nofeat_route_fold and nofeat_route_state is not None:
                route_source = str(nofeat_route_state.source)
                route_path = str(nofeat_route_state.path_order)
                route_hold = int(nofeat_route_state.hold_epochs)
                route_ratio = f"{float(nofeat_route_state.rank_digit_ratio):.4f}"
            elif use_online_gap_fold:
                route_source = str(online_gap_state.source)
                route_path = str(online_gap_state.path_order)
                route_hold = int(online_gap_state.hold_epochs)
                route_ratio = "None" if online_gap_state.last_gap is None else f"{float(online_gap_state.last_gap):.4f}"
            else:
                route_source = "fixed_path"
                route_path = str(cfg.progressive_path_order)
                route_hold = int(ep)
                route_ratio = f"{nofeat_rank_err / max(1e-6, nofeat_digit_err):.4f}"
            logging.info(
                "fold %d ep %d | train_loss=%.4f | route_src=%s path=%s hold=%d ratio=%s rank_err=%.4f digit_err=%.4f | ema_gap=%.4f obs_gap=%s raw_gap=%s feat_scale=%.3f feat_gate=%.3f gate_src=%s gate_ema=%s gate_cos=%s gate_updates=%d | %s_top1=%.4f %s_top2=%.4f %s_top3=%.4f | agg=%s",
                fold,
                ep,
                train_loss,
                route_source,
                route_path,
                route_hold,
                route_ratio,
                nofeat_rank_err,
                nofeat_digit_err,
                float(online_gap_state.ema_gap),
                "None" if online_gap_state.last_gap is None else f"{float(online_gap_state.last_gap):.4f}",
                "None" if online_gap_state.last_gap_raw is None else f"{float(online_gap_state.last_gap_raw):.4f}",
                float(feat_scale),
                float(feat_gate_epoch),
                str(feat_gate_source),
                "None" if feat_gate_ema is None else f"{float(feat_gate_ema):.4f}",
                "None" if feat_grad_cos_epoch is None else f"{float(feat_grad_cos_epoch):.4f}",
                int(feat_gate_updates),
                sel_name,
                cur["top1"],
                sel_name,
                cur["top2"],
                sel_name,
                cur["top3"],
                cur["agg_mode"],
            )

            if better_by_topk(cur, best_val):
                best_val = cur
                best_epoch = int(ep)
                best_agg = str(cur["agg_mode"])
                best_state = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                if align_head is not None:
                    best_state["align"] = {k: v.detach().cpu().clone() for k, v in align_head.state_dict().items()}

        if best_val is None or best_state is None:
            logging.warning("fold %d no valid best model", fold)
            continue

        model.load_state_dict(best_state["model"], strict=True)
        if align_head is not None and "align" in best_state:
            align_head.load_state_dict(best_state["align"], strict=True)
        met_test, eval_rows_test, eval_q_rows_test = evaluate_student(
            model,
            te_ld,
            device,
            agg_mode=best_agg,
            student_mode=str(cfg.student_mode),
        )
        fold_bin = compute_binary_metrics(
            y_true=[int(r["y_true"]) for r in eval_q_rows_test],
            y_prob=[float(r["y_prob"]) for r in eval_q_rows_test],
            threshold=0.5,
        )
        fold_comp = estimate_model_complexity(
            model=model,
            in_dim=int(in_dim),
            device=device,
            student_mode=str(cfg.student_mode),
        )

        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state["model"],
                "align_state_dict": best_state.get("align", None),
                "use_feat_align": bool(align_head is not None),
                "best_val": best_val,
                "best_epoch": int(best_epoch),
                "best_agg_mode": str(best_agg),
                "cfg": vars(cfg),
            },
            fold_dir / "best_student.pt",
        )
        if use_nofeat_route_fold and nofeat_route_state is not None:
            fold_gap_source = str(nofeat_route_state.source)
            fold_route_path = str(nofeat_route_state.path_order)
            fold_route_hold = int(nofeat_route_state.hold_epochs)
            fold_gap_value = float(nofeat_route_state.rank_digit_ratio)
            fold_gap_similarity: Optional[float] = None
        else:
            fold_gap_source = str(online_gap_state.source)
            fold_route_path = str(online_gap_state.path_order)
            fold_route_hold = int(online_gap_state.hold_epochs)
            fold_gap_value = float(online_gap_state.ema_gap)
            fold_gap_similarity = None if online_gap_state.last_gap is None else float(1.0 - online_gap_state.last_gap)
        if feature_gate_state is None:
            fold_feat_gate = 1.0
            fold_feat_gate_ema = None
            fold_feat_gate_source = "off"
            fold_feat_gate_updates = 0
        else:
            fold_feat_gate = float(feature_gate_state.gate)
            fold_feat_gate_ema = float(feature_gate_state.ema_grad_cos)
            fold_feat_gate_source = str(feature_gate_state.source)
            fold_feat_gate_updates = int(feature_gate_state.updates)

        fold_rows.append(
            {
                "fold": int(fold),
                "n_test": int(met_test["n"]),
                "select_on": str(cfg.select_on),
                "best_epoch": int(best_epoch),
                "best_agg_mode": str(best_agg),
                "in_dim": int(in_dim),
                "sel_top1": float(best_val["top1"]),
                "sel_top2": float(best_val["top2"]),
                "sel_top3": float(best_val["top3"]),
                "top1": float(met_test["top1"]),
                "top2": float(met_test["top2"]),
                "top3": float(met_test["top3"]),
                "bin_acc": float(fold_bin["acc"]),
                "bin_f1": float(fold_bin["pos_f1"]),
                "bin_auc": float(fold_bin["pos_auc"]),
                "gap_source": str(fold_gap_source),
                "gap_similarity": fold_gap_similarity,
                "gap_value": float(fold_gap_value),
                "route_path_order": str(fold_route_path),
                "route_hold_epochs": int(fold_route_hold),
                "feat_gate": float(fold_feat_gate),
                "feat_gate_ema_grad_cos": fold_feat_gate_ema,
                "feat_gate_source": str(fold_feat_gate_source),
                "feat_gate_updates": int(fold_feat_gate_updates),
            }
        )
        complexity_rows.append(
            {
                "fold": int(fold),
                "params_k": float(fold_comp["params_k"]),
                "flops_per_sample": float(fold_comp["flops_per_sample"]),
                "in_dim": int(in_dim),
            }
        )

        total_n += int(met_test["n"])
        total_h1 += int(round(float(met_test["top1"]) * float(met_test["n"])))
        total_h2 += int(round(float(met_test["top2"]) * float(met_test["n"])))
        total_h3 += int(round(float(met_test["top3"]) * float(met_test["n"])))

        for r in eval_rows_test:
            detail_rows.append(
                {
                    "fold": int(fold),
                    "subject_id": str(r["subject_id"]),
                    "y_true": int(r["y_true"]),
                    "y_hat": int(r["y_hat"]),
                    "top3": json.dumps([int(x) for x in r["top3"]], ensure_ascii=False),
                    "agg_mode": str(best_agg),
                }
            )
        for r in eval_q_rows_test:
            detail_q_rows.append(
                {
                    "fold": int(fold),
                    "subject_id": str(r["subject_id"]),
                    "qid": int(r["qid"]),
                    "q_digit": int(r["q_digit"]),
                    "y_true": int(r["y_true"]),
                    "y_prob": float(r["y_prob"]),
                    "y_pred": int(r["y_pred"]),
                }
            )

        logging.info(
            "fold %d test: top1=%.2f%% top2=%.2f%% top3=%.2f%% | ACC=%.2f%% F1=%.2f%% AUC=%.2f%% | selected_epoch=%d agg=%s on %s",
            fold,
            float(met_test["top1"]) * 100.0,
            float(met_test["top2"]) * 100.0,
            float(met_test["top3"]) * 100.0,
            float(fold_bin["acc"]) * 100.0,
            float(fold_bin["pos_f1"]) * 100.0,
            float(fold_bin["pos_auc"]) * 100.0,
            int(best_epoch),
            str(best_agg),
            str(cfg.select_on),
        )

    if len(fold_rows) == 0:
        raise RuntimeError("No fold completed")

    mean_top1 = float(np.mean([float(r["top1"]) for r in fold_rows]))
    mean_top2 = float(np.mean([float(r["top2"]) for r in fold_rows]))
    mean_top3 = float(np.mean([float(r["top3"]) for r in fold_rows]))
    mean_bin_acc = float(np.mean([float(r["bin_acc"]) for r in fold_rows]))
    mean_bin_f1 = float(np.mean([float(r["bin_f1"]) for r in fold_rows]))
    mean_bin_auc = float(np.mean([float(r["bin_auc"]) for r in fold_rows]))
    overall_top1 = float(total_h1) / max(1, int(total_n))
    overall_top2 = float(total_h2) / max(1, int(total_n))
    overall_top3 = float(total_h3) / max(1, int(total_n))
    overall_bin = compute_binary_metrics(
        y_true=[int(r["y_true"]) for r in detail_q_rows],
        y_prob=[float(r["y_prob"]) for r in detail_q_rows],
        threshold=0.5,
    )
    top1_stats = summarize_mean_std([float(r["top1"]) for r in fold_rows])
    top2_stats = summarize_mean_std([float(r["top2"]) for r in fold_rows])
    top3_stats = summarize_mean_std([float(r["top3"]) for r in fold_rows])
    acc_stats = summarize_mean_std([float(r["bin_acc"]) for r in fold_rows])
    f1_stats = summarize_mean_std([float(r["bin_f1"]) for r in fold_rows])
    auc_stats = summarize_mean_std([float(r["bin_auc"]) for r in fold_rows])
    params_stats = summarize_mean_std([float(r["params_k"]) for r in complexity_rows])
    flops_stats = summarize_mean_std([float(r["flops_per_sample"]) for r in complexity_rows])

    with (out_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "n_test",
                "select_on",
                "best_epoch",
                "best_agg_mode",
                "in_dim",
                "sel_top1",
                "sel_top2",
                "sel_top3",
                "top1",
                "top2",
                "top3",
                "bin_acc",
                "bin_f1",
                "bin_auc",
                "gap_source",
                "gap_similarity",
                "gap_value",
                "route_path_order",
                "route_hold_epochs",
                "feat_gate",
                "feat_gate_ema_grad_cos",
                "feat_gate_source",
                "feat_gate_updates",
            ],
        )
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    with (out_dir / "detail_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "subject_id", "y_true", "y_hat", "top3", "agg_mode"])
        w.writeheader()
        for r in detail_rows:
            w.writerow(r)

    with (out_dir / "detail_question_binary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "subject_id", "qid", "q_digit", "y_true", "y_prob", "y_pred"])
        w.writeheader()
        for r in detail_q_rows:
            w.writerow(r)

    with (out_dir / "model_complexity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "params_k", "flops_per_sample", "in_dim"])
        w.writeheader()
        for r in complexity_rows:
            w.writerow(r)

    with (out_dir / "model_complexity.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "params_k_mean": float(params_stats["mean"]),
                "params_k_std": float(params_stats["std"]),
                "flops_per_sample_mean": float(flops_stats["mean"]),
                "flops_per_sample_std": float(flops_stats["std"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "mean_fold_top1": mean_top1,
        "mean_fold_top2": mean_top2,
        "mean_fold_top3": mean_top3,
        "mean_fold_bin_acc": mean_bin_acc,
        "mean_fold_bin_f1": mean_bin_f1,
        "mean_fold_bin_auc": mean_bin_auc,
        "overall_top1": overall_top1,
        "overall_top2": overall_top2,
        "overall_top3": overall_top3,
        "overall_bin_acc": float(overall_bin["acc"]),
        "overall_bin_f1": float(overall_bin["pos_f1"]),
        "overall_bin_auc": float(overall_bin["pos_auc"]),
        "stats_mean_std": {
            "top1": top1_stats,
            "top2": top2_stats,
            "top3": top3_stats,
            "bin_acc": acc_stats,
            "bin_f1": f1_stats,
            "bin_auc": auc_stats,
            "params_k": params_stats,
            "flops_per_sample": flops_stats,
        },
        "model_complexity": {
            "params_k_mean": float(params_stats["mean"]),
            "params_k_std": float(params_stats["std"]),
            "flops_per_sample_mean": float(flops_stats["mean"]),
            "flops_per_sample_std": float(flops_stats["std"]),
        },
        "folds": fold_rows,
        "config": vars(cfg),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Per-fold metrics:")
    for r in fold_rows:
        logging.info(
            "fold %d: n=%d top1=%.2f%% top2=%.2f%% top3=%.2f%% | selected_epoch=%d agg=%s on %s",
            int(r["fold"]),
            int(r["n_test"]),
            float(r["top1"]) * 100.0,
            float(r["top2"]) * 100.0,
            float(r["top3"]) * 100.0,
            int(r["best_epoch"]),
            str(r["best_agg_mode"]),
            str(r["select_on"]),
        )
    logging.info(
        "Mean-fold: top1=%.2f%% top2=%.2f%% top3=%.2f%% | Overall: top1=%.2f%% top2=%.2f%% top3=%.2f%%",
        mean_top1 * 100.0,
        mean_top2 * 100.0,
        mean_top3 * 100.0,
        overall_top1 * 100.0,
        overall_top2 * 100.0,
        overall_top3 * 100.0,
    )
    logging.info(
        "mean+/-std: top1=%.1f+/-%.1f top2=%.1f+/-%.1f top3=%.1f+/-%.1f, ACC=%.1f+/-%.1f F1=%.1f+/-%.1f AUC=%.1f+/-%.1f",
        float(top1_stats["mean"]) * 100.0,
        float(top1_stats["std"]) * 100.0,
        float(top2_stats["mean"]) * 100.0,
        float(top2_stats["std"]) * 100.0,
        float(top3_stats["mean"]) * 100.0,
        float(top3_stats["std"]) * 100.0,
        float(acc_stats["mean"]) * 100.0,
        float(acc_stats["std"]) * 100.0,
        float(f1_stats["mean"]) * 100.0,
        float(f1_stats["std"]) * 100.0,
        float(auc_stats["mean"]) * 100.0,
        float(auc_stats["std"]) * 100.0,
    )
    logging.info(
        "Question-level binary overall: n=%d ACC=%.2f%% F1=%.2f%% AUC=%.2f%%",
        int(overall_bin["n"]),
        float(overall_bin["acc"]) * 100.0,
        float(overall_bin["pos_f1"]) * 100.0,
        float(overall_bin["pos_auc"]) * 100.0,
    )
    logging.info(
        "Model complexity mean+/-std: params=%.3f+/-%.3f K, flops/sample=%.0f+/-%.0f",
        float(params_stats["mean"]),
        float(params_stats["std"]),
        float(flops_stats["mean"]),
        float(flops_stats["std"]),
    )
    logging.info("Saved: %s", out_dir)


if __name__ == "__main__":
    main()
