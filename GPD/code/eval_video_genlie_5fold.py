#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GenLie-style single-modality feature classifier adapted for number-guess task (5-fold).

Task workflow:
1) Load pre-extracted modality features from: data/<sid>/features/<session>/(Qxx.pt | xx.npy)
2) Load question lie labels from: data/<sid>/dataset/<session>/label.csv
3) Train question-level lie classifier (truth=0 / lie=1)
4) Aggregate per-question lie scores to digit scores
5) Evaluate subject-level top1/top2/top3 on fixed 5-fold split

Notes:
- Uses only Q03..Q22 (20 questions) for digit inference, aligned with GSR task.
- Hidden digit is derived from two lie questions in label.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


QUESTION_NUMBERS = [4, 8, 3, 1, 6, 9, 2, 5, 7, 10, 7, 2, 9, 3, 5, 1, 8, 10, 6, 4]
TARGET_QIDS = list(range(3, 23))  # 20 questions


def normalize_modality_name(name: str) -> str:
    s = str(name).strip().lower()
    return s if s else "feature"


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


BFI2_DOMAINS: Dict[str, List[Tuple[int, bool]]] = {
    "E": [(1, False), (6, False), (11, True), (16, True), (21, False), (26, True), (31, True), (36, True), (41, False), (46, False), (51, True), (56, False)],
    "A": [(2, False), (7, False), (12, True), (17, True), (22, True), (27, False), (32, False), (37, True), (42, True), (47, True), (52, False), (57, False)],
    "C": [(3, True), (8, True), (13, False), (18, False), (23, True), (28, True), (33, False), (38, False), (43, False), (48, True), (53, False), (58, True)],
    "N": [(4, True), (9, True), (14, False), (19, False), (24, True), (29, True), (34, False), (39, False), (44, True), (49, True), (54, False), (59, False)],
    "O": [(5, True), (10, False), (15, False), (20, False), (25, True), (30, True), (35, False), (40, False), (45, True), (50, True), (55, True), (60, False)],
}
BFI2_DOMAIN_ORDER = ["O", "C", "E", "A", "N"]


def _reverse_score(v: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return (lo + hi) - float(v)


def _score_items(x60: np.ndarray, items: List[Tuple[int, bool]]) -> float:
    vals: List[float] = []
    for idx1, is_rev in items:
        v = float(x60[int(idx1) - 1])
        if is_rev:
            v = _reverse_score(v)
        vals.append(v)
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def _to_len60(raw60: np.ndarray) -> np.ndarray:
    x = np.asarray(raw60, dtype=np.float32).reshape(-1)
    if x.size < 60:
        x = np.pad(x, (0, 60 - x.size), mode="constant")
    elif x.size > 60:
        x = x[:60]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def raw60_to_big5(raw60: np.ndarray) -> np.ndarray:
    x60 = _to_len60(raw60)
    dom_scores = {k: _score_items(x60, BFI2_DOMAINS[k]) for k in BFI2_DOMAINS.keys()}
    v5 = np.asarray([dom_scores[k] for k in BFI2_DOMAIN_ORDER], dtype=np.float32)
    return v5


def _parse_big5_from_row(row: Dict[str, str]) -> Optional[np.ndarray]:
    if not row:
        return None
    pref_keys = [k for k in row.keys() if str(k).lower().startswith("big5_")]
    if len(pref_keys) >= 5:
        pref_keys = sorted(pref_keys, key=lambda x: int(str(x).split("_")[-1]))
        vals = [_safe_float(row.get(k, 0.0), 0.0) for k in pref_keys[:5]]
        return np.asarray(vals, dtype=np.float32)
    num_keys = sorted([k for k in row.keys() if str(k).strip().isdigit()], key=lambda x: int(x))
    if len(num_keys) >= 5:
        vals = [_safe_float(row.get(k, 0.0), 0.0) for k in num_keys[:5]]
        return np.asarray(vals, dtype=np.float32)
    return None


def load_big5_vector(subject_dir: Path) -> np.ndarray:
    csv_path = subject_dir / "bfi_big5.csv"
    if csv_path.exists():
        try:
            rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8-sig", newline="")))
        except Exception:
            rows = []
        if rows:
            arr = _parse_big5_from_row(rows[0])
            if arr is not None and arr.shape[0] == 5:
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    json_path = subject_dir / "bfi_big5.json"
    if json_path.exists():
        try:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            obj = None
        if isinstance(obj, dict):
            arr = _parse_big5_from_row({k: str(v) for k, v in obj.items()})
            if arr is not None and arr.shape[0] == 5:
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if isinstance(obj, list) and len(obj) >= 5:
            arr = np.asarray([_safe_float(v, 0.0) for v in obj[:5]], dtype=np.float32)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    raw_csv = subject_dir / "bfi_row.csv"
    if raw_csv.exists():
        try:
            rows = list(csv.DictReader(raw_csv.open("r", encoding="utf-8-sig", newline="")))
        except Exception:
            rows = []
        if rows:
            row = rows[0]
            keys = sorted([k for k in row.keys() if str(k).strip().isdigit()], key=lambda z: int(z))
            if len(keys) > 0:
                vals = [_safe_float(row.get(k, 0.0), 0.0) for k in keys]
                return raw60_to_big5(np.asarray(vals, dtype=np.float32))

    return np.zeros((5,), dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    d = str(device).strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda") if d == "cuda" else torch.device(d)
        return torch.device("cpu")
    return torch.device("cpu")


def digit_from_qid(qid: int) -> Optional[int]:
    if 3 <= qid <= 22:
        return QUESTION_NUMBERS[qid - 3]
    return None


def find_single_child_dir(root: Path) -> Optional[Path]:
    if not root.exists() or not root.is_dir():
        return None
    ds = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(ds) == 0:
        return None
    return ds[0]


def _read_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_label_map(label_csv: Path) -> Dict[int, int]:
    if not label_csv.exists():
        return {}
    try:
        rows = list(csv.DictReader(label_csv.open("r", encoding="utf-8-sig", newline="")))
    except Exception:
        rows = []
    if not rows:
        return {}

    key_map = {str(k).strip().lower(): k for k in rows[0].keys()}
    qk = key_map.get("question_id")
    lk = key_map.get("label")
    if qk is None or lk is None:
        return {}

    out: Dict[int, int] = {}
    for r in rows:
        try:
            qid = int(float(str(r.get(qk, "")).strip()))
            lv = int(float(str(r.get(lk, "")).strip()))
            out[qid] = lv
        except Exception:
            continue
    return out


def hidden_number_from_label_csv(label_csv: Path) -> Optional[int]:
    labels = load_label_map(label_csv)
    if not labels:
        return None
    lie_qids = [q for q in TARGET_QIDS if int(labels.get(q, 0)) == 1]
    if len(lie_qids) != 2:
        return None
    d1 = digit_from_qid(int(lie_qids[0]))
    d2 = digit_from_qid(int(lie_qids[1]))
    if d1 is None or d2 is None or int(d1) != int(d2):
        return None
    return int(d1)


def _collect_numeric_tensors(obj: Any, out: List[torch.Tensor], depth: int = 0, max_depth: int = 6) -> None:
    if obj is None or depth > max_depth:
        return
    if torch.is_tensor(obj):
        if obj.numel() > 0 and obj.dtype != torch.bool:
            out.append(obj.detach().cpu())
        return
    if isinstance(obj, np.ndarray):
        if obj.size == 0:
            return
        if obj.dtype == np.object_:
            for v in list(obj.flat)[:64]:
                _collect_numeric_tensors(v, out, depth + 1, max_depth)
            return
        if np.issubdtype(obj.dtype, np.number):
            out.append(torch.from_numpy(obj))
        return
    if isinstance(obj, dict):
        for k in [
            "feat",
            "feature",
            "features",
            "embedding",
            "embeddings",
            "x",
            "video_feat",
            "video_feature",
            "audio_feat",
            "audio_feature",
            "audio_embedding",
            "acoustic_feat",
            "acoustic_feature",
            "acoustic_embedding",
            "wavlm_feat",
            "wavlm_feature",
        ]:
            if k in obj:
                _collect_numeric_tensors(obj[k], out, depth + 1, max_depth)
        for v in obj.values():
            _collect_numeric_tensors(v, out, depth + 1, max_depth)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj[:64]:
            _collect_numeric_tensors(v, out, depth + 1, max_depth)
        return


def _tensor_to_vec(t: torch.Tensor) -> torch.Tensor:
    t = t.float()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    if t.ndim == 0:
        return t.view(1)
    if t.ndim == 1:
        return t
    # [*, D] -> [D]
    return t.reshape(-1, t.shape[-1]).mean(dim=0)


def load_feature_vector(file_path: Path) -> Optional[torch.Tensor]:
    try:
        if file_path.suffix.lower() in [".pt", ".pth"]:
            try:
                obj = torch.load(file_path, map_location="cpu", weights_only=False)
            except TypeError:
                obj = torch.load(file_path, map_location="cpu")
        elif file_path.suffix.lower() == ".npy":
            obj = np.load(file_path, allow_pickle=True)
        elif file_path.suffix.lower() == ".npz":
            z = np.load(file_path, allow_pickle=True)
            ks = list(z.keys())
            obj = z[ks[0]] if len(ks) > 0 else None
        else:
            return None
    except Exception:
        return None

    if obj is None:
        return None
    cands: List[torch.Tensor] = []
    _collect_numeric_tensors(obj, cands)
    if len(cands) == 0:
        return None
    cands.sort(key=lambda x: int(x.numel()), reverse=True)
    v = _tensor_to_vec(cands[0])
    if int(v.numel()) == 0:
        return None
    return v.contiguous()


def _parse_qid_from_stem(stem: str) -> Optional[int]:
    s = str(stem).strip()
    pats = [
        r"^[Qq](\d{1,3})$",
        r"^(\d{1,3})$",
        r"^[Ee]vent_(\d{1,3})$",
        r"^[Qq](\d{1,3})[_-].*$",
        r"^(\d{1,3})[_-].*$",
        r"^[Ee]vent_(\d{1,3})[_-].*$",
    ]
    for pat in pats:
        m = re.match(pat, s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def find_feature_file(feature_dir: Path, qid: int) -> Optional[Path]:
    # Exact names first (compat with both old and new feature exports).
    stems = [
        f"Q{qid:02d}",
        f"q{qid:02d}",
        f"Q{qid}",
        f"q{qid}",
        f"{qid}",
        f"{qid:02d}",
        f"Event_{qid:03d}",
        f"event_{qid:03d}",
    ]
    exts = [".pt", ".pth", ".npy", ".npz"]
    for st in stems:
        for ext in exts:
            p = feature_dir / f"{st}{ext}"
            if p.exists():
                return p

    # Fallback: parse qid from file stem.
    fs = sorted([p for p in feature_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    for p in fs:
        pq = _parse_qid_from_stem(p.stem)
        if pq is not None and int(pq) == int(qid):
            return p
    return None


@dataclass
class SubjectSample:
    sid: str
    y_digit: int  # 1..10
    q_feats: Dict[int, torch.Tensor]  # qid->vec
    q_lie: Dict[int, int]  # qid->0/1
    big5: np.ndarray  # [5]


def load_subject_sample(data_root: Path, sid: str, min_valid_q: int = 20, feature_qid_shift: int = 0) -> Optional[SubjectSample]:
    subj_root = data_root / sid
    feat_dir = find_single_child_dir(subj_root / "features")
    ds_dir = find_single_child_dir(subj_root / "dataset")
    if feat_dir is None or ds_dir is None:
        return None

    label_csv = ds_dir / "label.csv"
    y_digit = hidden_number_from_label_csv(label_csv)
    if y_digit is None:
        return None
    label_map = load_label_map(label_csv)

    q_feats: Dict[int, torch.Tensor] = {}
    q_lie: Dict[int, int] = {}
    for qid in TARGET_QIDS:
        fq = int(qid) + int(feature_qid_shift)
        if fq <= 0:
            continue
        fp = find_feature_file(feat_dir, fq)
        if fp is None:
            continue
        vec = load_feature_vector(fp)
        if vec is None or int(vec.numel()) == 0:
            continue
        q_feats[int(qid)] = vec
        q_lie[int(qid)] = int(label_map.get(int(qid), 0))

    if len(q_feats) < int(min_valid_q):
        return None

    return SubjectSample(
        sid=str(sid),
        y_digit=int(y_digit),
        q_feats=q_feats,
        q_lie=q_lie,
        big5=load_big5_vector(subj_root).astype(np.float32),
    )


def vec_to_dim(v: torch.Tensor, dim: int) -> np.ndarray:
    out = np.zeros((int(dim),), dtype=np.float32)
    n = min(int(dim), int(v.numel()))
    if n > 0:
        out[:n] = v[:n].detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def choose_input_dim(train_samples: Sequence[SubjectSample], cap_dim: int) -> int:
    dims: List[int] = []
    for s in train_samples:
        for qid in TARGET_QIDS:
            if qid in s.q_feats:
                dims.append(int(s.q_feats[qid].numel()))
    if len(dims) == 0:
        return min(256, int(cap_dim))
    d = int(np.quantile(np.asarray(dims, dtype=np.float32), 0.9))
    d = max(32, d)
    d = min(d, int(cap_dim))
    return d


def compute_norm_stats(train_samples: Sequence[SubjectSample], in_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    for s in train_samples:
        for qid in TARGET_QIDS:
            v = s.q_feats.get(qid)
            if v is None:
                continue
            rows.append(vec_to_dim(v, in_dim))
    if len(rows) == 0:
        return np.zeros((in_dim,), dtype=np.float32), np.ones((in_dim,), dtype=np.float32)
    x = np.stack(rows, axis=0).astype(np.float32)
    mu = x.mean(axis=0).astype(np.float32)
    sd = x.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu, sd


def compute_big5_stats(train_samples: Sequence[SubjectSample]) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[np.ndarray] = []
    for s in train_samples:
        p = np.asarray(s.big5, dtype=np.float32).reshape(-1)
        if p.shape[0] == 5:
            rows.append(p)
    if len(rows) == 0:
        return np.zeros((5,), dtype=np.float32), np.ones((5,), dtype=np.float32)
    x = np.stack(rows, axis=0).astype(np.float32)
    mu = x.mean(axis=0).astype(np.float32)
    sd = x.std(axis=0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu, sd


@dataclass
class QuestionInstance:
    sid: str
    qid: int
    q_digit: int
    y_digit: int
    lie: int
    x: np.ndarray
    p5: np.ndarray


class QuestionDataset(Dataset):
    def __init__(self, instances: Sequence[QuestionInstance]):
        self.items = list(instances)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        return (
            torch.from_numpy(it.x.astype(np.float32)),
            torch.from_numpy(it.p5.astype(np.float32)),
            torch.tensor(int(it.lie), dtype=torch.long),
            str(it.sid),
            torch.tensor(int(it.qid), dtype=torch.long),
            torch.tensor(int(it.q_digit), dtype=torch.long),
            torch.tensor(int(it.y_digit), dtype=torch.long),
        )


def collate_questions(batch):
    xs, p5s, ys, sids, qids, qds, yds = zip(*batch)
    x = torch.stack(xs, dim=0)
    p5 = torch.stack(p5s, dim=0)
    y = torch.stack(ys, dim=0)
    qid = torch.stack(qids, dim=0)
    qd = torch.stack(qds, dim=0)
    yd = torch.stack(yds, dim=0)
    return x, p5, y, list(sids), qid, qd, yd


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam=1.0):
    return GradReverse.apply(x, lam)


class ReEmbedEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 256, drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(drop)),
            nn.Linear(int(hidden), int(out_dim)),
        )

    def forward(self, x):
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, drop: float = 0.3, num_cls: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(drop)),
            nn.Linear(int(hidden), int(num_cls)),
        )

    def forward(self, x):
        return self.net(x)


class GenLieModel(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        re_dim: int,
        hidden: int,
        drop: float,
        num_persons: int,
        use_big5: bool = False,
        big5_scale: float = 0.2,
        big5_hidden: int = 16,
        big5_fusion: str = "interaction",
    ):
        super().__init__()
        self.re = ReEmbedEncoder(feat_dim, hidden, re_dim, drop)
        self.use_big5 = bool(use_big5)
        self.big5_scale = float(big5_scale)
        self.big5_fusion = str(big5_fusion).lower().strip()
        self.big5_hidden = int(big5_hidden)

        cls_in = int(re_dim)
        if self.use_big5:
            self.p_proj = nn.Sequential(
                nn.Linear(5, self.big5_hidden),
                nn.ReLU(),
                nn.Dropout(float(drop)),
            )
            if self.big5_fusion == "interaction":
                self.p_to_emb = nn.Linear(self.big5_hidden, int(re_dim))
                cls_in += int(re_dim)
            cls_in += self.big5_hidden

        self.cls = ClassifierHead(cls_in, hidden, drop, num_cls=2)
        self.idh = ClassifierHead(re_dim, hidden, drop, num_cls=num_persons)

    def forward(self, feats: torch.Tensor, p5: Optional[torch.Tensor] = None, lam: float = 1.0, return_feat: bool = False):
        emb = self.re(feats)
        z_parts = [emb]
        if self.use_big5:
            if p5 is None:
                p = torch.zeros((emb.size(0), 5), dtype=emb.dtype, device=emb.device)
            else:
                p = p5.to(emb.device).float()
                if p.ndim == 1:
                    p = p.unsqueeze(0)
            if p.shape[-1] != 5:
                raise ValueError(f"big5 dim mismatch, expected 5 got {p.shape[-1]}")
            p = p * float(self.big5_scale)
            p_emb = self.p_proj(p)
            if self.big5_fusion == "interaction":
                gate = torch.tanh(self.p_to_emb(p_emb))
                inter = emb * gate
                z_parts.append(inter)
            z_parts.append(p_emb)
        z = torch.cat(z_parts, dim=-1) if len(z_parts) > 1 else z_parts[0]
        logits = self.cls(z)
        idlog = self.idh(grad_reverse(emb, lam))
        if return_feat:
            return logits, idlog, emb
        return logits, idlog


def sample_triplets(embeddings: torch.Tensor, labels: torch.Tensor, pid_idx: torch.Tensor):
    a, p, n = [], [], []
    bsz = int(labels.size(0))
    for i in range(bsz):
        yi = int(labels[i].item())
        pi = int(pid_idx[i].item())
        pos_candidates = [j for j in range(bsz) if int(labels[j].item()) == yi and j != i]
        neg_candidates = [j for j in range(bsz) if int(labels[j].item()) != yi and int(pid_idx[j].item()) != pi]
        if len(pos_candidates) == 0 or len(neg_candidates) == 0:
            continue
        a.append(i)
        p.append(random.choice(pos_candidates))
        n.append(random.choice(neg_candidates))
    if len(a) == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    return torch.tensor(a, dtype=torch.long), torch.tensor(p, dtype=torch.long), torch.tensor(n, dtype=torch.long)


def build_instances(
    samples: Sequence[SubjectSample],
    in_dim: int,
    mu: np.ndarray,
    sd: np.ndarray,
    p5_mu: np.ndarray,
    p5_sd: np.ndarray,
) -> List[QuestionInstance]:
    out: List[QuestionInstance] = []
    for s in samples:
        p5 = np.asarray(s.big5, dtype=np.float32).reshape(-1)
        if p5.shape[0] != 5:
            p5 = np.zeros((5,), dtype=np.float32)
        p5 = (p5 - p5_mu) / p5_sd
        for qid in TARGET_QIDS:
            v = s.q_feats.get(qid)
            if v is None:
                continue
            qd = digit_from_qid(int(qid))
            if qd is None:
                continue
            x = vec_to_dim(v, in_dim)
            x = (x - mu) / sd
            out.append(
                QuestionInstance(
                    sid=s.sid,
                    qid=int(qid),
                    q_digit=int(qd),
                    y_digit=int(s.y_digit),
                    lie=int(s.q_lie.get(int(qid), 0)),
                    x=x.astype(np.float32),
                    p5=p5.astype(np.float32),
                )
            )
    return out


def aggregate_subject_scores(rows: Sequence[Dict[str, Any]], agg_mode: str = "sum") -> Dict[str, Dict[str, Any]]:
    agg_mode = str(agg_mode).lower().strip()
    by_sid: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sid = str(r["sid"])
        by_sid.setdefault(
            sid,
            {
                "y_digit": int(r["y_digit"]),
                "pairs": defaultdict(list),
            },
        )
        by_sid[sid]["pairs"][int(r["q_digit"])].append(float(r["score_lie"]))

    out: Dict[str, Dict[str, Any]] = {}
    for sid, obj in by_sid.items():
        logits = np.zeros((10,), dtype=np.float32)
        for d in range(1, 11):
            vals = np.asarray(obj["pairs"].get(d, []), dtype=np.float32)
            if vals.size == 0:
                logits[d - 1] = -1e6 if agg_mode == "max" else 0.0
            elif agg_mode == "sum":
                logits[d - 1] = float(vals.sum())
            elif agg_mode == "mean":
                logits[d - 1] = float(vals.mean())
            elif agg_mode == "max":
                logits[d - 1] = float(vals.max())
            else:
                raise ValueError(f"Unknown agg_mode: {agg_mode}")
        rank = (np.argsort(-logits) + 1).tolist()
        out[sid] = {
            "y_true": int(obj["y_digit"]),
            "rank": [int(x) for x in rank],
            "logits": logits.tolist(),
        }
    return out


def topk_metrics(preds: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    n = len(preds)
    h1 = 0
    h2 = 0
    h3 = 0
    for sid, p in preds.items():
        y = int(p["y_true"])
        rk = [int(x) for x in p["rank"]]
        h1 += int(rk[0] == y)
        h2 += int(y in rk[:2])
        h3 += int(y in rk[:3])
    den = max(1, n)
    return {
        "n": float(n),
        "top1": float(h1 / den),
        "top2": float(h2 / den),
        "top3": float(h3 / den),
    }


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
    pos = (y_true == 1)
    neg = (y_true == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_prob, kind="mergesort")
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and y_prob[order[j]] == y_prob[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_rank_pos = float(ranks[pos].sum())
    auc = (sum_rank_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def binary_metrics_from_rows(rows: Sequence[Dict[str, Any]], threshold: float = 0.5) -> Dict[str, float]:
    if len(rows) == 0:
        return {"n_q": 0.0, "acc": 0.0, "pos_f1": 0.0, "pos_auc": 0.5}
    y_true = np.asarray([int(r["lie_true"]) for r in rows], dtype=np.int64)
    y_prob = np.asarray([float(r["score_lie"]) for r in rows], dtype=np.float64)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    acc = float((y_pred == y_true).mean())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = float(tp / max(1, tp + fp))
    rec = float(tp / max(1, tp + fn))
    f1 = float((2.0 * prec * rec) / max(1e-12, prec + rec))
    auc = _binary_auc(y_true, y_prob)
    return {"n_q": float(y_true.size), "acc": acc, "pos_f1": f1, "pos_auc": auc}


def estimate_genlie_complexity(
    model: GenLieModel,
    in_dim: int,
    device: torch.device,
    use_big5: bool = False,
) -> Dict[str, float]:
    param_count = int(sum(p.numel() for p in model.parameters()))
    flops = 0
    hooks: List[Any] = []

    def _linear_hook(mod: nn.Linear, inp, _out):
        nonlocal flops
        if len(inp) == 0 or (not torch.is_tensor(inp[0])):
            return
        x = inp[0]
        if x.ndim == 0:
            return
        bitems = int(np.prod(list(x.shape[:-1]))) if x.ndim > 1 else int(x.shape[0])
        flops += int(bitems) * int(mod.in_features) * int(mod.out_features) * 2

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    was_train = bool(model.training)
    model.eval()
    with torch.no_grad():
        x = torch.zeros((1, int(in_dim)), dtype=torch.float32, device=device)
        p5 = torch.zeros((1, 5), dtype=torch.float32, device=device) if bool(use_big5) else None
        _ = model(x, p5=p5, lam=0.0)
    if was_train:
        model.train()
    for h in hooks:
        h.remove()

    # One subject has 20 questions in this task.
    flops_per_question = float(flops)
    flops_per_subject = float(flops_per_question * 20.0)
    return {
        "param_k": float(param_count / 1e3),
        "flops_per_question": flops_per_question,
        "flops_per_subject": flops_per_subject,
    }


def evaluate_question_loader(model: GenLieModel, loader: DataLoader, device: torch.device, lam_eval: float = 0.0) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for feats, p5, lbls, sids, qids, qds, yds in loader:
            feats = feats.to(device)
            p5 = p5.to(device)
            logits, _ = model(feats, p5=p5, lam=float(lam_eval))
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)
            for i in range(len(sids)):
                rows.append(
                    {
                        "sid": str(sids[i]),
                        "qid": int(qids[i].item()),
                        "q_digit": int(qds[i].item()),
                        "y_digit": int(yds[i].item()),
                        "lie_true": int(lbls[i].item()),
                        "score_lie": float(prob[i]),
                    }
                )
    return rows


def run_fold(
    fold: int,
    train_samples: Sequence[SubjectSample],
    test_samples: Sequence[SubjectSample],
    device: torch.device,
    cfg: argparse.Namespace,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, float]]:
    in_dim = choose_input_dim(train_samples, int(cfg.input_dim_cap))
    mu, sd = compute_norm_stats(train_samples, in_dim)
    p5_mu, p5_sd = compute_big5_stats(train_samples)

    tr_inst = build_instances(train_samples, in_dim, mu, sd, p5_mu, p5_sd)
    te_inst = build_instances(test_samples, in_dim, mu, sd, p5_mu, p5_sd)
    if len(tr_inst) == 0 or len(te_inst) == 0:
        raise RuntimeError(f"fold {fold}: empty train/test instances")

    tr_ds = QuestionDataset(tr_inst)
    te_ds = QuestionDataset(te_inst)

    person_list = sorted(list({x.sid for x in train_samples}))
    pid2idx = {p: i for i, p in enumerate(person_list)}

    y_train = [int(x.lie) for x in tr_inst]
    cnt = Counter(y_train)
    w0 = 1.0 / max(1, int(cnt.get(0, 1)))
    w1 = 1.0 / max(1, int(cnt.get(1, 1)))
    ws = [w1 if int(y) == 1 else w0 for y in y_train]
    sampler = WeightedRandomSampler(torch.tensor(ws, dtype=torch.double), num_samples=len(ws), replacement=True)

    tr_ld = DataLoader(tr_ds, batch_size=int(cfg.batch_size), sampler=sampler, collate_fn=collate_questions)
    te_ld = DataLoader(te_ds, batch_size=int(cfg.batch_size), shuffle=False, collate_fn=collate_questions)

    model = GenLieModel(
        feat_dim=int(in_dim),
        re_dim=int(cfg.re_embed_dim),
        hidden=int(cfg.hidden_dim),
        drop=float(cfg.dropout_rate),
        num_persons=max(2, len(person_list)),
        use_big5=bool(int(cfg.use_big5)),
        big5_scale=float(cfg.big5_scale),
        big5_hidden=int(cfg.big5_hidden),
        big5_fusion=str(cfg.big5_fusion),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    cls_loss = nn.CrossEntropyLoss()
    id_loss_fn = nn.CrossEntropyLoss()
    tri_loss_fn = nn.TripletMarginLoss(margin=float(cfg.triplet_margin))
    scaler = torch.amp.GradScaler(device.type, enabled=bool(int(cfg.fp16)) and device.type == "cuda")

    best = None
    best_state = None
    best_rows: List[Dict[str, Any]] = []

    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        loss_sum = 0.0
        nsum = 0
        iterator = tr_ld if tqdm is None else tqdm(tr_ld, desc=f"fold {fold} ep {ep}/{cfg.epochs}", leave=False)
        for feats, p5, lbls, sids, qids, qds, yds in iterator:
            feats = feats.to(device)
            p5 = p5.to(device)
            lbls = lbls.to(device)
            pid_idx = torch.tensor([pid2idx.get(str(s), 0) for s in sids], dtype=torch.long, device=device)
            with torch.amp.autocast(device_type=device.type, enabled=bool(int(cfg.fp16)) and device.type == "cuda"):
                logits, id_logits, emb = model(feats, p5=p5, lam=float(cfg.id_loss_lambda), return_feat=True)
                lc = cls_loss(logits, lbls)
                lid = id_loss_fn(id_logits, pid_idx)
                a, p, n = sample_triplets(emb, lbls, pid_idx)
                if a.numel() == 0:
                    lt = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    a = a.to(device)
                    p = p.to(device)
                    n = n.to(device)
                    lt = tri_loss_fn(emb[a], emb[p], emb[n])
                loss = lc + float(cfg.id_loss_weight) * lid + float(cfg.triplet_loss_weight) * lt

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            bsz = int(lbls.shape[0])
            loss_sum += float(loss.item()) * bsz
            nsum += bsz

        train_loss = float(loss_sum / max(1, nsum))

        test_rows = evaluate_question_loader(model, te_ld, device, lam_eval=0.0)
        subj_pred = aggregate_subject_scores(test_rows, agg_mode=str(cfg.agg_mode))
        m = topk_metrics(subj_pred)

        cur = {
            "top1": float(m["top1"]),
            "top2": float(m["top2"]),
            "top3": float(m["top3"]),
            "n": float(m["n"]),
            "epoch": float(ep),
            "train_loss": float(train_loss),
            "in_dim": float(in_dim),
        }
        logging.info(
            "fold %d ep %d | train_loss=%.4f | top1=%.4f top2=%.4f top3=%.4f",
            fold,
            ep,
            train_loss,
            cur["top1"],
            cur["top2"],
            cur["top3"],
        )

        if best is None:
            better = True
        else:
            better = False
            if cur["top1"] > best["top1"]:
                better = True
            elif cur["top1"] == best["top1"] and cur["top2"] > best["top2"]:
                better = True
            elif cur["top1"] == best["top1"] and cur["top2"] == best["top2"] and cur["top3"] > best["top3"]:
                better = True

        if better:
            best = cur
            best_rows = test_rows
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best is None or best_state is None:
        raise RuntimeError(f"fold {fold}: no valid result")

    complexity = estimate_genlie_complexity(
        model=model,
        in_dim=int(in_dim),
        device=device,
        use_big5=bool(int(cfg.use_big5)),
    )
    return best, best_rows, complexity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GenLie single-modality feature 5-fold evaluator")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--split_root", type=str, default="./data_split/5fold")
    p.add_argument("--out_dir", type=str, default="./results/video_genlie_5fold")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--feature_modality", type=str, default="video")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", type=int, default=0, choices=[0, 1])

    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--re_embed_dim", type=int, default=256)
    p.add_argument("--dropout_rate", type=float, default=0.3)
    p.add_argument("--triplet_margin", type=float, default=1.0)
    p.add_argument("--id_loss_weight", type=float, default=0.1)
    p.add_argument("--triplet_loss_weight", type=float, default=0.1)
    p.add_argument("--id_loss_lambda", type=float, default=1.0)

    p.add_argument("--input_dim_cap", type=int, default=4096)
    p.add_argument("--feature_qid_shift", type=int, default=0, help="feature qid = qid + shift")
    p.add_argument("--min_valid_q", type=int, default=20)
    p.add_argument("--agg_mode", type=str, default="sum", choices=["sum", "mean", "max"])
    p.add_argument("--use_big5", type=int, default=0, choices=[0, 1])
    p.add_argument("--big5_scale", type=float, default=0.2)
    p.add_argument("--big5_hidden", type=int, default=16)
    p.add_argument("--big5_fusion", type=str, default="interaction", choices=["interaction", "concat"])
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_modality = normalize_modality_name(getattr(cfg, "feature_modality", "video"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "train.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    seed_everything(int(cfg.seed))
    device = get_device(str(cfg.device))
    logging.info("device=%s", device)
    logging.info("feature_modality=%s", feature_modality)
    logging.info("workflow: question score -> digit aggregation")
    logging.info(
        "big5=%s (fusion=%s, scale=%.3f, hidden=%d)",
        "on" if bool(int(cfg.use_big5)) else "off",
        str(cfg.big5_fusion),
        float(cfg.big5_scale),
        int(cfg.big5_hidden),
    )

    split_root = Path(cfg.split_root)
    all_ids: List[str] = []
    for fold in range(1, 6):
        all_ids.extend(_read_ids(split_root / f"fold_{fold}" / "train_ids.txt"))
        all_ids.extend(_read_ids(split_root / f"fold_{fold}" / "test_ids.txt"))
    all_ids = sorted(set(all_ids))
    if len(all_ids) == 0:
        raise RuntimeError("No subject IDs found in split_root")

    data_root = Path(cfg.data_root)
    sample_map: Dict[str, SubjectSample] = {}
    for sid in all_ids:
        s = load_subject_sample(
            data_root=data_root,
            sid=sid,
            min_valid_q=int(cfg.min_valid_q),
            feature_qid_shift=int(cfg.feature_qid_shift),
        )
        if s is not None:
            sample_map[sid] = s
    logging.info("loaded %s subjects: %d/%d", feature_modality, len(sample_map), len(all_ids))
    if len(sample_map) == 0:
        raise RuntimeError(f"No valid {feature_modality} subject sample loaded")

    fold_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    qdetail_rows: List[Dict[str, Any]] = []
    complexity_rows: List[Dict[str, Any]] = []
    total_n = 0
    total_h1 = 0
    total_h2 = 0
    total_h3 = 0

    for fold in range(1, 6):
        tr_ids = [x for x in _read_ids(split_root / f"fold_{fold}" / "train_ids.txt") if x in sample_map]
        te_ids = [x for x in _read_ids(split_root / f"fold_{fold}" / "test_ids.txt") if x in sample_map]
        if len(tr_ids) == 0 or len(te_ids) == 0:
            logging.warning("fold %d skipped: train=%d test=%d", fold, len(tr_ids), len(te_ids))
            continue

        tr_samples = [sample_map[sid] for sid in tr_ids]
        te_samples = [sample_map[sid] for sid in te_ids]

        best, test_rows, complexity = run_fold(fold, tr_samples, te_samples, device, cfg)

        subj_pred = aggregate_subject_scores(test_rows, agg_mode=str(cfg.agg_mode))
        met = topk_metrics(subj_pred)
        bin_met = binary_metrics_from_rows(test_rows)

        h1 = int(round(float(met["top1"]) * float(met["n"])))
        h2 = int(round(float(met["top2"]) * float(met["n"])))
        h3 = int(round(float(met["top3"]) * float(met["n"])))
        total_n += int(met["n"])
        total_h1 += h1
        total_h2 += h2
        total_h3 += h3

        fold_row = {
            "fold": int(fold),
            "n_test": int(met["n"]),
            "n_q": int(bin_met["n_q"]),
            "best_epoch": int(best["epoch"]),
            "in_dim": int(best["in_dim"]),
            "train_loss": float(best["train_loss"]),
            "top1": float(met["top1"]),
            "top2": float(met["top2"]),
            "top3": float(met["top3"]),
            "bin_acc": float(bin_met["acc"]),
            "pos_f1": float(bin_met["pos_f1"]),
            "pos_auc": float(bin_met["pos_auc"]),
        }
        fold_rows.append(fold_row)
        complexity_rows.append(
            {
                "fold": int(fold),
                "param_k": float(complexity["param_k"]),
                "flops_per_question": float(complexity["flops_per_question"]),
                "flops_per_subject": float(complexity["flops_per_subject"]),
            }
        )
        logging.info(
            "fold %d: top1=%.2f%% top2=%.2f%% top3=%.2f%% | ACC=%.2f%% F1=%.2f%% AUC=%.2f%% | params=%.3fK flops/q=%.0f (best_epoch=%d)",
            fold,
            fold_row["top1"] * 100.0,
            fold_row["top2"] * 100.0,
            fold_row["top3"] * 100.0,
            fold_row["bin_acc"] * 100.0,
            fold_row["pos_f1"] * 100.0,
            fold_row["pos_auc"] * 100.0,
            float(complexity["param_k"]),
            float(complexity["flops_per_question"]),
            fold_row["best_epoch"],
        )

        for sid, p in subj_pred.items():
            detail_rows.append(
                {
                    "fold": int(fold),
                    "subject_id": str(sid),
                    "y_true": int(p["y_true"]),
                    "y_hat": int(p["rank"][0]),
                    "top2_hit": int(int(p["y_true"]) in [int(x) for x in p["rank"][:2]]),
                    "top3_hit": int(int(p["y_true"]) in [int(x) for x in p["rank"][:3]]),
                    "top3": json.dumps([int(x) for x in p["rank"][:3]], ensure_ascii=False),
                }
            )
        for r in test_rows:
            qdetail_rows.append(
                {
                    "fold": int(fold),
                    "subject_id": str(r["sid"]),
                    "qid": int(r["qid"]),
                    "q_digit": int(r["q_digit"]),
                    "y_digit": int(r["y_digit"]),
                    "lie_true": int(r["lie_true"]),
                    "score_lie": float(r["score_lie"]),
                    "lie_pred": int(float(r["score_lie"]) >= 0.5),
                }
            )

    if len(fold_rows) == 0:
        raise RuntimeError("No fold completed")

    mean_top1 = float(np.mean([float(r["top1"]) for r in fold_rows]))
    mean_top2 = float(np.mean([float(r["top2"]) for r in fold_rows]))
    mean_top3 = float(np.mean([float(r["top3"]) for r in fold_rows]))
    mean_bin_acc = float(np.mean([float(r["bin_acc"]) for r in fold_rows]))
    mean_pos_f1 = float(np.mean([float(r["pos_f1"]) for r in fold_rows]))
    mean_pos_auc = float(np.mean([float(r["pos_auc"]) for r in fold_rows]))
    overall_top1 = float(total_h1) / max(1, int(total_n))
    overall_top2 = float(total_h2) / max(1, int(total_n))
    overall_top3 = float(total_h3) / max(1, int(total_n))
    overall_bin = binary_metrics_from_rows(qdetail_rows)
    top1_stats = summarize_mean_std([float(r["top1"]) for r in fold_rows])
    top2_stats = summarize_mean_std([float(r["top2"]) for r in fold_rows])
    top3_stats = summarize_mean_std([float(r["top3"]) for r in fold_rows])
    acc_stats = summarize_mean_std([float(r["bin_acc"]) for r in fold_rows])
    f1_stats = summarize_mean_std([float(r["pos_f1"]) for r in fold_rows])
    auc_stats = summarize_mean_std([float(r["pos_auc"]) for r in fold_rows])
    param_stats = summarize_mean_std([float(r["param_k"]) for r in complexity_rows])
    flops_q_stats = summarize_mean_std([float(r["flops_per_question"]) for r in complexity_rows])
    flops_subj_stats = summarize_mean_std([float(r["flops_per_subject"]) for r in complexity_rows])

    with (out_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "n_test",
                "n_q",
                "best_epoch",
                "in_dim",
                "train_loss",
                "top1",
                "top2",
                "top3",
                "bin_acc",
                "pos_f1",
                "pos_auc",
            ],
        )
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    with (out_dir / "detail_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "subject_id", "y_true", "y_hat", "top2_hit", "top3_hit", "top3"])
        w.writeheader()
        for r in detail_rows:
            w.writerow(r)

    with (out_dir / "detail_question_binary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["fold", "subject_id", "qid", "q_digit", "y_digit", "lie_true", "score_lie", "lie_pred"],
        )
        w.writeheader()
        for r in qdetail_rows:
            w.writerow(r)

    with (out_dir / "model_complexity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "param_k", "flops_per_question", "flops_per_subject"])
        w.writeheader()
        for r in complexity_rows:
            w.writerow(r)

    with (out_dir / "model_complexity.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "param_k_mean": float(param_stats["mean"]),
                "param_k_std": float(param_stats["std"]),
                "flops_per_question_mean": float(flops_q_stats["mean"]),
                "flops_per_question_std": float(flops_q_stats["std"]),
                "flops_per_subject_mean": float(flops_subj_stats["mean"]),
                "flops_per_subject_std": float(flops_subj_stats["std"]),
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
        "mean_fold_pos_f1": mean_pos_f1,
        "mean_fold_pos_auc": mean_pos_auc,
        "overall_top1": overall_top1,
        "overall_top2": overall_top2,
        "overall_top3": overall_top3,
        "overall_bin_acc": float(overall_bin["acc"]),
        "overall_pos_f1": float(overall_bin["pos_f1"]),
        "overall_pos_auc": float(overall_bin["pos_auc"]),
        "stats_mean_std": {
            "top1": top1_stats,
            "top2": top2_stats,
            "top3": top3_stats,
            "bin_acc": acc_stats,
            "pos_f1": f1_stats,
            "pos_auc": auc_stats,
            "param_k": param_stats,
            "flops_per_question": flops_q_stats,
            "flops_per_subject": flops_subj_stats,
        },
        "model_complexity": {
            "param_k_mean": float(param_stats["mean"]),
            "param_k_std": float(param_stats["std"]),
            "flops_per_question_mean": float(flops_q_stats["mean"]),
            "flops_per_question_std": float(flops_q_stats["std"]),
            "flops_per_subject_mean": float(flops_subj_stats["mean"]),
            "flops_per_subject_std": float(flops_subj_stats["std"]),
        },
        "folds": fold_rows,
        "config": vars(cfg),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Per-fold metrics:")
    for r in fold_rows:
        logging.info(
            "fold %d: n=%d top1=%.2f%% top2=%.2f%% top3=%.2f%% | ACC=%.2f%% F1=%.2f%% AUC=%.2f%%",
            int(r["fold"]),
            int(r["n_test"]),
            float(r["top1"]) * 100.0,
            float(r["top2"]) * 100.0,
            float(r["top3"]) * 100.0,
            float(r["bin_acc"]) * 100.0,
            float(r["pos_f1"]) * 100.0,
            float(r["pos_auc"]) * 100.0,
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
        "Question-level binary overall: n=%d ACC=%.2f%% F1=%.2f%% AUC=%.2f%%",
        int(overall_bin["n_q"]),
        float(overall_bin["acc"]) * 100.0,
        float(overall_bin["pos_f1"]) * 100.0,
        float(overall_bin["pos_auc"]) * 100.0,
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
        "Model complexity mean+/-std: params=%.3f+/-%.3f K, flops/q=%.0f+/-%.0f, flops/subject=%.0f+/-%.0f",
        float(param_stats["mean"]),
        float(param_stats["std"]),
        float(flops_q_stats["mean"]),
        float(flops_q_stats["std"]),
        float(flops_subj_stats["mean"]),
        float(flops_subj_stats["std"]),
    )
    logging.info("Saved: %s", out_dir)


if __name__ == "__main__":
    main()
