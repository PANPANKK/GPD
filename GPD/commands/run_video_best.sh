#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="$ROOT_DIR/code/train_video_genlie_q2d_progressive_kd_5fold.py"

OUT_ROOT="${1:-$ROOT_DIR/results}"
OUT_DIR="${OUT_DIR:-$OUT_ROOT/video_best}"
mkdir -p "$OUT_DIR"

DATA_ROOT="${VIDEO_DATA_ROOT:-/mnt/**/data}"
SPLIT_ROOT="${VIDEO_SPLIT_ROOT:-/mnt/**/repro_gap_adaptive_20260319/data_split/5fold}"
TEACHER_REPO_ROOT="${VIDEO_TEACHER_REPO_ROOT:-/mnt/**/repro_gap_adaptive_20260319/gsr_time2graphplus}"
TEACHER_CKPT_ROOT="${VIDEO_TEACHER_CKPT_ROOT:-/mnt/**/repro_gap_adaptive_20260319/results/kd_staged/xkd_exp01/stage1_teacher}"

"$PYTHON_BIN" "$TRAIN_SCRIPT" \
  --data_root "$DATA_ROOT" \
  --split_root "$SPLIT_ROOT" \
  --out_dir "$OUT_DIR" \
  --device cuda \
  --student_modality video \
  --teacher_repo_root "$TEACHER_REPO_ROOT" \
  --teacher_ckpt_root "$TEACHER_CKPT_ROOT" \
  --teacher_type time2graph \
  --teacher_infer_batch_size 256 \
  --teacher_num_workers 8 \
  --reuse_teacher_score_cache 1 \
  --reuse_teacher_feat_cache 1 \
  --batch_size 16 --epochs 120 --lr 1e-4 --weight_decay 1e-4 \
  --seed 42 --fp16 0 \
  --hidden_dim 512 --re_embed_dim 256 --dropout_rate 0.3 \
  --student_mode per_q --triplet_margin 1.0 --id_loss_weight 0.0 --triplet_loss_weight 0.0 --id_loss_lambda 1.0 \
  --input_dim_cap 4096 --feature_qid_shift 0 --min_valid_q 20 \
  --val_ratio 0.0 --select_on test \
  --agg_modes sum,mean,max,logsumexp --kd_digit_agg_mode sum \
  --lambda_lie_ce 1.0 --lambda_digit_ce 0.3 --lambda_rank_kd 0.0 --lambda_digit_kd 0.7 --lambda_feat_align 0.2 \
  --temp_rank 2.0 --temp_digit 2.0 \
  --stage1_epochs 8 --stage2_epochs 12 --stage3_digit_ramp_epochs 10 \
  --progressive_mode sigmoid --progressive_path_order joint \
  --progressive_overlap_ratio 0.0 --progressive_overlap_width 6.0 \
  --rank_warm_width 4.0 --feat_warm_width 6.0 --digit_warm_width 8.0 --feat_max_weight 1.0 \
  --kd_conf_mode none --kd_conf_high 0.35 --kd_conf_low 0.05 --kd_conf_min_keep_frac 0.15 \
  --logitstd_mode none --logitstd_eps 1e-6 \
  --online_gap_mode ema_hard --gap_update_warmup_epochs 5 --gap_update_interval 2 --gap_ema_momentum 0.8 \
  --gap_route_profile manual --gap_adaptive_feat_weight 0 --gap_obs_scale 1.0 --gap_obs_bias 0.0 \
  --gap_route_no_feature_enter 0.34 --gap_route_no_feature_exit 0.40 \
  --gap_route_digit_enter 0.40 --gap_route_digit_exit 0.48 \
  --gap_route_feature_exit 0.52 --gap_route_feature_enter 0.60 \
  --gap_route_min_hold_epochs 3 --gap_online_max_rows 4096 --gap_online_min_reliable_similarity 0.02 \
  --nofeat_dynamic_route 0 --nofeat_route_require_rank_kd 0 --rank0_force_online_gap_fallback 0 \
  --feat_gate_mode off \
  --route_feat_scale_enable 1 --route_feat_scale_feature_first 1.0 --route_feat_scale_joint 1.0 \
  --route_feat_scale_digit_first 1.0 --route_feat_scale_no_feature 0.0

echo "[DONE] video summary: $OUT_DIR/summary.json"
