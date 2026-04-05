# GPD Release Package

## 1. Package Contents

- `GPD/code/train_video_genlie_q2d_progressive_kd_5fold.py`
  - Main training script (supports `--student_modality video|audio`)
- `GPD/code/eval_video_genlie_5fold.py`
  - Data loader and shared model utilities
- `GPD/commands/run_video_best.sh`
  - Example command for video runs
- `GPD/commands/run_audio_best.sh`
  - Example command for audio runs
- `GPD/commands/run_all.sh`
  - Run video and audio commands sequentially
- `GPD/requirements.txt`
  - Python dependencies

## 2. Environment

- Python 3.9+
- PyTorch 2.x
- CUDA GPU is recommended

Install dependencies:

```bash
pip install -r GPD/requirements.txt
```

## 3. Required Paths

Set these paths in shell environment variables before running:

- `VIDEO_DATA_ROOT`
- `VIDEO_SPLIT_ROOT`
- `VIDEO_TEACHER_REPO_ROOT`
- `VIDEO_TEACHER_CKPT_ROOT`
- `AUDIO_DATA_ROOT`
- `AUDIO_SPLIT_ROOT`
- `AUDIO_TEACHER_REPO_ROOT`
- `AUDIO_TEACHER_CKPT_ROOT`

## 4. Run

Run video:

```bash
bash GPD/commands/run_video_best.sh /path/to/output_root
```

Run audio:

```bash
bash GPD/commands/run_audio_best.sh /path/to/output_root
```

Run both:

```bash
bash GPD/commands/run_all.sh /path/to/output_root
```

Optional selection controls:

- `SELECT_ON` (default `val`, choices: `val|test`)
- `VAL_RATIO` (default `0.2`)

Example:

```bash
SELECT_ON=val VAL_RATIO=0.2 bash GPD/commands/run_video_best.sh /path/to/output_root
```

## 5. Outputs

Each run produces:

- `summary.json`
- `fold_metrics.csv`
- `detail_predictions.csv`
- `detail_question_binary.csv`
- `model_complexity.csv`
- `model_complexity.json`
- `fold_i/best_student.pt`
