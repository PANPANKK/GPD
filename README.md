
## 1. Package Contents

- `code/train_video_genlie_q2d_progressive_kd_5fold.py`
  - Main training script (supports `--student_modality video|audio`)
- `code/eval_video_genlie_5fold.py`
  - Data loading and base model dependencies imported by the main script
- `commands/run_video_best.sh`
  - One-click script for the video baseline
- `commands/run_audio_best.sh`
  - One-click script for the audio baseline
- `commands/run_all.sh`
  - Run both video and audio baselines
- `meta/expected_metrics.json`
  - Reference metrics from historical best checkpoints
- `requirements.txt`
  - Python dependencies

## 2. Environment Requirements

Recommended environment:

- Python 3.9+
- PyTorch 2.x with CUDA support
- A single high-memory GPU is recommended for the 120-epoch setting

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Data and Teacher Model Paths

The scripts were originally developed with fixed internal paths.  
For public release, these paths should be replaced by user-defined local paths or environment variables.

Suggested directory structure:

- Video data: `/path/to/data/video`
- Audio data: `/path/to/data/audio`
- Video split files: `/path/to/data_split/video_5fold`
- Audio split files: `/path/to/data_split/audio_5fold`
- Video teacher repository: `/path/to/teacher_repo/video`
- Audio teacher repository: `/path/to/teacher_repo/audio`
- Video teacher checkpoint: `/path/to/checkpoints/video_teacher`
- Audio teacher checkpoint: `/path/to/checkpoints/audio_teacher`

Before running, please update the scripts or shell commands to match your local environment.

## 4. Running the Experiments

### 4.1 Run Each Modality Separately

```bash
bash commands/run_video_best.sh /path/to/output_root
bash commands/run_audio_best.sh /path/to/output_root
```

Default output directories:

- Video: `/path/to/output_root/video_best`
- Audio: `/path/to/output_root/audio_best`

### 4.2 Run Both Modalities

```bash
bash commands/run_all.sh /path/to/output_root
```

## 5. Output Files

Each modality produces:

- `summary.json` — final aggregated results
- `fold_metrics.csv`
- `detail_predictions.csv`
- `detail_question_binary.csv`
- `model_complexity.csv`
- `model_complexity.json`
- `fold_i/best_student.pt`

## 6. Baseline Configuration

### Video Baseline

- `progressive_mode=sigmoid`
- `progressive_path_order=joint`
- `online_gap_mode=ema_hard`
- `lambda_rank_kd=0.0`
- `lambda_digit_kd=0.7`
- `lambda_feat_align=0.2`

### Audio Baseline

- `progressive_mode=sigmoid`
- `progressive_path_order=feature_first`
- `online_gap_mode=ema_hard`
- `lambda_rank_kd=0.0`
- `lambda_digit_kd=0.7`
- `lambda_feat_align=0.2`

## 7. Reference Metrics (Historical Snapshot)

See `meta/expected_metrics.json`.  

> Note: Minor variations may occur across hardware, drivers, and runtime environments.  
> Please use `summary.json` as the final reference for reproduced results.
