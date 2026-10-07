# GPD: Physiological Signal-Guided Progressive Cross-Modal Distillation for Non-Contact Deception Detection

**Official PyTorch implementation of MM'26 paper**

[![Paper](https://img.shields.io/badge/Paper-MM'26-blue)](docs/GPD_MM26_Paper.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-MuDD-green)](#-dataset-access)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📖 Overview

This repository contains the official implementation of our MM'26 paper: **"GPD: Physiological Signal-Guided Progressive Cross-Modal Distillation for Non-Contact Deception Detection"**.

### Key Contributions

- **MuDD Dataset**: A new **M**ultimodal **D**eception **D**etection dataset collected from 130 participants performing a number-guessing game, with synchronized video, audio, and physiological signals (GSR, PPG, HR), and Big-Five personality annotations (six modalities in total).
- **GPD Framework**: A progressive cross-modal knowledge distillation method that transfers knowledge from contact-based physiological sensors to non-contact audio-visual modalities.
- **State-of-the-art Performance**: Achieves superior deception detection using only video or audio, without requiring physiological sensors at inference time.

---

## 🗂️ Table of Contents

- [Installation](#-installation)
- [Dataset Access](#-dataset-access)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🔧 Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended)

The supplied `.sh` commands require Bash (Linux, macOS, WSL, or Git Bash). In Windows PowerShell, use the Python command below and pass paths explicitly. `requirements.txt` covers the student scripts; the external teacher repository has additional dependencies.

### Setup

1. Clone this repository:

```bash
git clone https://github.com/PANPANKK/GPD.git
cd GPD
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🗄️ Dataset Access

The **MuDD (Multimodal Deception Detection) Dataset** is available for academic research purposes only.

### Dataset Overview

- **130 participants**
- **~690 minutes in total** with number-guessing game paradigm
- **Multimodal recordings**:
  - 📹 Video 
  - 🎵 Audio
  - 🧠 Physiological signals: GSR, PPG, HR 
- **Ground-truth annotations** for binary truth/deception classification and 10-class hidden-number inference

### How to Access

1. **Read the Data Use Agreement**: [MuDD_Data_Use_Agreement.docx](docs/MuDD_Data_Use_Agreement.docx)
2. **Complete the Application Form**: Download and fill out the form in the agreement document
3. **Submit your application**: Email the signed form to **liuyao@uestc.edu.cn**, the dataset contact listed in the agreement
4. **Student applicants**: Must obtain supervisor/PI signature

### Terms of Use

- ✅ Academic and non-commercial research only
- ❌ No commercial use or operational deployment
- ❌ No redistribution or sharing with unauthorized parties
- ✅ Must cite our paper in any publications using MuDD

For full terms, see [MuDD_Data_Use_Agreement.docx](docs/MuDD_Data_Use_Agreement.docx).

---

## 🚀 Quick Start

### Environment Variables

Before running the Bash experiment wrappers, set the required data paths. These variables are read by the wrappers, not by Python directly:

```bash
# For video-based student model
export VIDEO_DATA_ROOT=/path/to/video/data
export VIDEO_SPLIT_ROOT=/path/to/video/splits
export VIDEO_TEACHER_REPO_ROOT=/path/to/video/teacher/repo
export VIDEO_TEACHER_CKPT_ROOT=/path/to/video/teacher/checkpoints

# For audio-based student model
export AUDIO_DATA_ROOT=/path/to/audio/data
export AUDIO_SPLIT_ROOT=/path/to/audio/splits
export AUDIO_TEACHER_REPO_ROOT=/path/to/audio/teacher/repo
export AUDIO_TEACHER_CKPT_ROOT=/path/to/audio/teacher/checkpoints
```

### Required Inputs and External Teacher

Raw video/audio files cannot be passed directly to these scripts. Provide pre-extracted question features and matching labels:

```text
data/<subject_id>/features/<session>/Q03.npy ... Q22.npy
data/<subject_id>/dataset/<session>/label.csv
splits/fold_1/train_ids.txt
splits/fold_1/test_ids.txt
... through fold_5
teacher_checkpoints/fold_1/best_ckpt.pt
... through fold_5
```

`label.csv` needs `question_id,label` columns, with labels 0 (truth) or 1 (lie). Q03–Q22 must contain exactly two lie questions corresponding to the same hidden number. Split text files contain one subject ID per line; training and test IDs must be disjoint. Each subject should have one matching session in `features` and `dataset`; the loader currently chooses the first subdirectory in each independently.

Feature formats supported: `.pt`, `.pth`, `.npy`, `.npz`. Raw-media preprocessing, split files, teacher source, and teacher checkpoints are not bundled. The default teacher requires `gsr_t2g.data` (`build_segment_table`, `load_subject_samples`, `records_to_arrays`) and `gsr_t2g.model.Time2GraphPlusForGSR` from the configured external repository. Teacher loading also needs its physiological input files under `data_root`; use the layout required by that repository. Obtain these inputs and its dependencies before training.

The paper DOI is retained in the citation, but it returned HTTP 404 during the 2026-10-07 link audit. The paper badge opens the [local paper copy](docs/GPD_MM26_Paper.pdf).

### Run Pre-configured Experiments

The following scripts use the paper's Section 5.1 settings: Adam, 120 epochs, batch size 16, learning rate/weight decay 1e-4, hidden/re-embedding dimensions 512/256, dropout 0.3, digit KD weight 0.7, feature KD weight 0.2 for both modalities, and temperature 2.0. Test-based best-epoch selection follows the author's configuration.

```bash
# Train video-based student model
bash commands/run_video_best.sh /path/to/output_root

# Train audio-based student model
bash commands/run_audio_best.sh /path/to/output_root

# Run both sequentially
bash commands/run_all.sh /path/to/output_root
```

---

## 🏋️ Training

### Basic Usage

```bash
python code/train_video_genlie_q2d_progressive_kd_5fold.py \
  --data_root /path/to/video/data \
  --split_root /path/to/video/splits \
  --teacher_repo_root /path/to/teacher/repo \
  --teacher_ckpt_root /path/to/teacher/checkpoints \
  --student_modality video \
  --out_dir /path/to/output \
  --select_on test \
  --lambda_digit_kd 0.7 --temp_digit 2.0 \
  --epochs 120 --batch_size 16
```

For Windows PowerShell, use a single line (Bash `\` line continuation and `export` do not work in PowerShell):

```powershell
python code/train_video_genlie_q2d_progressive_kd_5fold.py --data_root "F:/path/to/video/data" --split_root "F:/path/to/video/splits" --teacher_repo_root "F:/path/to/teacher/repo" --teacher_ckpt_root "F:/path/to/teacher/checkpoints" --student_modality video --out_dir "F:/path/to/output" --select_on test --epochs 120 --batch_size 16
```

### Paper Preprocessing and Repeated Runs

Section 5.1 specifies GSR low-pass filtering at 2 Hz, downsampling 256→32 Hz, and resizing each segment to 448 points; the external teacher produces 80-dimensional features. Visual features use VideoMAEv2 (16 frames per temporal segment, 768 dimensions). Audio features use pretrained WavLM and local average pooling aligned to visual segment boundaries (768 dimensions). These preprocessing pipelines are described by the paper but are not bundled in this release.

The main comparison requires five independent random-seed runs, each containing five folds. The paper does not list the five seed values; set `SEEDS` to the actual experiment list before running:

```bash
# Replace these example values with the actual five distinct experiment seeds.
export SEEDS="42 43 44 45 46"
bash commands/run_paper_5seeds.sh /path/to/new/output_root
```

The output `paper_5seed_summary.json` reports mean/std across seed runs; folds are not treated as seed repetitions. `STD_DDOF=1` uses sample standard deviation (default); `STD_DDOF=0` uses population standard deviation. The paper does not specify this convention. Each seed's F1/AUC is a sample-weighted average of per-fold metrics, as stated in Section 5.1. Pooled binary metrics are preserved separately for diagnostics.

The paper describes small-gap feature-first routing and large-gap no-feature routing. The implementation now follows that direction, Eq.(9)'s decreasing gap factor, and Eq.(10)'s squared L2 feature loss. These corrections change training behavior relative to the earlier release; rerun experiments in a fresh output directory. Numerical agreement with the paper has not been verified without the original inputs and teacher.

### Key Arguments

- `--student_modality`: Choose `video` or `audio` as the non-contact modality
- `--out_dir`: Output directory
- `--teacher_ckpt_root`: Required teacher checkpoint directory
- `--lambda_rank_kd`, `--lambda_digit_kd`, `--lambda_feat_align`: Loss weights (defaults: 0.0, 0.7, 0.2)
- `--temp_rank`, `--temp_digit`: Distillation temperatures (both default to 2.0)
- `--progressive_mode`, `--stage1_epochs`, `--stage2_epochs`: Progressive schedule
- `--select_on`: `test` (default) follows the author's configured test-based best-epoch selection; `val` enables an optional inner validation split
- `--val_ratio`: Inner validation split ratio when `--select_on val` (default: 0.2)

### Advanced Configuration

For the full experiment configuration, refer to:
- `commands/run_video_best.sh` for video configuration
- `commands/run_audio_best.sh` for audio configuration

---

## 📊 Evaluation

After training, the script automatically evaluates on the test set and produces:

- `summary.json`: Overall metrics across all folds
- `fold_metrics.csv`: Per-fold performance breakdown
- `detail_predictions.csv`: Sample-level predictions
- `detail_question_binary.csv`: Question-level binary predictions
- `model_complexity.{csv,json}`: Model size and computational cost

### Checkpoints and Baseline Script

The KD training script saves `fold_1/best_student.pt` through `fold_5/best_student.pt` and evaluates the selected student automatically. New checkpoints include the feature normalization statistics needed for reuse. A standalone checkpoint evaluation command is not provided in this release.

Despite its filename, `code/eval_video_genlie_5fold.py` trains a separate baseline; it does not load `best_student.pt`. It selects epochs using test metrics, following the author's chosen experiment setting. The KD script also defaults to `--select_on test`; `val` remains available as an optional protocol.

Run `python code/train_video_genlie_q2d_progressive_kd_5fold.py --help` to see all supported arguments.

---


## 📝 Citation

If you use MuDD dataset or GPD code in your research, please cite:

```bibtex
@inproceedings{jiang2026gpd,
  title={GPD: Physiological Signal-Guided Progressive Cross-Modal Distillation for Non-Contact Deception Detection},
  author={Jiang, Peiyuan and Liu, Yao and Gan, Yanglei and Yang, Jiaye and Ahmad, Khwaja Mutahir and Liao, Zhenlong and Liu, Lu and Peng, Xuefeng and Xue, Yuewei and Yao, Daibing and Liu, Qiao},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026},
  organization={ACM},
  doi={10.1145/3767308.3835225}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The MuDD dataset is governed by a separate [Data Use Agreement](docs/MuDD_Data_Use_Agreement.docx) and is available only for approved academic research.

---

## 👥 Contact

For questions about the code or dataset access:

- **Peiyuan Jiang**: darcy981020@gmail.com
- **Yao Liu** (Corresponding Author): liuyao@uestc.edu.cn

**Affiliation**: University of Electronic Science and Technology of China (UESTC)

---


## 📌 Repository Structure

```
GPD/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── docs/
│   ├── MuDD_Data_Use_Agreement.docx # Dataset access agreement
│   ├── GPD_MM26_Paper.pdf          # Local paper copy
│   ├── DATASET_ACCESS_GUIDE.md     # Dataset application guide
│   └── TECHNICAL_DOCUMENTATION.md  # Implementation notes
├── code/
│   ├── train_video_genlie_q2d_progressive_kd_5fold.py  # Main training script
│   ├── eval_video_genlie_5fold.py                      # Baseline training and shared utilities
│   └── summarize_paper_runs.py                        # Five-seed mean/std summary
└── commands/
    ├── run_video_best.sh     # Video experiment script
    ├── run_audio_best.sh     # Audio experiment script
    ├── run_all.sh            # Run both modalities for one seed
    └── run_paper_5seeds.sh    # Five independent 5-fold runs
```

---

**⭐ If you find this work useful, please star the repository!**
