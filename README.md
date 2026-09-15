# GPD: Physiological Signal-Guided Progressive Cross-Modal Distillation for Non-Contact Deception Detection

**Official PyTorch implementation of MM'26 paper**

[![Paper](https://img.shields.io/badge/Paper-MM'26-blue)](https://doi.org/10.1145/3767308.3835225)
[![Dataset](https://img.shields.io/badge/Dataset-MuDD-green)](#dataset-access)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📖 Overview

This repository contains the official implementation of our MM'26 paper: **"GPD: Physiological Signal-Guided Progressive Cross-Modal Distillation for Non-Contact Deception Detection"**.

### Key Contributions

- **MuDD Dataset**: A new **M**ultimodal **D**eception **D**etection dataset collected from 130 participants performing a number-guessing game, with synchronized video, audio, and physiological signals (GSR, PPG, HR).
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

### Setup

1. Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/GPD.git
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
- **~690 minutes per session** with number-guessing game paradigm
- **Multimodal recordings**:
  - 📹 Video (facial expressions, head movements)
  - 🎵 Audio (speech, prosody)
  - 🧠 Physiological signals: GSR, PPG, HR (contact-based sensors)
- **Ground-truth annotations** for Bin.+T-10cls

### How to Access

1. **Read the Data Use Agreement**: [MuDD_Data_Use_Agreement.pdf](docs/MuDD_Data_Use_Agreement.pdf)
2. **Complete the Application Form**: Download and fill out the form in the agreement document
3. **Submit your application**: Email the signed form to **darcy981020@gmail.com**
4. **Student applicants**: Must obtain supervisor/PI signature

### Terms of Use

- ✅ Academic and non-commercial research only
- ❌ No commercial use or operational deployment
- ❌ No redistribution or sharing with unauthorized parties
- ✅ Must cite our paper in any publications using MuDD

For full terms, see [MuDD_Data_Use_Agreement.pdf](docs/MuDD_Data_Use_Agreement.pdf).

---

## 🚀 Quick Start

### Environment Variables

Before running experiments, set the required data paths:

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

### Run Pre-configured Experiments

We provide shell scripts with our best hyperparameters:

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
  --student_modality video \
  --output_root /path/to/output \
  --progressive_weight 0.3 \
  --distill_temperature 4.0 \
  --epochs 50 \
  --batch_size 16
```

### Key Arguments

- `--student_modality`: Choose `video` or `audio` as the non-contact modality
- `--progressive_weight`: Weight for progressive distillation loss (default: 0.3)
- `--distill_temperature`: Temperature for knowledge distillation (default: 4.0)
- `--val_ratio`: Validation split ratio (default: 0.2)

### Advanced Configuration

For custom hyperparameter tuning, modify the training script or refer to:
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

### Load and Evaluate a Trained Model

```bash
python code/eval_video_genlie_5fold.py \
  --checkpoint_path /path/to/best_student.pt \
  --modality video \
  --output_dir /path/to/eval_output
```

---


## 📝 Citation

If you use MuDD dataset or GPD code in your research, please cite:

```bibtex
@inproceedings{jiang2026gpd,
  title={GPD: Physiological Signal-Guided Progressive Cross-Modal Distillation for Non-Contact Deception Detection},
  author={Jiang, Peiyuan and Liu, Yao and Gan, Yanglei and Yang, Jiaye and Ahmad, Khwaja Mutahir and Liao, Zhenlong and Liu, Lu and Peng, Xuefeng and Xue, Yuewei and Yao, Daibing and Liu, Qiao},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  pages={},
  year={2026},
  organization={ACM},
  doi={10.1145/3767308.3835225}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The MuDD dataset is governed by a separate [Data Use Agreement](docs/MuDD_Data_Use_Agreement.pdf) and is available only for approved academic research.

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
│   └── MuDD_Data_Use_Agreement.pdf  # Dataset access agreement
├── code/
│   ├── train_video_genlie_q2d_progressive_kd_5fold.py  # Main training script
│   └── eval_video_genlie_5fold.py                      # Evaluation utilities
└── commands/
    ├── run_video_best.sh     # Video experiment script
    ├── run_audio_best.sh     # Audio experiment script
    └── run_all.sh            # Run all experiments
```

---

**⭐ If you find this work useful, please star the repository!**
