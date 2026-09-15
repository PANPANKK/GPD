# Quick Start Guide

This guide will help you get started with GPD in 5 minutes.

---

## Prerequisites

- Linux or macOS (Windows with WSL2 also works)
- Python 3.9 or higher
- CUDA-capable GPU (8GB+ VRAM recommended)
- Git

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/GPD.git
cd GPD
```

---

## Step 2: Set Up Environment

### Option A: Using pip (recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using conda

```bash
# Create conda environment
conda create -n gpd python=3.9
conda activate gpd

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

---

## Step 3: Apply for MuDD Dataset Access

**You need the MuDD dataset to run experiments.**

1. Read the [Dataset Access Guide](docs/DATASET_ACCESS_GUIDE.md)
2. Complete the [Data Use Agreement](docs/MuDD_Data_Use_Agreement.docx)
3. Email your signed application to: **liuyao@uestc.edu.cn**
4. Wait for approval (typically 1-2 weeks)

---

## Step 4: Prepare Dataset

After receiving dataset access, organize your data:

```
/path/to/data/
├── video/
│   ├── participant_001/
│   └── ...
├── audio/
│   ├── participant_001/
│   └── ...
├── physiological/
│   ├── participant_001/
│   └── ...
└── splits/
    ├── fold_1.json
    └── ...
```

---

## Step 5: Set Environment Variables

Create a script `setup_paths.sh`:

```bash
#!/bin/bash

# Video paths
export VIDEO_DATA_ROOT=/path/to/data/video
export VIDEO_SPLIT_ROOT=/path/to/data/splits
export VIDEO_TEACHER_REPO_ROOT=/path/to/teacher/video/repo
export VIDEO_TEACHER_CKPT_ROOT=/path/to/teacher/video/checkpoints

# Audio paths
export AUDIO_DATA_ROOT=/path/to/data/audio
export AUDIO_SPLIT_ROOT=/path/to/data/splits
export AUDIO_TEACHER_REPO_ROOT=/path/to/teacher/audio/repo
export AUDIO_TEACHER_CKPT_ROOT=/path/to/teacher/audio/checkpoints
```

Load the variables:

```bash
source setup_paths.sh
```

---

## Step 6: Run Your First Experiment

### Quick Test (Video-based Student)

```bash
bash commands/run_video_best.sh ./outputs/video_test
```

### Quick Test (Audio-based Student)

```bash
bash commands/run_audio_best.sh ./outputs/audio_test
```

---

## Step 7: Check Results

After training completes, check the output directory:

```bash
ls outputs/video_test/
```

You should see:
- `summary.json` - Overall metrics
- `fold_metrics.csv` - Per-fold results
- `fold_0/`, `fold_1/`, ... - Checkpoints and logs

---

## Common First-Time Issues

### Issue 1: "CUDA out of memory"

**Solution**: Reduce batch size in the shell script:

```bash
# Edit commands/run_video_best.sh
# Change: --batch_size 16
# To:     --batch_size 8
```

### Issue 2: "Teacher checkpoint not found"

**Solution**: Verify your environment variables:

```bash
echo $VIDEO_TEACHER_CKPT_ROOT
# Should print the correct path
```

### Issue 3: "FileNotFoundError: Data file not found"

**Solution**: Check data paths and file structure:

```bash
ls $VIDEO_DATA_ROOT
# Should list participant directories
```

---

## Next Steps

1. **Read the full documentation**: [TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)
2. **Explore the code**: Start with `code/train_video_genlie_q2d_progressive_kd_5fold.py`
3. **Tune hyperparameters**: Modify shell scripts in `commands/`
4. **Run experiments**: Try different configurations
5. **Visualize results**: Use the evaluation scripts

---

## Getting Help

- **Documentation**: Check [docs/](docs/) directory
- **GitHub Issues**: Report bugs or ask questions
- **Email**: Contact liuyao@uestc.edu.cn for dataset issues

---

## Example Output

After a successful run, `summary.json` might look like:

```json
{
  "modality": "video",
  "accuracy": 0.827,
  "precision": 0.815,
  "recall": 0.839,
  "f1_score": 0.827,
  "auc_roc": 0.891,
  "num_folds": 5
}
```

Congratulations! You've successfully run GPD. 🎉

---

## What's Next?

- Try audio-based student: `bash commands/run_audio_best.sh ./outputs/audio_test`
- Experiment with hyperparameters
- Visualize attention maps (if visualization scripts are available)
- Compare with baseline methods
- Adapt to your own dataset

---

**Need more details?** Read the [full documentation](docs/TECHNICAL_DOCUMENTATION.md) or open an issue on GitHub.
