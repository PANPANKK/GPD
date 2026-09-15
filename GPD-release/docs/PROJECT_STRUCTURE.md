# Project Structure

This document describes the organization of the GPD repository.

---

## Directory Tree

```
GPD/
├── .github/                          # GitHub configuration
│   ├── ISSUE_TEMPLATE/               # Issue templates
│   │   ├── bug_report.md             # Bug report template
│   │   ├── dataset_access.md         # Dataset access inquiries
│   │   └── feature_request.md        # Feature request template
│   └── pull_request_template.md      # PR template
│
├── code/                             # Source code
│   ├── train_video_genlie_q2d_progressive_kd_5fold.py  # Main training script
│   └── eval_video_genlie_5fold.py    # Data loaders and utilities
│
├── commands/                         # Experiment scripts
│   ├── run_video_best.sh             # Video student training
│   ├── run_audio_best.sh             # Audio student training
│   └── run_all.sh                    # Run all experiments
│
├── docs/                             # Documentation
│   ├── DATASET_ACCESS_GUIDE.md       # How to access MuDD dataset
│   ├── FAQ.md                        # Frequently asked questions
│   ├── GPD_MM26_Paper.pdf            # Published paper
│   ├── MuDD_Data_Use_Agreement.docx  # Dataset access agreement
│   ├── QUICK_START.md                # Quick start guide
│   └── TECHNICAL_DOCUMENTATION.md    # Detailed technical docs
│
├── .env_gpd.example                  # Example environment config
├── .gitignore                        # Git ignore rules
├── CHANGELOG.md                      # Version history
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # MIT License (code only)
├── README.md                         # Main project README
├── requirements.txt                  # Python dependencies
└── setup_env.sh                      # Environment setup script
```

---

## Key Files

### Root Level

- **README.md**: Main entry point, project overview, installation, and usage instructions
- **LICENSE**: MIT License for code (dataset has separate agreement)
- **CHANGELOG.md**: Version history and future roadmap
- **CONTRIBUTING.md**: Guidelines for contributing to the project
- **requirements.txt**: Python package dependencies
- **setup_env.sh**: Interactive script to set up environment variables
- **.env_gpd.example**: Template for environment configuration
- **.gitignore**: Files and directories to exclude from git

### Code Directory (`code/`)

#### `train_video_genlie_q2d_progressive_kd_5fold.py`
**Purpose**: Main training script for progressive knowledge distillation

**Key Components**:
- Student model definition (video or audio)
- Progressive distillation loss functions
- Training loop with validation
- 5-fold cross-validation
- Model checkpoint saving
- Metrics computation and logging

**Usage**:
```bash
python code/train_video_genlie_q2d_progressive_kd_5fold.py \
  --student_modality video \
  --output_root ./outputs
```

#### `eval_video_genlie_5fold.py`
**Purpose**: Data loading, preprocessing, and evaluation utilities

**Key Components**:
- Dataset classes for video, audio, and physiological data
- Data preprocessing pipelines
- Evaluation metrics computation
- Utility functions for data loading

**Usage**: Imported by training script; can also be used standalone for evaluation

---

### Commands Directory (`commands/`)

Shell scripts with pre-configured hyperparameters for best performance.

#### `run_video_best.sh`
Train video-based student model with optimal hyperparameters.

**Example**:
```bash
bash commands/run_video_best.sh ./outputs/video_exp1
```

#### `run_audio_best.sh`
Train audio-based student model with optimal hyperparameters.

**Example**:
```bash
bash commands/run_audio_best.sh ./outputs/audio_exp1
```

#### `run_all.sh`
Run both video and audio experiments sequentially.

**Example**:
```bash
bash commands/run_all.sh ./outputs/all_experiments
```

---

### Documentation Directory (`docs/`)

Comprehensive project documentation.

#### Core Documentation

1. **QUICK_START.md**: Get started in 5 minutes
2. **TECHNICAL_DOCUMENTATION.md**: Detailed technical information
   - Architecture overview
   - Model details
   - Training configuration
   - Evaluation metrics
   - File formats
3. **FAQ.md**: Frequently asked questions

#### Dataset Documentation

4. **DATASET_ACCESS_GUIDE.md**: How to apply for MuDD dataset access
5. **MuDD_Data_Use_Agreement.docx**: Official dataset agreement
6. **GPD_MM26_Paper.pdf**: Published paper (MM'26)

---

### GitHub Configuration (`.github/`)

#### Issue Templates

- **bug_report.md**: For reporting bugs
- **feature_request.md**: For suggesting enhancements
- **dataset_access.md**: For dataset access inquiries (redirects to email)

#### Pull Request Template

- **pull_request_template.md**: Standardized PR format

---

## Code Organization Principles

### Modularity

- **Separation of concerns**: Data loading, model definition, training, and evaluation are separated
- **Reusability**: Utility functions are in `eval_video_genlie_5fold.py` for reuse

### Configuration

- **Environment variables**: All paths configured via env vars
- **Command-line arguments**: Hyperparameters passed as CLI args
- **Shell scripts**: Pre-configured experiments in `commands/`

### Reproducibility

- **5-fold CV**: Ensures robust evaluation
- **Random seeds**: Should be set for reproducibility
- **Detailed logging**: Metrics and checkpoints saved

---

## File Naming Conventions

### Code Files

- Use descriptive names: `train_video_genlie_q2d_progressive_kd_5fold.py`
- Avoid abbreviations unless well-known (e.g., `kd` for knowledge distillation)

### Shell Scripts

- Action-based names: `run_video_best.sh`, `setup_env.sh`
- Use `.sh` extension

### Documentation

- All caps for special files: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE`
- Descriptive names for guides: `DATASET_ACCESS_GUIDE.md`, `QUICK_START.md`

---

## Output Directory Structure

After running experiments, the output directory will contain:

```
outputs/
└── experiment_name/
    ├── summary.json                  # Overall metrics
    ├── fold_metrics.csv              # Per-fold results
    ├── detail_predictions.csv        # Sample-level predictions
    ├── detail_question_binary.csv    # Question-level predictions
    ├── model_complexity.csv          # Model size analysis
    ├── model_complexity.json         # Model size (JSON format)
    ├── fold_0/                       # Fold 0 results
    │   ├── best_student.pt           # Best model checkpoint
    │   ├── train.log                 # Training log
    │   └── ...
    ├── fold_1/
    │   └── ...
    ├── fold_2/
    │   └── ...
    ├── fold_3/
    │   └── ...
    └── fold_4/
        └── ...
```

---

## Data Directory Structure (Expected)

### Video Data

```
VIDEO_DATA_ROOT/
├── participant_001/
│   ├── session_01.mp4
│   ├── session_02.mp4
│   └── ...
├── participant_002/
│   └── ...
└── ...
```

### Audio Data

```
AUDIO_DATA_ROOT/
├── participant_001/
│   ├── session_01.wav
│   ├── session_02.wav
│   └── ...
├── participant_002/
│   └── ...
└── ...
```

### Physiological Data

```
PHYSIO_DATA_ROOT/
├── participant_001/
│   ├── gsr.csv
│   ├── ppg.csv
│   └── skt.csv
├── participant_002/
│   └── ...
└── ...
```

### Split Files

```
SPLIT_ROOT/
├── fold_1.json
├── fold_2.json
├── fold_3.json
├── fold_4.json
└── fold_5.json
```

Each split file contains:
```json
{
  "train": ["participant_001", "participant_002", ...],
  "test": ["participant_015", "participant_016", ...]
}
```

---

## Development Workflow

### 1. Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/GPD.git
cd GPD
pip install -r requirements.txt
bash setup_env.sh
```

### 2. Configure Environment

```bash
cp .env_gpd.example .env_gpd
# Edit .env_gpd with your paths
source .env_gpd
```

### 3. Run Experiments

```bash
bash commands/run_video_best.sh ./outputs/my_experiment
```

### 4. Analyze Results

```bash
cat outputs/my_experiment/summary.json
```

---

## Adding New Features

### Adding a New Modality

1. Extend data loader in `code/eval_video_genlie_5fold.py`
2. Add preprocessing pipeline
3. Update training script to accept new modality
4. Add shell script in `commands/`
5. Update documentation

### Adding New Loss Functions

1. Define loss in `code/train_video_genlie_q2d_progressive_kd_5fold.py`
2. Add command-line argument for loss weight
3. Update shell scripts with new parameter
4. Document in technical documentation

### Adding Visualization Tools

1. Create new script in `code/` (e.g., `visualize_attention.py`)
2. Add dependencies to `requirements.txt`
3. Add usage example in documentation
4. Update README with visualization section

---

## Testing (Future Work)

Planned test structure:

```
tests/
├── test_data_loading.py      # Test data loaders
├── test_models.py             # Test model architectures
├── test_losses.py             # Test loss functions
├── test_evaluation.py         # Test metrics computation
└── fixtures/                  # Test data fixtures
```

---

## Dependencies Management

### Core Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **scikit-learn**: Metrics and utilities
- **OpenCV** (cv2): Video processing
- **librosa**: Audio processing

### Optional Dependencies

- **tensorboard**: Logging (future)
- **wandb**: Experiment tracking (future)
- **pytest**: Testing (future)

---

## Version Control Best Practices

### Commits

- Use clear, descriptive commit messages
- Follow format: `<type>: <subject>`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Branches

- `main`: Stable release branch
- `develop`: Development branch
- `feature/<name>`: New features
- `bugfix/<name>`: Bug fixes

### Tags

- Use semantic versioning: `v1.0.0`, `v1.1.0`, etc.
- Tag releases for reproducibility

---

## Contact

For questions about project structure:
- Open an issue on GitHub
- Email: liuyao@uestc.edu.cn
