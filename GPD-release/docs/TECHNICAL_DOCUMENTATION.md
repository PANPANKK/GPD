# Technical Documentation

## Architecture Overview

GPD (Physiological Signal-Guided Progressive Cross-Modal Distillation) is a knowledge distillation framework that transfers deception detection capability from contact-based physiological sensors to non-contact audio-visual modalities.

### Key Components

1. **Teacher Model**: Multi-modal fusion network trained on physiological signals (GSR, PPG, SKT)
2. **Student Model**: Unimodal network (video-based or audio-based)
3. **Progressive Distillation**: Staged knowledge transfer with increasing difficulty
4. **Cross-Modal Alignment**: Bridging the semantic gap between physiological and audio-visual features

---

## Model Architecture

### Teacher Network

The teacher network processes contact-based physiological signals:

```
Input: [GSR, PPG, SKT] → Feature Extraction → Temporal Modeling → Fusion → Classification
```

**Components**:
- Signal preprocessing (normalization, filtering)
- 1D CNN for feature extraction
- LSTM/Transformer for temporal modeling
- Multi-modal fusion layer
- Binary classifier (truth/deception)

### Student Network (Video)

```
Input: Video Frames → CNN Backbone → Temporal Aggregation → FC Layers → Classification
```

**Components**:
- ResNet-50 or EfficientNet backbone (pre-trained on ImageNet)
- Temporal pooling or LSTM
- Fully connected layers
- Binary classifier

### Student Network (Audio)

```
Input: Audio Waveform → Spectrogram → CNN/Transformer → Temporal Pooling → Classification
```

**Components**:
- Mel-spectrogram extraction
- CNN or Transformer encoder
- Attention-based pooling
- Binary classifier

---

## Progressive Distillation Strategy

### Stage 1: Easy Samples
- High-confidence teacher predictions
- Clear deceptive/truthful patterns
- Lower distillation weight

### Stage 2: Medium Difficulty
- Moderate-confidence predictions
- Intermediate distillation weight

### Stage 3: Hard Samples
- Low-confidence or ambiguous cases
- Higher distillation weight
- Fine-grained knowledge transfer

### Loss Function

```python
L_total = L_task + λ_kd * L_kd + λ_prog * L_prog

where:
- L_task: Cross-entropy loss (ground truth)
- L_kd: KL divergence between teacher and student logits
- L_prog: Progressive distillation loss
- λ_kd, λ_prog: Hyperparameters
```

---

## Data Processing Pipeline

### Video Processing

1. **Face Detection**: Extract facial region using MTCNN or RetinaFace
2. **Frame Sampling**: Sample frames at 1-3 fps
3. **Preprocessing**:
   - Resize to 224×224
   - Normalize with ImageNet mean/std
   - Data augmentation (horizontal flip, color jitter)
4. **Temporal Windowing**: Group frames into sequences (e.g., 10 frames per sample)

### Audio Processing

1. **Resampling**: Convert to 16 kHz
2. **Feature Extraction**:
   - Mel-spectrogram (128 mel bins, window=25ms, hop=10ms)
   - Optional: MFCCs, pitch, energy
3. **Normalization**: Mean-variance normalization
4. **Segmentation**: Fixed-length segments (e.g., 3 seconds)

### Physiological Signal Processing

1. **Filtering**: Bandpass filtering to remove noise
2. **Artifact Removal**: Motion artifacts, sensor disconnections
3. **Normalization**: Z-score normalization per session
4. **Windowing**: Aligned with video/audio segments

---

## Training Configuration

### Hyperparameters (Best Configuration)

| Parameter | Video | Audio |
|-----------|-------|-------|
| Batch Size | 16 | 16 |
| Learning Rate | 1e-4 | 1e-4 |
| Optimizer | Adam | Adam |
| Weight Decay | 1e-5 | 1e-5 |
| Epochs | 50 | 50 |
| Distillation Temperature (τ) | 4.0 | 4.0 |
| Progressive Weight (λ_prog) | 0.3 | 0.3 |
| KD Weight (λ_kd) | 1.0 | 1.0 |

### Data Splitting

- **5-fold cross-validation**
- **Validation ratio**: 20% of training data
- **Subject-independent**: Participants in test set are unseen during training

### Model Selection

- Early stopping based on validation accuracy
- Best checkpoint saved per fold
- Final test evaluation uses best validation model

---

## Evaluation Metrics

### Binary Classification Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve

### Reported Format

- Mean ± Standard Deviation across 5 folds
- Per-fold breakdown available in output files

---

## File Formats

### Input Data

**Video**: `.mp4`, `.avi` (H.264 codec recommended)
**Audio**: `.wav`, `.flac` (16-bit PCM recommended)
**Physiological**: `.csv` with columns `[timestamp, value]`

### Output Files

1. **summary.json**: Aggregated metrics across folds
   ```json
   {
     "accuracy": 0.827,
     "precision": 0.815,
     "recall": 0.839,
     "f1": 0.827,
     "auc": 0.891
   }
   ```

2. **fold_metrics.csv**: Per-fold results
   ```csv
   fold,accuracy,precision,recall,f1,auc
   0,0.825,0.810,0.840,0.825,0.888
   1,0.830,0.820,0.835,0.828,0.895
   ...
   ```

3. **detail_predictions.csv**: Sample-level predictions
   ```csv
   sample_id,true_label,predicted_label,confidence_truth,confidence_deception
   001_q1,0,0,0.75,0.25
   001_q2,1,1,0.20,0.80
   ...
   ```

---

## Environment Variables

Required paths for training:

```bash
# Video modality
export VIDEO_DATA_ROOT=/path/to/video/data
export VIDEO_SPLIT_ROOT=/path/to/video/splits
export VIDEO_TEACHER_REPO_ROOT=/path/to/video/teacher/repo
export VIDEO_TEACHER_CKPT_ROOT=/path/to/video/teacher/checkpoints

# Audio modality
export AUDIO_DATA_ROOT=/path/to/audio/data
export AUDIO_SPLIT_ROOT=/path/to/audio/splits
export AUDIO_TEACHER_REPO_ROOT=/path/to/audio/teacher/repo
export AUDIO_TEACHER_CKPT_ROOT=/path/to/audio/teacher/checkpoints
```

---

## Computational Requirements

### Minimum Requirements

- **GPU**: NVIDIA GPU with 8GB VRAM (e.g., GTX 1080, RTX 2070)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space (for dataset + checkpoints)
- **CUDA**: Version 11.0 or higher

### Recommended Setup

- **GPU**: NVIDIA RTX 3090 or A100 (24GB VRAM)
- **RAM**: 32GB or more
- **Storage**: SSD with 100GB+ free space

### Training Time Estimates

- **Video model**: ~2-3 hours per fold on RTX 3090
- **Audio model**: ~1-2 hours per fold on RTX 3090
- **Full 5-fold CV**: ~10-15 hours total

---

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
**Solution**: Reduce batch size, use gradient accumulation, or use a GPU with more VRAM

**Issue**: Teacher model checkpoint not found
**Solution**: Verify `VIDEO_TEACHER_CKPT_ROOT` and `AUDIO_TEACHER_CKPT_ROOT` paths

**Issue**: Data loading error
**Solution**: Check data paths, file permissions, and file formats

**Issue**: Poor performance on custom dataset
**Solution**: Ensure proper preprocessing, verify label quality, check data distribution

---

## Extending the Framework

### Adding a New Modality

1. Implement data loader in `code/eval_video_genlie_5fold.py`
2. Define feature extractor network
3. Modify training script to accept new `--student_modality` option
4. Update configuration files

### Custom Loss Functions

Override the loss computation in `train_video_genlie_q2d_progressive_kd_5fold.py`:

```python
def compute_loss(student_logits, teacher_logits, labels, stage):
    # Your custom loss implementation
    pass
```

### Hyperparameter Tuning

Modify shell scripts in `commands/` or use tools like:
- Optuna for Bayesian optimization
- Ray Tune for distributed hyperparameter search

---

## Citation

If you use this codebase or dataset, please cite:

```bibtex
@inproceedings{jiang2026gpd,
  title={GPD: Physiological Signal-Guided Progressive Cross-Modal Distillation for Non-Contact Deception Detection},
  author={Jiang, Peiyuan and Liu, Yao and Gan, Yanglei and Yang, Jiaye and Ahmad, Khwaja Mutahir and Liao, Zhenlong and Liu, Lu and Peng, Xuefeng and Xue, Yuewei and Yao, Daibing and Liu, Qiao},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026},
  doi={10.1145/3767308.3835225}
}
```

---

## Contact

For technical questions or support:
- GitHub Issues: [https://github.com/YOUR_USERNAME/GPD/issues](https://github.com/YOUR_USERNAME/GPD/issues)
- Email: liuyao@uestc.edu.cn
