# Frequently Asked Questions (FAQ)

## General Questions

### Q1: What is GPD?

**A**: GPD (Physiological Signal-Guided Progressive Cross-Modal Distillation) is a knowledge distillation framework for non-contact deception detection. It transfers knowledge from contact-based physiological sensors (GSR, PPG, SKT) to non-contact audio-visual modalities (video, audio).

### Q2: What is the MuDD dataset?

**A**: MuDD (Multimodal Deception Detection Dataset) is a multimodal dataset collected from 72 participants performing a number-guessing game. It includes synchronized video, audio, and physiological signals with ground-truth deception labels.

### Q3: What's the difference between GPD and other deception detection methods?

**A**: Unlike traditional methods that rely on physiological sensors at deployment time, GPD only requires audio or video input after training, making it more practical for real-world applications.

---

## Dataset Access

### Q4: How do I get access to the MuDD dataset?

**A**: Follow these steps:
1. Read the [Data Use Agreement](MuDD_Data_Use_Agreement.docx)
2. Complete the application form
3. Get supervisor approval (for students)
4. Email to liuyao@uestc.edu.cn
5. Wait for approval (1-2 weeks)

See [DATASET_ACCESS_GUIDE.md](DATASET_ACCESS_GUIDE.md) for details.

### Q5: Can I use MuDD for commercial purposes?

**A**: No. MuDD is available only for non-commercial academic research. Contact liuyao@uestc.edu.cn for commercial inquiries.

### Q6: Can I share the dataset with collaborators?

**A**: No. Each person needing access must apply separately. You cannot share credentials or dataset files.

### Q7: What if I'm a student? Do I need permission?

**A**: Yes, student applicants must obtain their supervisor's or PI's signature on the application form.

### Q8: How long does dataset access last?

**A**: Access is typically granted for the duration of your approved research project. You may need to renew for extended projects.

---

## Technical Questions

### Q9: What hardware do I need?

**Minimum**:
- GPU: 8GB VRAM (e.g., GTX 1080, RTX 2070)
- RAM: 16GB
- Storage: 50GB

**Recommended**:
- GPU: 24GB VRAM (e.g., RTX 3090, A100)
- RAM: 32GB
- Storage: 100GB SSD

### Q10: Can I run GPD on CPU only?

**A**: Yes, but it will be very slow. A GPU is strongly recommended for training. You can use CPU for inference on small batches.

### Q11: What Python version do I need?

**A**: Python 3.9 or higher is required. Python 3.9 or 3.10 is recommended.

### Q12: How long does training take?

**A**: On an RTX 3090:
- Video model: ~2-3 hours per fold
- Audio model: ~1-2 hours per fold
- Full 5-fold CV: ~10-15 hours total

### Q13: Can I use a different backbone network?

**A**: Yes, you can modify the code to use different backbone networks (e.g., EfficientNet, Vision Transformer). See [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for details.

---

## Training and Evaluation

### Q14: What does "5-fold cross-validation" mean?

**A**: The dataset is split into 5 parts. The model is trained 5 times, each time using 4 parts for training and 1 for testing. Final results are averaged across all 5 folds.

### Q15: How do I tune hyperparameters?

**A**: Modify the shell scripts in `commands/` or use hyperparameter optimization tools like Optuna or Ray Tune.

### Q16: What if I get "CUDA out of memory" errors?

**A**: Try these solutions:
1. Reduce batch size
2. Use gradient accumulation
3. Use a GPU with more VRAM
4. Reduce input resolution

### Q17: How do I evaluate a trained model?

**A**: Use the evaluation script:
```bash
python code/eval_video_genlie_5fold.py \
  --checkpoint_path /path/to/best_student.pt \
  --modality video
```

### Q18: What metrics are reported?

**A**: Accuracy, Precision, Recall, F1-Score, and AUC-ROC. Results are reported as mean ± standard deviation across 5 folds.

---

## Dataset and Preprocessing

### Q19: What is the input format for video?

**A**: Video files should be in `.mp4` or `.avi` format. Face detection and frame extraction are handled by the preprocessing pipeline.

### Q20: What is the input format for audio?

**A**: Audio files should be in `.wav` or `.flac` format. Recommended: 16 kHz sampling rate, 16-bit PCM.

### Q21: Do I need to preprocess the data myself?

**A**: Basic preprocessing is handled by the data loaders. However, you may need to:
1. Organize files in the expected directory structure
2. Generate split files (or use provided splits)
3. Ensure proper file formats

### Q22: Can I use my own dataset?

**A**: Yes, but you'll need to:
1. Adapt the data loaders in `code/eval_video_genlie_5fold.py`
2. Create your own split files
3. Modify the training script if your labels or format differ

---

## Results and Publication

### Q23: What performance should I expect?

**A**: On the MuDD dataset:
- Video-based student: ~82-83% accuracy
- Audio-based student: ~79-80% accuracy
- Teacher (physiological): ~85% accuracy

Results may vary depending on hyperparameters and random seed.

### Q24: Can I publish results using MuDD?

**A**: Yes, for non-commercial academic research. You must cite the official paper in any publication.

### Q25: Can I publish sample images or audio from MuDD?

**A**: No. Raw or identifiable participant-level data (facial images, audio, physiological signals) cannot be publicly released without prior written permission. You can publish aggregate statistics and evaluation results.

### Q26: What citation format should I use?

**A**: Use the BibTeX entry provided in the README:

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

## Code and Development

### Q27: Where is the main training code?

**A**: The main training script is `code/train_video_genlie_q2d_progressive_kd_5fold.py`.

### Q28: How do I modify the model architecture?

**A**: Edit the model definition in `code/train_video_genlie_q2d_progressive_kd_5fold.py`. Look for the student model definition and modify the network layers.

### Q29: Can I contribute to the project?

**A**: Yes! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on how to contribute code, documentation, or report issues.

### Q30: How do I report a bug?

**A**: Open an issue on GitHub using the bug report template. Include:
1. Description of the bug
2. Steps to reproduce
3. Expected vs. actual behavior
4. Environment details (OS, Python version, GPU, etc.)
5. Error messages or logs

---

## Troubleshooting

### Q31: Teacher checkpoint not found

**Solution**: Verify environment variables:
```bash
echo $VIDEO_TEACHER_CKPT_ROOT
echo $AUDIO_TEACHER_CKPT_ROOT
```

Ensure the paths exist and contain checkpoint files.

### Q32: Data loading errors

**Solution**: Check:
1. Data paths are correct (`$VIDEO_DATA_ROOT`, `$AUDIO_DATA_ROOT`)
2. Files are in expected format (`.mp4`, `.wav`)
3. Directory structure matches expected layout
4. File permissions allow reading

### Q33: Poor performance on custom dataset

**Solution**:
1. Verify label quality and balance
2. Check data preprocessing pipeline
3. Tune hyperparameters for your dataset
4. Ensure sufficient training data
5. Consider domain adaptation techniques

### Q34: Training is very slow

**Solution**:
1. Check GPU utilization (`nvidia-smi`)
2. Increase batch size if possible
3. Use faster data loading (increase `num_workers`)
4. Profile code to find bottlenecks
5. Consider using mixed precision training (fp16)

### Q35: ImportError or ModuleNotFoundError

**Solution**:
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Check Python version compatibility

---

## Contact and Support

### Q36: Where can I get help?

**A**: 
- **Code issues**: Open a GitHub issue
- **Dataset access**: Email liuyao@uestc.edu.cn
- **Technical questions**: Check documentation or open an issue
- **Paper questions**: Contact the authors

### Q37: How do I contact the authors?

**A**: 
- **Corresponding author**: Yao Liu (liuyao@uestc.edu.cn)
- **GitHub issues**: For code-related questions

### Q38: Is there a paper supplement or appendix?

**A**: Check the `docs/` directory for additional materials. The full paper is available at the ACM Digital Library (DOI: 10.1145/3767308.3835225).

---

## Miscellaneous

### Q39: What does "progressive distillation" mean?

**A**: Progressive distillation gradually increases the difficulty of knowledge transfer, starting with easy samples (high-confidence predictions) and progressing to hard samples (low-confidence or ambiguous cases).

### Q40: Why use physiological signals as teacher if they're not available at test time?

**A**: Physiological signals provide ground-truth insights into internal states (e.g., arousal, stress) that correlate with deception. By using them during training (as a teacher), we can guide the student model to learn similar patterns from audio-visual cues.

### Q41: Can I use GPD for other tasks (e.g., emotion recognition)?

**A**: The framework is designed for deception detection but can be adapted for other cross-modal knowledge distillation tasks. You'll need to modify the data loaders, labels, and possibly the model architecture.

### Q42: Is there a pre-trained model available?

**A**: Pre-trained models trained on MuDD cannot be publicly released due to privacy concerns (they may encode participant-level information). You need to train your own model after obtaining dataset access.

---

**Didn't find your question here?** Open an issue on GitHub or contact the authors.
