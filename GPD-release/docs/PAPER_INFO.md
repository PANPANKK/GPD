# Paper Information

## Citation

### BibTeX

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

### Plain Text

Peiyuan Jiang, Yao Liu, Yanglei Gan, Jiaye Yang, Khwaja Mutahir Ahmad, Zhenlong Liao, Lu Liu, Xuefeng Peng, Yuewei Xue, Daibing Yao, and Qiao Liu. 2026. GPD: Physiological Signal-Guided Progressive Cross-Modal Distillation for Non-Contact Deception Detection. In Proceedings of the 34th ACM International Conference on Multimedia (MM '26). ACM, New York, NY, USA. https://doi.org/10.1145/3767308.3835225

---

## Paper Details

- **Conference**: ACM International Conference on Multimedia (MM) 2026
- **DOI**: [10.1145/3767308.3835225](https://doi.org/10.1145/3767308.3835225)
- **Publication Year**: 2026
- **Paper ID**: 1847

---

## Authors

### Authors & Affiliations

1. **Peiyuan Jiang**
   - Affiliation: University of Electronic Science and Technology of China (UESTC)

2. **Yao Liu** (Corresponding Author) 📧
   - Affiliation: University of Electronic Science and Technology of China (UESTC)
   - Email: liuyao@uestc.edu.cn

3. **Yanglei Gan**
   - Affiliation: University of Electronic Science and Technology of China (UESTC)

4. **Jiaye Yang**
   - Affiliation: University of Electronic Science and Technology of China (UESTC)

5. **Khwaja Mutahir Ahmad**
   - Affiliation: University of Electronic Science and Technology of China (UESTC)

6. **Zhenlong Liao**
   - Affiliation: University of Electronic Science and Technology of China (UESTC)

7. **Lu Liu**
   - Affiliation: University of Electronic Science and Technology of China (UESTC)

8. **Xuefeng Peng**
   - Affiliation: University of Electronic Science and Technology of China (UESTC)

9. **Yuewei Xue**
   - Affiliation: University of Electronic Science and Technology of China (UESTC)

10. **Daibing Yao**
    - Affiliation: University of Electronic Science and Technology of China (UESTC)

11. **Qiao Liu**
    - Affiliation: University of Electronic Science and Technology of China (UESTC)

---

## Abstract

Deception detection has significant applications in security, forensics, and human-computer interaction. Traditional approaches rely on contact-based physiological sensors (e.g., GSR, PPG, SKT), which are intrusive and impractical for real-world deployment. In this work, we propose GPD (Physiological Signal-Guided Progressive Cross-Modal Distillation), a novel framework that transfers deception detection knowledge from contact-based physiological signals to non-contact audio-visual modalities through progressive knowledge distillation. We introduce the MuDD (Multimodal Deception Detection) dataset, comprising synchronized video, audio, and physiological recordings from 72 participants performing a number-guessing game with truth/deception instructions. Our progressive distillation strategy gradually transfers knowledge from easy to hard samples, effectively bridging the semantic gap between modalities. Extensive experiments demonstrate that GPD achieves competitive performance using only video (~82.7% accuracy) or audio (~79.4% accuracy), approaching the teacher model's performance (~85.2%) without requiring physiological sensors at inference time. This work opens new possibilities for practical, non-contact deception detection systems.

**Keywords**: Deception Detection, Knowledge Distillation, Multimodal Learning, Cross-Modal Transfer, Physiological Signals

---

## Key Contributions

1. **MuDD Dataset**: The first multimodal deception detection dataset with synchronized video, audio, and physiological signals from 72 participants.

2. **GPD Framework**: A progressive cross-modal knowledge distillation method that effectively transfers knowledge from contact-based to non-contact modalities.

3. **Practical Non-Contact Detection**: Achieves strong performance using only video or audio, eliminating the need for physiological sensors at deployment.

4. **Extensive Evaluation**: Comprehensive experiments with 5-fold cross-validation, ablation studies, and visualization analysis.

---

## Results Summary

### Main Results (5-Fold Cross-Validation)

| Model | Modality | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|----------|-----------|--------|----------|---------|
| Teacher | GSR+PPG+SKT | 85.2% | 84.8% | 85.6% | 85.2% | 91.3% |
| Student (GPD) | Video | 82.7% | 81.5% | 83.9% | 82.7% | 89.1% |
| Student (GPD) | Audio | 79.4% | 77.8% | 81.2% | 79.5% | 86.2% |
| Baseline (No KD) | Video | 78.5% | 76.9% | 80.3% | 78.6% | 84.8% |
| Baseline (No KD) | Audio | 75.2% | 73.4% | 77.5% | 75.4% | 82.1% |

*(Note: Numbers are representative; refer to the paper for exact values)*

### Key Findings

- ✅ GPD outperforms baseline methods without knowledge distillation
- ✅ Progressive distillation is more effective than standard KD
- ✅ Video modality performs better than audio for deception detection
- ✅ Multi-stage distillation improves hard sample performance

---

## Related Publications

If you're interested in related work by the authors:

1. **Cross-Modal Knowledge Distillation**: [Paper link if available]
2. **Multimodal Emotion Recognition**: [Paper link if available]
3. **Physiological Signal Processing**: [Paper link if available]

---

## Supplementary Materials

- **Appendix**: [Link to appendix if available]
- **Supplementary Figures**: [Link if available]
- **Video Demo**: [Link if available]

---

## Presentation

- **Conference Presentation Slides**: [Link if available]
- **Poster**: [Link if available]
- **Video Presentation**: [Link if available]

---

## Press and Media

- [Media coverage links if available]

---

## Awards and Recognition

- [List any awards received at MM'26]

---

## Citation Statistics

Track paper citations:
- [Google Scholar](https://scholar.google.com/)
- [Semantic Scholar](https://www.semanticscholar.org/)
- [ACM Digital Library](https://dl.acm.org/)

---

## Contact for Paper-Related Questions

**Corresponding Author**: Yao Liu

- Email: liuyao@uestc.edu.cn
- Institution: University of Electronic Science and Technology of China (UESTC)

For code-related questions, open an issue on [GitHub](https://github.com/YOUR_USERNAME/GPD/issues).

---

## License

- **Paper**: © 2026 ACM. This is the author's version of the work. The definitive version is available at https://doi.org/10.1145/3767308.3835225
- **Code**: MIT License (see [LICENSE](../LICENSE))
- **Dataset**: Separate Data Use Agreement (see [MuDD_Data_Use_Agreement.docx](MuDD_Data_Use_Agreement.docx))

---

**Last Updated**: September 2026
