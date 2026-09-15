# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-09-16

### Added
- Initial public release of GPD codebase
- Main training script for progressive cross-modal distillation
- Evaluation utilities and data loaders
- Pre-configured shell scripts for video and audio experiments
- Comprehensive documentation:
  - README with project overview
  - Dataset access guide and agreement
  - Technical documentation
  - Quick start guide
  - FAQ
  - Contributing guidelines
- GitHub templates for issues and pull requests
- MIT License for code (separate agreement for dataset)
- Requirements file with all dependencies
- .gitignore for Python/PyTorch projects

### Features
- 5-fold cross-validation support
- Progressive distillation with configurable stages
- Support for video-based student models
- Support for audio-based student models
- Automated model selection and evaluation
- Comprehensive metrics reporting (accuracy, precision, recall, F1, AUC)
- Model complexity analysis (parameters, FLOPs)

### Documentation
- Detailed installation instructions
- Environment setup guide
- Training and evaluation tutorials
- Hyperparameter tuning guidelines
- Troubleshooting section
- Code of conduct for contributors

---

## [Unreleased]

### Planned
- Visualization tools for attention maps
- Additional baseline comparisons
- Multi-GPU training support
- Mixed precision (FP16) training
- Hyperparameter optimization scripts
- Pre-trained feature extractors
- Additional modality support (text, physiological only)
- Real-time inference demo
- Web interface for testing
- Docker container for reproducibility

### Under Consideration
- Integration with Hugging Face Hub
- TensorBoard logging
- Weights & Biases integration
- Automated hyperparameter search
- Model quantization for deployment
- ONNX export for cross-platform inference

---

## Version History

### Version 1.0.0 (September 2026)
- First official release accompanying MM'26 paper publication
- Includes core training and evaluation pipeline
- MuDD dataset access application process established

---

## Future Roadmap

### Short-term (3-6 months)
- [ ] Add visualization scripts for model interpretability
- [ ] Provide example notebooks for common use cases
- [ ] Expand documentation with more examples
- [ ] Add unit tests for core functionality
- [ ] Implement multi-GPU distributed training

### Medium-term (6-12 months)
- [ ] Support for additional datasets
- [ ] Real-time inference optimization
- [ ] Mobile deployment examples
- [ ] Cloud deployment guides (AWS, GCP, Azure)

### Long-term (12+ months)
- [ ] Extended modality support (multimodal fusion)
- [ ] Online learning capabilities
- [ ] Federated learning support for privacy-sensitive applications
- [ ] Continual learning for domain adaptation

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Report Issues

If you encounter bugs or have suggestions:
1. Check existing issues to avoid duplicates
2. Use the appropriate issue template
3. Provide detailed information (environment, steps to reproduce, etc.)

### How to Submit Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request with a detailed description

---

## Citation

If you use this code or dataset, please cite:

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

## Acknowledgments

- MM'26 reviewers for valuable feedback
- All participants in the MuDD dataset collection
- UESTC for providing research facilities
- Open-source community for tools and libraries used in this project

---

## License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Dataset**: Separate Data Use Agreement (see [docs/MuDD_Data_Use_Agreement.docx](docs/MuDD_Data_Use_Agreement.docx))

---

**Maintained by**: University of Electronic Science and Technology of China (UESTC)

**Contact**: liuyao@uestc.edu.cn
