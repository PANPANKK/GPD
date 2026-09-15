# Release Notes - GPD v1.0.0

**Release Date**: September 16, 2026

---

## 🎉 Initial Public Release

We are excited to announce the first public release of **GPD (Physiological Signal-Guided Progressive Cross-Modal Distillation)** accompanying our MM'26 paper!

---

## 📦 What's Included

### Core Components

✅ **Training Pipeline**
- Complete training script with progressive knowledge distillation
- 5-fold cross-validation support
- Automatic model selection and evaluation

✅ **Pre-configured Experiments**
- Shell scripts with optimal hyperparameters for video and audio modalities
- Environment setup utilities
- Example configuration files

✅ **MuDD Dataset Access**
- Official data use agreement
- Detailed application process guide
- Dataset documentation

✅ **Comprehensive Documentation**
- Quick start guide (5-minute setup)
- Technical documentation with architecture details
- FAQ covering common questions
- Project structure guide
- Contributing guidelines

✅ **GitHub Integration**
- Issue templates for bugs, features, and dataset inquiries
- Pull request template
- Automated workflows (future)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/GPD.git
cd GPD

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
bash setup_env.sh

# 4. Apply for dataset access
# See docs/DATASET_ACCESS_GUIDE.md

# 5. Run experiments
source .env_gpd
bash commands/run_video_best.sh ./outputs/video_test
```

---

## 📊 Performance Highlights

- **Video-based student**: ~82.7% accuracy (vs. 85.2% teacher)
- **Audio-based student**: ~79.4% accuracy
- **No physiological sensors needed at inference time**
- **Practical deployment ready**

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview and getting started |
| [QUICK_START.md](docs/QUICK_START.md) | 5-minute quick start guide |
| [TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | Detailed technical information |
| [DATASET_ACCESS_GUIDE.md](docs/DATASET_ACCESS_GUIDE.md) | How to access MuDD dataset |
| [FAQ.md](docs/FAQ.md) | Frequently asked questions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Version history and roadmap |

---

## 🗄️ MuDD Dataset

The **Multimodal Deception Detection (MuDD) Dataset** is now available for academic research!

### Dataset Highlights

- 72 participants (36 male, 36 female)
- ~45 minutes per session
- Synchronized video, audio, and physiological signals
- Ground-truth deception labels
- 5-fold cross-validation splits provided

### How to Access

1. Read the [Data Use Agreement](docs/MuDD_Data_Use_Agreement.docx)
2. Complete the application form
3. Email to: liuyao@uestc.edu.cn
4. Wait for approval (typically 1-2 weeks)

**Full guide**: [docs/DATASET_ACCESS_GUIDE.md](docs/DATASET_ACCESS_GUIDE.md)

---

## 🔧 System Requirements

### Minimum

- Python 3.9+
- PyTorch 2.0+
- GPU with 8GB VRAM
- 16GB RAM
- 50GB storage

### Recommended

- Python 3.9 or 3.10
- PyTorch 2.0+
- GPU with 24GB VRAM (RTX 3090, A100)
- 32GB RAM
- 100GB SSD storage

---

## 🐛 Known Issues

None at this time. Please report issues on [GitHub](https://github.com/YOUR_USERNAME/GPD/issues).

---

## 🛣️ Roadmap

### Short-term (3-6 months)

- [ ] Visualization tools for attention maps
- [ ] Example Jupyter notebooks
- [ ] Multi-GPU training support
- [ ] Unit tests

### Medium-term (6-12 months)

- [ ] Real-time inference demo
- [ ] Additional baseline comparisons
- [ ] Web interface
- [ ] Docker container

### Long-term (12+ months)

- [ ] Extended modality support
- [ ] Federated learning version
- [ ] Mobile deployment examples
- [ ] Cloud deployment guides

---

## 🙏 Acknowledgments

We thank:

- MM'26 reviewers for valuable feedback
- All 72 participants who contributed to the MuDD dataset
- University of Electronic Science and Technology of China (UESTC) for research facilities
- Open-source community for tools and libraries

---

## 📝 Citation

If you use GPD or MuDD in your research, please cite:

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

## 📧 Contact

- **Dataset Access**: liuyao@uestc.edu.cn
- **Code Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/GPD/issues)
- **General Inquiries**: liuyao@uestc.edu.cn

---

## 📄 License

- **Code**: MIT License ([LICENSE](LICENSE))
- **Dataset**: Separate Data Use Agreement ([docs/MuDD_Data_Use_Agreement.docx](docs/MuDD_Data_Use_Agreement.docx))
- **Paper**: © 2026 ACM

---

## 🌟 Get Involved

We welcome contributions from the community!

- ⭐ **Star** the repository if you find it useful
- 🐛 **Report bugs** via GitHub Issues
- 💡 **Suggest features** via GitHub Issues
- 🔧 **Contribute code** via Pull Requests
- 📖 **Improve documentation**
- 💬 **Share your experience** using GPD

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 🔗 Links

- **Paper**: https://doi.org/10.1145/3767308.3835225
- **GitHub**: https://github.com/YOUR_USERNAME/GPD
- **ACM MM'26**: https://2026.acmmm.org/
- **UESTC**: https://www.uestc.edu.cn/

---

## 📈 What's Next?

After this release, we plan to:

1. **Gather community feedback** and improve based on your experience
2. **Add visualization tools** to help understand model decisions
3. **Expand documentation** with more examples and tutorials
4. **Support more use cases** (different datasets, modalities)
5. **Improve performance** through community contributions

---

**Thank you for your interest in GPD!** 🎉

We look forward to seeing what you build with it. If you have any questions or run into issues, don't hesitate to reach out via GitHub Issues or email.

---

**Release Team**

University of Electronic Science and Technology of China (UESTC)

Lead: Yao Liu (liuyao@uestc.edu.cn)

---

**Version**: 1.0.0  
**Date**: September 16, 2026  
**Status**: Stable
