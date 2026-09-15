# MuDD Dataset Access Guide

## Dataset Overview

The **MuDD (Multimodal Deception Detection) Dataset** is a comprehensive multimodal dataset for deception detection research, collected at the University of Electronic Science and Technology of China (UESTC).

### Dataset Statistics

- **Participants**: 72 (36 male, 36 female)
- **Session Duration**: ~45 minutes per participant
- **Paradigm**: Number-guessing game with truth/deception instructions
- **Data Modalities**:
  - 📹 **Video**: Frontal facial recordings (1920×1080, 30 fps)
  - 🎵 **Audio**: Speech recordings (48 kHz, 16-bit)
  - 🧠 **Physiological Signals**:
    - GSR (Galvanic Skin Response)
    - PPG (Photoplethysmography)
    - SKT (Skin Temperature)
- **Annotations**: Ground-truth labels for deceptive vs. truthful responses

### Data Structure

After approval, you will receive access to:

```
MuDD/
├── video/
│   ├── participant_001/
│   │   ├── session_01.mp4
│   │   └── ...
│   └── ...
├── audio/
│   ├── participant_001/
│   │   ├── session_01.wav
│   │   └── ...
│   └── ...
├── physiological/
│   ├── participant_001/
│   │   ├── gsr.csv
│   │   ├── ppg.csv
│   │   └── skt.csv
│   └── ...
├── annotations/
│   ├── labels.csv
│   └── metadata.json
└── splits/
    ├── fold_1.json
    ├── fold_2.json
    ├── fold_3.json
    ├── fold_4.json
    └── fold_5.json
```

---

## How to Apply for Dataset Access

### Step 1: Read the Data Use Agreement

Carefully read the [MuDD Data Use Agreement](MuDD_Data_Use_Agreement.docx) document. Key terms include:

✅ **Permitted Uses**:
- Academic and educational research
- Non-commercial scientific studies
- Method development and evaluation

❌ **Prohibited Uses**:
- Commercial products or services
- Operational decision-making about individuals
- Surveillance, screening, or law enforcement
- Redistribution to unauthorized parties

### Step 2: Complete the Application Form

The application form is included in the Data Use Agreement document. You need to provide:

**Section A: Applicant Information**
- Full name
- Email address
- Affiliation (university/research institution)
- Position (student, postdoc, faculty, etc.)
- ORCID or researcher ID (if available)
- Brief research description (1-2 paragraphs)

**Section B: Supervisor/PI Approval (for students)**
- Supervisor's name and email
- Supervisor's signature and date

### Step 3: Submit Your Application

**Email to**: liuyao@uestc.edu.cn

**Subject**: MuDD Dataset Access Request - [Your Name]

**Attach**:
1. Completed and signed Data Use Agreement form
2. (For students) Supervisor/PI signature page

### Step 4: Wait for Approval

- Applications are typically reviewed within **1-2 weeks**
- You will receive an email with:
  - Download instructions
  - Access credentials (if applicable)
  - Additional technical documentation

---

## Responsibilities After Access

### ✅ DO:
- Use the dataset only for your approved research purpose
- Store data securely with appropriate access controls
- Cite the official paper in any publications
- Report unauthorized access or data breaches immediately

### ❌ DON'T:
- Share access credentials or dataset files with others
- Upload raw data to public repositories
- Attempt to identify or contact participants
- Use the dataset for commercial purposes

---

## Citation Requirements

Any publication using MuDD **must cite**:

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

## Contact

For questions about dataset access or technical issues:

**Dataset Administrator & Corresponding Author**:
- **Name**: Yao Liu
- **Affiliation**: University of Electronic Science and Technology of China
- **Email**: liuyao@uestc.edu.cn

For code-related questions:
- Open an issue on [GitHub](https://github.com/YOUR_USERNAME/GPD/issues)

---

## Frequently Asked Questions

### Q1: Can I use MuDD for my company's research project?

**A**: MuDD is available only for non-commercial academic research. Commercial use requires separate authorization. Contact liuyao@uestc.edu.cn for inquiries.

### Q2: Can I share the dataset with my collaborators?

**A**: Collaborators who need direct access must submit their own applications. You cannot share credentials or dataset files.

### Q3: Can I publish sample images or audio clips from MuDD?

**A**: Raw or identifiable participant-level data (facial images, audio clips, physiological signals) cannot be publicly released without prior written permission. You may publish aggregate statistics, evaluation results, and non-identifiable research artifacts.

### Q4: What if I'm a student? Do I need my supervisor's approval?

**A**: Yes, student applicants must obtain their supervisor's or principal investigator's (PI) review and signature on the application form.

### Q5: How long does dataset access last?

**A**: Access is typically granted for the duration of your approved research project. If your project extends beyond the initial timeframe, you may need to renew your application.

### Q6: What happens if I violate the agreement?

**A**: Violations may result in immediate termination of access and a requirement to delete all dataset copies. Serious violations may be reported to your institution.

### Q7: Can I use MuDD to train commercial models?

**A**: No. Training or validating models for commercial deployment is not permitted under the current agreement.

---

## Updates and Announcements

Check the [GitHub repository](https://github.com/YOUR_USERNAME/GPD) for:
- Dataset updates or corrections
- New baseline results
- Code improvements
- Community contributions

---

**Last Updated**: September 2026
