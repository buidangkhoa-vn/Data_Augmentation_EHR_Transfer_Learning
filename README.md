# EHR Transfer Learning for Chronic Respiratory Disease Diagnosis

This repository presents a machine learning framework for Chronic Respiratory Disease (CRD) diagnosis by integrating chest X-ray (CXR) features with Electronic Health Records (EHR).
The work focuses on transfer learning and multimodal learning under severe class imbalance settings.
---

## Overview

Transfer learning with DenseNet for chest X-ray feature extraction

Multimodal learning combining CXR features and structured EHR data

Emphasis on class imbalance handling for rare respiratory diseases

Evaluation on multiple CRD prediction tasks
---

## Data Sources and Access Notice

This project relies on the [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) and
[MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) datasets, which contain a wide range
of clinical records and chest X-ray images collected from intensive care unit patients.

⚠️ **Important Notice on Data Access**

The datasets used in this project are **not publicly downloadable**.

Access to MIMIC-IV and MIMIC-CXR requires:
- Completion of the official
  [PhysioNet credentialed health data training](https://physionet.org/about/citi-course/)
- An approved
  [PhysioNet account](https://physionet.org/)

All users must comply with the data use agreements and ethical guidelines defined by PhysioNet.
---

## Pipeline (Baseline – Instructor)
- Cohort selection & EDA  
- CXR feature extraction (18 features)
- EHR feature extraction (12 features)
- Disease labels from ICD codes (8 CRDs)

Notebooks:
- `EDA.ipynb`
- `cohortselection.ipynb`
- `cxrretrieval.ipynb`
- `finalisedcohortdata.ipynb`

---

## Data Augmentation

### Model 1 – Synthetic Minority Over-sampling Technique (SMOTE) + XGBoost  
**Contributor:** Trần Thành Trọng  
- SMOTE on image features  
- XGBClassifier / XGBRegressor for EHR reconstruction  

### Model 2 – Random Over-Sampling (ROS) + XGBClassifier  
**Contributor:** Bùi Đăng Khoa
- Random Over-Sampling on training set only  
- Evaluated with:
  - Image features only
  - Image + EHR features  

---

## Results (Key Findings)
- Image-only models suffer from severe class imbalance
- SMOTE combined with XGBClassifier/XGBRegressor helps rare diseases overcome near-zero Recall, while simultaneously optimizing the F1-score and enhancing disease detection capability.
- ROS + XGBClassifier combined with EHR features demonstrates a favorable Precision–Recall trade-off.
- EHR features significantly improve detection of rare CRDs

Metrics: Accuracy, Precision, Recall, F1-score

---

## Repository Structure
```text
code/
├── EDA.ipynb
├── cohortselection.ipynb
├── cxrretrieval.ipynb
├── SMOTE+XGB.ipynb
├── Model_1.ipynb
├── Model_2.ipynb
└── finalisedcohortdata.ipynb

```
---

## Authors
- Trần Thành Trọng – SMOTE + XGBoost (Model 1)
- Bùi Đăng Khoa – ROS + XGBClassifier (Model 2)

**Instructor**: Nguyễn Tuấn Khôi  
**Co-Senior Supervisor**: Ngô Hoàng Anh

---

## Reference
See `CDR_detection_via_TL_with_data_augmentation.pdf` for full details.
