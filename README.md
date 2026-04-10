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
├── EDA.ipynb                         # Exploratory Data Analysis on cohort data
├── cohortselection.ipynb             # Filter cohort by existence in MIMIC-CXR
├── cxrretrieval.ipynb                # Selecting 1 CXR for each patient
├── SMOTE+XGB.ipynb        
├── Model_1.ipynb
├── Model_2.ipynb                     
└── finalisedcohortdata.ipynb         # Feature extraction on MIMIC-IV, finalising data 

```
---
## Requirements

This project was developed with Python 3.10.12 and the dependencies listed in `requirements.txt`.

The notebooks were originally executed on Google Colab.
A minimum of 4 GB RAM is recommended.
GPU is not required.

## Environment Setup

There are two ways to run the notebooks: on Google Colab or on a local machine.

### Running on Colab (Recommended)
This is the recommended method. **Hyperlinks that make reference to plots will only work on Colab!**
1. Install [Google Colaboratory](https://colab.research.google.com/) on your [Google Drive](https://drive.google.com/drive/u/0/).
2. Upload your `.ipynb` notebook file to Google Drive. The notebook should automatically open on Colab.
**Note**: In case of any incompatibilities due to Colab changing package versions, add the following code cell to the start of your notebook and run it:

```python
!wget https://github.com/clemence-mottez/mimic_iv/raw/main/requirements.txt
!pip install -r requirements.txt
```
**Running on local**
In most cases, the notebook can be run locally by just installing the required package versions with the following command (`requirements.txt` is assumed to be in the same directory as your notebook):
```bash
pip install -r requirements.txt
```
To replicate Colab’s operating system on local computer, this is the specifications of the Colab environment as of October 2023:

```text
Python implementation: CPython
Python version       : 3.10.12
IPython version      : 7.34.0

Compiler    : GCC 11.4.0
OS          : Linux
Release     : 5.15.120+
Machine     : x86_64
Processor   : x86_64
CPU cores   : 2
Architecture: 64bit
```
Make sure to do removal/add exception handler for any line in the form of `from google.colab import ...`.

---
## Authors
- Trần Thành Trọng – SMOTE + XGBoost (Model 1)
- Bùi Đăng Khoa – ROS + XGBClassifier (Model 2)

**Instructor**: Nguyễn Tuấn Khôi  
**Co-Senior Supervisor**: Ngô Hoàng Anh

---

## Reference
See `CDR_detection_via_TL_with_data_augmentation.pdf` for full details.
