<div align="center">

# 🧠 Facial Expression
### A Full End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Amazon Rekognition](https://img.shields.io/badge/Amazon-Rekognition-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/rekognition/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Predicting cognitive stress levels from facial expression dynamics during Arabic text reading sessions — using unsupervised clustering and supervised Gradient Boosting.*

</div>

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Objectives](#-objectives)
- [Dataset Description](#-dataset-description)
- [ML Pipeline Overview](#-ml-pipeline-overview)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Clustering Phase — GMM](#-clustering-phase--gaussian-mixture-model)
- [Supervised Learning](#-supervised-learning)
- [Model Evaluation](#-model-evaluation)
- [Technologies Used](#-technologies-used)
- [Scientific Validity Notes](#-scientific-validity-notes)

---

## 🌐 Introduction

### The Problem

Stress is a pervasive psychological phenomenon that significantly impacts cognitive performance, reading comprehension, and mental health. Yet detecting stress non-invasively and in real time — without physiological sensors or self-reporting — remains an open challenge in human-computer interaction and cognitive science.

### Why Facial Expressions?

The human face is one of the most information-rich channels of non-verbal communication. Research in affective computing has demonstrated that subtle changes in facial pose, gaze direction, and expressed emotion can serve as reliable proxies for internal cognitive states, including stress, confusion, and cognitive load.

Amazon Rekognition's deep learning-powered facial analysis API enables extraction of high-fidelity emotion confidence scores and spatial pose metrics from video frames — making it possible to build a data-driven stress detection pipeline without specialized medical hardware.

### Why Arabic Reading Sessions?

Arabic is a morphologically rich and visually complex language. Studies indicate that reading cognitive load differs significantly across language scripts. Arabic text processing — particularly for participants reading at varying speeds or comprehension levels — generates distinctive emotional and postural signatures that can be used to differentiate stress states.

This project captures these dynamics frame-by-frame, creating a temporal dataset that reflects real cognitive engagement under varying levels of reading difficulty.

### The Role of AI and Machine Learning

Rather than relying on pre-defined clinical thresholds, this project leverages machine learning to **discover** stress patterns from data. By combining unsupervised clustering (to generate pseudo-labels) with supervised classification (to generalize predictions), the pipeline learns the latent emotional structure of stress from the bottom up — making it adaptable and data-driven.

> ⚠️ **Important Note:** Stress level labels in this project are *pseudo-labels* derived from unsupervised clustering, not ground-truth psychological assessments. This is a data-driven approximation, not a clinical diagnostic tool.

---

## 🎯 Objectives

- 📊 Perform comprehensive **Exploratory Data Analysis** on facial emotion and pose features
- 🔧 Engineer meaningful **derived features** (pose magnitude, eye magnitude, interaction terms) to enrich the feature space
- 🔍 Apply **unsupervised clustering** (Gaussian Mixture Model) to discover hidden stress-related groupings without labeled data
- 🏷️ Generate **pseudo-labels** from cluster assignments to enable supervised learning
- 🤖 Train a high-performance **Gradient Boosting classifier** on the pseudo-labeled data
- ✅ Evaluate model robustness using **stratified cross-validation** and held-out test set metrics
- 📈 Visualize decision boundaries, feature importances, and cluster structures
- 🔬 Critically analyze the **scientific validity** and limitations of the pseudo-labeling approach

---

## 📦 Dataset Description

Facial expression data was extracted frame-by-frame from video recordings of participants reading Arabic text passages, using **Amazon Rekognition's DetectFaces API**.

### Raw Dataset Statistics

| Split | Samples | Features |
|-------|---------|----------|
| Training Set | 1,263 | 8 |
| Testing Set | 1,706 | 8 |

### Feature Schema

| Feature | Type | Description |
|---------|------|-------------|
| `Frame` | Integer | Sequential frame index |
| `Emotion` | Categorical | Dominant detected emotion (`CALM`, `SURPRISED`, `SAD`, `CONFUSED`, `DISGUSTED`) |
| `Emotion_Confidence` | Float (0–100) | Rekognition's confidence score for the dominant emotion |
| `Pose_Roll` | Float (degrees) | Head rotation around the Z-axis (tilt) |
| `Pose_Yaw` | Float (degrees) | Head rotation around the Y-axis (left/right) |
| `Pose_Pitch` | Float (degrees) | Head rotation around the X-axis (up/down) |
| `Eye_Yaw` | Float (degrees) | Estimated horizontal gaze deviation |
| `Eye_Pitch` | Float (degrees) | Estimated vertical gaze deviation |

```

> ⚠️ **Class Imbalance:** The training set exhibits a 52.4× imbalance (CALM vs. CONFUSED); the test set reaches 267.5×. This is carefully addressed during supervised learning via balanced sample weighting.

---

## 🔄 ML Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA                                 │
│              Amazon Rekognition Frame Outputs                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA LOADING & VALIDATION                     │
│        Missing value check · Duplicate detection · EDA          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                          │
│   One-hot encoding · Pose Magnitude · Eye Magnitude             │
│   Pose × Eye Interaction · StandardScaler normalization         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             EXPLORATORY DATA ANALYSIS (EDA)                     │
│   Correlation heatmaps · Distribution plots  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           UNSUPERVISED CLUSTERING  (GMM)                        │
│   BIC-based optimal k selection · EM algorithm                  │
│   Soft probabilistic cluster assignment → Pseudo-labels         │
│   PCA visualization · Cluster profiling                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE SELECTION                               │
│        Mutual Information scoring → Top-6 features selected     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           SUPERVISED LEARNING (Gradient Boosting)               │
│   Balanced sample weights · 5-fold Stratified CV                │
│   200 estimators · Learning rate 0.1 · Max depth 4              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION                             │
│   Accuracy · Precision · Recall · F1-score                      │
│   Confusion matrix · Per-class analysis                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Exploratory Data Analysis

EDA was conducted systematically before any modeling to understand data quality, distribution, and inter-feature relationships.

### Correlation Analysis

A correlation matrix was computed over all numeric features in the training set. The analysis revealed several structurally important relationships:

- `Pose_Magnitude` exhibits strong correlation with its constituent components (`Pose_Roll`, `Pose_Yaw`, `Pose_Pitch`), as expected by construction
- `Eye_Magnitude` is similarly correlated with `Eye_Yaw` and `Eye_Pitch`
- Pose and eye movement features show modest inter-correlation, motivating the creation of the `Pose_x_Eye` interaction term

Pairs with Pearson |r| > 0.80 were identified and flagged to avoid redundancy in the feature space.

### Distribution Analysis

Statistical summaries (mean, std, min/max, IQR) were generated for all numeric features. Notable observations:

- `Emotion_Confidence` shows high variance (std ≈ 18.3), with values ranging from 29 to 99.9, indicating frames where the model was uncertain
- `Pose_Roll` is tightly distributed around −4.4°, suggesting a consistent slight head tilt in the reading posture
- `Eye_Yaw` has a mean of −4.2° with std ≈ 4.5°, reflecting consistent left-ward gaze bias during Arabic right-to-left reading

### Class Imbalance Analysis

Emotion class distributions were visualized using bar charts for both train and test splits. The severe imbalance (52–267× in train vs. test) was diagnosed as a critical preprocessing concern, addressed downstream by computing balanced sample weights during classification.

---

## 🔵 Clustering Phase — Gaussian Mixture Model

### Why Clustering Was Used

Before any supervised training, the dataset contained **no stress-level labels**. The target variable — cognitive stress — is a latent construct that cannot be directly observed from raw Rekognition outputs.

Clustering was applied to:

- Discover hidden emotional and behavioral structure in the data
- Identify latent groups of participants with similar facial expression profiles
- Reveal complex probabilistic relationships between emotional and postural features
- Generate **pseudo-labels** (data-driven stress categories) to enable downstream supervised learning

Since facial emotion features from Rekognition are inherently correlated and overlapping by nature — emotions rarely occur in isolation — clustering provides a principled way to uncover the latent distribution structure.

---

### Initial Experiments: K-Means

K-Means clustering was evaluated as a baseline to establish a reference point for unsupervised performance.

However, K-Means carries several assumptions that are incompatible with the structure of this dataset:

- **Spherical clusters:** K-Means minimizes Euclidean distance and assumes roughly globular, equally-sized clusters
- **Hard assignments:** Every sample is assigned exclusively to one cluster with no uncertainty estimate
- **Sensitivity to scale:** Although features were normalized, K-Means still struggled with the overlapping distributions in the emotional feature space

In practice, K-Means produced **lower Silhouette Scores** and less semantically meaningful cluster groupings. The boundaries between stress states — especially between adjacent medium-stress categories — were blurred in ways that K-Means could not represent, resulting in suboptimal cluster separability and less interpretable groupings.

---

### Why Gaussian Mixture Model (GMM) Was Chosen

The Gaussian Mixture Model was selected as the final clustering algorithm based on both theoretical and empirical grounds.

**Theoretical rationale:**

Human emotional states are inherently continuous and overlapping. A person transitioning between "calm" and "slightly stressed" does not flip a binary switch — they exist probabilistically in a region of emotional state space that shares characteristics of both. GMM reflects this reality by:

- **Modeling each cluster as a multivariate Gaussian distribution** with its own mean vector and covariance structure
- **Allowing elliptical cluster shapes** rather than forcing spherical boundaries
- **Assigning soft membership probabilities** — each sample receives a probability of belonging to each cluster, reflecting genuine uncertainty
- **Capturing correlated feature structures** through full or tied covariance matrices

**Empirical rationale:**

- GMM achieved better BIC (Bayesian Information Criterion) scores, indicating superior model fit relative to model complexity
- The resulting clusters showed more coherent emotional profiles when examined per-cluster
- Cluster assignments produced more balanced and interpretable pseudo-label distributions

---
**Pseudo-label generation:**

Each training and test sample was assigned to the cluster with the **highest posterior probability** (argmax of the responsibility vector), yielding 6 distinct stress-level pseudo-labels:

```
High Stress       →  152 samples (train)
Low Stress        →  376 samples (train)
Medium Stress 1   →   34 samples (train)
Medium Stress 2   →  344 samples (train)
Medium Stress 3   →  334 samples (train)
Medium Stress 4   →   23 samples (train)
```

PCA-based 2D projection was used to visualize cluster separability, confirming that GMM produced more coherent groupings than K-Means in the reduced feature space.

---

## 🤖 Supervised Learning

### Algorithm: Gradient Boosting Classifier

Following pseudo-label generation, a **Gradient Boosting Classifier** was trained on the training set using the GMM-derived stress labels as targets.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Sufficient boosting rounds without overfitting |
| `max_depth` | 4 | Limits individual tree complexity |
| `learning_rate` | 0.1 | Conservative step size for stable convergence |
| `subsample` | 0.8 | Stochastic boosting — reduces variance |
| `min_samples_split` | 10 | Prevents splits on very small node populations |

### Handling Class Imbalance

Given the severe class imbalance in the pseudo-labeled training set (ratio up to 16× between `Low Stress` and `Medium Stress 1`), **balanced sample weights** were computed using `sklearn.utils.class_weight.compute_sample_weight('balanced')`.

This re-weights each training sample inversely proportional to its class frequency, ensuring the model is not biased toward majority stress categories.

### Feature Selection

Prior to training, **Mutual Information (MI)** scores were computed between each feature and the pseudo-labels. Features with MI scores above the mean threshold were selected:

| Rank | Feature | MI Score |
|------|---------|----------|
| 1 | `Eye_Yaw` | 0.838 |
| 2 | `Pose_Yaw` | 0.698 |
| 3 | `Pose_x_Eye` | 0.565 |
| 4 | `Eye_Magnitude` | 0.409 |
| 5 | `Pose_Magnitude` | 0.363 |
| 6 | `Pose_Pitch` | 0.328 |

Gaze direction (`Eye_Yaw`) and head yaw rotation (`Pose_Yaw`) are the most discriminative features — consistent with the expectation that reading stress manifests through gaze instability and postural shifts.

### Cross-Validation

A **5-fold Stratified K-Fold** cross-validation was applied on the training set to estimate generalization performance before final evaluation:

```
Fold 1: 0.9644
Fold 2: 0.9526
Fold 3: 0.9526
Fold 4: 0.9643
Fold 5: 0.9603
────────────────
Mean:   0.9588 ± 0.0053
```

Stratified splitting ensures each fold preserves the original class distribution, critical given the imbalanced pseudo-labels.

---

## 📈 Model Evaluation

### Test Set Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **75.73%** |
| Weighted Precision | 0.77 |
| Weighted Recall | 0.76 |
| Weighted F1-Score | 0.75 |

### Per-Class Report

| Stress Level | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| High Stress | 0.91 | 0.74 | 0.81 | 729 |
| Low Stress | 0.66 | 0.97 | 0.79 | 254 |
| Medium Stress 1 | 0.05 | 0.04 | 0.04 | 56 |
| Medium Stress 2 | 0.78 | 0.83 | 0.80 | 369 |
| Medium Stress 3 | 0.64 | 0.70 | 0.67 | 287 |
| Medium Stress 4 | 0.00 | 0.00 | 0.00 | 11 |

### Metric Interpretations

**Accuracy (75.73%)** — The model correctly classifies roughly 3 out of 4 test samples. High overall accuracy is driven by strong performance on the majority classes (High Stress, Low Stress, Medium Stress 2).

**Precision** — Among samples predicted as a given class, what fraction truly belongs to it. High precision for `High Stress` (0.91) indicates few false alarms for the most critical category.

**Recall** — Among all true members of a class, what fraction the model recovered. `Low Stress` achieves near-perfect recall (0.97), meaning almost no true low-stress samples are missed.

**F1-Score** — Harmonic mean of precision and recall; robust to class imbalance. The macro-averaged F1 (0.52) reveals the challenge of classifying minority classes (`Medium Stress 1`, `Medium Stress 4`), both of which had very few training samples and likely represent edge-case GMM clusters.

**Confusion Matrix** — Visualizes the distribution of predictions vs. ground-truth pseudo-labels across all 6 stress categories. Off-diagonal entries highlight systematic confusions, particularly between adjacent medium-stress classes — an expected artifact of the continuous nature of emotional states.

> The gap between cross-validation accuracy (~95.9%) and test accuracy (~75.7%) is primarily attributable to distribution shift between the train and test splits in the pseudo-label space, and the inherent difficulty of minority stress classes.

---

## 🛠️ Technologies Used

| Technology | Version | Role |
|------------|---------|------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) **Python** | 3.10+ | Core programming language |
| **Pandas** | Latest | Data loading, manipulation, aggregation |
| **NumPy** | Latest | Numerical computation, array operations |
| **Scikit-learn** | 1.3+ | GMM, Gradient Boosting, preprocessing, evaluation |
| **Matplotlib** | Latest | Static charts, confusion matrix visualization |
| **Seaborn** | Latest | Correlation heatmaps, distribution plots |
| **Plotly** | Latest | Interactive visualizations |
| **Amazon Rekognition** | AWS API | Facial emotion and pose extraction |


---

## 🔬 Scientific Validity Notes

This project employs a **hybrid unsupervised-then-supervised** strategy. Understanding its epistemological boundaries is essential for responsible interpretation:

| Aspect | What It Means |
|--------|---------------|
| **Pseudo-labels ≠ Ground Truth** | Stress levels are data-driven approximations from GMM clusters, not clinically validated psychological states |
| **Pattern Learning** | High classification accuracy reflects how well the model has learned the clustering structure — not that it measures genuine psychological stress |

---

<div align="center">

**Graduation Project**


</div>
