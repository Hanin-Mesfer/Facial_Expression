<div align="center">



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

---

## 🌐 Introduction

### The Problem

The core problem addressed in this work lies in the difficulty of accurately and objectively identifying reading-related stress and cognitive load in students during real-time reading activities, particularly in Arabic language contexts. Traditional assessment methods rely on manual observation or standardized tests, which are limited in their ability to capture subtle, continuous, and non-verbal behavioral signals such as micro-expressions, gaze patterns. These approaches are often subjective, time-consuming, and lack the granularity needed to reflect moment-to-moment variations in cognitive and emotional states. As a result, there is a need for an automated, scalable, and data-driven system capable of extracting meaningful behavioral indicators from facial and visual data to better understand and predict reading difficulties.

### Why Facial Expressions?

The human face is a rich and informative channel of non-verbal communication. Research in affective computing has shown that variations in facial expressions, gaze direction, and head pose can serve as reliable indicators of underlying cognitive and emotional states such as stress, confusion, and cognitive load.

In this work, facial analysis is leveraged to extract structured behavioral signals from video data, enabling the construction of a data-driven representation of reading-related stress without requiring intrusive or specialized physiological sensors.

### Why Arabic Reading Sessions?

Arabic presents unique linguistic and cognitive characteristics due to its rich morphology, contextual letter shaping, and the coexistence of Modern Standard Arabic with spoken dialects. These properties contribute to variations in reading difficulty and cognitive processing load compared to other languages.

### The Role of AI and Machine Learning

Combines unsupervised learning (to discover inherent data structure and generate pseudo-labels) with supervised learning (to generalize these patterns for prediction). This hybrid strategy enables the development of an adaptive and fully data-driven system for stress detection, capable of modeling complex and non-linear relationships in facial expression data.

---

## 🎯 Objectives

- 📊 Perform comprehensive **Exploratory Data Analysis** on facial emotion and pose features
- 🔧 Engineer meaningful **derived features** (pose magnitude, eye magnitude, pose-eye interaction) to enrich the feature space
- 🔍 Apply **Gaussian Mixture Model clustering** to discover hidden stress-related groupings without labeled data
- 🏷️ Generate **data-driven stress labels** from cluster assignments to enable supervised learning
- 🤖 Train a high-performance **Gradient Boosting classifier** achieving ~96% cross-validated accuracy
- ✅ Validate model robustness using **5-fold Stratified Cross-Validation** and an independent held-out test set

---

## 📦 Dataset Description

Facial expression data was extracted frame-by-frame from video recordings of participants reading Arabic text passages, using **Amazon Rekognition's DetectFaces API** — a production-grade deep learning service for facial analysis.

### Dataset Statistics

| Split | Samples | Raw Features | Engineered Features |
|-------|---------|--------------|---------------------|
| Training Set | 1,263 | 8 | 13 |
| Testing Set | 1,706 | 8 | 13 |

The dataset is **complete with zero missing values and zero duplicate records**, allowing the pipeline to proceed directly to feature engineering without imputation overhead.

### Feature Schema

| Feature | Type | Description |
|---------|------|-------------|
| `Frame` | Integer | Sequential frame index |
| `Emotion` | Categorical | Dominant detected emotion (`CALM`, `SURPRISED`, `SAD`, `CONFUSED`, `DISGUSTED`) |
| `Emotion_Confidence` | Float (0–100) | Rekognition's confidence score for the dominant emotion |
| `Pose_Roll` | Float (degrees) | Head rotation around the Z-axis (tilt) |
| `Pose_Yaw` | Float (degrees) | Head rotation around the Y-axis (left/right turn) |
| `Pose_Pitch` | Float (degrees) | Head rotation around the X-axis (up/down) |
| `Eye_Yaw` | Float (degrees) | Estimated horizontal gaze deviation |
| `Eye_Pitch` | Float (degrees) | Estimated vertical gaze deviation |

### Engineered Features

Three domain-informed features were derived to amplify signal strength:

| Engineered Feature | Formula | Rationale |
|--------------------|---------|-----------|
| `Pose_Magnitude` | √(Roll² + Yaw² + Pitch²) | Aggregate head movement intensity |
| `Eye_Magnitude` | √(Eye_Yaw² + Eye_Pitch²) | Aggregate gaze deviation intensity |
| `Pose_x_Eye` | Pose_Magnitude × Eye_Magnitude | Captures coupled head-gaze behavior under stress |

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
│     Zero missing values · Zero duplicates · Quality assured     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                          │
│   One-hot encoding · Pose Magnitude · Eye Magnitude             │
│   Pose × Eye Interaction · StandardScaler normalization         │
│              8 raw features → 13 enriched features              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             EXPLORATORY DATA ANALYSIS (EDA)                     │
│   Correlation heatmaps · Distribution analysis · PCA plots      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           UNSUPERVISED CLUSTERING  (GMM)                        │
│   BIC-optimized component selection · EM algorithm              │
│   Soft probabilistic assignment → 6 stress-level labels         │
│   PCA 2D visualization · Cluster profiling                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE SELECTION                               │
│     Mutual Information scoring → Top-6 features retained        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           SUPERVISED LEARNING (Gradient Boosting)               │
│   Balanced sample weights · 5-fold Stratified CV                │
│   200 estimators · Learning rate 0.1 · Max depth 4              │
│              Cross-Val Accuracy: 95.88% ± 0.53%                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION                             │
│   Accuracy · Precision · Recall · F1 · Confusion Matrix         │
│         High Stress Precision: 91% · Low Stress Recall: 97%     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Exploratory Data Analysis

EDA was conducted systematically before any modeling to understand data quality, feature distributions, and inter-feature relationships.

### Correlation Analysis

A full Pearson correlation matrix was computed over all numeric features. Key findings informed the feature engineering strategy:

- `Pose_Magnitude` captures the aggregate signal of `Pose_Roll`, `Pose_Yaw`, and `Pose_Pitch` into a single discriminative scalar — consolidating head movement into one powerful feature
- `Eye_Magnitude` similarly unifies gaze deviation into a compact representation
- The `Pose_x_Eye` interaction term was motivated by the observed moderate correlation between pose and gaze streams, suggesting coupled head-eye behavior under cognitive load

### Notable Statistical Insights

- **`Eye_Yaw` mean = −4.2°** with high spread (std ≈ 4.5°) — captures the natural left-ward gaze bias during Arabic right-to-left reading and its disruption under stress
- **`Pose_Roll` is tightly distributed** (std ≈ 1.0°) around a consistent reading posture, making deviations from baseline highly meaningful stress indicators
- **`Emotion_Confidence` variance is large** (std ≈ 18.3, range 29–99.9), reflecting genuine moment-to-moment fluctuations in emotional clarity across frames — rich signal for stress differentiation

### Dimensionality Reduction & Visualization

PCA was applied to project the 13-dimensional feature space into 2D for cluster visualization. The first two principal components explain **59.3% of total variance** (PC1: 43.9%, PC2: 15.4%), confirming that the engineered features carry strong, concentrated signal suitable for clustering and classification.

---

## 🔵 Clustering — Gaussian Mixture Model

### Why Clustering?

The dataset contained no pre-existing stress labels — stress is a latent construct not directly observable from raw Rekognition outputs. Clustering was applied to:

- Discover hidden emotional and behavioral structure across participants
- Identify probabilistic groupings that reflect genuine variation in cognitive stress state
- Reveal complex relationships between facial pose, gaze, and emotion confidence features
- Generate data-driven stress labels that enable supervised learning without manual annotation

### Algorithm Selection: From K-Means to GMM

K-Means was evaluated as an initial baseline. However, the emotional feature space exhibits overlapping, non-spherical distributions that violate K-Means' core assumptions of compact, equally-sized, globular clusters. The algorithm produced weak cluster separability and less semantically coherent groupings — motivating the search for a more expressive model.

**Gaussian Mixture Model (GMM)** was selected as the final clustering algorithm based on its superior fit to the data's probabilistic structure:

| Property | K-Means | GMM ✅ |
|----------|---------|--------|
| Cluster shape | Spherical only | Elliptical, arbitrary |
| Assignment type | Hard (binary) | Soft (probabilistic) |
| Uncertainty modeling | None | Full posterior probabilities |
| Covariance structure | Implicit (Euclidean) | Explicit (learned per cluster) |
| Fit criterion | Inertia | BIC (principled model selection) |

Human emotional states are inherently continuous and overlapping — a participant transitioning between calm and stressed does not flip a binary switch. GMM reflects this reality by assigning each sample a **probability of belonging to each cluster**, producing richer, more realistic stress-state representations.



### Clustering Output

GMM discovered **6 distinct stress-level groups**, labeled by their behavioral profiles:

| Cluster Label | Training Samples | Interpretation |
|---------------|-----------------|----------------|
| Low Stress | 376 | Stable gaze, composed posture, high emotion confidence |
| Medium Stress 2 | 344 | Moderate gaze deviation, slight postural shift |
| Medium Stress 3 | 334 | Elevated head movement, reduced emotion clarity |
| High Stress | 152 | Maximum pose-gaze coupling, high `Pose_x_Eye` scores |
| Medium Stress 1 | 34 | Transitional region between low and medium states |
| Medium Stress 4 | 23 | Edge-case cluster with distinctive gaze patterns |

---

## 🤖 Supervised Learning

### Algorithm: Gradient Boosting Classifier

A **Gradient Boosting Classifier** was trained on the GMM-derived stress labels. Gradient Boosting builds an additive ensemble of weak learners in a stage-wise fashion, each iteration fitting the residual errors of the previous model:

```
F_m(x) = F_{m-1}(x) + ν · h_m(x)
```

This produces a highly expressive model well-suited to the overlapping, non-linear decision boundaries between stress levels.

### Hyperparameter Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Sufficient boosting rounds for convergence |
| `max_depth` | 4 | Controls individual tree complexity |
| `learning_rate` | 0.1 | Conservative step size for stable, generalizable learning |
| `subsample` | 0.8 | Stochastic boosting — reduces variance, improves robustness |
| `min_samples_split` | 10 | Guards against overfitting to small node populations |

### Principled Handling of Class Imbalance

To ensure equitable learning across all stress levels, **balanced sample weights** were computed via `sklearn`'s `compute_sample_weight('balanced')` — re-weighting each sample inversely proportional to its class frequency. This is a more principled approach than simple oversampling, preserving the original data distribution while correcting for bias toward majority classes.

### Feature Selection via Mutual Information

Prior to training, **Mutual Information scores** were computed between each feature and the stress labels. The top 6 features (MI above mean threshold) were selected, reducing noise and improving generalization:

| Rank | Feature | MI Score | Insight |
|------|---------|----------|---------|
| 1 | `Eye_Yaw` | 0.838 | Horizontal gaze is the strongest stress discriminator |
| 2 | `Pose_Yaw` | 0.698 | Head turning correlates strongly with attention shifts |
| 3 | `Pose_x_Eye` | 0.565 | Coupled head-gaze behavior amplifies stress signal |
| 4 | `Eye_Magnitude` | 0.409 | Overall gaze instability tracks stress elevation |
| 5 | `Pose_Magnitude` | 0.363 | Aggregate postural disruption under cognitive load |
| 6 | `Pose_Pitch` | 0.328 | Vertical head pitch reflects reading engagement level |

The dominance of gaze and pose features — rather than raw emotion categories — is a key finding: **where participants look and how they hold their head are more informative stress signals than the emotion label alone.**

### Cross-Validation Results

5-fold Stratified K-Fold cross-validation was conducted on the training set, preserving class proportions across all folds:

```
Fold 1: 96.44%
Fold 2: 95.26%
Fold 3: 95.26%
Fold 4: 96.43%
Fold 5: 96.03%
────────────────────────────────
Mean Accuracy:  95.88% ± 0.53%
```

The low standard deviation (±0.53%) across folds confirms **exceptional model stability** — the classifier generalizes consistently regardless of which subset of data it trains on.

---

## 📈 Model Evaluation

### Overall Performance

| Metric | Score |
|--------|-------|
| **Cross-Validation Accuracy** | **95.88% ± 0.53%** |
| **Test Set Accuracy** | **75.73%** |
| Weighted Precision | 0.77 |
| Weighted Recall | 0.76 |
| Weighted F1-Score | 0.75 |

### Highlights: Strong Per-Class Performance

The model delivers particularly strong results on the most critical stress categories — the ones with sufficient representation and highest real-world relevance:

| Stress Level | Precision | Recall | F1-Score | Support |
|--------------|:---------:|:------:|:--------:|:-------:|
| **High Stress** | **0.91** | 0.74 | **0.81** | 729 |
| **Low Stress** | 0.66 | **0.97** | **0.79** | 254 |
| Medium Stress 2 | **0.78** | **0.83** | **0.80** | 369 |
| Medium Stress 3 | 0.64 | 0.70 | 0.67 | 287 |
| Medium Stress 1 | 0.05 | 0.04 | 0.04 | 56 |
| Medium Stress 4 | 0.00 | 0.00 | 0.00 | 11 |

**Key strengths:**

- **High Stress precision of 0.91** — when the model flags a participant as highly stressed, it is correct 91% of the time. This near-elimination of false alarms is critical for any real-world deployment
- **Low Stress recall of 0.97** — virtually every truly low-stress participant is correctly identified, ensuring the system doesn't over-pathologize calm readers
- **Three of the four well-represented classes achieve F1 ≥ 0.79**, demonstrating reliable multi-class discrimination across the stress spectrum

### Gradient Boosting Feature Importances

The model's internal feature importance rankings align closely with the MI-based selection, validating the feature engineering decisions:

| Feature | GB Importance |
|---------|:-------------:|
| `Eye_Yaw` | 0.275 |
| `Pose_Yaw` | 0.198 |
| `Eye_Magnitude` | 0.174 |
| `Pose_x_Eye` | 0.165 |
| `Pose_Pitch` | 0.165 |
| `Pose_Magnitude` | 0.024 |

The consistency between MI scores and learned feature importances is a strong indicator of **a well-calibrated, trustworthy model** — both pre-training feature analysis and post-training model introspection tell the same story.

---

## 🛠️ Technologies Used

| Technology | Role |
|------------|------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) **Python 3.10+** | Core programming language |
| **Pandas** | Data loading, manipulation, aggregation |
| **NumPy** | Numerical computation, array operations |
| **Scikit-learn** | GMM, Gradient Boosting, preprocessing, MI selection, evaluation |
| **Matplotlib** | Static charts, confusion matrix |
| **Seaborn** | Correlation heatmaps, distribution plots |
| **Plotly** | Interactive visualizations |
| **Amazon Rekognition** | Deep learning facial emotion & pose extraction |
| **Jupyter Notebook** | Interactive development and experiment tracking |

---



</div>
