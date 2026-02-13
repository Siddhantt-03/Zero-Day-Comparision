# Zero-Day Vulnerability Detection - Machine Learning Comparison

A comparative study and implementation of Machine Learning and Deep Learning models for detecting Zero-Day vulnerabilities using static, behavioral, and hybrid detection approaches.

## Overview

Zero-day vulnerabilities are unknown software flaws exploited before patches are released. Traditional signature-based detection systems fail against such attacks because they rely on known patterns.

This project evaluates multiple Machine Learning and Neural Network models to detect zero-day threats and compares their performance using standard evaluation metrics.

The goal is simple:
Build an intelligent detection framework that outperforms traditional systems.

---

## Architecture

This project follows a **Multi-Tier Detection Architecture**:

### Tier 1 - Rapid Pre-Filtering

* Static Analysis
* Signature Filtering
* Code Complexity Analysis

### Tier 2 - Behavioral Analysis

* Runtime Monitoring
* Memory Profiling
* API Behavior Tracking

### Tier 3 - Contextual Intelligence

* Threat Feeds Integration
* Historical Exploit Analysis
* Real-Time SIEM Support

The hybrid output from all tiers enhances detection accuracy while minimizing false positives.

---

## Models Implemented

### Machine Learning Models

* Random Forest
* Decision Tree
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* AdaBoost

### Neural Network Models

* Multi-Layer Perceptron (MLP)
* Deep Neural Networks (Multiple Hidden Layers)

---

## Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* False Positive Rate
* Cross-Validation (k-fold)

---

## Results Summary

| Model            | Accuracy |
| ---------------- | -------- |
| Random Forest    | 99.51%   |
| Decision Tree    | 99.24%   |
| KNN              | 99.12%   |
| MLP              | 99.25%   |
| Deep Learning    | 99.33%   |
| Hybrid Framework | 99.7%    |

Random Forest performed best among traditional ML models, while hybrid integration achieved the highest overall accuracy.

---

## Tech Stack

* Python
* Scikit-learn
* TensorFlow / Keras
* Pandas
* NumPy
* Matplotlib

---

## Project Structure

```
Zero-Day-Comparision/
│
├── dataset/
├── preprocessing/
├── models/
│   ├── random_forest.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── neural_network.py
│
├── evaluation/
├── results/
├── architecture_diagram.png
└── README.md
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Siddhantt-03/Zero-Day-Comparision.git
cd Zero-Day-Comparision
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run a model:

```bash
python models/random_forest.py
```

---

## Key Insights

* Ensemble methods outperform standalone classifiers.
* Hybrid detection (static + behavioral + contextual) improves robustness.
* False positives reduce significantly when behavioral data is integrated.
* Real-time intelligence integration is critical for zero-day defense.

---

## Research Basis

This repository is based on the research paper:

**“Exploring Zero-Day Vulnerabilities: Techniques, Impact, and Mitigation Strategies”**

The study integrates machine learning with threat intelligence to improve detection of unknown attack vectors.

---

## Limitations

* Dependent on quality of training data
* Requires computational resources for deep models
* Real-time deployment requires further optimization

---

## Future Improvements

* Integration with real-time SIEM
* Use of GAN-based malware generation
* Transformer-based vulnerability detection
* Larger real-world dataset validation

---

## Author

Siddhant Pandey

Manipal University Jaipur

Computer Science & Engineering

---
