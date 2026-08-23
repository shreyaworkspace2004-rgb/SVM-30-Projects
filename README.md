#  30 Support Vector Machine (SVM) 

A comprehensive collection of **30 end-to-end Support Vector Machine (SVM/SVC) projects** covering linear separation, non-linear RBF kernels, class balancing, multi-class decision functions, and feature scaling using **Python** and **Scikit-Learn**.

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

##  30 Projects Catalog

| # | Project Name | Description | Key Concept / Features |
|---|---|---|---|
| **01** | **Iris Species Multi-Class Classifier** | Multi-class target classification using baseline support vector classifier[cite: 32]. | `load_iris()`, Default SVC[cite: 32] |
| **02** | **Breast Cancer Detection Pipeline** | Binary medical diagnostics evaluated with train-test split accuracy score[cite: 33]. | `load_breast_cancer()`, SVC[cite: 33] |
| **03** | **Scaled RBF Kernel Classifier** | Optimized radial basis function classifier with StandardScaler normalization[cite: 35]. | `StandardScaler()`, `kernel='rbf'`[cite: 35] |
| **04** | **Non-Linear Concentric Circles Classifier** | Separation of synthetic non-linear circular boundary data[cite: 34]. | `make_circles(noise=0.1)`[cite: 34] |
| **05** | **Handwritten Digits Multi-Classifier** | Recognition and classification of 8x8 pixel optical digit patterns[cite: 37]. | `load_digits()`, SVC[cite: 37] |
| **06** | **Decision Function Margin Explorer** | Calculates distance margins from separating hyperplane for confidence scoring[cite: 36]. | `model.decision_function()`[cite: 36] |
| **07** | **Standardized Feature Preprocessor** | Full data scaling pipeline ensuring zero mean and unit variance before SVM fit[cite: 38]. | `StandardScaler().fit_transform()`[cite: 38] |
| **08** | **Imbalanced Class Weight Optimizer** | Balances unequal class distributions automatically using adjusted weighting[cite: 31]. | `SVC(class_weight='balanced')`[cite: 31] |
| **09** | **C-Regularization Parameter Tuner** | Controls soft margin tolerance and misclassification penalty[cite: 40]. | `SVC(C=10)`[cite: 40] |
| **10** | **K-Best Feature Selection Pipeline** | Selects top discriminatory features using univariate statistical tests before SVM[cite: 39]. | `SelectKBest(k=2)`[cite: 39] |
| **11** | **Polynomial Kernel Classifier** | Solves high-dimensional non-linear boundaries using poly kernel mapping. | `kernel='poly'`, Degree |
| **12** | **Linear SVC vs Kernel SVC** | Compares execution speed and decision surface against pure `LinearSVC`. | Linear Boundary vs RBF |
| **13** | **Customer Churn Predictor** | Classifies high-risk customer churn utilizing customer account metrics. | Usage Metrics, Account Age |
| **14** | **Credit Card Fraud Detection** | High-precision fraud detector handling severe class imbalances. | Transaction Amount, Latency |
| **15** | **Spam SMS Filtering Model** | High-dimensional text categorization using TF-IDF and linear SVM. | Text Word Vectors |
| **16** | **Wine Quality Tier Classifier** | Predicts chemical quality classes based on alcohol, pH, and sulfates. | Acidity, Density, Alcohol % |
| **17** | **Heart Disease Risk Classifier** | Biometric cardiovascular assessment predicting vessel blockage risk. | Blood Pressure, Cholesterol |
| **18** | **Parkinson's Disease Voice Screener**| Biomedical voice acoustic classification detecting early motor symptoms. | Jitter, Shimmer, Pitch |
| **19** | **Stock Market Trend Classifier** | Binary direction predictor forecasting daily price movements. | Technical Indicators, RSI |
| **20** | **Signature Authenticity Verifier** | Geometric coordinate classifier detecting forged signature patterns. | Stroke Width, Pressure |
| **21** | **Loan Eligibility Classifier** | Predicts loan approvals based on credit score, income, and liabilities. | Applicant Income, Credit History |
| **22** | **Customer Review Sentiment SVM** | Analyzes customer review polarities into positive or negative tiers. | N-gram Features, Sentiment Score |
| **23** | **Employee Attrition Forecaster** | Models corporate turnover likelihood based on workplace satisfaction scores. | Satisfaction, Tenure, Overtime |
| **24** | **Vehicle Silhouette Classifier** | Multiclass classification of vehicle types using 2D silhouette descriptors. | Compactness, Circularity |
| **25** | **Bioinformatics Protein Folding** | Structural biology classifier categorizing amino acid folding sequences. | Sequence Features |
| **26** | **Water Quality Potability Rater** | Determines water safety by evaluating hardness, chloramines, and solids. | Solids, Sulfate, pH |
| **27** | **Network Intrusion Detection** | Cyber defense classifier flagging malicious TCP network packets. | Packet Length, Error Rate |
| **28** | **E-Commerce Purchase Intent** | Classifies visitor session traffic into buying or browsing intent. | Page Duration, Bounce Rate |
| **29** | **Crop Disease Symptom Classifier** | Classifies agricultural crop health states from environmental metrics. | Temperature, Humidity, Soil pH |
| **30** | **Medical Hospital Readmission** | Predicts 30-day patient readmission likelihood from diagnosis records. | Stay Duration, Lab Test Counts |

---

##  Project Structure

```text
├── projects/
│   ├── 01_iris_svm.py
│   ├── 02_breast_cancer_svm.py
│   ├── 03_scaled_rbf_svm.py
│   ├── ...
│   └── 30_hospital_readmission_svm.py
├── requirements.txt
└── README.md
