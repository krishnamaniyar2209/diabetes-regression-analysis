# 🩺 Diabetes Disease Progression Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Linear%20Regression-red?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS675-blue)

> A machine learning project predicting the **progression of diabetes** one year after baseline using **Linear Regression** and **XGBoost** — built for CS675: Introduction to Data Science at Pace University.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Extra Credit — XGBoost](#-extra-credit--xgboost)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Author](#-author)
- [License](#-license)

---

## 🔬 Overview

This project implements and evaluates **Linear Regression** models on the classic **Diabetes dataset** (Efron, Hastie, Johnstone & Tibshirani, *Least Angle Regression*, 2004). The goal is to predict a quantitative measure of disease progression one year after baseline from 10 patient health features, then benchmark a gradient-boosted model (XGBoost) against the linear baseline.

The notebook covers:
- ✅ **Exploratory Data Analysis (EDA)** — distributions, correlations, missing values
- ✅ **Single-feature regression** — finding the best individual predictor
- ✅ **Pair-feature regression** — best two-feature combination (all 45 pairs)
- ✅ **Full-feature regression** — all 10 features together
- ✅ **Learning-curve analysis** — training vs. validation MSE across dataset sizes
- ✅ **XGBoost comparison** — gradient boosting vs. linear regression (extra credit)

---

## 📊 Dataset

**Source:** [Stanford University — Diabetes Dataset (LARS Paper)](https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data)

**Reference Paper:**
> Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004)
> *"Least Angle Regression"*, Annals of Statistics, 407–499.

| Property | Details |
|---|---|
| Records | 442 patients |
| Features | 10 numeric predictors |
| Target | `y` — disease progression (1 year after baseline) |
| Missing Values | None |

### Feature Descriptions

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex |
| `bmi` | Body Mass Index |
| `bp` | Average blood pressure |
| `s1` | TC — Total serum cholesterol |
| `s2` | LDL — Low-density lipoproteins |
| `s3` | HDL — High-density lipoproteins |
| `s4` | TCH — Total cholesterol / HDL ratio |
| `s5` | LTG — Log of serum triglycerides level |
| `s6` | GLU — Blood sugar level |
| `y` | **Target** — Quantitative disease progression |

---

## 📁 Project Structure
```
diabetes-regression-analysis/
│
├── Diabetes_Progression_Prediction.ipynb   # Main Jupyter notebook
├── README.md                               # Project documentation
└── requirements.txt                        # Python dependencies
```

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis (EDA)
Shape and dtype inspection, descriptive statistics, missing-value check, per-feature histograms and boxplots, a correlation heatmap, and a feature–target correlation ranking (top correlations with `y`: **bmi 0.59, s5 0.57, bp 0.44**).

### Step 2 — Single Feature Regression
Trains a Linear Regression model for each of the 10 features and selects the one with the **lowest MSE**:
```python
for feature in X.columns:
    model = LinearRegression()
    model.fit(X[[feature]], y)
    mse = mean_squared_error(y, model.predict(X[[feature]]))
```

### Step 3 — Pair Feature Regression
Tests all **45 feature pairs** via `itertools.combinations` and selects the best pair:
```python
for pair in combinations(X.columns, 2):
    model = LinearRegression()
    model.fit(X[list(pair)], y)
    mse = mean_squared_error(y, model.predict(X[list(pair)]))
```

### Step 4 — All Features Regression
Fits one Linear Regression model on all 10 features and reports every coefficient.

### Step 5 — Learning Curve Analysis
Computes training and validation MSE at four training-set sizes (n = 20, 50, 100, 200) to study the bias–variance tradeoff.

### Step 6 — XGBoost (Extra Credit)
Trains an XGBoost regressor (`n_estimators=100, learning_rate=0.1, max_depth=3`) and compares it to Linear Regression via test MSE, R², residual analysis, and 5-fold cross-validation.

---

## 📈 Models & Results

### Task 1 — Best Single Feature
| Best Feature | Slope | Intercept | MSE | R² |
|---|---|---|---|---|
| `bmi` | 10.23 | -117.77 | 3890.46 | 0.344 |

### Task 2 — Best Feature Pair
| Feature 1 | Feature 2 | Intercept | Coef 1 | Coef 2 | MSE | R² |
|---|---|---|---|---|---|---|
| `bmi` | `s5` | -299.96 | 7.28 | 56.06 | 3205.19 | 0.459 |

### Task 3 — All 10 Features
| Metric | Value |
|---|---|
| Intercept | -334.57 |
| MSE | 2859.70 |
| R² | 0.518 |

### Task 4 — Learning Curve (Training vs. Validation MSE)
| Training Size | Training MSE | Validation MSE |
|---|---|---|
| 20 | 2,066 | 15,906 |
| 50 | 2,864 | 3,907 |
| 100 | 3,151 | 3,423 |
| 200 | 2,855 | 3,015 |

> Tiny training sets overfit dramatically (n=20: validation MSE ≈ 8× training MSE). By n=200 the gap nearly closes — classic bias–variance behavior.

---

## 🚀 Extra Credit — XGBoost

An XGBoost Regressor was trained and benchmarked against Linear Regression.

### Test Set Performance
| Model | Test MSE | Test R² |
|---|---|---|
| **Linear Regression** | **2900.19** | **0.453** ✅ |
| XGBoost | 2959.46 | 0.441 |

### 5-Fold Cross-Validation
| Model | Mean CV MSE | Std CV MSE | Mean CV R² |
|---|---|---|---|
| **Linear Regression** | **3015.38** | 291.24 | **0.478** ✅ |
| XGBoost | 3350.09 | 424.85 | 0.423 |

### Additional Analysis
- 📊 **Predicted vs. Actual** scatter for both models
- 📊 **Feature Importance** — XGBoost ranks **`bmi` (0.32)** and **`s5` (0.22)** as top predictors, consistent with the linear analysis
- 📊 **Residual plots** — roughly normal errors for both models

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Clone & Setup
```bash
git clone https://github.com/krishnamaniyar2209/diabetes-regression-analysis.git
cd diabetes-regression-analysis
pip install -r requirements.txt
jupyter notebook Diabetes_Progression_Prediction.ipynb
```

---

## 🚀 Usage

1. Open `Diabetes_Progression_Prediction.ipynb` in Jupyter or Google Colab
2. The dataset loads **automatically** from Stanford's URL — no download needed:
```python
df = pd.read_csv(
    "https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data",
    sep=r"\s+", header=0, names=column_names
)
```
3. Run all cells sequentially from top to bottom

---

## 💡 Key Findings

- **`bmi`** is the single strongest predictor of disease progression (R² = 0.34), consistent with medical literature
- **`bmi` + `s5`** (log triglycerides) is the best feature pair, cutting MSE from 3890 → 3205
- Using **all 10 features** gives the best linear model (MSE 2860, R² 0.52)
- **Learning curves** confirm bias–variance behavior — small samples overfit badly, larger samples generalize
- **Linear Regression outperformed XGBoost** on both the test set (R² 0.453 vs. 0.441) and 5-fold CV (R² 0.478 vs. 0.423). On only 442 rows with a largely linear feature–target relationship, gradient boosting slightly overfit — a reminder that the more complex model is not always better
- **Residual analysis** shows roughly normal errors, supporting the linear-model assumptions

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| scikit-learn | Linear Regression, metrics, cross-validation |
| XGBoost | Gradient boosting model |
| pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Jupyter Notebook | Development environment |

---

## 👤 Author

**Krishna Maniyar** — Data Analyst
- 🎓 Pace University — Seidenberg School of CSIS, MS in Data Science
- 📘 CS675: Introduction to Data Science
- 📧 krishnamaniyarkm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for CS675 @ Pace University
</p>
