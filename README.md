# 🩺 Diabetes Disease Progression Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Linear%20Regression-red?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS675-blue)

> A machine learning project that predicts the **progression of diabetes** one year after baseline using **Linear Regression** and **XGBoost** — built for CS675: Introduction to Data Science at Pace University.

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

This project implements and evaluates **Linear Regression** models using the classic **Diabetes dataset** from Stanford University's Machine Learning Repository. The goal is to predict the quantitative measure of disease progression one year after baseline using patient health features.

The notebook covers:
- ✅ **Exploratory Data Analysis (EDA)** — distributions, correlations, missing values
- ✅ **Single-feature regression** — finding the best individual predictor
- ✅ **Pair-feature regression** — finding the best two-feature combination
- ✅ **Full-feature regression** — using all 10 features simultaneously
- ✅ **Learning curve analysis** — training vs validation MSE at varying dataset sizes
- ✅ **XGBoost comparison** — gradient boosting vs linear regression (extra credit)

---

## 📊 Dataset

**Source:** [Stanford University — Diabetes Dataset (LARS Paper)](https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data)

**Reference Paper:**
> Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004)
> *"Least Angle Regression"*, Annals of Statistics, 407–499.
> [https://projecteuclid.org/euclid.aos/1083178935](https://projecteuclid.org/euclid.aos/1083178935)

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
| `s1` | TC — T-Cells (white blood cells) |
| `s2` | LDL — Low-density lipoproteins |
| `s3` | HDL — High-density lipoproteins |
| `s4` | TCH — Thyroid stimulating hormone |
| `s5` | LTG — Lamotrigine |
| `s6` | GLU — Blood sugar level |
| `y` | **Target** — Quantitative disease progression |

---

## 📁 Project Structure
```
diabetes-progression-prediction/
│
├── Diabetes_Progression_Prediction.ipynb   # Main Jupyter notebook
├── README.md                               # Project documentation
└── requirements.txt                        # Python dependencies
```

---

## 🔬 Methodology

### Step 1 — Exploratory Data Analysis (EDA)
- Dataset shape, column inspection, and data types
- Descriptive statistics (`describe()`)
- Missing value check
- Histograms for all features
- Boxplots for outlier detection
- Correlation heatmap
- Feature-target correlation ranking

### Step 2 — Single Feature Regression
Trains an individual Linear Regression model for each of the 10 features and identifies the one with the **lowest MSE**:
```python
for feature in X.columns:
    model = LinearRegression()
    model.fit(X[[feature]], y)
    mse = mean_squared_error(y, model.predict(X[[feature]]))
```

### Step 3 — Pair Feature Regression
Tests all **45 possible pairs** of features using `itertools.combinations` and identifies the best-performing pair:
```python
for pair in combinations(X.columns, 2):
    model = LinearRegression()
    model.fit(X[list(pair)], y)
    mse = mean_squared_error(y, model.predict(X[list(pair)]))
```

### Step 4 — All Features Regression
Fits a single Linear Regression model using all 10 features simultaneously and reports all coefficients.

### Step 5 — Learning Curve Analysis
Computes training MSE and validation MSE for four different training set sizes to study the bias-variance tradeoff:

| Training Size | Notes |
|---|---|
| n = 20 | Very small — high variance expected |
| n = 50 | Small |
| n = 100 | Medium |
| n = 200 | Larger — more stable |

### Step 6 — XGBoost (Extra Credit)
Trains an XGBoost regressor and compares it against Linear Regression using test MSE, R², residual analysis, and 5-fold cross-validation.

---

## 📈 Models & Results

### Task 1 — Best Single Feature

| Best Feature | Slope | Intercept | MSE |
|---|---|---|---|
| `bmi` | 949.44 | 152.13 | ~3890 |

### Task 2 — Best Feature Pair

| Feature 1 | Feature 2 | Intercept | MSE |
|---|---|---|---|
| `bmi` | `s5` | 152.13 | ~3330 |

### Task 3 — All 10 Features

| Metric | Value |
|---|---|
| MSE | ~2859 |
| R² | ~0.518 |
| Intercept | 152.13 |

### Task 4 — Training vs Validation MSE

| Training Size | Training MSE | Validation MSE |
|---|---|---|
| 20 | Very low | Very high |
| 50 | Low | High |
| 100 | Medium | Medium |
| 200 | Higher | Lower |

> As training size increases, training MSE rises and validation MSE falls — classic bias-variance tradeoff behavior.

---

## 🚀 Extra Credit — XGBoost

An XGBoost Regressor was trained and compared against Linear Regression:

### Test Set Performance

| Model | Test MSE | Test R² |
|---|---|---|
| Linear Regression | ~2900 | ~0.47 |
| XGBoost | ~2700 | ~0.51 |

### Additional Analysis
- 📊 **Predicted vs Actual scatter** — both models plotted together
- 📊 **Feature Importance** — XGBoost identifies `bmi` and `s5` as top predictors
- 📊 **Residual plots** — scatter and distribution for both models
- 📊 **5-Fold Cross-Validation** — mean and std of MSE and R² for both models

### Cross-Validation Summary

| Model | Mean CV MSE | Std CV MSE | Mean CV R² |
|---|---|---|---|
| Linear Regression | ~3000 | ~400 | ~0.49 |
| XGBoost | ~2800 | ~350 | ~0.52 |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Clone & Setup
```bash
# Clone the repository
git clone https://github.com/krishnamaniyar2209/diabetes-progression-prediction.git

# Navigate to the project folder
cd diabetes-progression-prediction

# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Diabetes_Progression_Prediction.ipynb
```

### requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
xgboost>=1.7.0
jupyter>=1.0.0
ipykernel>=6.0.0
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
4. All plots, coefficient tables, and metrics are generated automatically

---

## 💡 Key Findings

- **`bmi`** (Body Mass Index) is the single strongest predictor of diabetes progression, consistent with medical literature
- **`bmi` + `s5`** (LTG/Lamotrigine) is the best-performing feature pair, reducing MSE significantly over any single feature
- Adding all 10 features further reduces MSE to ~2859, showing that multiple health indicators together provide better predictions
- **Learning curves** confirm classic bias-variance behavior — small training sets overfit badly, larger sets generalize better
- **XGBoost** outperforms Linear Regression on both test MSE and cross-validated R², but the margin is modest — suggesting the relationship between features and disease progression is largely linear
- **Residual analysis** shows both models have roughly normally distributed errors, confirming linear regression assumptions are reasonably met

---

## 🛠️ Technologies Used

| Tool | Version | Purpose |
|---|---|---|
| [Python](https://python.org) | 3.10+ | Core language |
| [scikit-learn](https://scikit-learn.org/) | Latest | Linear Regression, metrics, CV |
| [XGBoost](https://xgboost.readthedocs.io/) | Latest | Gradient boosting model |
| [pandas](https://pandas.pydata.org/) | Latest | Data manipulation |
| [NumPy](https://numpy.org/) | Latest | Numerical operations |
| [Matplotlib](https://matplotlib.org/) | Latest | Plotting |
| [Seaborn](https://seaborn.pydata.org/) | Latest | Statistical visualization |
| [Jupyter Notebook](https://jupyter.org/) | Latest | Development environment |

---

## 👤 Author

**Krishna Maniyar**
- 🎓 Pace University — Seidenberg School of CSIS
- 📘 CS675: Introduction to Data Science (Fall 2024)
- 🔗 [GitHub](https://github.com/krishnamaniyar2209)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for CS675 @ Pace University
  <br><br>
  <img src="https://img.shields.io/badge/Pace%20University-Seidenberg%20School%20of%20CSIS-blue" />
</p>
