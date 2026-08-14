# 🩺 Diabetes Disease Progression Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Linear%20Regression-red?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![University](https://img.shields.io/badge/Pace%20University-CS675-blue)

> A machine learning project predicting the **progression of diabetes** one year after baseline using **Linear Regression** and **XGBoost**, built for CS675: Introduction to Data Science at Pace University.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Highlights](#-highlights)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Extra Credit: XGBoost](#-extra-credit-xgboost)
- [Limitations & Next Steps](#-limitations--next-steps)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Key Findings](#-key-findings)
- [Technologies Used](#️-technologies-used)
- [Author](#-author)

---

## 🔬 Overview

This project implements and evaluates **Linear Regression** models on the classic **Diabetes dataset** (Efron, Hastie, Johnstone & Tibshirani, *Least Angle Regression*, 2004). The goal is to predict a quantitative measure of disease progression one year after baseline from 10 patient health features, then benchmark a gradient-boosted model (XGBoost) against the linear baseline.

The notebook covers:
- ✅ **Exploratory Data Analysis (EDA)**, distributions, correlations, missing values
- ✅ **Single-feature regression**, finding the best individual predictor
- ✅ **Pair-feature regression**, best two-feature combination (all 45 pairs)
- ✅ **Full-feature regression**, all 10 features together
- ✅ **Learning-curve analysis**, training vs. validation MSE across dataset sizes
- ✅ **XGBoost comparison**, gradient boosting vs. linear regression with test split and 5-fold CV (extra credit)

---

## ✨ Highlights

- Showed that a **more complex model isn't automatically better**: on 5-fold cross-validation, plain Linear Regression beat XGBoost on 4 of 5 folds (R² 0.478 vs. 0.423) with 31% lower fold-to-fold variance — and explained why, rather than just reporting the win: 442 rows and a largely linear signal give gradient boosting nothing to exploit and room to overfit.
- Caught a subtle evaluation flaw in its own methodology: the test-set split and cross-validation fold 1 turned out to be identical partitions (same `random_state=42`), which would have made the two "independent" evaluations non-independent. Flagged it explicitly rather than letting the CV mean look more validated than it was.
- Systematically searched every one of the **45 possible feature pairs** (not just the two best features individually) to find the true best two-feature model, and explained *why* the winning pair worked (`bmi` for magnitude, `s5` for the variance `bmi` alone can't explain).
- Distinguished in-sample fit from genuine predictive accuracy throughout — explicitly separating "this feature-selection MSE describes fit quality" from "this cross-validated number is what to trust for predictive performance," a distinction a lot of student projects blur.

---

## 🎥 Demo

*The best-pair scatter plot (predicted vs. actual for `bmi` + `s5`) and the learning-curve chart are the most immediately informative visuals here — either would work well as a preview image for the repo.*

```
![Best model fit and learning curve](docs/results_preview.png)
```

---

## 📊 Dataset

**Source:** [Stanford University, Diabetes Dataset (LARS Paper)](https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data)

**Reference Paper:**
> Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004)
> *"Least Angle Regression"*, Annals of Statistics, 407-499.

| Property | Details |
|---|---|
| Records | 442 patients |
| Features | 10 numeric predictors |
| Target | `y`, disease progression 1 year after baseline |
| Missing Values | None |

This is the **raw, unstandardized** version of the dataset from Stanford, not the mean-centered and scaled version bundled with scikit-learn. Coefficients below are therefore on the original feature scales and are not directly comparable in magnitude.

### Feature Descriptions

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex |
| `bmi` | Body Mass Index |
| `bp` | Average blood pressure |
| `s1` | TC, Total serum cholesterol |
| `s2` | LDL, Low-density lipoproteins |
| `s3` | HDL, High-density lipoproteins |
| `s4` | TCH, Total cholesterol / HDL ratio |
| `s5` | LTG, Log of serum triglycerides level |
| `s6` | GLU, Blood sugar level |
| `y` | **Target**, quantitative disease progression |

### Feature Correlation with Target

| Feature | Correlation with `y` |
|---|---|
| `bmi` | **0.586** |
| `s5` | **0.566** |
| `bp` | 0.441 |
| `s4` | 0.430 |
| `s3` | -0.395 |
| `s6` | 0.382 |
| `s1` | 0.212 |
| `age` | 0.188 |
| `s2` | 0.174 |
| `sex` | 0.043 |

Note that `s1` and `s2` correlate at 0.897 with each other, and `s3` with `s4` at -0.738. This multicollinearity is why several coefficients in the full model carry counterintuitive signs.

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

### Step 1: Exploratory Data Analysis (EDA)
Shape and dtype inspection, descriptive statistics, missing-value check, per-feature histograms and boxplots, a correlation heatmap, and a feature-target correlation ranking.

### Step 2: Single Feature Regression
Trains a Linear Regression model for each of the 10 features and selects the one with the **lowest MSE**:
```python
for feature in X.columns:
    model = LinearRegression()
    model.fit(X[[feature]], y)
    mse = mean_squared_error(y, model.predict(X[[feature]]))
```

### Step 3: Pair Feature Regression
Tests all **45 feature pairs** via `itertools.combinations` and selects the best pair:
```python
for pair in combinations(X.columns, 2):
    model = LinearRegression()
    model.fit(X[list(pair)], y)
    mse = mean_squared_error(y, model.predict(X[list(pair)]))
```

### Step 4: All Features Regression
Fits one Linear Regression model on all 10 features and reports every coefficient.

### Step 5: Learning Curve Analysis
Computes training and validation MSE at four training-set sizes (n = 20, 50, 100, 200) to study the bias-variance tradeoff.

### Step 6: XGBoost (Extra Credit)
Trains an XGBoost regressor (`n_estimators=100, learning_rate=0.1, max_depth=3`) and compares it to Linear Regression via an 80/20 test split, R², residual analysis, and 5-fold cross-validation.

> ⚠️ **On evaluation.** Tasks 1 through 4 fit and score on the **full 442-row dataset**, so their MSE and R² describe *fit quality*, not predictive accuracy. This is the intended scope of those tasks, which are about feature selection and model form rather than generalization. The XGBoost section (Step 6) uses a proper held-out test set and 5-fold cross-validation, and those are the numbers to read as predictive performance.

---

## 📈 Models & Results

### Task 1: Best Single Feature

All 10 features ranked by in-sample MSE:

| Feature | Slope | Intercept | MSE | R² |
|---|---|---|---|---|
| **`bmi`** | **10.23** | **-117.77** | **3890.46** | **0.344** ✅ |
| `s5` | 83.51 | -235.48 | 4031.00 | 0.320 |
| `bp` | 2.46 | -80.77 | 4774.11 | 0.195 |
| `s4` | 25.72 | 47.46 | 4831.14 | 0.185 |
| `s3` | -2.35 | 269.29 | 5005.66 | 0.156 |
| `s6` | 2.56 | -81.94 | 5062.38 | 0.146 |
| `s1` | 0.47 | 62.80 | 5663.32 | 0.045 |
| `age` | 1.10 | 98.52 | 5720.55 | 0.035 |
| `s2` | 0.44 | 101.20 | 5750.24 | 0.030 |
| `sex` | 6.65 | 142.38 | 5918.89 | 0.002 |

### Task 2: Best Feature Pair

Top 5 of all 45 combinations:

| Feature 1 | Feature 2 | Intercept | Coef 1 | Coef 2 | MSE | R² |
|---|---|---|---|---|---|---|
| **`bmi`** | **`s5`** | **-299.96** | **7.28** | **56.06** | **3205.19** | **0.459** ✅ |
| `bmi` | `bp` | -203.62 | 8.52 | 1.38 | 3581.69 | 0.396 |
| `bmi` | `s4` | -129.72 | 8.60 | 13.54 | 3638.17 | 0.386 |
| `bmi` | `s3` | -21.12 | 8.90 | -1.24 | 3669.26 | 0.381 |
| `bp` | `s5` | -302.28 | 1.44 | 68.48 | 3695.05 | 0.377 |

`bmi` appears in **eight of the ten** best-performing pairs, confirming it as the dominant predictor. The two exceptions in the top ten, `bp`+`s5` and `s3`+`s5`, both pair a moderate predictor with `s5`, which is the second-strongest feature on its own.

### Task 3: All 10 Features

| Metric | Value |
|---|---|
| Intercept | -334.57 |
| MSE | 2859.70 |
| R² | 0.518 |

| Feature | Coefficient |
|---|---|
| `s5` | 68.48 |
| `s4` | 6.53 |
| `bmi` | 5.60 |
| `bp` | 1.12 |
| `s2` | 0.75 |
| `s3` | 0.37 |
| `s6` | 0.28 |
| `age` | -0.04 |
| `s1` | -1.09 |
| `sex` | -22.86 |

### Task 4: Learning Curve (Training vs. Validation MSE)

| Training Size | Validation Size | Training MSE | Validation MSE | Ratio |
|---|---|---|---|---|
| 20 | 422 | 2,066 | 15,906 | 7.7x |
| 50 | 392 | 2,864 | 3,907 | 1.4x |
| 100 | 342 | 3,151 | 3,423 | 1.1x |
| 200 | 242 | 2,855 | 3,015 | 1.1x |

> Tiny training sets overfit dramatically: at n=20, validation MSE is nearly 8 times training MSE. That size fits 10 predictors plus an intercept on 20 observations — barely two rows per parameter — so the model is near-singular and memorizes its training sample. By n=200 the gap has essentially closed. Textbook bias-variance behavior, and a direct illustration of why 442 rows is a small dataset.
>
> **Caveat:** `train_test_split(train_size=size)` assigns everything not used for training to validation, so the validation set shrinks as the training set grows (422 → 242 rows). The training sets are nested, but each row's validation MSE is measured on a different sample, so the four figures show a real trend without being strictly like-for-like. A fixed holdout with `learning_curve()` would make them directly comparable.

---

## 🚀 Extra Credit: XGBoost

An XGBoost Regressor was trained and benchmarked against Linear Regression on a held-out test set and via 5-fold cross-validation.

### Test Set Performance (80/20 split)

| Model | Test MSE | Test R² |
|---|---|---|
| **Linear Regression** | **2900.19** | **0.453** ✅ |
| XGBoost | 2959.46 | 0.441 |

### 5-Fold Cross-Validation

| Fold | LinReg MSE | LinReg R² | XGBoost MSE | XGBoost R² |
|---|---|---|---|---|
| 1 | 2900.19 | 0.453 | 2959.46 | 0.441 |
| 2 | 2662.64 | 0.573 | 2968.17 | 0.524 |
| 3 | 3312.31 | 0.391 | 4107.07 | 0.245 |
| 4 | 2797.88 | 0.584 | 3482.12 | 0.483 |
| 5 | 3403.89 | 0.391 | 3233.65 | 0.421 |
| **Mean** | **3015.38** | **0.478** ✅ | 3350.09 | 0.423 |
| **Std** | **291.24** | | 424.85 | |

Linear Regression wins on **every fold except one**, and its fold-to-fold standard deviation is 31% lower (291 vs. 425). XGBoost's worst fold collapses to R² 0.245. On 442 rows with a largely linear feature-target relationship, the extra capacity buys nothing and costs stability.

> **Note:** `train_test_split(random_state=42)` and `KFold(shuffle=True, random_state=42)` happen to produce the same partition, so the test-set result above is identical to CV fold 1. The two are not independent evaluations. The 5-fold mean is the figure to quote.

### Additional Analysis
- 📊 **Predicted vs. Actual** scatter for both models
- 📊 **Feature importance**, XGBoost ranks `bmi` (0.317) and `s5` (0.217) highest, consistent with the linear analysis
- 📊 **Residual plots**, roughly normal errors for both models

---

## ⚠️ Limitations & Next Steps

1. **Tasks 1 to 3 report in-sample fit.** Feature and pair selection is performed using training MSE on the same 442 rows used for fitting. With 45 pair candidates this biases the selection slightly toward whichever pair best fits noise. Nested cross-validation would give an unbiased estimate of the selection procedure itself.
2. **Multicollinearity is unaddressed.** `s1` and `s2` correlate at 0.897, `s3` and `s4` at -0.738. This destabilizes coefficient estimates in the full model and explains the negative `s1` coefficient despite its positive correlation with the target. Ridge or Lasso would be the natural next step, and the LARS paper this dataset comes from exists precisely to address that problem.
3. **XGBoost was not tuned.** A single hyperparameter configuration was tested. The conclusion that linear regression wins is sound for these settings but a grid search over `max_depth`, `learning_rate`, and `n_estimators` would make the comparison fairer.
4. **The learning curve uses a varying validation set.** Validation size falls from 422 to 242 rows as training size grows, so the curve is indicative rather than a controlled measurement. `sklearn.model_selection.learning_curve()` with a fixed CV strategy would fix this.
5. **442 rows is small.** The learning curve shows validation error still declining at n=200, so more data would likely help more than any modeling change.
6. **No feature engineering.** Interaction terms and polynomial features are untested, and BMI in particular often shows nonlinear effects in clinical literature.

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
2. The dataset loads **automatically** from Stanford's URL, no download needed:
```python
df = pd.read_csv(
    "https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data",
    sep=r"\s+", header=0, names=column_names
)
```
3. Run all cells sequentially from top to bottom

---

## 💡 Key Findings

- **`bmi`** is the single strongest predictor of disease progression (R² 0.34), consistent with medical literature
- **`bmi` + `s5`** (log triglycerides) is the best of all 45 feature pairs, cutting MSE from 3890 to 3205, and `bmi` appears in eight of the ten best-performing pairs
- Using **all 10 features** gives the best in-sample linear fit (MSE 2860, R² 0.52), though honest cross-validated performance lands lower at R² 0.478
- **Learning curves** confirm bias-variance behavior: at n=20 validation error is 8 times training error, closing to near parity by n=200
- **Linear Regression outperformed XGBoost** on 5-fold CV (R² 0.478 vs. 0.423), winning four of five folds with 31% lower variance. On a small, largely linear dataset, gradient boosting overfits. The more complex model is not automatically the better one
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

**Krishna Maniyar**, Data Analyst
- 🎓 Pace University, Seidenberg School of CSIS, MS in Data Science
- 📘 CS675: Introduction to Data Science
- 📧 maniyarkrishnakm22@gmail.com
- 🔗 [GitHub](https://github.com/krishnamaniyar2209) · [LinkedIn](https://www.linkedin.com/in/krishnamaniyar/) · [Portfolio](https://krishnamaniyar2209.github.io/)

---

<p align="center">
  Made with ❤️ for CS675 @ Pace University
</p>
