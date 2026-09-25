# Module 2 — Analytics (`/analytics`)

Titanic exploratory data analysis and predictive modeling. This module covers dataset profiling, missing-value handling, univariate and bivariate analysis, a multivariate survival data story, exploratory standardization, three classification models, class-imbalance comparison, Random Forest hyperparameter tuning, a linear-regression side task for fare prediction, and a saved end-to-end inference pipeline.

---

## Setup

Python 3.9+ is required.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

Required packages:

```text
pandas
numpy
seaborn
matplotlib
scikit-learn
imbalanced-learn
joblib
```

---

## Module Structure

```text
analytics/
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── titanic.csv
├── titanic_survival_pipeline.joblib
├── requirements.txt
├── README.md
└── charts/
```

`01_eda.ipynb` contains Part A: profiling, cleaning, univariate/bivariate analysis, the multivariate data story, and the exploratory standardization check.

`02_modeling.ipynb` contains Part B: stratified train/test splitting, leakage-safe preprocessing, three classifiers, imbalance handling, Random Forest tuning, regression, model comparison, and pipeline persistence.

`titanic.csv` is the committed offline fallback copy of the raw dataset.

`titanic_survival_pipeline.joblib` is the fitted end-to-end classification pipeline containing both preprocessing and the final tuned Random Forest estimator.

---

# Part A — Profiling, Cleaning, and the Data Story

## 1. Dataset Loading and Profiling

The raw Titanic dataset is loaded once in `01_eda.ipynb` using:

```python
df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)
```

The resulting `titanic.csv` is committed inside `/analytics` as the offline fallback.

If the online load is unavailable, `01_eda.ipynb` reads the committed CSV instead.

The raw dataset contains:

* **891 rows**
* **15 columns**

The notebook reports:

```python
df.info()
df.describe()
df.shape
```

The raw dataset contains missing values in four columns:

| Column        | Missing values | Missing % |
| ------------- | -------------: | --------: |
| `deck`        |            688 |    77.22% |
| `age`         |            177 |    19.87% |
| `embarked`    |              2 |     0.22% |
| `embark_town` |              2 |     0.22% |

The modeling notebook does not reload the raw dataset from the network. It reads the committed `titanic.csv` instead and reapplies the cleaning steps.

---

## 2. Missing-Value Handling

The missing-value strategy follows the required percentage-based threshold rule.

| Column        | Missing % | Strategy                      | Justification                                                                                                                  |
| ------------- | --------: | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `embarked`    |     0.22% | Drop affected rows            | Below 5%, so the affected rows are removed rather than imputed.                                                                |
| `embark_town` |     0.22% | Drop affected rows            | Below 5%; the same two rows are missing both embarkation fields.                                                               |
| `age`         |    19.87% | Median imputation             | Falls in the 5–30% range. The median value of 28.0 is used.                                                                    |
| `deck`        |    77.22% | Encode missing as `"Missing"` | Missingness is too high for reliable value imputation, so it is retained as its own category instead of inventing deck values. |

After cleaning:

```text
Rows:    889
Columns: 15
Remaining null values: 0
```

The two rows missing `embarked` and `embark_town` are removed. The 177 missing `age` values are replaced by the median age of 28.0. The 688 missing `deck` values are encoded as `"Missing"`.

---

## 3. Univariate Analysis

Histograms and box plots are produced for both `age` and `fare`.

### Age

Using the 1.5×IQR rule:

```text
Q1 = 22.00
Q3 = 35.00
IQR = 13.00
Lower bound = 2.50
Upper bound = 54.50
```

**IQR outliers: 65**

The outlier rule flags observations below 2.5 or above 54.5 years.

### Fare

Using the same IQR rule:

```text
Q1 = 7.90
Q3 = 31.00
IQR = 23.10
Lower bound = -26.76
Upper bound = 65.66
```

**IQR outliers: 114**

All identified fare outliers are on the high side because fares above £65.66 exceed the upper IQR bound.

### Fare statistics

```text
Mean   = £32.10
Median = £14.45
Mode   = £8.05
```

Because:

```text
Mean > Median > Mode
```

the fare distribution is **right-skewed (positively skewed)**. Most passengers paid relatively low fares, while a smaller number of expensive tickets pull the mean upward.

---

## 4. Bivariate Analysis

### Survival rate by sex

Boolean masks are used to separate male and female passengers.

| Sex    | Survival rate |
| ------ | ------------: |
| Male   |        18.89% |
| Female |        74.04% |

Survival was substantially higher among female passengers than male passengers in the cleaned dataset.

### Survival rate by passenger class

| Passenger class | Survival rate |
| --------------- | ------------: |
| 1st             |        62.62% |
| 2nd             |        47.28% |
| 3rd             |        24.24% |

Observed survival decreased from 1st to 3rd class.

### Survival rate by sex and passenger class

Boolean `&` masking is used to combine sex and passenger-class conditions.

| Sex    | 1st class | 2nd class | 3rd class |
| ------ | --------: | --------: | --------: |
| Female |    96.74% |    92.11% |    50.00% |
| Male   |    36.89% |    15.74% |    13.54% |

Survival was higher for females than males within every class. Survival also generally decreased from 1st to 3rd class for both sexes.

---

## Correlation Analysis

The correlation matrix is calculated using exactly these six numeric columns:

```python
[
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]
```

The boolean-derived columns `adult_male` and `alone` are intentionally excluded because they are derived/redundant indicators rather than independent measured features.

### Two strongest absolute off-diagonal correlations

The feature pairs are ranked by the absolute value of their Pearson correlation coefficient.

1. **`pclass` ↔ `fare`: r = -0.548**

   This is the strongest absolute correlation in the six-feature matrix. Higher numerical class values correspond to lower fares, so passenger class and fare contain related information about the socioeconomic structure represented in the dataset.

2. **`sibsp` ↔ `parch`: r = +0.415**

   Passengers traveling with more siblings/spouses also tended to travel with more parents/children. The two family-count variables therefore show a moderate positive association.

The target `survived` has its strongest correlations with `pclass` (`r ≈ -0.34`) and `fare` (`r ≈ +0.26`), indicating that passenger class and fare are associated with survival.

---

# Multivariate Data Story

Five distinct visualizations are used to build a coherent picture of survival differences.

### Chart 1 — Survival by passenger class and sex

Survival was substantially higher for female passengers than male passengers within every passenger class. Passenger class also mattered within each sex, with survival generally decreasing from 1st to 3rd class. This shows a strong association between sex, class, and survival.

### Chart 2 — Fare by survival outcome

Survivors paid a higher median fare than non-survivors: approximately £26.00 versus £10.50. Fare is also associated with passenger class, so the relationship should be interpreted as an association rather than evidence that paying a higher fare itself caused survival.

### Chart 3 — Age versus fare colored by survival

Survivors appear across a broad range of ages, so there is no single age threshold that separates survivors from non-survivors. Survivors are also more concentrated toward higher fares, consistent with the observed relationship between survival, fare, and passenger class.

### Chart 4 — Survival heatmap by passenger class and sex

The heatmap provides a compact view of the combined sex/class pattern. Female 1st-class passengers had the highest observed survival rate at 96.74%, while male 3rd-class passengers had the lowest at 13.54%.

### Chart 5 — Survival by family size

Survival varies non-linearly with family size. Passengers traveling with small numbers of family members had higher observed survival rates than solo travelers and several larger-family groups, indicating an association rather than establishing a causal effect.

The notebooks display these charts inline; chart image files may also be stored under `charts/` as supporting artifacts.

---

# Exploratory Standardization Check

As an EDA sanity check, `age` and `fare` are standardized using:

```text
z = (x - mean) / std
```

using `StandardScaler`.

Before transformation:

```text
age:
    mean ≈ 29.32
    std  ≈ 14.50

fare:
    mean ≈ 32.10
    std  ≈ 49.70
```

After transformation:

```text
age_zscore:
    mean ≈ 0
    std  ≈ 1

fare_zscore:
    mean ≈ 0
    std  ≈ 1
```

This standardization is an **EDA-only sanity check**. These transformed columns are not used as precomputed inputs to the classification pipeline.

---

# Part B — Predictive Modeling

## 1. Stratified Train/Test Split

The cleaned dataset contains:

```text
Not survived: 549 (61.75%)
Survived:     340 (38.25%)
```

A stratified 80/20 train/test split is used:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This produces:

```text
Training rows: 711
Testing rows: 178
```

Stratification is important because the target classes are not evenly balanced. It preserves approximately the same survived/not-survived proportion in both training and test sets, making evaluation more representative of the original dataset.

---

## 2. Modeling Features and Preprocessing

The classification features are:

### Numeric

```text
pclass
age
sibsp
parch
fare
```

### Categorical

```text
sex
embarked
```

Excluded features include redundant or label-leaking variables such as:

```text
alive
class
embark_town
who
adult_male
alone
deck
```

The preprocessing is implemented using a `ColumnTransformer` inside a scikit-learn `Pipeline`.

### Numeric preprocessing

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler()
```

### Categorical preprocessing

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing is fit only during `.fit(X_train, y_train)` and is subsequently applied to `X_test` through transformation. No preprocessing step is fit on the test set or on the full dataset before splitting.

This structure also makes the saved pipeline usable with raw future input.

---

# 3. Classification Models

Three classifiers are trained on the same stratified train/test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The Decision Tree is constrained to:

```text
max_depth = 5
```

For visualization, only the top three levels are displayed using `plot_tree` so that the tree remains readable.

---

# 4. Classifier Evaluation

Each classifier is evaluated using:

* Confusion matrix
* Accuracy
* Precision
* Recall
* F1 score
* ROC curve
* AUC

### Classification results

| Model               | Accuracy | Precision | Recall |     F1 |    AUC |
| ------------------- | -------: | --------: | -----: | -----: | -----: |
| Logistic Regression |    0.809 |    0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree       |    0.764 |    0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest       |    0.809 |    0.7656 | 0.7206 | 0.7424 | 0.8196 |

Logistic Regression and Random Forest have the same accuracy on the held-out test set. Logistic Regression has the highest AUC, while Random Forest has the highest baseline recall and F1 among the three classifiers.

---

# 5. Class-Imbalance Comparison

The target distribution is:

```text
0 / Not survived → 549 (61.75%)
1 / Survived     → 340 (38.25%)
```

Random Forest is evaluated using three approaches.

| Strategy                   | Precision | Recall |     F1 |
| -------------------------- | --------: | -----: | -----: |
| Baseline                   |    0.7656 | 0.7206 | 0.7424 |
| `class_weight="balanced"`  |    0.7656 | 0.7206 | 0.7424 |
| SMOTE — training fold only |    0.7612 | 0.7500 | 0.7556 |

`class_weight="balanced"` produced the same reported metrics as the baseline in this experiment.

SMOTE produced a small increase in recall and F1 while causing a small decrease in precision. SMOTE is applied through an imbalanced-learn pipeline, so the oversampling operation is performed only during training and does not modify the held-out test set.

---

# 6. Random Forest Hyperparameter Tuning

`GridSearchCV` is applied to Random Forest using:

```text
n_estimators: [100, 200, 300]
max_depth: [None, 5, 10]
max_features: ['sqrt', 'log2']
```

The search uses:

```text
5-fold cross-validation
scoring = F1
```

The Random Forest estimator is constructed with:

```python
oob_score=True
bootstrap=True
```

### Best parameters

```text
n_estimators = 300
max_depth = None
max_features = 'sqrt'
```

### Tuning results

```text
Best 5-fold CV F1 = 0.7449
OOB score         = 0.8073
```

The OOB score is the Random Forest's out-of-bag score and should not be numerically compared directly with the cross-validated F1 because they use different scoring definitions.

On the held-out test set, the tuned Random Forest achieved:

```text
Accuracy  = 0.8034
Precision = 0.7619
Recall    = 0.7059
F1        = 0.7328
AUC       = 0.8237
```

The tuned model's test-set F1 is slightly below the untuned Random Forest's test-set F1 on this particular split, illustrating that a single held-out split can vary from the cross-validation estimate.

---

# 7. Regression Side Task — Predicting Fare

A separate multivariate linear regression predicts `fare`.

### Predictors

Numeric:

```text
pclass
age
sibsp
parch
survived
```

Categorical:

```text
sex
embarked
```

The regression uses its own preprocessing pipeline with:

* median imputation + standardization for numeric features
* most-frequent imputation + one-hot encoding for categorical features

The regression is evaluated on a separate 80/20 train/test split because the original classification stratification is designed around the binary `survived` target.

### Regression metrics

```text
MAE          = £21.10
RMSE         = £41.70
R²           = 0.3482
Adjusted R²  = 0.3091
```

The model explains approximately 35% of the variance in fare, so the available features capture some, but not all, of the variation in ticket price.

### Residual analysis

The residual plot shows **heteroscedasticity** rather than a constant-width random residual spread.

The residual standard deviation is approximately:

```text
Low-predicted-fare half:  13.14
High-predicted-fare half: 56.88
```

The correlation between predicted fare and absolute residual is approximately:

```text
0.337
```

The increasing spread of residuals at higher predicted fares indicates non-constant error variance. This is consistent with the strong right-skew observed in the original fare distribution.

---

# 8. Final Model Comparison

Classification and regression metrics are kept as separate metric groups because they represent different tasks and are not directly comparable on a common scale.

| Model Type     | Model                    | Accuracy | Precision | Recall |     F1 |    AUC |     MAE |    RMSE |     R² | Adjusted R² |
| -------------- | ------------------------ | -------: | --------: | -----: | -----: | -----: | ------: | ------: | -----: | ----------: |
| Classification | Logistic Regression      |    0.809 |    0.7833 | 0.6912 | 0.7344 | 0.8610 |       — |       — |      — |           — |
| Classification | Decision Tree            |    0.764 |    0.7600 | 0.5588 | 0.6441 | 0.8374 |       — |       — |      — |           — |
| Classification | Random Forest            |    0.809 |    0.7656 | 0.7206 | 0.7424 | 0.8196 |       — |       — |      — |           — |
| Regression     | Linear Regression — fare |        — |         — |      — |      — |      — | 21.0986 | 41.7021 | 0.3482 |      0.3091 |

---

# 9. Final Recommendation

The **tuned Random Forest** is selected as the deployment model because it provides a strong overall classification balance and is supported by cross-validation and out-of-bag evaluation. Its 5-fold cross-validated F1 is 0.7449 and its OOB score is 0.8073, while its held-out test set achieves 0.803 accuracy, 0.762 precision, 0.706 recall, 0.733 F1, and 0.824 AUC. Logistic Regression achieves the highest test-set AUC at 0.8610, so it remains a strong alternative when ranking performance by discrimination alone. The saved deployment artifact is therefore the tuned Random Forest pipeline, including all preprocessing steps required to accept raw input.

---

# 10. Saved End-to-End Pipeline

The final model is saved as:

```text
titanic_survival_pipeline.joblib
```

The saved object is the **complete fitted scikit-learn Pipeline**, not the bare Random Forest estimator.

It contains:

```text
Raw input
   ↓
ColumnTransformer
   ├── Numeric imputation
   ├── Numeric scaling
   ├── Categorical imputation
   └── One-hot encoding
   ↓
Tuned Random Forest
   ↓
Prediction
```

The pipeline is saved with:

```python
joblib.dump(
    tuned_rf_pipeline,
    "titanic_survival_pipeline.joblib"
)
```

It is reloaded with:

```python
loaded_pipeline = joblib.load(
    "titanic_survival_pipeline.joblib"
)
```

The reloaded pipeline reproduces the original pipeline's predictions on raw test rows.

It is also tested with a new raw passenger record containing:

```text
pclass
age
sibsp
parch
fare
sex
embarked
```

This confirms that the saved artifact can perform end-to-end prediction directly from raw, unpreprocessed input.

---

## Execution Order

Run the module in this order:

```text
01_eda.ipynb
    ↓
Load raw Titanic dataset once
    ↓
Immediately save analytics/titanic.csv
    ↓
Profile + clean + analyze
    ↓
02_modeling.ipynb
    ↓
Read analytics/titanic.csv
    ↓
Reapply required cleaning
    ↓
Stratified train/test split
    ↓
Train-only preprocessing
    ↓
3 classifiers
    ↓
Imbalance comparison
    ↓
Random Forest GridSearchCV + OOB
    ↓
Fare regression
    ↓
Final comparison
    ↓
Save complete pipeline
    ↓
Reload and verify predictions
```

The committed `titanic.csv` is the single offline snapshot used by the modeling stage; the modeling notebook must not call `sns.load_dataset()` again.
