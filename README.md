# Mock2 - Machine Quality Data Science Project

## Project Overview

This notebook is a practical Data Science and Machine Learning workflow
based on a machine quality dataset.

The dataset contains machine-related measurements and a binary `defect`
target. The notebook covers data generation, data checking, statistics,
preprocessing, feature engineering, supervised learning, and
unsupervised learning.

## Dataset

The notebook creates a synthetic dataset with **300 original records**
and temporarily adds **5 duplicate rows**, giving **305 rows** before
cleaning.

### Columns

  Column          Description
  --------------- ---------------------------------
  `record_id`     Unique record identifier
  `temperature`   Machine temperature measurement
  `vibration`     Machine vibration measurement
  `pressure`      Machine pressure measurement
  `hours`         Machine operating hours
  `group`         Machine group (`G1` or `G2`)
  `defect`        Binary target: `0` or `1`

Missing values are introduced in `temperature` and `vibration`.

## Main Workflow

### 1. Data Generation and Loading

-   Creates a synthetic dataset using NumPy.
-   Stores the raw data in `data/raw/set_c.csv`.
-   Loads the CSV using pandas.
-   Checks the shape and first few records.

### 2. Initial Data Analysis

The notebook checks:

-   Dataset shape
-   First rows
-   Missing values
-   Duplicate records
-   Target distribution

The generated dataset initially contains:

-   305 rows
-   7 columns
-   15 missing values in `temperature`
-   15 missing values in `vibration`
-   5 duplicate rows

After removing duplicates, the dataset contains 300 unique records.

## Maths and Advanced Statistics

This section uses the `temperature` variable and includes:

-   Sample size
-   Mean
-   Median
-   Standard deviation
-   Histogram
-   Comparison of temperature between groups `G1` and `G2`
-   Independent two-sample t-test
-   95% confidence interval

### Hypothesis Testing

The notebook uses:

-   **H0:** Mean temperature is the same for G1 and G2.
-   **H1:** Mean temperature is different for G1 and G2.
-   **Significance level:** `alpha = 0.05`

## Data Preprocessing and Feature Engineering

The preprocessing section includes:

### Duplicate Removal

Duplicate rows are removed and the cleaned dataset is checked for 300
unique records.

### Train/Validation/Test Split

The data is split using stratification on the `defect` target.

The notebook creates:

-   Fit/training data
-   Validation data
-   Test data

Record IDs are also saved into:

-   `fit_ids.csv`
-   `validation_ids.csv`
-   `test_ids.csv`

The notebook checks that the three ID sets do not overlap.

### Missing Value Imputation

Numerical missing values are handled using:

``` python
SimpleImputer(strategy="median")
```

The numerical columns are:

``` text
temperature
vibration
pressure
hours
```

### Feature Engineering

A new feature is created:

``` text
engineered_feature = vibration × pressure
```

### Feature Scaling

`StandardScaler` is applied to:

-   temperature
-   vibration
-   pressure
-   hours
-   engineered_feature

## Supervised Learning

The notebook uses **Logistic Regression** for binary defect
classification.

### Model

``` python
LogisticRegression(max_iter=1000, random_state=42)
```

### Evaluation Metrics

The model is evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion matrix

The confusion matrix is saved as:

``` text
outputs/figures/confusion_matrix.png
```

## Unsupervised Learning

The notebook also applies **K-Means clustering** to the scaled machine
measurements.

The notebook tests:

``` text
k = 2
k = 3
k = 4
```

For each value of `k`, inertia is calculated.

A final K-Means model is then used to assign cluster labels, and a
cluster profile is created using the average values of the machine
features.

## Project Structure

``` text
Mock2/
│
├── Mock2.ipynb
├── README.md
│
├── data/
│   └── raw/
│       └── set_c.csv
│
├── outputs/
│   └── figures/
│       └── confusion_matrix.png
│
├── fit_ids.csv
├── validation_ids.csv
└── test_ids.csv
```

Some files and folders are created by the notebook when the
corresponding cells are executed.

## Requirements

Python 3.x with the following libraries:

``` text
numpy
pandas
matplotlib
scipy
scikit-learn
```

## Installation

Install the required packages with:

``` bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## How to Run

1.  Open `Mock2.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2.  Run the notebook cells in order.
3.  The dataset will be generated automatically.
4.  The notebook performs data analysis and preprocessing.
5.  Logistic Regression is trained and evaluated.
6.  K-Means clustering is performed at the end.
7.  Generated CSV files and figures are saved in the project folders.

## Technologies Used

-   Python
-   NumPy
-   Pandas
-   Matplotlib
-   SciPy
-   Scikit-learn
-   Jupyter Notebook / Google Colab

## Conclusion

This project demonstrates an end-to-end machine quality analysis
workflow, starting from synthetic data creation and exploratory checks
and continuing through statistical analysis, preprocessing, feature
engineering, classification, and clustering.
