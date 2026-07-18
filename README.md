# Bank Customer Churn Prediction using ANN

## Project overview

This project develops an end-to-end deep learning solution to predict whether a bank customer is likely to churn.

The workflow covers:

- Exploratory data analysis and visualisation
- Distribution, skewness and outlier assessment
- Leakage-safe preprocessing and feature scaling
- Basic and regularised Artificial Neural Networks
- Dropout regularisation
- Early stopping
- Model checkpointing
- KerasTuner hyperparameter tuning
- Class-weight experiments
- Decision-threshold tuning
- Validation-based model comparison
- Final reserved-test evaluation
- Model, preprocessor and metadata persistence
- Streamlit prediction and analytics dashboard

**Final model configuration:** Improved ANN with Dropout, EarlyStopping and ModelCheckpoint, used with the tuned classification threshold of **0.30**.

---

## Dataset

**Source:** [Bank Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/santoshd3/bank-customers/data)

**Dataset file:** `Churn Modeling.csv`

The dataset contains **10,000 bank customers**.

| Target class | Customers | Percentage |
|---|---:|---:|
| Not churned | 7,963 | 79.63% |
| Churned | 2,037 | 20.37% |
| **Total** | **10,000** | **100.00%** |

The target variable is:

- `Exited = 0`: Not churned
- `Exited = 1`: Churned

The following identifier columns were excluded from model training:

- `RowNumber`
- `CustomerId`
- `Surname`

### Model input features

| Feature | Type |
|---|---|
| `CreditScore` | Numerical |
| `Geography` | Categorical |
| `Gender` | Categorical |
| `Age` | Numerical |
| `Tenure` | Numerical |
| `Balance` | Numerical |
| `NumOfProducts` | Numerical/discrete |
| `HasCrCard` | Binary |
| `IsActiveMember` | Binary |
| `EstimatedSalary` | Numerical |

The model receives **10 raw input features**, which are transformed into **13 processed features** by the saved preprocessing pipeline.

---

## Project workflow

1. Load and inspect the raw dataset
2. Validate data types, categories, and target distribution
3. Perform exploratory data analysis
4. Analyse numerical distributions and skewness
5. Detect outliers using boxplots and the IQR method
6. Define modelling features
7. Create fixed training, validation, and test partitions
8. Fit preprocessing only on modelling-training data
9. Train a Basic ANN
10. Train an Improved ANN with regularisation and callbacks
11. Tune ANN hyperparameters using KerasTuner
12. Test class-weighted ANN variants
13. Tune classification thresholds using validation F1-score
14. Compare all approved candidates using validation metrics only
15. Lock the final model and threshold
16. Evaluate the reserved test partition
17. Save and reload all inference artefacts
18. Build and validate the Streamlit application

### Data partitions

| Partition | Records |
|---|---:|
| Modelling training | 6,400 |
| Validation | 1,600 |
| Final test | 2,000 |

---

## Exploratory data analysis

### Distribution and skewness decisions

- `Age` had the highest skewness at **1.011**, indicating right skew.
- `NumOfProducts` had moderate right skew at **0.746**.
- `CreditScore`, `Tenure`, `Balance` and `EstimatedSalary` had low skewness values.
- No skewness-correcting transformation, such as a log was applied.
- Numerical features were later standardised using `StandardScaler`. Standardisation changes feature scale but does not correct skewness.
- ANN models do not require all input variables to follow a normal distribution.
- `StandardScaler` was applied later to standardise numerical feature scales; scaling does not remove skewness.

### Outlier assessment

| Feature | IQR outliers | Percentage | Decision |
|---|---:|---:|---|
| CreditScore | 15 | 0.15% | Retained |
| Age | 359 | 3.59% | Retained |
| Tenure | 0 | 0.00% | No treatment |
| Balance | 0 | 0.00% | No treatment |
| NumOfProducts | 60 | 0.60% | Retained |
| EstimatedSalary | 0 | 0.00% | No treatment |

The detected outliers represented valid customer records rather than confirmed data errors. No rows were removed or capped.

### Important historical churn patterns

- Overall churn rate: **20.37%**
- Germany had the highest geography-level churn rate at **32.44%**
- Spain churn rate: **16.67%**
- France churn rate: **16.15%**
- The `51–60` age group had the highest churn rate at **56.21%**
- Four-product customers had a 100% observed churn rate in both activity groups:
  - Active: **29 of 29**
  - Inactive: **31 of 31**
- The four-product result is based on only **60 customers** and must therefore be interpreted cautiously.

These are descriptive cohort patterns and should not be interpreted as causal findings or individual ANN feature contributions.

---

## Preprocessing

A fitted preprocessing object was used for both notebook evaluation and Streamlit inference.

The preprocessing process includes:

- Numerical feature standardisation using `StandardScaler`
- One-hot encoding of `Geography` and `Gender`
- Retention of binary variables
- Consistent raw feature order
- No fitting or preprocessing duplication inside `app.py`

The preprocessor was fitted only on modelling-training data and then applied to validation and test data.

---

## ANN experiments

Six approved ANN configurations were compared:

1. Basic ANN
2. Improved ANN with Dropout, EarlyStopping and ModelCheckpoint
3. Tuned ANN using KerasTuner
4. Class-weighted Improved ANN
5. Class-weighted Improved ANN with tuned threshold
6. Improved ANN with tuned threshold

### Regularisation and callbacks

The Improved ANN used:

- Multiple hidden layers
- Dropout regularisation
- `EarlyStopping`
- `ModelCheckpoint`
- Best-checkpoint restoration before evaluation

Model weights were saved after each training epoch, and the best complete checkpoint was saved separately.

### KerasTuner search

KerasTuner evaluated combinations of:

- Number of hidden layers
- Units per hidden layer
- Activation functions
- Dropout rates
- Optimiser
- Learning rate
- Batch size

Validation PR-AUC was used as the tuning objective because churn is the minority class.

The loss function remained fixed as `binary_crossentropy`, which is appropriate for binary classification with a sigmoid output.

The best-tuned configuration used:

- 4 hidden layers
- Adam optimiser
- Learning rate: `0.01`
- Batch size: `64`

The KerasTuner model achieved the highest validation precision at **0.8067**, but its validation recall, F1-score and balanced accuracy were lower than those of the final threshold-tuned Improved ANN configuration.

---

## Validation model comparison

Final model selection used:

1. Validation F1-score
2. Validation PR-AUC
3. Validation ROC-AUC
4. Validation balanced accuracy
5. Training-validation gaps as evidence of overfitting

| Model | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Improved ANN with tuned threshold | **0.30** | 0.8462 | 0.7654 | 0.6212 | 0.6288 | **0.6250** | **0.8602** | **0.6920** |
| Class-weighted Improved ANN with tuned threshold | 0.60 | 0.8375 | 0.7610 | 0.5954 | 0.6319 | 0.6131 | 0.8578 | 0.6842 |
| Class-weighted Improved ANN | 0.50 | 0.7987 | **0.7789** | 0.5041 | **0.7454** | 0.6015 | 0.8578 | 0.6842 |
| Improved ANN with Dropout, EarlyStopping and ModelCheckpoint | 0.50 | **0.8625** | 0.7174 | 0.7624 | 0.4724 | 0.5833 | **0.8602** | **0.6920** |
| Basic ANN | 0.50 | 0.8519 | 0.7061 | 0.7109 | 0.4601 | 0.5587 | 0.8399 | 0.6500 |
| Tuned ANN using KerasTuner | 0.50 | 0.8538 | 0.6742 | **0.8067** | 0.3712 | 0.5084 | 0.8554 | 0.6804 |

The **Improved ANN with threshold 0.30** achieved the highest validation F1-score while maintaining competitive balanced accuracy, ROC-AUC and PR-AUC.

The threshold-tuned Improved ANN uses the same trained Improved ANN checkpoint; only its classification threshold changes from 0.50 to 0.30.

---

## Why the threshold was changed to 0.30

At the default threshold of `0.50`, the Improved ANN achieved:

- Validation accuracy: **0.8625**
- Validation precision: **0.7624**
- Validation recall: **0.4724**
- Validation F1-score: **0.5833**

At the tuned threshold of `0.30`, it achieved:

- Validation accuracy: **0.8462**
- Validation precision: **0.6212**
- Validation recall: **0.6288**
- Validation F1-score: **0.6250**
- Validation balanced accuracy: **0.7654**

Lowering the threshold improved churn detection and produced the highest validation F1-score among the approved candidates.

The trade-off was a reduction in precision and overall accuracy, but the model detected a substantially larger proportion of customers who actually churned.

---

## Final selected model

**Model:** Improved ANN with Dropout, EarlyStopping and ModelCheckpoint  
**Classification threshold:** `0.30`

Classification rule:

```text
Model-estimated churn score >= 0.30 → Likely to churn
Model-estimated churn score < 0.30  → Likely to stay
```

---

## Final internal test re-evaluation

After locking the corrected model-selection decision and threshold, the fixed internal test partition was re-evaluated without using its results to alter the final model or threshold.

| Metric | Final test result |
|---|---:|
| Accuracy | **0.8475** |
| Balanced accuracy | **0.7689** |
| Precision | **0.6226** |
| Recall | **0.6364** |
| F1-score | **0.6294** |
| ROC-AUC | **0.8644** |
| PR-AUC | **0.7125** |
| Log loss | **0.3316** |

### Final confusion matrix

|  | Predicted not churned | Predicted churned |
|---|---:|---:|
| **Actual not churned** | 1,436 | 157 |
| **Actual churned** | 148 | 259 |

The model correctly identified **259 of 407 churned customers**.

It missed **148 churned customers** and generated **157 false-positive churn alerts**.

### Validation versus final test

| Metric | Validation | Final test |
|---|---:|---:|
| Accuracy | 0.8462 | 0.8475 |
| Balanced accuracy | 0.7654 | 0.7689 |
| Precision | 0.6212 | 0.6226 |
| Recall | 0.6288 | 0.6364 |
| F1-score | 0.6250 | 0.6294 |
| ROC-AUC | 0.8602 | 0.8644 |
| PR-AUC | 0.6920 | 0.7125 |
| Loss | 0.3423 | 0.3316 |

The close validation and final-test results indicate stable generalisation without material performance deterioration.

---

## Saved artefacts

The final inference package contains:

```text
artifacts/
├── final_churn_ann_model.keras
├── final_churn_preprocessor.pkl
├── final_churn_metadata.json
├── final_ann_validation_comparison.csv
└── final_churn_test_metrics.csv
```

The metadata file stores:

- Final model name
- Locked threshold
- Raw input feature order
- Numerical, categorical and binary features
- Processed feature names
- Expected processed feature count
- Categorical options
- Target mapping
- Validation metrics
- Final-test metrics
- Library versions
- Model and preprocessor filenames

### Reload validation

The saved model, preprocessor and metadata were reloaded and tested on all **2,000 final-test records**.

| Reload check | Result |
|---|---|
| Original and reloaded model-output scores match | `True` |
| Original and reloaded classes match | `True` |
| Maximum absolute probability difference | `0.0` |
| Reloaded threshold | `0.30` |
| Processed feature count | `13` |

This confirms that saving and reloading the inference artefacts did not change the model-output scores or final predicted classes.

---

## Streamlit application

The Streamlit application contains two tabs:

### 1. Churn prediction

Users can enter a customer profile and view:

- Model-estimated churn score
- Final threshold
- Churn/stay classification
- Risk category
- Historical EDA context
- Retention action suggestions
- Input preview

The application uses the saved model, fitted preprocessor and metadata. It does not refit or recreate preprocessing.

Strict compatibility checks verify:

- Final model identity
- Threshold value
- Raw feature order
- Processed feature count
- ANN input dimension
- Categorical options
- Model and preprocessor filenames
- Missing or unexpected input features

### Default prediction

The default customer profile produces:

- Model-estimated churn score: **17.01%**
- Final threshold: **0.30**
- Prediction: **Likely to stay**
- Risk category: **Low churn risk**

### 2. Churn dashboard

The dashboard displays:

- Total customers: **10,000**
- Churned customers: **2,037**
- Retained customers: **7,963**
- Overall churn rate: **20.37%**
- Churn rate by geography
- Churn rate by age group
- Churn rate by product count and activity status
- Business interpretation and management takeaways

The dashboard interpretations are descriptive historical patterns and are not presented as causal conclusions.

**Streamlit App Link**: Click here 

---

## Project structure

```text
.
├── code.ipynb
├── app.py
├── Churn Modeling.csv
├── requirements.txt
│
├── assets/
│   └── customer_churn_banner.png
│
├── artifacts/
│   ├── final_churn_ann_model.keras
│   ├── final_churn_preprocessor.pkl
│   ├── final_churn_metadata.json
│   ├── final_ann_validation_comparison.csv
│   └── final_churn_test_metrics.csv
│
├── checkpoints/
├── models/
└── tuner_results/
```

The `checkpoints`, `models` and `tuner_results` directories are generated during ANN training and hyperparameter tuning.

---

## Tested environment

```text
Python==3.12.0
ipykernel==7.3.0
numpy==1.26.4
pandas==3.0.3
matplotlib==3.11.0
seaborn==0.13.2
scipy==1.17.1
scikit-learn==1.9.0
tensorflow==2.16.2
keras-tuner==1.4.8
joblib==1.5.3
streamlit==1.58.0
```

---

## Limitations

- The dataset contains 10,000 historical customer records from one source.
- No external validation dataset was used.
- The reserved internal test partition had been inspected during an earlier development workflow; therefore, the final figures should not be interpreted as performance on a completely unseen external dataset.
- The ANN output is a model-estimated churn score and was not separately calibrated as a formal probability estimate.
- Historical EDA patterns do not prove causation.
- The application does not provide individual ANN feature attributions.
- Four-product customers showed 100% observed churn, but this result is based on only 60 customers.
- The final model still missed 148 churned customers and produced 157 false-positive alerts.
- Predictions should support human decision-making and should not be used alone for customer rejection, penalties or unfair treatment.

---

## Final conclusion

The final Improved ANN with a tuned threshold of **0.30** provided the strongest validation F1-score among the evaluated candidates.

It achieved:

- **84.75% final-test accuracy**
- **76.89% balanced accuracy**
- **62.94% F1-score**
- **63.64% churn recall**
- **86.44% ROC-AUC**
- **71.25% PR-AUC**

The saved model, preprocessing pipeline, and metadata reproduced the original predictions exactly after reload, confirming that the final inference package is internally consistent and ready for use in the validated Streamlit application.

----
