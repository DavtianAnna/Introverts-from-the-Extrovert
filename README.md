# 🧠 Predict the Introverts from the Extroverts (Playground Series S5E7)

This repository contains my solution for the Kaggle competition:  
**[Playground Series - Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7/overview)**  
The task is to predict whether a person is an introvert or extrovert based on synthetic tabular data.

---


## 📁 Project Structure

```

project-root/
├── 📄 Task.ipynb               # Main notebook: full pipeline from preprocessing to model training & submission.
├── 📊 train.csv                # Training dataset including target (Depression).
├── 🧪 test.csv                 # Test dataset for prediction.
├── 📝 sample_submission.csv    # Kaggle submission format.
├── 🚀 sub.csv                  # Predictions from SCV model
├── 🚀 sub1.csv                 # Predictions from Sequential model
└── 📜 README.md                # Project documentation.

```

---

## 💻 Technologies Used

- **Programming Language**: Python 3.x
- **Notebook Environment**: Jupyter Notebook


- **Libraries**:
  - `pandas`, `numpy`: Data manipulation
  - `seaborn`, `matplotlib`: Visualization (e.g. boxplots, distributions)
  - `scikit-learn`: Machine learning framework
    - `StandardScaler`: Feature scaling
    - `SVC` (Support Vector Classifier): Core ML model
    - `Pipeline`: To streamline preprocessing + model
    - `GridSearchCV`: Hyperparameter tuning
    - `LabelEncoder`: Encoding categorical features
    - `train_test_split`, `classification_report`, `ConfusionMatrixDisplay`: Model evaluation
  - `keras`:  Deep Learning
    -  `Sequential`: Layer-by-layer models
    -  `Adam`: Optimizer for training
    -  `Dense`: Fully connected layers
    -  `to_categorical`: Convert labels for classification

---


## 🔁 Project Workflow

The `Task.ipynb` notebook includes the following steps:

### 1. 📥 Data Loading & Cleaning
- Loaded the data from `.csv` files.
- Dropped non-informative columns like `id`.
- Cleaned categorical values by ensuring test set categories are consistent with training set.
- Handled missing or anomalous values using a **probability-based approach**:
  - Instead of using median/mode imputation, anomalies were identified by their statistical improbability.
  - Low-probability values were either corrected or replaced using distribution-based thresholds.
  - This helped preserve informative variance while reducing noise.
 

### 2. 🔡 Encoding
- Applied `LabelEncoder` to convert categorical variables to numeric form.
- Ensured label consistency between train and test sets.


### 3. 🤖 Model Building
 
  #### Support Vector Classifier (SVC)
  
- Used `Pipeline` with `StandardScaler` and `SVC`.
- Tuned hyperparameters with `GridSearchCV`:
  - `C`: `[1, 5, 10]`
  - `kernel`: `['linear', 'poly', 'rbf', 'sigmoid']`
  - `gamma`: `['scale', 'auto']`

  #### Neural Network (Sequential)
    - Used `Sequential` with `StandardScaler`.
    -  `Adam` optimizer – efficient gradient-based optimization for training
    -  `Dense` layers – fully connected layers for learning complex patterns
    -  `to_categorical` – convert categorical labels for multi-class classification


### 4. 🧪 Evaluation
- Split data into training and validation sets (80/20).
- Evaluated model using:
  - `classification_report` (precision, recall, F1)
  - `ConfusionMatrixDisplay`

### 5. 📤 Prediction & Submission
- Final predictions generated on the test dataset.
- Used `.predict` to obtain class labels for submission.
- Saved SVC model predictions in `sub.csv` for Kaggle upload.
- Saved Sequential model predictions in `sub1.csv` for Kaggle upload.


---


## 📈 Performance Summary

| Model                     | Algorithm                   | Score    | Output File |
| ------------------------- | --------------------------- | -------- | ----------- |
| SVC (with scaling)        | Support Vector Machine      | 0.967611 | `sub.csv`   |
| Sequential (with scaling) | Neural Network (Sequential) | 0.968218 | `sub1.csv`  |


---

## ⚙️ Installation

Install all required dependencies:
