# Titanic Survival Prediction using Random Forest

## Project objective
Predict whether a Titanic passenger survived (1) or did not survive (0) using passenger details.

## Dataset
This project expects a Kaggle-style `titanic.csv` file in the same folder as `main.py`.
Required columns:

- `survived`
- `pclass`
- `sex`
- `age`
- `sibsp`
- `parch`
- `fare`
- `embarked`

The full Kaggle Titanic training CSV contains these columns. If your file uses different column names, update `FEATURES` and `TARGET` in `main.py`.

## Setup (Windows)
1. Install Python.
2. Open Command Prompt in this project folder.
3. Create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Copy `titanic.csv` into this folder.
6. Run:

   ```bash
   python main.py
   ```

## Outputs
The program creates an `outputs` folder containing:

- `confusion_matrix.png`
- `feature_importance.png`
- `feature_importance.csv`
- `test_predictions.csv`

## Methodology
1. Load and validate the dataset.
2. Separate input features and target.
3. Split data into 80% training and 20% testing sets.
4. Impute missing numeric and categorical values inside a pipeline.
5. One-hot encode categorical columns.
6. Train a Random Forest classifier.
7. Evaluate test accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC, and 5-fold stratified cross-validation.
8. Predict survival for an example passenger.

## Important note
Preprocessing is fitted only on the training folds by using a scikit-learn Pipeline. This helps prevent data leakage. Model scores depend on the dataset and split; they are not hard-coded.
