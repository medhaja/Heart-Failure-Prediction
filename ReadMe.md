# Heart Failure Prediction Model

This project uses the XGBoost algorithm to predict heart failure from clinical records. It includes data preprocessing, model training, evaluation, and feature importance visualization. Additionally, MLflow is used for tracking experiments, managing models, and logging metrics.

## Dataset

The dataset contains medical records of patients who had heart failure, with 13 clinical features such as age, anaemia, diabetes, and blood pressure. The target variable is `DEATH_EVENT`, which indicates if the patient died during the follow-up period.

## Requirements

Before running this script, ensure you have the following packages installed:

- pandas
- numpy
- xgboost
- sklearn
- matplotlib
- mlflow

You can install them using pip:

```bash
pip install pandas numpy xgboost sklearn matplotlib mlflow
```

## Usage

1. **Data Loading and Preprocessing**: 
   - Load the data from `heart_failure_clinical_records.csv`.
   - Check for any missing values in the dataset.
   - Split the data into training and testing sets.

2. **Model Training**:
   - Configure and train an XGBoost model using the training data.
   - Parameters and the number of boosting rounds can be adjusted as needed.

3. **Model Evaluation**:
   - Predict the test set outcomes.
   - Evaluate the model using accuracy, ROC AUC score, and a confusion matrix.

4. **Feature Importance Visualization**:
   - Visualize the importance of different features in the dataset using plots.

5. **MLflow Tracking**:
   - The script includes MLflow integration for tracking models, parameters, and metrics.
   - Ensure MLflow is running by executing `mlflow ui` in your terminal and visiting `http://localhost:5000` in your browser.

6. **Hyperparameter Tuning**:
   - The script also includes a hyperparameter tuning example using MLflow, which logs different experiments and finds the best parameters.

## Files Included

- `heart_failure_clinical_records.csv`: Dataset file.
- `confusion_matrix.csv`: Output file containing the confusion matrix of the model predictions.

## Notes

- The script assumes that the dataset `heart_failure_clinical_records.csv` is in the same directory as the script.
- Ensure that MLflow is properly set up and configured in your environment for tracking and logging.

## Additional Information

For more details on the project, contact medhajahlbhat@gmail.com or visit medhaja.me
