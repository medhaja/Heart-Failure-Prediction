import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.xgboost
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval

DATA_PATH = "heart_failure_clinical_records.csv"

def load_and_preprocess(data_path):
    """ Load and preprocess data from CSV file. """
    data = pd.read_csv(data_path)
    print(data.isnull().sum())
    X = data.drop("DEATH_EVENT", axis=1)
    y = data["DEATH_EVENT"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_objective(X_train, y_train):
    """ Create the objective function for hyperparameter tuning using XGBoost and MLflow. """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            evals_result = {}
            model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtrain, 'train')], evals_result=evals_result, early_stopping_rounds=10)
            mlflow.xgboost.log_model(model, "xgboost_model")
            loss = evals_result['train']['logloss'][-1]
            return {'loss': loss, 'status': STATUS_OK, 'model': model}
    return objective

def hyperparameter_optimization(X_train, y_train):
    """ Perform hyperparameter optimization. """
    space = {
        'max_depth': hp.choice('max_depth', range(3, 10)),
        'eta': hp.uniform('eta', 0.01, 0.3),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    trials = Trials()
    objective = create_objective(X_train, y_train)
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )
    best_params = space_eval(space, best)
    best_model = trials.best_trial['result']['model']
    return best_params, best_model

def evaluate_model(model, X_test, y_test):
    """ Evaluate the trained model on the test set. """
    dtest = xgb.DMatrix(X_test)
    pred_probs = model.predict(dtest)
    predictions = [1 if x > 0.5 else 0 for x in pred_probs]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, pred_probs)
    conf_matrix = confusion_matrix(y_test, predictions)
    return accuracy, roc_auc, conf_matrix

def plot_feature_importance(model):
    """ Plot feature importance of the model. """
    xgb.plot_importance(model, importance_type='weight', title='Feature Importance: Weight')
    plt.show()
    xgb.plot_importance(model, importance_type='gain', title='Feature Importance: Gain')
    plt.show()
    xgb.plot_importance(model, importance_type='cover', title='Feature Importance: Cover')
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess(DATA_PATH)
    best_params, best_model = hyperparameter_optimization(X_train, y_train)
    accuracy, roc_auc, conf_matrix = evaluate_model(best_model, X_test, y_test)
    print(f"Best Hyperparameters: {best_params}")
    print(f"Accuracy: {accuracy}, ROC AUC: {roc_auc}")
    print("Confusion Matrix:\n", conf_matrix)
    plot_feature_importance(best_model)
