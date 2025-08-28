The project aims at predicting the likelihood of a credit card defualt, using the following:

Dataset:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Model: 
https://xgboost.readthedocs.io/en/release_3.0.0/

Bayesian Hyperparameter Tuning: 
https://medium.com/@rithpansanga/optimizing-xgboost-a-guide-to-hyperparameter-tuning-77b6e48e289d

# Credit Card Default Prediction with XGBoost

This project uses **XGBoost** to predict credit card defaults. It's a machine learning model that predicts whether a customer will default on their credit card payments based on their financial and demographic information. This project is a great example of an end-to-end machine learning pipeline, including data loading, preprocessing, model training, and hyperparameter tuning.

---

### How it Works

The Python script `default_credit_cards.py` implements the following steps:

1.  **Data Loading and Preprocessing:**
    * It loads the `default_credit_cards.csv` dataset.
    * It separates the features (`X`) from the target variable (`Y`), which indicates whether a customer defaulted.
    * It converts all data to integer type for model compatibility.

2.  **Dataset Splitting:**
    * The data is split into three sets:
        * **Training Set (80%):** Used to train the model.
        * **Validation Set (10%):** Used for hyperparameter tuning.
        * **Test Set (10%):** Used to evaluate the final model's performance.

3.  **Model Training & Evaluation:**
    * An initial **XGBoost** classifier is trained on the training data.
    * Its accuracy is calculated on the test set to establish a baseline performance.

4.  **Hyperparameter Tuning:**
    * The script uses **Bayesian Hyperparameter Tuning** via the `hyperopt` library to find the optimal hyperparameters for the XGBoost model.
    * The algorithm, specifically **Tree of Parzen Estimators (TPE)**, intelligently searches for the best combination of `max_depth`, `learning_rate`, and `subsample` to improve accuracy.
    * The model is re-trained with the best-found parameters, and its improved accuracy is calculated on the test set.

---

### Prerequisites

To run this project, you'll need the following Python libraries installed. You can install them using pip:

```bash
pip install pandas scikit-learn xgboost matplotlib hyperopt numpy


