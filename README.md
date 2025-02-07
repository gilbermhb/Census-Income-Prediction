
# Census Income Prediction

This project aims to predict whether an individual's income exceeds $50K per year based on census data. The code provided includes data preprocessing, model training, evaluation, and an ensemble model for prediction. 

## Prerequisites

Make sure you have Python installed, along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib

## How to Use

1. **Load Data**: The `load_data` function loads the dataset from a CSV file and prepares it for further processing.
   
2. **Prepare Data**: The `prepare_data` function performs data cleaning, preprocessing, and feature engineering. It converts categorical variables into numerical form, handles missing values, and prepares the dataset for model training.

3. **Model Training and Evaluation**:
    - **K Nearest Neighbors (KNN)**: Trains a KNN classifier with a specified number of neighbors (`k`) and evaluates its performance using accuracy, mean squared error, confusion matrix, and classification report.
    - **Random Forest Classifier**: Trains a Random Forest classifier and evaluates its performance using accuracy, confusion matrix, and classification report. You can optionally perform a grid search for hyperparameter tuning.
    - **Decision Tree Classifier**: Trains a Decision Tree classifier and evaluates its performance using accuracy, confusion matrix, and classification report. You can optionally perform a grid search for hyperparameter tuning.
    - **Ensemble Model (Voting Classifier)**: Combines the predictions of the KNN, Random Forest, and Decision Tree classifiers using a Voting Classifier and evaluates its performance using accuracy, confusion matrix, and classification report.

4. **Learning Curve**: The learning curve for the ensemble model is plotted to visualize the model's performance with different training sizes.

## Files

- `census-income.data.csv`: Training data file path.
- `census-income.test.csv`: Test data file path.

## Usage

```
train_data = load_data('path_to_training_data')
test_data = load_data('path_to_test_data')

X_train, y_train = prepare_data(train_data, delimiter="")
X_test, y_test = prepare_data(test_data, delimiter=".")
```



#### Notes:
* Uncomment relevant sections to train and evaluate specific models or perform parameter tuning.
* The delimeter is used to transform the income column given that in the test set that column ends with dot (.)
