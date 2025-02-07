# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:01:31 2024

@author: Gilbert Hernandez
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
# from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def load_data(file):
    data = pd.read_csv(file, header=None)
    column_names = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','gender','capital_gain','capital_loss','hours_per_week','native_country','income']
    data.columns = column_names
    
    return data


def prepare_data(data, d='.'):
    """This function perfoms Data Cleaning, where the argument delimiter is used as a paramenter
    to clean the label column where it can have a dot at the end, the default value is empty"""
    #Remove spaces from categorical features
    data[['workclass','education','marital_status','occupation','relationship','race','gender','native_country']] = data[['workclass','education','marital_status','occupation','relationship','race','gender','native_country']].applymap(lambda x: x.lstrip())
        
    # Working on '?' occurrences: Workclass "?" - 1836, occupation "?" - 1843, native-country "?" - 583
    data['workclass'] = data['workclass'].apply(lambda x: np.nan if x == '?' else x)
    data['occupation'] = data['occupation'].apply(lambda x: np.nan if x == '?' else x)
    data['native_country'] = data['native_country'].apply(lambda x: np.nan if x == '?' else x)
    
    #Replacing nan values with mode.
    data[['workclass', 'occupation', 'native_country']] = data[['workclass', 'occupation', 'native_country']].fillna(data.mode().iloc[0])
    # data = data.dropna(axis=0)
    
    # Convert the annual_income into binomial where <=50k == 1 and >50k == 0
    data['income'] = data['income'].apply(lambda x: x.replace(f"<=50K{d}", "0").replace(f">50K{d}", "1"))
    data['income'] = data['income'].astype(int)
    
    #Grouping age by ranges
    data['age_map'] = data['age'].apply(lambda x: 1 if x > 16 and x <= 25 
                                          else 2 if x > 25 and x <= 32 
                                          else 3 if x > 32 and x <= 40
                                          else 4 if x > 40 and x <= 50 
                                          else 5)
    
    #Workclass
    data['workclass'] = data['workclass'].apply(lambda x: 'Gov' if x in ['State-gov', 'Federal-gov','Local-gov'] else ('Self-emp' if x in ['Self-emp-not-inc', 'Self-emp-inc'] else ('No-income' if x in ['Without-pay','Never-worked'] else x)))
    
    # turn education into school, HS-grad, some-college, bachelors, Prof-School, Associate, Masters, doctorate
    data['education'] = data['education'].apply(lambda x: "School" if x in ["11th", "10th", "7th-8th", "9th", "12th", "5th-6th", "1st-4th", "Preschool"] else ("Associate" if x in ["Assoc-voc","Assoc-acdm"] else x))
    
    # Reduce the categorical values for the marital_status column
    data['marital_status'] = data['marital_status'].apply(lambda x: "Married" if x in ["Married-civ-spouse", "Married-spouse-absent","Married-AF-spouse"] else ('Prev-married' if x in ['Divorced','Widowed'] else ('Single' if x in ['Never-married'] else x)))
    
    #"""In this case most of the data is skeweed toward White, so It will actually make more sense or provide more information if we group Asian-Pac-Islander and Amer-Indian-Esimo as Others"""
    data['race'] = data['race'].apply(lambda x: 'Other' if x in ['Asian-Pac-Islander','Amer-Indian-Eskimo'] else x)
    
    #Arrange Native_country by grouping some
    data['native_country'] = data['native_country'].apply(lambda x: 'United-States' if x=='Outlying-US(Guam-USVI-etc)' else x)
    data = data[data['native_country'] != 'Holand-Netherlands']
    #Spliting into X and y
    
    cat_var = data[['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country']]
    num_var = data[['education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'age_map']]
    
    
    
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    
    # Step 2: Fit and transform the OneHotEncoder on the categorical variables
    cat_encoded = one_hot_encoder.fit_transform(cat_var)
    
    # Step 3: Get the feature names from the encoder
    feature_names = one_hot_encoder.get_feature_names_out(input_features=cat_var.columns)
    
    # Step 4: Convert the encoded sparse matrix to a DataFrame
    cat_encoded_df = pd.DataFrame(cat_encoded.toarray(), columns=feature_names, index=cat_var.index)
    
    # Concatenate numerical and one-hot encoded variables
    X_encoded = pd.concat([num_var, cat_encoded_df], axis=1)
    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_encoded)
    X = pd.DataFrame(X, columns = X_encoded.columns)
    y = data['income']
    
    
    return X, y


# def plot_learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring=None):
#     train_sizes, train_scores, valid_scores = learning_curve(
#         estimator, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring)
    
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     valid_scores_mean = np.mean(valid_scores, axis=1)
#     valid_scores_std = np.std(valid_scores, axis=1)
    
#     plt.figure(figsize=(10, 6))
#     plt.title("Learning Curve")
#     plt.xlabel("Training Examples")
#     plt.ylabel("Score")
#     plt.grid()
    
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
#                      valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
    
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, valid_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
    
#     plt.legend(loc="best")
#     plt.show()



train_data = load_data('C:/Users/Gilbert Hernandez/OneDrive - Fordham University/CISC 5790 - Data Mining/Project/census-income.data.csv')
test_data = load_data('C:/Users/Gilbert Hernandez/OneDrive - Fordham University/CISC 5790 - Data Mining/Project/census-income.test.csv')
X_train, y_train = prepare_data(train_data, d="")
X_test, y_test = prepare_data(test_data, d=".")



#################################################
k=46
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nKNN Classifier\n")
print(f"For k: {k}, Accuracy: {accuracy * 100:.2f}%")
# print(f"The MSE: {mse}")

#Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n")
print(conf_matrix)

# Generate classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:\n")
print(class_report)

####################################
""" Uncomment to run the code that helped to verify the optimum K for the KNN Classifier """

# # from sklearn.model_selection import cross_val_score

# # k_values = list(range(20, 300))  # You can adjust the range as needed

# # # Create an empty list to store the mean cross-validation scores for each K
# # cv_scores = []

# # # Iterate over each K value
# # for k in k_values:
# #     # Create a KNN classifier with the current value of K
# #     knn_classifier = KNeighborsClassifier(n_neighbors=k)
    
# #     # Perform cross-validation and calculate the mean accuracy
# #     scores = cross_val_score(knn_classifier, X, y, cv=5, scoring='accuracy')  # Adjust cv value as needed
# #     mean_accuracy = scores.mean()
    
# #     # Append the mean accuracy to the list of cross-validation scores
# #     cv_scores.append(mean_accuracy)

# # # Find the optimal value of K with the highest cross-validation score
# # optimal_k = k_values[cv_scores.index(max(cv_scores))]

# # print("Optimal K value:", optimal_k)
# # # Optimal K value: 46


# #####################################

rf_model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, min_samples_split=2, random_state=42)

# Step 5: Train Model
rf_model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred_rf = rf_model.predict(X_test)

# Calculate accuracy
print("\nRandom Forest Classifier\n")
accuracy = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:\n")
print(conf_matrix)

# Generate a classification report
report = classification_report(y_test, y_pred_rf)
print("\nClassification Report:\n", report)


#####################################
""" Uncomment to run the code that helped to verify the best parameter for the Random Forest Classifier """

# print('\nGetting the best parameter for the random forest classifier:\n')

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Instantiate the Random Forest classifier
# rf_classifier = RandomForestClassifier(random_state=42)

# # Instantiate the Grid Search
# grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# # Fit the Grid Search
# grid_search.fit(X_train, y_train)

# # Retrieve the best parameters
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# # Optionally, evaluate performance on test set
# best_rf_classifier = grid_search.best_estimator_
# test_score = best_rf_classifier.score(X_test, y_test)
# print("Test Accuracy:", test_score)


#####################################

decision_tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=5)

# Train the Decision Tree classifier on the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_dt = decision_tree.predict(X_test)

# Calculate accuracy
print("\nDecision Tree Classifier\n")
accuracy = accuracy_score(y_test, y_pred_dt)
print("\nAccuracy:\n", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred_dt)
print("\nConfusion Matrix:\n")
print(conf_matrix)

# Generate a classification report
report = classification_report(y_test, y_pred_dt)
print("\nClassification Report:\n", report)


#####################################
""" Uncomment to run the code that helped to verify the best parameter for the Decision Tree Classifier """ 

# print('\nGetting the best parameter for the decision tree classifier:\n') 

# param_grid = {
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Instantiate the decision tree classifier
# dt_classifier = DecisionTreeClassifier(random_state=42)

# # Instantiate the Grid Search
# grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# # Fit the Grid Search
# grid_search.fit(X_train, y_train)

# # Retrieve the best parameters
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# # Optionally, evaluate performance on test set
# best_rf_classifier = grid_search.best_estimator_
# test_score = best_rf_classifier.score(X_test, y_test)
# print("Test Accuracy:", test_score)


#####################################


# Define the base classifiers
classifier1 = knn_classifier
classifier2 = rf_model
classifier3 = decision_tree

# Create the ensemble model
ensemble_model = VotingClassifier(estimators=[
    ('KNN', classifier1),
    ('Random_Forest', classifier2),
    ('Decision_Tree', classifier3)
], voting='hard')  # Adjust voting parameter as needed (e.g., 'hard' for majority vote, 'soft' for probability averaging)

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Predict the labels using the ensemble model
y_pred_ensemble = ensemble_model.predict(X_test)

# Calculate the accuracy of the ensemble model
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print("\nEnsemble Model Accuracy:", accuracy_ensemble)

# Calculate the confusion matrix
conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)
print("\nConfusion Matrix:\n", conf_matrix_ensemble)

# Generate the classification report
class_report_ensemble = classification_report(y_test, y_pred_ensemble)
print("\nClassification Report:\n", class_report_ensemble)



train_sizes, train_scores, test_scores = learning_curve(ensemble_model, X_train, y_train, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="blue")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="orange")
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="orange", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.show()
