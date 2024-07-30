import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

#Load the dataset
df = pd.read_csv("nba2021.csv")

#Drop the players who have Minutes Played(MP) less than 15
updated_df = df[df.MP >= 15]

#Drop irrelevant features
updated_df = updated_df.drop(['Player', 'Age', 'Tm'], axis=1)

#Derive new features from existing features, avoid division by zero by giving deafult value as 0 if denominator is 0 
updated_df['AST/PTS'] = np.where(updated_df['PTS'] != 0, updated_df['AST'] / updated_df['PTS'], 0)
updated_df['TRB/AST'] = np.where(updated_df['AST'] != 0, updated_df['TRB'] / updated_df['AST'], 0)
updated_df['STL/BLK'] = np.where(updated_df['BLK'] != 0, updated_df['STL'] / updated_df['BLK'], 0)

#Have X as target class and y as all the features. Also, encode class labels into numerical values
X = updated_df.drop('Pos', axis=1)
y = updated_df.loc[:, 'Pos']
y = updated_df['Pos']
posi = {
    "PG": 1,
    "SG": 2,
    "SF": 3,
    "PF": 4,
    "C" : 5
}
y = y.map(posi).values.reshape(-1,1)

#Calculate the top features using MI
discrete_features = X.dtypes == int
def cal_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

#Drop features whose MI score is less than 0.1
mi_scores = cal_mi_scores(X, y, discrete_features)
def drop_irrelevant(df, mi_scores):
    return df.loc[:, mi_scores > 0.1]
X = drop_irrelevant(X, mi_scores)

#Split the dataset into 75% training and rest as testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.75)

#Standadize the training data
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

#Use GridSearch to find the best hyperparameters
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
grid_search = GridSearchCV(SVC(random_state=0), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_svc = grid_search.best_estimator_
print("Best Model: ", best_svc)


#User Linear SVC to fit and predict for testing data
linearsvm = SVC(kernel='linear', C=0.3, random_state=0, class_weight='balanced').fit(X_train_scaled, y_train)
loaded_model = joblib.load('svm_model.pkl')
y_train_pred = linearsvm.predict(X_train_scaled)
y_test_pred = linearsvm.predict(X_test_scaled)

#Get the accuracy of training and testing data
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

model_filename = 'svm_model.pkl'
joblib.dump(linearsvm, model_filename)

print("Training set accuracy: {:.3f}".format(train_accuracy))
print("Test set accuracy: {:.3f}".format(test_accuracy))

#Use 10-fold Stratified Cross-Validation instead of Test-Train-Split
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
accuracy_scores = cross_val_score(linearsvm, X, y, cv=kf, scoring='accuracy')
print("Cross-validation scores: {}" .format(accuracy_scores))
average_accuracy = np.mean(accuracy_scores)
print("Average Cross-Validation Score: {:.2f}" .format(average_accuracy))


#Compute the Confusion matrix
confusion_mat = confusion_matrix(y_test, y_test_pred)
class_labels = ["PG", "SG", "SF", "PF", "C"]
lab_confusion = pd.DataFrame(confusion_mat, index=class_labels, columns=class_labels)
row_sum = lab_confusion.sum(axis=1)
col_sum = lab_confusion.sum(axis=0)
lab_confusion["All"] = row_sum
lab_confusion.loc["All"] = col_sum

print("Confusion Matrix:")
print(lab_confusion)