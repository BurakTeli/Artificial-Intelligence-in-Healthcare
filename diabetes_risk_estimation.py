# import libraries
import pandas as pd # data science library
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# import data and EDA
# loading data
df = pd.read_csv("diabetes.csv")
df_name = df.columns

# sutun isimleri (buyuk kucuk fark, bosluk, ingilizce olmayan karakterler)
# sample sayisi ve kayip veri problemi
# veri tipleri mesela age -> string olsaydi problemdi
df.info() 

describe = df.describe()

sns.pairplot(df, hue = "Outcome")
plt.show()

def plot_correlation_heatmap(df):
    
    corr_matrix = df.corr()
    
    plt.figure(figsize = (10,8))
    sns.heatmap(corr_matrix, annot=True, cmap = "coolwarm", fmt = ".2f", linewidths=0.5)
    plt.title("Correlation of Features")
    plt.show()

plot_correlation_heatmap(df)

# Outlier Detection
def detect_outliers_iqr(df):
    
    outlier_indices = []
    outliers_df = pd.DataFrame()
    
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        
        Q1 = df[col].quantile(0.25) # first quartile 
        Q3 = df[col].quantile(0.75) # third quartile 

        IQR = Q3 - Q1 # interquartile range

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outlier_indices.extend(outliers_in_col.index)
        
        outliers_df = pd.concat([outliers_df, outliers_in_col], axis = 0)
        
    # remove duplicate indices
    outlier_indices = list(set(outlier_indices))
    
    # remove duplicate rows in the outliers dataframe
    outliers_df = outliers_df.drop_duplicates()
    
    return outliers_df, outlier_indices

outliers_df, outlier_indices = detect_outliers_iqr(df)

# remove outliers from the dataframe
df_cleaned = df.drop(outlier_indices).reset_index(drop=True)

# Train test split
X = df_cleaned.drop(["Outcome"], axis = 1)
y = df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

# standartizasyon
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training and evaluation
"""
LogisticRegression
DecisionTreeClassifier
KNeighborsClassifier
GaussianNB
SVC
AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
"""

def getBasedModel():
    basedModels = []
    basedModels.append(("LR", LogisticRegression()))
    basedModels.append(("DT", DecisionTreeClassifier()))
    basedModels.append(("KNN", KNeighborsClassifier()))
    basedModels.append(("NB", GaussianNB()))
    basedModels.append(("SVM", SVC()))
    basedModels.append(("AdaB", AdaBoostClassifier()))
    basedModels.append(("GBM", GradientBoostingClassifier()))
    basedModels.append(("RF", RandomForestClassifier()))
    
    return basedModels

def baseModelsTraning(X_train, y_train, models):
    
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name}: accuracy: {cv_results.mean()}, std: {cv_results.std()}")
    
    return names, results

def plot_box(names, results):
    df = pd.DataFrame({names[i]: results[i] for i in range(len(names))})
    plt.figure(figsize = (12,8))
    sns.boxplot(data=df)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
    
models = getBasedModel()
names, results = baseModelsTraning(X_train, y_train, models)
plot_box(names, results)

# hyperparameter tuning
# DT hyperparameter set
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":[1,2,4]
    }

dt = DecisionTreeClassifier()
# grid search cv
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring="accuracy")
# training
grid_search.fit(X_train, y_train)
print("En iyi parametreleri: ", grid_search.best_params_)
best_dt_model = grid_search.best_estimator_
y_pred = best_dt_model.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
"""
[[88 21]
 [28 23]]
"""
print("classification_report")
print(classification_report(y_test, y_pred))

# Model testing with real data
new_data = np.array([[6,149,72,35,0,34.6,0.627,51]])
new_prediction = best_dt_model.predict(new_data)
print("New prediction: ", new_prediction)




































