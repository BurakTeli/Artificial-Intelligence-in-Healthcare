# Libraries
import pandas as pd  # data science library
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

import argparse

parser = argparse.ArgumentParser(description="Grafiklerin belli bir bölümünü çalıştırma")
parser.add_argument('--all', action='store_true', help="Train all models")
args = parser.parse_args()


import warnings
warnings.filterwarnings("ignore")

# data and eda

df = pd.read_csv('/Users/buraktelli/Desktop/Sağlık_Proje/3_kodlar_verisetleri/4_SagliktaMakineOgrenmesiUygulamalari/diabetes.csv')
df_name = df.columns
print(df.head())

# Görselleştirme işlemi: Veri kümesinin ısı haritasını oluşturma
plt.figure(figsize=(12, 8))  # Grafik boyutunu ayarlıyoruz
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")  # Korelasyon ısı haritası
plt.title("Feature Correlation Heatmap")  # Başlık ekliyoruz

# Fotoğraf olarak kaydetme
plt.savefig('/Users/buraktelli/Desktop/Sağlık_Proje/diabetes_correlation_heatmap.png')  # Resmi kaydediyoruz
plt.show()  # Grafiği gösteriyoruz

df.info()

describe = df.describe()
sns.pairplot(df, hue= "Outcome")
plt.show()

def plot_correlation_heatmap(df):

    corr_matrix = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=0.5) # görselleştirme için eğer annot=True, fmt=".2f", linewidths=0.5 bunları eklemeseydik
    # arasıdaki boşluk rakamların gözükmesi ve aynı zamanda , sonra fazla sıfır oalcaktı
    plt.title("Correlation of Features")
    plt.show()

plot_correlation_heatmap(df)


# Outliner detection

def detect_outliers_iqr(df):
    outlier_indices = []
    outliers_df = pd.DataFrame()
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        Q1 = df[col].quantile(0.25)  # first quartile
        Q3 = df[col].quantile(0.75)  # third quartile

        IQR = Q3 - Q1  # interquartile range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_in_col = df[(df[col] <= lower_bound) | (df[col] >= upper_bound)]
        outlier_indices.extend(outliers_in_col.index)
        outliers_df = pd.concat([outliers_df, outliers_in_col], axis=0)

# remove duplicate indices
    outlier_indices = list(set(outlier_indices))

# remove duplicate rows in the outliers dataframe
    outliers_df = outliers_df.drop_duplicates()

    return outliers_df, outlier_indices

outliers_df, outlier_indices = detect_outliers_iqr(df)

# remove outliers from the dataframe

df_cleaned = df.drop(outlier_indices).reset_index(drop=True)

# train test

X = df_cleaned.drop(["Outcome"], axis = 1)
y = df_cleaned["Outcome"]
X_train, X_test, Y_train, Y_test = train_test_split(X,y ,test_size = 0.25, random_state=42)

# standartizasyon
scaler = StandardScaler()

X_test_scaled = scaler.fit_transform(X_test)
X_train_scaled = scaler.fit_transform(X_train)


# Öğrenme işlemleri
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
    basedModels.append(("LR",LogisticRegression()))                       # En yüksek
    basedModels.append(("DT",DecisionTreeClassifier()))                   # En düşük
    basedModels.append(("KNN",KNeighborsClassifier()))
    basedModels.append(("NB",GaussianNB()))
    basedModels.append(("SVM",SVC()))
    basedModels.append(("AdaB",AdaBoostClassifier()))
    basedModels.append(("GBM",GradientBoostingClassifier()))
    basedModels.append(("RFC",RandomForestClassifier()))

    return basedModels

def basedModelsTraining(X_train, Y_train, models):

    results = []
    names = []
    for name , model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name}: accuracy:{cv_results.mean()}, std: {cv_results.std()}")

    return names, results

def plot_box(names, results):
    df = pd.DataFrame({names[i]: results[i]for i in range(len(names))})
    plt.figure(figsize = (12,8))
    sns.boxplot(data=df)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.show()

models = getBasedModel()
names, results = basedModelsTraining(X_train, Y_train, models)
plot_box(names, results)

# hyperparameter tuning for DecisionTreeClassifier

param_grid = {
    "criterion": ["gini","entropy"],
    "max_depth": [10, 20, 30, 40, 50,],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
    }

dt = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid , cv=5 , scoring="accuracy")
grid_search.fit(X_train, Y_train)

print("En iyi parametreler :", grid_search.best_params_)

best_dt_model = grid_search.best_estimator_
y_pred = best_dt_model.predict(X_test)

# Confusion Matrix Görselleştirmesi

cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix\n\"[[88 21]\n[28 23]]\"")  # Confusion matrix sonucu
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

report = classification_report(Y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Precision, Recall, F1-Score için bar grafiği
report_df = report_df.drop('accuracy', axis=1)  # accuracy metric'i grafikte yer almayacak
report_df.plot(kind='bar', figsize=(10, 6))
plt.title("Classification Report Metrics\n\"precision, recall, f1-score için grafiği bar ile görselleştirdik.\"")
plt.xlabel("Classes")
plt.ylabel("Scores")
plt.xticks(rotation=0)
plt.show()

# Sonuçları yazdırma
print("Confusion Matrix")
print(cm)

print("classification_report")
print(classification_report(Y_test, y_pred))
