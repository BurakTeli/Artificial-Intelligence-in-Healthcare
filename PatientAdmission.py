import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Veriyi yükle
df = pd.read_csv('/Users/buraktelli/Desktop/Sağlık_Proje/3_kodlar_verisetleri/4_SagliktaMakineOgrenmesiUygulamalari/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv')
df_ = df.head(50)

# EDA - Görselleştirmeler
f, ax = plt.subplots()
sns.boxplot(x="Payment Typology 1", y="Length of Stay", data=df)
plt.title("Payment Typology 1 vs Length of Stay")
plt.xticks(rotation=60)
plt.show()

f, ax = plt.subplots()
sns.countplot(x="Age Group", data=df[df["Payment Typology 1"] == "Medicare"], order=["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"])
plt.title("Medicare Patients for Age Group")
plt.show()

f, ax = plt.subplots()
sns.boxplot(x="Type of Admission", y="Length of Stay", data=df)
plt.title("Type of Admission vs Length of Stay")
plt.xticks(rotation=60)
plt.show()

f, ax = plt.subplots()
sns.boxplot(x="Age Group", y="Length of Stay", data=df, order=["0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older"])
plt.title("Age Group vs Length of Stay")
plt.xticks(rotation=60)
ax.set(ylim=(0, 25))
plt.show()

# "los_bin" dağılımı
df["los_bin"] = pd.cut(x=df["Length of Stay"], bins=[0, 5, 10, 20, 30, 50, 120], labels=[5, 10, 20, 30, 50, 120])
f, ax = plt.subplots()
sns.countplot(x="los_bin", data=df)
plt.title("Length of Stay Binned Distribution")
plt.show()

# Karar Ağacı modelinin tahmin sonuçlarını görselleştirme
X = df.drop(["Length of Stay", "los_bin", "los_label"], axis=1)
y = df["los_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtree = DecisionTreeClassifier(max_depth=10)
dtree.fit(X_train, y_train)

train_prediction = dtree.predict(X_train)
test_prediction = dtree.predict(X_test)

# Gerçek ve tahmin edilen değerler arasındaki farkları gösteren grafik
f, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=y_test, y=test_prediction)
plt.title("True vs Predicted Length of Stay Categories (Test Set)")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()
