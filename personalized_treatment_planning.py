# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# %% data create

"""
1000 adet hasta
age: 20-80
gender: K(0)-E(1) 
disease: 0(yok), 1(hipertansiyon), 2(diyabet), 3(her ikisi)
semptomlar: ates + oksuruk + bas agrisi
    ates: 0(yok), 1(hafif) , 2(siddetli)
    oksuruk: 0(yok), 1(hafif) , 2(siddetli)
    bas agrisi: 0(yok), 1(hafif) , 2(siddetli)
kan basinci: 90 - 180
kan sekeri: 70 - 200
onceki tedavi yaniti: 0(kotu), 1(orta), 2(iyi)
"""

"""
0(tedavi - 1) (basit tedavi) : kan basinci < 120, kan sekeri < 100, semptomlar <= 1
1(tedavi - 2) (orta tedavi)  : kan basinci > 120, kan sekeri < 140, semptomlar == 1
2(tedavi - 3): (ileri seviye): kan basinci > 140, kan sekeri >= 150, semptomlar == 2
"""

def assign_treatment_1(blood_pressure, blood_sugar, symptom):
    return (blood_pressure < 120) & (blood_sugar < 100) & (symptom <= 1)

def assign_treatment_2(blood_pressure, blood_sugar, symptom):
    return (blood_pressure > 120) & (blood_sugar < 140) & (symptom == 1)

def assign_treatment_3(blood_pressure, blood_sugar, symptom):
    return (blood_pressure > 140) & (blood_sugar >= 150) & (symptom == 2)

num_samples = 1000

age = np.random.randint(20, 80, size = num_samples)
gender = np.random.randint(0, 2, size = num_samples)
disease = np.random.randint(0, 4, size = num_samples)
symptom_fever = np.random.randint(0, 3, size = num_samples)
symptom_cough = np.random.randint(0, 3, size = num_samples)
symptom_headache = np.random.randint(0, 3, size = num_samples)
blood_pressure = np.random.randint(90, 180, size = num_samples)
blood_sugar = np.random.randint(70, 200, size = num_samples)
previous_treatment_responce = np.random.randint(0, 3, size = num_samples)

symptom = symptom_fever + symptom_cough + symptom_headache

treatment_plan = np.zeros(num_samples)

for i in range(num_samples):
    if assign_treatment_1(blood_pressure[i], blood_sugar[i], symptom[i]):
        treatment_plan[i] = 0 # tedavi 1
    elif assign_treatment_2(blood_pressure[i], blood_sugar[i], symptom[i]):
        treatment_plan[i] = 1 # tedavi 2
    else:
        treatment_plan[i] = 2 # tedavi 3

data = pd.DataFrame({
    "age":age,
    "gender":gender,
    "disease":disease,
    "symptom_fever":symptom_fever,
    "symptom_cough":symptom_cough,
    "symptom_headache":  symptom_headache,
    "blood_pressure":blood_pressure,
    "blood_sugar":blood_sugar,
    "previous_treatment_responce":previous_treatment_responce,
    "symptom":symptom,
    "treatment_plan":treatment_plan})

# %% training: DL -> Artificial Neural Network
X = data.drop(["treatment_plan"], axis = 1).values
y = to_categorical(data["treatment_plan"], num_classes = 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size = 32)

# %% evaluation
val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f"Validation accuracy: {val_accuracy}, validation loss: {val_loss}")

plt.figure()

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], "bo-", label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], "r^-", label = "Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["loss"], "bo-", label = "Training Loss")
plt.plot(history.history["val_loss"], "r^-", label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)





































