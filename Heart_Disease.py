import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn. ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings. filterwarnings("ignore")

#load dataset ve EDA

#df = pd.read_csv("/Users/buraktelli/Desktop/Sağlık_Proje/heart_disease_uci.csv")

df = pd. read_csv("heart_disease_uci.csv")

# handling missing value
# train test split
# standardizasyon
# kategorik kodlama
# modelling: RF, KNN, Voting Classifier train ve test
# CM