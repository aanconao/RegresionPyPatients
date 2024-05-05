# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:40:15 2024

@author: alan_
"""

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

start = time.time()

df = pd.read_csv("dataset.csv", delimiter=",")

##---------------------------Eliminar las columnas que no vamos a usar-------------------------##
columns_to_delete = ['encounter_id', 'patient_id', 'elective_surgery',
                     'apache_2_diagnosis', 'apache_post_operative', 'Unnamed: 83']

columns_to_delete = columns_to_delete + \
    list(df.columns[df.columns.str.contains('h1')])
columns_to_delete = columns_to_delete + \
    list(df.columns[df.columns.str.contains('noninvasive')])
columns_to_delete = columns_to_delete + \
    list(df.columns[df.columns.str.contains('icu')])


df.drop(columns_to_delete, axis=1, inplace=True)
##---------------------------------------------------------------------------------------------##


##----------------Eliminar pacientes para tener la relacion de 70/30 vivos/muertos-------------##

# Eliminamos los primeros 68000 supervivientes del dataframe y nos los quedamos en otro dataframe
# para poder usarlos en el futuro para evaluar como funciona el modelo

survivors_to_delete = df[df['hospital_death'] == 0].iloc[:68000]
df.drop(survivors_to_delete.index, inplace=True)


# del(survivors_to_delete) #Descomentar si no nos quisieramos quedar con los que quitamos del df

# Eliminamos los primeros 1500 muertos del dataframe y nos los quedamos en otro dataframe para poder
# usarlos en el futuro para evaluar como funciona el modelo

deaths_to_delete = df[df['hospital_death'] == 1].iloc[:1500]
df.drop(deaths_to_delete.index, inplace=True)


# del(deaths_to_delete) #Descomentar si no nos quisieramos quedar con los que quitamos del df

print(f"{df[df['hospital_death'] == 0].shape[0] / df.shape[0]} de supervivientes")
##---------------------------------------------------------------------------------------------##

##----------------------Crear el modelo de regresion y validad su precision--------------------##

##---------------------------------------------------------------------------------------------##

##----------------------------------------REGRESIÓN LOGÍSTICA--------------------------------------

# Codificación de la variable categórica 'hospital_death'
encoder = LabelEncoder()
y = encoder.fit_transform(df['hospital_death'])

# Separación de datos en conjuntos de entrenamiento y prueba
X = df.drop('hospital_death', axis=1)
x = pd.get_dummies(X)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Imputación de valores faltantes
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

# Creación del modelo de regresión logística
model = LogisticRegression()

# Entrenamiento del modelo
model.fit(x_train_imputed, y_train)

# Predicciones sobre el conjunto de prueba
y_pred = model.predict(x_test_imputed)

# Evaluación del rendimiento
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Precisión: {accuracy:.3f}")
print(f"Precisión: {precision:.3f}")

#-------------------------------------------

print(f"Total time: {round(time.time() - start,3)}s")
