# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:40:15 2024

@author: alan_
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

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

# Se deben dividir los datos para variable dependiente (Y) y variables independientes (X)

y = df['hospital_death']
x = df.drop('hospital_death', axis=1)

#codificar los valores string para que se pueda usar en el analisis
x = pd.get_dummies(x)

##----------------------------------------REGRESION LINEAL------------------------------------------

## Imputacion de valores ya que el LinearRegression no puede trabajar con valores faltantes
imputer = SimpleImputer(strategy='mean')
x_imputed = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

x_train, x_test, y_train, y_test = train_test_split(x_imputed, y, test_size=0.25, random_state=42)
#modelo de regresion
reg = LinearRegression().fit(x_train, y_train)

#prediccion sobre los datos con los que haremos las pruebas
y_pred = reg.predict(x_test)

##Precision
r2 = r2_score(y_test, y_pred)
print(f"Precision con Regresion lineal: {r2}")

#--------------------------------------HIST REGRESSOR-------------------------------------------
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.25, random_state=42)

#modelo de regresion
reg2 = HistGradientBoostingRegressor().fit(x_train, y_train)


#prediccion sobre los datos con los que haremos las pruebas
y_pred2 = reg2.predict(x_test)

##Precision

r2H = r2_score(y_test, y_pred2)

print(f"Precision con Hist: {r2H}")

#--------------------------------

print(f"Total time: {round(time.time() - start,3)}s")
