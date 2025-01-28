# Proyecto Final: Machine Learning para Clasificar la variedad de los Vinos de la empresa JP 🍇🍷
# Desarrollado por: Julio Paredes

# En este proyecto, realizaremos un análisis exploratorio de datos sobre el conjunto de datos de vinos. 
# El objetivo es comprender mejor las características que influyen en la calidad del vino y explorar posibles patrones 
# y relaciones entre estas características. El conjunto de datos contiene información sobre diversas propiedades 
# físico-químicas de vinos, así como su calidad percibida. Utilizaremos técnicas de visualización y 
# resúmenes estadísticos para analizar los datos y responder preguntas clave que permitan clasificar 
# los vinos eficientemente mediante un modelo ML.

# Preguntas e Hipótesis de Interés (partimos por lo menos con tres preguntas)
# ¿Existe alguna relación entre las propiedades físico-químicas de los vinos y su calidad percibida?
# ¿Qué características tienen mayor impacto en la calidad del vino?
# ¿Cómo varían las propiedades del vino en función de su calidad?
# ¿Se podrá aplicar algún algoritmo ML que permita conocer la calidad de los vinos y ver a que clase pertenece?
# El objetivo de este código es realizar un ejercicio de clasificación utilizando un modelo de regresión logística en 
# un conjunto de datos de vinos.

# Análisis Exploratorio de Datos
# Realizaremos un análisis exploratorio de datos utilizando gráficos interactivos y resúmenes numéricos para responder 
# nuestras preguntas de interés y explorar insights preliminares sobre el conjunto de datos de vinos.

# Verificación de Valores Nulos
# Primero, verificaremos si hay valores nulos en el conjunto de datos y manejaremos los valores faltantes si es necesario.

# Visualizaciones Interactivas
# Utilizaremos gráficos interactivos para explorar las relaciones entre las características del vino y su calidad percibida. 
# Esto nos permitirá visualizar patrones y tendencias de manera más dinámica.

## LIBRERÍAS ##
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as pex
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

## ACCESO A LA BASE DE DATOS ##

# El archivo ya se envuentra en la misma carpeta del proyecto asi que llamo directamente
data = pd.read_csv('wine.csv')

# Mostrar las primeras filas
print(data.head())

# Muestra la dimension del dataset: folas * columnas
print(data.shape)

# Para conocer el tipo de dato que hay en cada columna
data.info()

# POR COMODIDAD CONTINUAMOS EN EL NOTEBOOK