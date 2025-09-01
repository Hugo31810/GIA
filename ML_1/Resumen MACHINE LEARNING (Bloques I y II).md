# Índice de contenidos:

^7ed2e5
```insta-toc
---
title:
  name: ""
  level: 1
  center: true
exclude: ""
style:
  listType: dash
omit: []
levels:
  min: 1
  max: 6
---

# <center></center>

- Índice de contenidos:
- Bloque I - Preprocesado de datos
    - 0 - Extra
        - Identificar columnas categóricas y numéricas
        - Responder a preguntas comunes
    - 1 - Tipos de datos
        - Creación de un DF random
        - Codificación de los datos categóricos
            - Codificación con la librería Pandas
            - Codificación con diccionarios
            - Codificación One-Hot (.getdummies)
            - Codificación con enteros
            - Obtener el DF final/codificado
    - 2 - Separar, explorar y rellenar
        - Separar un DF en test y train
        - Descripción estadística básica de los datos
        - Preprocesado de datos
            - Tratamiento de valores perdidos
            - Localización de valores perdidos
            - Imputación univariada
            - Imputación multivariada
    - 3 - Ingeniería de características
        - Aumento de la dimensionalidad
        - Reducción de la dimensionalidad
        - Transformaciones sin modificar la dimensionalidad
            - Escalado al intervalo unidad
            - Escalado al máximo en valor absoluto
            - Estandarización
    - 4 - Filtrado de características
        - Reducción de dimensionalidad mediante filtrado
            - Filtrado por varianza
            - Filtrado por correlación
            - Filtrado por información mutua
    - 5 - Análisis de Componentes principales (PCA)
    - 6 - Visualización de elementos básicos
        - Puntos
            - Representación de puntos en 2D
            - Representación de puntos en 3D
        - Rectas
        - Planos
        - Hiperplanos
- Bloque II - Aprendizaje supervisado
    - 7 - Modelos lineales
        - Modelo lineal de regresión - Regresión lineal
            - Formas de aplicar el modelo lineal de regresión
                - Usando sklearn.linearmodel.LinearRegression
                - Utilizando la fórmula deducida:
        - Modelo lineal de clasificación binaria
    - 8 - Regresión logística
    - 9 - Evaluación de modelos
        - Calidad de los modelos de clasificación binaria
            - Matriz de confusión:
            - Calidad de los modelos de clasificación binaria
            - Medidas derivación de la matriz de confusión
                - Accuracy (Exactitud)
                - Precision (Precisión)
                - Recall (Sensibilidad o Exhaustividad)
                - F1 - Score
    - 10 - Función de pérdida
        - Funciones de pérdida para regresión
    - Funciones de pérdida para regresión
        - Funciones de pérdida para la clasificación
    - 11 - Descenso de gradiente
        - Algoritmo Descenso de gradiente
        - Cómo saber según el gráfico de la función de pérdida que regularización se esta usando
        - Sin términos de regularización, únicamente la función de pérdida
    - 11.2 - Hiperparámetros
        - Tasa de aprendizaje
        - Tamaño del lote
        - Ciclos de entrenamientos (Épocas)
    - 11.3 - Aplicación del descenso de gradiente
    - 12 - Regularización
        - Regularización Lasso - Regularización L1
        - Regularización Ridge - Regularización L2
        - Regularización Elastic Net
    - Repaso general
    - Resolución de ejercicios prácticos
        - 1. Datos
        - 2. Separar e imputar
        - 3. Ingeniería de características
        - 4. Reducción de características
        - 5. Estandarización, Filtrado por varianza y PCA
        - 6. Rectas , planos e hiperplanos
        - 7. Regresión lineal
        - 8. Problema completo de clasificación binaria
        - 9. Regularización
```



# Bloque I - Preprocesado de datos
## 0 - Extra
### Identificar columnas categóricas y numéricas
```python
categorial_cols = df.select_dtypes(include=['objetcs']).columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
```
### Responder a preguntas comunes
```python
#@title Double-click or run to view answers about dataset statistics

answer = '''
What is the maximum fare? 				              Answer: $159.25
What is the mean distance across all trips? 		Answer: 8.2895 miles
How many cab companies are in the dataset? 		  Answer: 31
What is the most frequent payment type? 		    Answer: Credit Card
Are any features missing data? 				          Answer: No
'''

# You should be able to find the answers to the questions about the dataset
# by inspecting the table output after running the DataFrame describe method.
#
# Run this code cell to verify your answers.

# What is the maximum fare?
max_fare = training_df['FARE'].max()
print("What is the maximum fare? \t\t\t\tAnswer: ${fare:.2f}".format(fare = max_fare))

# What is the mean distance across all trips?
mean_distance = training_df['TRIP_MILES'].mean()
print("What is the mean distance across all trips? \t\tAnswer: {mean:.4f} miles".format(mean = mean_distance))

# How many cab companies are in the dataset?
num_unique_companies =  training_df['COMPANY'].nunique()
print("How many cab companies are in the dataset? \t\tAnswer: {number}".format(number = num_unique_companies))

# What is the most frequent payment type?
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("What is the most frequent payment type? \t\tAnswer: {type}".format(type = most_freq_payment_type))

# Are any features missing data?
missing_values = training_df.isnull().sum().sum()
print("Are any features missing data? \t\t\t\tAnswer:", "No" if missing_values == 0 else "Yes")

```
## 1 - Tipos de datos
[[#^7ed2e5|Home]]

[[1 y 2- Tipos de datos y separar, explorar y rellenar.]]

### Creación de un DF `random`

```python
import pandas as pd  
import random

genero = ['H', 'M', 'U']  
material = ['Algodón', 'Lana', 'Pana', 'Cuero', 'Microfibra', 'Poliester']  
talla = ['-XS', 'XS', 'S', 'M', 'L', 'XL', '+XL']  
estilo = ['Deporte', 'Fiesta', 'Joven', 'Trabajo', 'Etiqueta']

N = 10  
df_ejemplo = pd.DataFrame({  
    'Genero': [random.choice(genero) for _ in range(N)],  
    'Material': [random.choice(material) for _ in range(N)],  
    'Talla': [random.choice(talla) for _ in range(N)],  
    'Estilo': [random.choice(estilo) for _ in range(N)],  
})
```
### Codificación de los datos categóricos
#### Codificación con la librería `Pandas`
- Si queremos codificar todos los atributos categóricos de un DF debemos recorrer sus columnas ejecutando `cat.codes`.
- Para saber a qué categoría se corresponde cada entero lo mejor es ir creando un diccionario al mismo tiempo. Para ello utilizaremos primero `cat.categories` y luego juntaremos códigos y categorías en la estructura de datos `dict` de Python.
```python
#Codificamos el primer atributo
codes1 = df_categorico['Atributo 1'].cat.codes
categ1 = df_categorico['Atributo 1'].cat.categories #opcional
#generamos el diccionario:
code_to_categ1 = dict(zip(codes1,df_categorico['Atributo 1']))

#Codificamos el segundo atributo
codes2 = df_categorico['Atributo 2'].cat.codes
categ2 = df_categorico['Atributo 2'].cat.categories #opcional
#generamos el diccionario:
code_to_categ2 = dict(zip(codes2,df_categorico['Atributo 2']))
```
#### Codificación con diccionarios 
- Primero creamos el diccionario y su inverso (para descodificar si lo necesitamos más tarde)
```python
genero = ['H', 'M', 'U']
categ_to_code_genero= {string: i for i, string in enumerate(genero)}
code_to_categ_genero = {i: string for string, i in categ_to_code_genero.items()}
```
- Cambiamos los datos categóricos dados por los del diccionario (solo para columnas)
```python
x = df.copy()
x['Genero'] = x['Genero'].map(categ_to_code_genero)
```
#### Codificación *One-Hot* (`.get_dummies`)
- En esta codificación se crean tantas columnas como categorías diferentes hay por cada atributo categórico.  La codificación se realiza escribiendo un 1 en aquella columna que se corresponde con la categoría y un 0 en todas las demás.
```python
one_hot = pd.get_dummies(df_categorico['Atributo 1'])
# Unir el dataframe original con el dataframe codificado
df = pd.concat([df_categorico, one_hot], axis=1)
```

#### Codificación con enteros
```python
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
train_cat_enc = pd.DataFrame(index=train_df.index)
for col in cat_cols:
    le = LabelEncoder()
    train_cat_enc[col] = le.fit_transform(train_df[col].astype(str))
    label_encoders[col] = le
```
#### Obtener el DF final/codificado 
- Si el DF con el que estamos trabajando contiene a parte de datos categóricos, otro tipo debemos se debe hacer una selección de cuales codificar.
- Si queremos cambiar todas las columnas categóricas por un código debemos hacerlo en un bucle PERO primero hay que identificar en cuales hay que actuar.
- Listamos las columnas que son categóricas y luego las codificamos
- Se puede editar el ``for`` para realizar la codificación con los distintos métodos que hemos visto antes
```python
# 1) averiguamos las columnas categóricas
cat_cols = df_categorico.select_dtypes(include='category').columns.tolist()
# 2) creamos un dataframe con las columnas categóricas pero sin filas
df_cat_coded = pd.DataFrame(columns=cat_cols)
# 3) creamos un bucle que las recorra y las codifique, a la vez que creamos una diccionario de diccionarios para descodificar en el futuro
dict_decode={}
for col in cat_cols:
  codes = df_categorico[col].cat.codes #se crean los codigos
  code_to_categ = dict(zip(codes,df_categorico[col])) # se crean los diccionarios
  df_cat_coded[col] = codes #los codigos generados se añaden al DF
  dict_decode[col] = code_to_categ #se añade el diccionario a un df

print(df_cat_coded)
print(dict_decode)
```
## 2 - Separar, explorar y rellenar
[[#^7ed2e5|Home]]
[[1 y 2- Tipos de datos y separar, explorar y rellenar.]]
### Separar un DF en `test` y `train`
- Debemos de separar los datos que nos dan en dos grupos un de `train` y otro de `test`. La separación irá guiada normalmente por un porcentaje.
```python
from sklearn.model_selection import train_test_split
credit_train_df, credit_test_df = train_test_split(credit_df, test_size=0.2)
```
- Si queremos que la separación debe ser tal que se preserve la proporción de ejemplos de clase 1 y clase 2 en todos los conjuntos `stratify`
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True,stratify=Y, random_state=1460)
```
### Descripción estadística básica de los datos
- Es importante, en la medida de lo posible, conocer como son nuestro datos, para ello tenemos los siguientes comandos: 
```python
N, D = df.shape #N= numero de ejemplos, D = numero de columnas
df.dtypes #imprime el tipo de dato de cada atributo
print(df.info()) #similar a lo anterior
df.describe() # descripción estadistica básica
```
### Preprocesado de datos
#### Tratamiento de valores perdidos
#### Localización de valores perdidos
```python
credit_df = credit_df.replace('?', pd.NA) #para cambiar el "?" por NA
missing_data = credit_df.isna() #para localizar los valores peridos (tabla que pone true donde falta algun valor)

#para ver de forma visual los datos faltantes por columna
missing_values_per_column = missing_data.sum(axis=0)
mask_mayorq0 = missing_values_per_column > 0
print(missing_values_per_column[mask_mayorq0])

#otra forma mas sencilla
missing_count = missing_values_per_row.value_counts().sort_index()
print(missing_count)
```
#### Imputación univariada
- Es rellenar el valor que falte con un estadístico como la media, la moda o la mediana.
- Se puede dar el caso de que esta técnica genere ejemplos imposibles, para evitarlo podemos usar el método `interpolate()` o hacer una imputación multivariada
```python
df.fillna(dfoo.median(axis=0),inplace=True)
df.fillna(dfoo.mean(axis=0),inplace=True)
```
- Si nuestro DF contiene atributos categóricos necesitaremos primero codificarlos, aún así podemos hacer una imputación sin codificar
```python
df.fillna(df.select_dtypes(include=['number']).mean(), inplace=True)
```
#### Imputación multivariada
- Calcula el valor imputado en función del resto de atributos (como una especie de predicción en base al resto del DF)
- Para poder hacer imputación univariada necesitamos saber sobre modelos lineales de ML.
## 3 - Ingeniería de características
[[#^7ed2e5|Home]]
[[3 - Ingeniería de características]]
```PYTHON
#Librerías necesarias
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
```
### Aumento de la dimensionalidad
- Aumentar la dimensionalidad significa añadir nuevas características calculadas a partir de las que tenemos.
- Ejemplo de como hacer aumentado de dimensionalidad mediante características polinómicas:
	- Comando `fit` se usa en: 
		- Conjunto de `TRAIN`
	- Comando `transform` se usa en:
		- Conjunto de `TRAIN`
		- Conjunto de `TEST`
- Si queremos añadir características polinómicas de todas las columnas del DF
```python
degree = 2 #máximo grado de los polinomios resultantes
interaction_only = True #Añadir o no x^2

df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})

polyf = PolynomialFeatures(degree=degree,interaction_only=interaction_only)

polyf.set_output(transform="pandas")
polyf.fit(df)
df_poly = polyf.transform(df)

print('Dataframe inicial:')
print(df)
print('\nDataframe aumentado:')
print(df_poly)
```

- Si queremos añadir características polinómicas de columnas específicas del DF
```PYTHON
degree = 2 #máximo grado de los polinomios resultantes
interaction_only = True #Añadir o no x^2

df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})

polyf = PolynomialFeatures(degree=degree,interaction_only=interaction_only)

actuation = ['x1', 'x2']
polyf.set_output(transform="pandas")
polyf.fit(df[actuation])
df_poly = polyf.transform(df[actuation])

#Para obtener el DF original+nuevas columnas hacemos lo siguiente
df_poly = df_poly.drop(columns = actuation) #eliminamos las columnas sobre las que actuamos
df_poly_final = pd.concat([df, df_poly], axis = 1) #concatenamos el df original con la ampliacion que le hicimos
print(df_poly_final)
```

- Aumentar la dimensionalidad añadiendo características del tipo $log()$ y $exp()$
```python
# Crear el dataframe de juguete  
N = 10  
data = {  
    'x1': np.random.rand(N) * 10000,  
    'x2': np.random.rand(N) * 100,  
    'x3': np.random.randint(1, 100, N),  
    'x4': np.random.randint(0, 5, N),  
    'x5': np.random.randn(N),  
    'x6': np.random.randint(0, 2, N),  
}  
df = pd.DataFrame(data)  
   
  
# 1. Añadir características log(·) a todas las características  
# Evitamos log(0) añadiendo un pequeño valor a los datos que pueden ser 0  
log_df = df.copy()  
for col in log_df.columns:  
    log_df[f'log_{col}'] = np.log(log_df[col] + 1e-6)  
  
# 2. Añadir características exp(·) solo a las características originales  
exp_df = df.copy()  
for col in data.keys():  # Solo columnas originales  
    exp_df[f'exp_{col}'] = np.exp(exp_df[col])  
  
# Mostrar los dataframes transformados   
print("\nDataFrame con transformaciones logarítmicas:")  
print(log_df)  
print("\nDataFrame con transformaciones exponenciales:")  
print(exp_df)
```
### Reducción de la dimensionalidad 
Lo veremos en los puntos 4 y 5
### Transformaciones sin modificar la dimensionalidad
Hacer modificaciones sobre un atributo aunque no se cree otro nuevo también se considera ingeniería de características.
#### Escalado al intervalo unidad
Dada una columna $x$, donde $x_{\rm min}~$ y $x_{\rm max}$ son el valor mínimo y máximo alcanzados, entonces la siguiente formula escala todos los valores al intervalo $[0,1]$
$$x_{\rm esc} = \frac{x - x_{\rm min}}{x_{\rm max}-x_{\rm min}}$$
- Esta operación se debe hacer columna a columna ya que los valores máximo y mínimo de $x_i$ pueden ser diferentes a los de $x_j$.
- Como resultado tenemos todas las características a la misma escala.
- Podemos utilizar `sklearn.preprocessing.MinMaxScaler`
```python
# Escalado al intervalo unidad
#suponemos que df es nuestro dataframe
scalerUnit = MinMaxScaler().set_output(transform="pandas")
scalerUnit.fit(df)
scaleUnit_df = scalerUnit.transform(df)
```
#### Escalado al máximo en valor absoluto
Dada una columna $x$, la siguiente formula escala todos los valores de modo que:
- si todos son positivos, el valor máximo es $1$ y el mínimo es mayor que cero
- si todos son negativos, el valor mínimo es $-1$ y todos son menores que cero
- si hay positivos y negativos, todos quedan transformados dentro del intervalo $[-1,1]$.
$$x_{\rm esc} = \frac{x}{\max(|x|)}$$
- Esta operación se también se debe hacer columna a columna ya que el máximo del valor absoluto de los valores de dos columnas puede ser diferente.
- Podemos utilizar `sklearn.preprocessing.MaxAbsScaler`
```python
# Escalado al maximo de los valores absolutos
#suponemos que df es nuestro dataframe
scalerMaxabs = MaxAbsScaler().set_output(transform="pandas")
scalerMaxabs.fit(df)
scalerMaxabs_df = scalerMaxabs.transform(df)
```
#### Estandarización
Dada una columna $x$, con media $\mu$ y desviación $\sigma$, entonces la siguiente fórmula *estandariza* todos los valores de dicha columna
$$x_{\rm std} = \frac{x - \mu}{\sigma}$$
- Estandarizar significa que tiene media cero y desviación unidad.
- Esto **NO** significa que hayamos convertido la distribución de la columna $x$ en una normal.
$$¿ ~~ x_{\rm std} \sim \mathcal{N}(0,1) ~~ ?~~ \Large \leftarrow\text{ ¡NO!}$$
- Podemos utilizar `sklearn.preprocessing.StandardScaler`
```python
# Esrandarización
#suponemos que df es nuestro dataframe
scalerStd = StandardScaler().set_output(transform="pandas")
scalerStd.fit(df)
scaleStd_df = scalerStd.transform(df)
```
## 4 - Filtrado de características
[[#^7ed2e5|Home]]
[[4 - Reducción y filtrado]]
```python
# Librerías necesarías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
```
- Métodos de filtrado univariados
- Para cada característica de la tabla de datos se calcula una puntuación (score) de cuanta relevacia tiene 
- En el filtrado se eliminan todas a quellas características que no superan un umbral (threshold) elegido por nosotros.
### Reducción de dimensionalidad mediante filtrado
#### Filtrado por varianza
- Solo miro una columna
- Se ve afectado por la escala
- ``Pandas``: Usaremos el método `var()` de un DF. Por defecto se calcula la varianza de la muestra que divide entre $N-1$, lo que se puede modificar con la opción `ddof = 0` (degree of freedom), estaríamos dividiendo entre $N$ y calculando la varianza poblacional.
- `Scikit-Learn`: usamos `sklearn.feature_selection.VarianceThreshold`. El propio método calcula la varianza poblacional de los datos
	- Es interesante escalar al intervalo unidad antes del filtrado, aunque luego podemos usar las columnas originales para continuar con nuestro análisis de datos.
	- En este caso hacemos un escalado respecto del máximo de cada característica con el comando `MaxAbsScaler()`
```python
# ejemplo de filtrado por varianza
#dado el datafame dfoo
threshold=0.1
selector = VarianceThreshold(threshold=threshold)
selector.set_output(transform="pandas")
####### Escalado respecto del maximo de cada caracteristica########
####### Podría ser una estandarización de los datos ###############
scaler = MaxAbsScaler()
scaler.set_output(transform='pandas')
scaler.fit(dfoo)
df_scaled = scaler.transform(dfoo)
###################################################################

selector.fit(dfoo)
df_filtered = selector.transform(dfoo)

print(f'El dataframe original es:')
print_info(dfoo)
print(f'\n-----------\n')
print(f'El dataframe filtrado por varianza > {threshold}:')
print_info(df_filtered)
```
#### Filtrado por correlación
- No se ve afectado por la escala
- Es más eficiente en problemas de regresión lineal que en problemas de clasificación.
- Otro método de filtrar características es obtener la matriz de correlación y buscar las celdas de la matriz a +1 o -1.
```python
# matriz de correlación
print('Matriz de correlación de dfoo')
print(dfoo.corr())
```
- A veces, es interesante conservar aquellas características que tengan una correlación muy alta respecto al `target`, nuestro vector `y`. Para ello podemos usar ``colorbar()`` y según los colores ver que columnas nos interesa conservar. PARA FROMA VISUAL
```python
df = pd.DataFrame({
    'x1': [1, 5, 3, 2, 8],
    'x2': [3, -2, 8, -0.5, -1],
    'x3': [-50, 200, 300, -400, -300],
    'y' : [1, 1, 1, 0, 0] #etiqueta en un problema supervisado

})
corr_mat = df.corr()
print("\nMatriz de correlación")
print(corr_mat) 

plt.matshow(corr_mat, cmap='Blues')
plt.colorbar()
plt.show()
```
- PARA ELIMINAR LAS QUE NO CUMPLEN CIERTO UMBRAL:

```python
threshold = 0.95
corr = train_cont_var.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
train_cont_final = train_cont_var.drop(columns=to_drop)
```
#### Filtrado por información mutua
- Relacionada con árboles de decisión 
- Se deja para temas más adelantes
## 5 - Análisis de Componentes principales (PCA)
[[#^7ed2e5|Home]]
[[5 - Análisis de componentes principales (PCA)]]
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```
- Es una técnica que se usa para reducir la dimensión de conjuntos de datos
- Se crean NUEVAS características que:
	- Sustituyen a las que había antes
	- Están ordenada, por lo que si elijo las $d$ primeras, estaré eligiendo las que contienen mayor información.
- Antes de hacer PCA **hay que estandarizar**. Porque es un método que surge de un problema de optimización y hay que estandarizar los autovalores. (Para ello en Python usamos `StandardScaled`)
	- Recordamos que estandarizar un conjunto de datos significa hacer que la media de los datos se acerque a 0 y que la desviación se acerque a 1.

- Ejemplo de PCA seleccionando directamente el número de componentes
```python 
#dado el df wine_df

#estandarizamos los datos
scaler = StandardScaler().set_output(transform="pandas")
scaler.fit(wine_df)
wine_std = scaler.transform(wine_df)

#aplicamos PCA seleccionando directamente en número de componentes
n_components = 2
pca = PCA(n_components =n_components).set_output(transform="pandas")
pca.fit(wine_std)
df_pca = pca.transform(wine_std)
```

- Ejemplos de PCA seleccionando directamente el ratio de varianza explicada total
```python
n_components = .9

pca = PCA(n_components = n_components).set_output(transform="pandas")
pca.fit(wine_std)
df_pca = pca.transform(wine_std)
print(f'Tabla con los componentes principales hasta explicar el {n_components*100}% de la varianza')

print(df_pca)
```
## 6 - Visualización de elementos básicos
[[#^7ed2e5|Home]]
[[6 - Visualizar elementos básicos]]
```python
# Librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
```
### Puntos
#### Representación de puntos en 2D
```python
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
print(wine_df.info())

x_feature = 'ash'
y_feature = 'hue'
mark_size = 10
alpha = 0.4
wine_df.plot.scatter(x=x_feature, y=y_feature, s=mark_size, alpha=alpha)
```
#### Representación de puntos en 3D
```python
x_feature = 'magnesium'
y_feature = 'ash'
z_feature = 'flavanoids'
mark_size = 10
alpha = 0.4

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(wine_df[x_feature], wine_df[y_feature], wine_df[z_feature],s=mark_size, alpha = alpha)
ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.set_zlabel(z_feature)

plt.show()
```
### Rectas
```python
# Ejemplo de recta con vector director _v_ y un punto de intercepción _x0_
x0 = 3
v = [-1., -2.]
m = v[1]/v[0]
x1 = np.linspace(-3,3,10)
x2 = x0 + m*x1
  
# recta
plt.plot(x1,x2,'b', alpha=0.4)

# punto de intercepción
plt.scatter(0,x0,c='k',s=30)
plt.text( 0,x0, ' $x_0$', fontsize=12, color='black')

# origen de coordenadas
plt.scatter(0, 0,  s=60, marker='+', c='k')

# vector director v
plt.arrow(0, 0, v[0], v[1],head_width=0.2, head_length=0.3, fc='black', ec='black')
plt.text( v[0], v[1], ' v', fontsize=12, color='black')

# vector característico c
plt.arrow(0,0, -m, 1., head_width=0.2, head_length=0.3, fc='red', ec='red')
plt.text( -m, 1., ' c', fontsize=12, color='red')
  
plt.grid()
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.show()
```
### Planos
```python
# Ejemplo de recta con vector director _c_ y término independiente _c0_

c0 = 3
c = [2,.3,1]

x1 = np.linspace(-5,5,5)
x2 = np.linspace(-5,5,5)
xx1,xx2 = np.meshgrid(x1,x2)

intercep = (-c0/c[2])
x3 = intercep + (-c[0]/c[2])*xx1 + (-c[1]/c[2])*xx2

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# recta
surf = ax.plot_surface(xx1,xx2,x3, cmap='jet', alpha=0.5,
                       linewidth=0, antialiased=False)
# punto de intercepción
ax.scatter(0,0,intercep,c='k',s=30)
ax.text( 0, 0, intercep, ' $x_0$', fontsize=12, color='black')
# origen de coordenadas
ax.scatter(0, 0, 0, s=60, marker='+', c='k')
# vector director c
ax.quiver(0, 0, 0, c[0], c[1], c[2])
ax.text( c[0], c[1], c[2], ' c', fontsize=12, color='black')

ax.grid()
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.axes.set_xlim3d(-5, 5)
ax.axes.set_ylim3d(-5, 5)
ax.axes.set_zlim3d(-5, 5)
ax.set_box_aspect([1.0, 1.0, 1.0])
plt.show()
```
### Hiperplanos
- No podemos representarlo en una pantalla

# Bloque II - Aprendizaje supervisado
## 7 - Modelos lineales
[[#^7ed2e5|Home]]
[[7 - Modelos lineales]]
```python
# Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import add_dummy_feature
from sklearn import linear_model
import matplotlib.pyplot as plt
```
### Modelo lineal de regresión - Regresión lineal
La regresión lineal es una técnica que se usa para encontrar la relación entre las variables. La regresión lineal encuentra la relación entre funciones y una etiqueta.
![[car-data-points-with-model.png]]
- El modelo lineal de regresión es lo que obtenemos al ajustar por mínimos cuadrados un conjunto de pares de puntos, pero generalizado a $D \geq 2$ dimensiones.
- Utilizando el método `sklearn.linear_model.LinearRegression`
- Calculado el vector de pesos óptimo ${\bf w}^*$ con la fórmula que hemos deducido.

- Preparamos unos datos de muestra "especiales": 
```python
#Comenzamos fijando el númeoro de ejemplos del conjunto de entrenamiento (N) y las semilla del generador de número aleatorios
N = 20
seed = 1460
np.random.seed(seed=seed)

#Creamos una función que devuelve los pares(x,y) de entrenamiento (aleatorios pero dependientes de la semilla)
def generate_XY(N):
  m = np.random.rand()
  b = np.random.randint(5)
  X = np.linspace(0,10,N) + np.random.random(N)
  Y = m*X + b
  Y = Y + np.random.random(N)*2
  X = X.reshape(-1,1) #<- necesario para tener vectores columna
  Y = Y.reshape(-1,1) #<- necesario para tener vectores columna
  return X,Y


X, Y = generate_XY(N)
```

#### Formas de aplicar el modelo lineal de regresión
##### Usando `sklearn.linear_model.LinearRegression`
- Se crea un objeto de la clase ``LinearRegression`` y después se usa:
	- `fit` para encontrar los parámetros
	- `predict` para hacer predicciones
<small>
En este caso las predicciones las haremos sobre el mismo conjunto $\bf X$, pero lo habitual es que sea sobre el conjunto de Test (obviamente distinto del conjunto de entrenamiento).
</small>
```python
reg = linear_model.LinearRegression()
reg.fit(X,Y)
y_hat1 = reg.predict(X)
```

##### Utilizando la fórmula deducida:
![[equation.png|275]]

- Se crea una función que calcula $w^*$ 
- Se crea el modelo lineal $f({\bf x};{\bf w}^*) = w_0^* + w_1^*x_1$ y obtenemos la predicción $\hat y$ para cada ejemplo.
```python
# obtener w*
def matrix_solution(X, Y):
  X = add_dummy_feature(X)
  XtX = np.matmul(X.T, X)
  invXtX = np.linalg.inv(XtX)
  w_star = np.matmul(invXtX, X.T)
  w_star = np.matmul(w_star, Y)
  return w_star

w_star = matrix_solution(X,Y)

#obtener y^
y_hat2 = w_star[0] + np.matmul(X,w_star[1:])
```
- Sacamos conclusiones pintando
	- los pares $(x^{(i)}, y^{(i)})$, con $i=1,2,\ldots,N$, con aspas negras
	- los pares $(x^{(i)}, \hat y^{(i)})$ obtenidos con `LinearRegression` en forma de línea roja que une los puntos.<br> Obviamente esta línea es una recta.
	- los pares $(x^{(i)}, \hat y^{(i)})$ obtenidos con la fórmula, con puntos amarillos, un poco más grandes que los puntos negros.

```python
plt.plot(X, y_hat1,'r')
plt.scatter(X, y_hat2,c='y',s=40)
plt.scatter(X,Y,c='k',s=10, marker = 'x')
plt.xlabel('$x_1$'); plt.ylabel('$y$')
plt.show()

print('\nUtilizando LinearRegression()')
print(pd.DataFrame({'w_star': [reg.intercept_, reg.coef_]}, index=['intercept', 'coef']))
print('\nUtilizando la fórmula')
print(pd.DataFrame(w_star, columns=['w_star']))
```
### Modelo lineal de clasificación binaria
$$\hat y = \mathrm{Signo}\big(w_0 + {\bf w}^\top{\bf x}\big).$$
- Este modelo asume que tenemos dos clases etiquetadas como $\{+,-\}.$
- De nuevo $\hat y$ representa la estimación que hace el modelo.
- Tendremos una función discriminante
	- En general se utiliza una **función discriminante** que transforme el resultado de $f({\bf x};{\bf w})$  (ya sea esta lineal o no) en la etiqueta estimada.
		- Función _Umbral_ 

- La etiqueta puede ser {0, 1}.
- La función discriminante consiste en superar el umbral $\theta$.
```python
reg = linear_model.LinearRegression()

def matrix_solution(X, Y):
  X = add_dummy_feature(X)
  XtX = np.matmul(X.T, X)
  invXtX = np.linalg.inv(XtX)
  w_star = np.matmul(invXtX, X.T)
  w_star = np.matmul(w_star, Y)
  return w_star

# Fijamos valores globales del ejemplo
N = 20
seed = 1460
np.random.seed(seed=seed)

# Nueva función "generate_XY"
def generate_XY(N):
  m = np.random.rand()
  b = np.random.randint(5)
  X = np.linspace(0,10,N) + np.random.random(N)
  Y = np.sign(np.random.normal(size=N))
  X = X.reshape(-1,1) #<- necesario para tener vectores columna
  Y = Y.reshape(-1,1) #<- necesario para tener vectores columna
  return X,Y

# Función discriminante
def discriminant(fun_out):
  return np.sign(fun_out)

#--Creamos N pares (x,y)
X, Y = generate_XY(N)

#--Utilizamos el objeto "reg" con la función discriminante
reg.fit(X,Y)
y_hat1 = discriminant( reg.predict(X) )

#--Utilizamos la fórmula con la función discriminante
w_star = matrix_solution(X,Y)
y_hat2 = discriminant( w_star[0] + np.matmul(X,w_star[1]) )

#--Pintamos
plt.plot(X, y_hat1,'.r')
plt.scatter(X, y_hat2,c='y',s=40)
plt.scatter(X,Y,c='k',s=10, marker = 'x')
plt.xlabel('$x_1$'); plt.ylabel('$y$')
plt.grid()
plt.show()
```
## 8 - Regresión logística
[[#^7ed2e5|Home]]
Hemos explorado cómo construir un modelo para realizar predicciones numéricas continuas, como la eficiencia del combustible de un automóvil. Pero ¿Qué sucede si quieres crear un modelo para responder preguntas como "¿Lloverá hoy?" o "¿Es este correo electrónico spam?"?

La regresión logística es un modelo estadístico usado para predecir la posibilidad de que ocurra un evento binario, es decir, un resultado que solo tiene dos posibles salidas: sí o no, 1 o 0, aprobado o suspenso, enfermo o sano.

A diferencia de la regresión lineal, que predice un valor continuo, la regresión logística predice una probabilidad (un numero entre 0 y 1)

- Funciones sigmoide:
![[linear_to_logistic.png]]

| Nombre                               | Expresión                                                   |
| ------------------------------------ | ----------------------------------------------------------- |
| **Logística**                        | $S(x)$  =  $\frac{1}{1+e^{-x}}$                             |
| Tangente hiperbólica                 | $S(x)$ = $\tanh$$(x)$                                       |
| Arcotangente                         | $S(x)$ = $\arctan$$(x)$                                     |
| Función error                        | $S(x)$ = $\frac{2}{\sqrt{\pi}}$$\int\limits_0^x e^{-t^2}dt$ |
| y también funciones algebraicas como | $S(x)$=$\frac{x}{\sqrt{1+x^2}}$                             |

```python
# Librerías necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import add_dummy_feature
from sklearn import linear_model
import matplotlib.pyplot as plt
```
- Ejemplo con `LinearRegression` + Función logística
```python
seed = 1460
np.random.seed(seed=seed)

# Función que genera pares (x_1, x_2) con la misma etiqueta
def generateX_2D(N, mu=0, sigma=1, label=1):
  data = np.random.normal(mu, sigma, (N, 2))
  df = pd.DataFrame(data, columns=["x1", "x2"])
  df["y"] = np.ones(N)*label
  return df

# Función que genera pares (x_1, x_2) de varias etiquetas
def generate_XY(N, mu, sigma, label):
  df0 = generateX_2D(N, mu[0], sigma[0], label[0])
  df1 = generateX_2D(N, mu[1], sigma[1], label[1])
  df = pd.concat([df0,df1], axis=0)
  df = df.sample(2*N, ignore_index=True)
  return df

#-- generación del conjunto de datos para el ejemplo
N = 20
mu = [-1,1]
sigma = [1,1]
label = [-1,1]
df = generate_XY(N,mu,sigma,label)

#-- Aprendizaje del modelo lineal
X = df[['x1','x2']]
Y = df['y']
reg = linear_model.LinearRegression()
reg.fit(X,Y)
print(f'Los parámetros aprendidos para un modelo lineal son:')
print(f' w = {reg.coef_} , w0 = {reg.intercept_:0.2f}')

#-- Obtenemos los logits
logit = reg.predict(X)

#-- Convertimos los logits en probabilidad de pertenecer a la clase "1"
def logistic_fun(x):
    return 1 / (1 + np.exp(-x))

df["logit"]=logit
df['p(y=1)']=logistic_fun(logit)
df.head()
```

- Ejemplo con `Logistic Regression`
```python
#- Instanciamos un objeto de Regresion logística
logreg = linear_model.LogisticRegression()
#- Lo entrenamos con los datos dados
logreg.fit(X,Y)
#- Hacemos la predicción de la clase y también
#  la estimación de la probabilidad de pertenecer a cada clase
y_hat = logreg.predict(X)
prob  = logreg.predict_proba(X)

df['y_hat']=y_hat
df['prob y=-1'] =prob[:,0]
df['prob y=+1'] =prob[:,1]
df.head(10)
```
## 9 - Evaluación de modelos
[[#^7ed2e5|Home]]
### Calidad de los modelos de clasificación binaria
#### Matriz de confusión:
Cuando solo hay que predecir entre dos clases (positiva y negativa) podemos acertar de dos maneras y fallas de dos maneras.
#### Calidad de los modelos de clasificación binaria
```python
#Librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix
seed = 1460
```

Cuando sólo hay que predecir entre dos clases (positiva y negativa) podemos acertar de dos maneras y fallar de dos maneras también.<br>
Dado un ejemplo:
- Aciertos:
  - **Verdadero positivo**, _True Positive_, TP:
  $\hat y = + \quad,\quad y = +$ 
  Mi estimación es "Positivo" y la clase es "Positiva"
  - **Verdadero negativo**, _True Negative_, TN:
  $\hat y = - \quad,\quad y = -$ 
  Mi estimación es "Negativo" y la clase es "Negativa"
- Fallos:
  - **Falso positivo**, _False Positive_, FP:
  $\hat y = + \quad,\quad y = -$ 
  Mi estimación es "Positivo" y la clase es "Negativa"
  - **Falso negativo**, _False Negative_, FN:
  $\hat y = - \quad,\quad y = +$ 
  Mi estimación es "Negativo" y la clase es "Positiva"

Haciendo un recuento para cada ejemplo del conjunto de test, podemos escribir los cuatro casos en una tabla llamada **matriz de confusión**.

Todas ellas se ven reflejadas en la siguiente tabla a la que llamamos matriz de confusión:

#### Medidas derivación de la matriz de confusión
En la figura de abajo se muestran varias medidas que se calculan a partir de la terna (TP, TN, FP, FN).
Las más frecuentes son:
- **_Precision_**
_Precision_=1 significa que todos los ejemplo estimados como positivos efectivamente lo eran.
- **_Recall_**
_Recall_=1 significa que todos los ejemplos positivos han sido estimados correctamente.
- **_F1-score_**
Da una media entre _Precision_ y _Recall_.
- **_Accuracy_**
Es el porcentaje total de aciertos.

##### Accuracy (Exactitud)
Mide la proporción de predicciones que fueron correctas (tanto positivas como negativas)
(Aciertos totales sobre el total de casos)
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}​
$$

##### Precision (Precisión)
Se los casos que el modelo dijo que eran **positivos** ¿Cuántos lo son realmente?
"**PRE**cisión = ¿Cuántos de los **PRE**dichos positivos eran realmente positivos?"
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

##### Recall (Sensibilidad o Exhaustividad)
De todos los positivos reales, ¿Cuántos detectó el modelo?
"**RE**call = ¿cuántos positivos **RE**ales fueron encontrados?"
$$
\text{Recall} = \frac{TP}{TP + FN}

$$
##### F1 - Score
Es la media armónica entre precisión y recall. Un equilibrio entre ambos
"**F1** es el **Fair 1**, el punto medio justo entre precisión y recall."

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}

$$
| Métrica   | Pregunta clave                                   | Regla mnemotécnica                |
| --------- | ------------------------------------------------ | --------------------------------- |
| Accuracy  | ¿Qué tan bien acierta el modelo en general?      | Aciertos totales / Casos totales  |
| Precision | ¿De los positivos predichos, cuántos lo son?     | **PRE**dichos positivos correctos |
| Recall    | ¿De los reales positivos, cuántos se detectaron? | **RE**ales positivos encontrados  |
| F1        | ¿Qué tan equilibrados están precision y recall?  | **F1** = **Fair 1**               |

![[Sin título2.png]]

```python
# Entrenamiento del modelo de regresión logística
model = LogisticRegression()
model.fit(trainX, trainY)

# Cálculo de las probabilidades
y_score = model.predict_proba(testX)[:, 1]

# Cálculo de la curva ROC y AUC
fpr, tpr, thresholds = roc_curve(testY, y_score)
roc_auc = auc(fpr, tpr)

# Visualización de la curva ROC
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Mostrar resultados

print('\n', pd.concat([testY,
                       pd.DataFrame({'y_score':y_score}).set_index(testY.index)],
                      axis=1).sort_values(['y_score'])
      )

print('\n',
      pd.DataFrame({'fpr':fpr,
                    'tpr':tpr,
                    'th':thresholds}))

print(f'\nAUROC {roc_auc:0.3f}')

```
- Resultado habitual de la curva AUROC (lo perfecto es que vaya pegado a la izquierda y arriba)
![[Pasted image 20250303233523.png|300]]
## 10 - Función de pérdida
[[#^7ed2e5|Home]]
Tenemos a nuestra disposición (dado por el "cliente"):
- un conjunto de datos $\bf X$ en forma de tabla $N\times D$
- un conjunto $\bf Y$ de valores objetivos o etiquetas, que puede ser:
  - Para regresión: un vector columna de $N$ elementos en el primer caso.
  - Para clasificación: un vector columna o una matriz si se utiliza una representación _one-hot._
  A este conjunto frecuentemente se le denomina por su término en inglés, _ground truth_.

Evidentemente también se supone que tenemos un modelo, o sea una función paramétrica $~f({\bf x};{\bf w})~$. (Aportado por "nosotros")

El objetivo es encontrar los parámetros óptimos del modelo. Es decir, $w^*$

Cuando los valores perteneces al conjunto de entrenamiento conocemos el valor objetivo o la etiqueta real de cada ejemplo
La función de pérdida devuelve una medida relacionada con los fallos cometidos por el podemos respecto del *ground truth*.
**Sea $~\mathcal L \big( {\bf Y}, f({\bf X};{\bf w}) \big)~$ dicha función.**


![[loss-lines.png]]
### Funciones de pérdida para regresión
## Funciones de pérdida para regresión

En estos casos
$$
f({\bf x};{\bf w}): (\mathbb R^{D}; \mathbb R^{D}) \rightarrow \mathbb R,
$$
y el valor objetivo $y \in \mathbb R$ también.

Por tanto, una función de pérdida para regresión debe tener las siguientes características:
- Calcular $\delta_{(i)} = \left(y^{(i)} - f({\bf x}^{(i)},{\bf w}) \right)$ para cada $i = 1,2,\ldots,N.$
- Convertir $\delta_{(i)}$ en un valor positivo para todo $i$.
- Hacer el promedio para todos los ejemplos.

Algunas funciones de pérdida con estas características son:

**MSE** (_Mean Squared Error_)
$$
 \mathcal L \big( {\bf Y}, f({\bf X};{\bf w}) \big)
 =\frac{1}{N}\sum\limits_{i=1}^{N}\left(y^{(i)} - f({\bf x}^{(i)},{\bf w})\right)^2
$$

**MAE** (_Mean Absolute Error_)
$$
 \mathcal L \big( {\bf Y}, f({\bf X};{\bf w}) \big)
 =\frac{1}{N}\sum\limits_{i=1}^{N}\left\vert y^{(i)} - f({\bf x}^{(i)},{\bf w})\right\vert
$$

**Log-cosh**
$$
 \mathcal L \big( {\bf Y}, f({\bf X};{\bf w}) \big)
 =\frac{1}{N}\sum\limits_{i=1}^{N} \log\left(\cosh\big( y^{(i)} - f({\bf x}^{(i)})\big)\right)
$$
### Funciones de pérdida para la clasificación

A diferencia de la regresión, la clasificación puede tener varias formas.
1. Clasificación binaria:
_un ejemplo sólo puede pertenecer a 2 clases excluyentes_
1. Clasificación multi-clase:
_un ejemplo puede pertenecer a varias clases pero excluyentes_
1. Clasificación multi-etiqueta:
_un ejemplo puede pertenecer a una o varias clases al mismo tiempo_


Los tres casos se pueden abordar como un problema de regresión y después utilizar una función discriminante o también usar la función logística para obtener

PERO ya hemos aprendido que la regresión logística es un modo mucho más natural de abordar la clasificación porque incorpora una medida de la probabilidad de pertenecer a una cierta clase.

Sin embargo, cuando trabajamos con probabilidades las pérdidas citadas arriba **NO** son apropiadas.

Esta cuestión es tan importante que merece un cuaderno para ella sola, más adelante en el curso.


## 11 - Descenso de gradiente
[[#^7ed2e5|Home]]
El [**descenso de gradientes**](https://developers.google.com/machine-learning/glossary?hl=es-419#gradient-descent) es una técnica matemática que encuentra de forma iterativa los pesos y sesgos que producen el modelo con la pérdida más baja. El descenso por gradiente encuentra el mejor peso y sesgo repitiendo el siguiente proceso para una serie de iteraciones definidas por el usuario.

El descenso por gradiente es más antiguo que el aprendizaje automático.
El modelo comienza a entrenarse con ponderaciones y sesgos aleatorios cercanos a cero y, luego, repite los siguientes pasos:

1. Calcula la pérdida con el peso y el sesgo actuales.
2. Determina la dirección para mover los pesos y el sesgo que reducen la pérdida.
3. Mueve los valores de peso y sesgo una pequeña cantidad en la dirección que reduzca la pérdida.
4. Regresa al paso uno y repite el proceso hasta que el modelo no pueda reducir la pérdida más.

En el siguiente diagrama, se describen los pasos iterativos que realiza el descenso de gradientes para encontrar los pesos y el sesgo que producen el modelo con la pérdida más baja.


![[gradient-descent.png]]

- Curva de pérdida e instantánea del modelo al comienzo del proceso de entrenamiento
![[large-loss.png]]
- Curva de pérdida y captura del modelo a mitad del entrenamiento
![[med-loss.png]]
- Curva de pérdida y instantánea del modelo cerca del final del proceso de entrenamiento.
![[low-loss.png]]
- Modelo graficado con los valores de peso y sesgo que producen la pérdida más baja
![[graphed-model.png]]

### Algoritmo Descenso de gradiente
- Es un algoritmo para localizar el mínimo de una función de manera iterativa
- En cada iteración  nos desplazamos por w en la dirección que marca el gradiente de L 
- El algoritmo comienza en un punto aleatorio, y el desplazamiento produce un nuevo punto.
- Este proceso se repite, tomando como punto de partida el punto alcanzado en la iteración anterior, hasta que se cumpla una condición de parada
![[Pasted image 20250429124117.png]]![[Pasted image 20250429124140.png]]

---

![[Pasted image 20250429124334.png]]

---

![[Pasted image 20250429124316.png]]


### Cómo saber según el gráfico de la función de pérdida que regularización se esta usando 
- **L₁ (Lasso)**
    
    - Los contornos son **rombos** (los “círculos” de la norma-1 son diamantes).
        
    - La trayectoria tiende a moverse a lo largo de los ejes coordenados y a veces “salta” en las esquinas del rombo, promoviendo soluciones con pesos exactamente cero.
- **L₂ (Ridge)**
    
    - Los contornos (curvas de nivel) son **elipses o círculos** concéntricos alrededor del origen.
        
    - El descenso de gradiente “rodea” esos contornos de forma suave.
 ![[ChatGPT Image 29 abr 2025, 12_51_30.png|250]]


### Sin términos de regularización, únicamente la función de pérdida
![[ChatGPT Image 29 abr 2025, 12_57_38.png|475]]
## 11.2 - Hiperparámetros
[[#^7ed2e5|Home]]
- Los Hiperparámetros son variables que controlan diferentes aspectos del entrenamiento. Estos son los tres hiperparámetros más comunes:
	- Tasa de aprendizaje
	- Tamaño del lote
	- Épocas

### Tasa de aprendizaje
Es un numero que establecemos nostros y que influye en la rapidez con la que onverge el modelo.
Si la tasa de aprendizaje es demasiado baja el modelo puede tardar mucho en converger, pero si es muy alta el modelo puede que nunca converja.

| Imagen                   | Desripción                                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| ![[correct-lr 1.png]]    | Gráfico de pérdida que muestra un modelo entrenado con una tasa de aprendizaje que converge rápidamente. |
| ![[small-lr.png]]        | Gráfico de pérdida que muestra un modelo entrenado con una tasa de aprendizaje que converge rápidamente. |
| ![[high-lr.png]]         | Gráfico de pérdida que muestra un modelo entrenado con una tasa de aprendizaje que converge rápidamente. |
| ![[increasing-loss.png]] | Gráfico de pérdida que muestra un modelo entrenado con una tasa de aprendizaje que converge rápidamente. |
### Tamaño del lote
- Se refiere a la cantida de ejemplos que el modelo procesas antes de actualizar sus pesos y sesgos. 
- Hay dos técnicas comunes para obtener el gradiente correcto en el promedio sin necesidad de ver todos los ejemplos del conjunto de datos antes de actualizar los pesos y la polarización:
	- Descenso estocástico del gradiente (SGD): Solo usa un ejemplo por iteración. Cuando se dan suficientes iteraciones el SGD funciona, pero es muy ruidoso.
	![[noisy-gradient.png|300]]
	
	- Descenso estocástico del gradiente en minilotes (SGD de minilote): Es un equilibrio entre el SGD y el lote completo. Para una cantidad de datos de, el tamaño del lote puede ser cualquier número mayor que 1 y menor que. El modelo elige los ejemplos incluidos en cada lote de forma aleatoria, promedia sus gradientes y, luego, actualiza los pesos y sesgos una vez por iteración.
	![[mini-batch-sgd.png|300]]
### Ciclos de entrenamientos (Épocas)
Durante el entrenamiento, un ciclo de entrenamiento significa que el modelo procesó cada ejemplo del conjunto de entrenamiento _una vez_. Por ejemplo, dado un conjunto de entrenamiento con 1,000 ejemplos y un tamaño de minilote de 100 ejemplos, el modelo necesitará 10 iteraciones para completar una época.

El entrenamiento suele requerir muchas épocas. Es decir, el sistema debe procesar cada ejemplo del conjunto de entrenamiento varias veces.

La cantidad de épocas es un hiperparámetro que estableces antes de que el modelo comience a entrenarse. En muchos casos, deberás experimentar con la cantidad de épocas que tarda el modelo en converger. En general, más épocas producen un modelo mejor, pero también requieren más tiempo de entrenamiento.
![[batch-size.png]]

## 11.3 - Aplicación del descenso de gradiente
[[#^7ed2e5|Home]]
De este notebook de código no preguntará nada, solo teoría
## 12 - Regularización
[[#^7ed2e5|Home]]
La regularización consiste en añadir reglas para obtener $w^*$.
Antes, la única reglas que imponíamos para aprender el modelo es que los parámetros elegidos minimicen $\mathcal L \big( {\bf Y}, f({\bf X};{\bf w}) \big)$.
Puesto que $X$ e $Y$ vienen dados por el "cliente", lo único sobre lo que podemos imponer reglas es sobre los pesos.
Es decir, en vez de viajar libremente por el espacio de w buscando el punto $w^*$ donde se alcanza el mínimo de la pérdida, añadimos términos que restringen ese proceso de conseguir $w^*$.
De esa forma la función objetivos del problema de regularización se convierte en:
$$

\mathcal L \big( {\bf Y}, f({\bf X};{\bf w}) \big) +

\mathcal R \big( {\bf w} \big)

$$


### Regularización *Lasso* - Regularización L1
Consiste en añadir el término

$$

\mathcal R_{L} = \alpha \sum\limits_{i=1}^{D} \vert w_i\vert

$$

a la función de pérdida; donde $\alpha\ge 0$ es un valor que elegimos nosotros para contralar su efecto (es un hiper-parámetro).

- $\alpha \rightarrow 0:\quad$ Obtenemos el mismo $\bf w^*$ que con la regresión lineal, es decir desaparece el término de regularización.
- $\alpha > 0:\quad$ Algunos parámetros tenderán a ir hacia 0
- $\alpha \rightarrow \infty:~~$ Todos los parámetros se anulan.
En definitiva, quitando los dos casos extremos, el efecto de este regularizador es lograr que haya menos parámetros, porque muchos de ellos tenderán hacia 0.

- Dado que la norma-1 de un vector $\bf w$ es
$$\Vert {\bf w} \Vert_1 = \sum\limits_{i=1}^{D} \vert w_i\vert$$ a esta regularización también se le llama **Regularización L1**

- El nombre proviene de _Least Absolute Shrinkage and Selection Operator_, precisamente por la propiedad de seleccionar y reducir el número de parámetros.

- La clase `sklearn.linear_model.Lasso` implementa la pérdida con esta regularización para el modelo lineal.

También se puede usar en otros modelos que ofrecen la opción `penalty = "l1"`.
### Regularización Ridge - Regularización L2
Consiste en añadir el término

$$

\mathcal R_{R} = \frac{\alpha}{2} \sum\limits_{i=1}^{D} ( w_i )^2

$$

a la función de pérdida; donde $\alpha\ge 0$ es un valor que elegimos nosotros para contralar su efecto.
- $\alpha \rightarrow 0:\quad$Obtenemos el mismo $\bf w^*$ que con la regresión lineal, es decir desaparece el término de regularización, igual que con _Lasso_,

- $\alpha > 0:\quad$ Impide que los parámetros se separen mucho o que alguno crezca demasiado.<br>

- $\alpha \rightarrow \infty:~~$ Todos los parámetros se anulan, igual que con _Lasso_.
En definitiva, quitando los dos casos extremos, el efecto de este regularizador es lograr que todos los parámetros tengan valores pequeños, pero no necesariamente nulos.

>- Dado que la norma-2 al cuadrado de un vector $\bf w$ es

$$\Vert {\bf w} \Vert_2 = \sum\limits_{i=1}^{D} ( w_i )^2$$ a esta regularización también se le llama **Regularización L2**

- El factor $1/2$ se añade para que al derivar el término el exponente se cancele con él.

- La clase `sklearn.linear_model.Ridge` implementa la pérdida con esta regularización para el modelo lineal.

 También se puede usar en otros modelos que ofrecen la opción `penalty = "l2"`.

### Regularización *Elastic Net*
Consiste en añadir una combinación lineal de ambos
$$
\mathcal R_{E} = \alpha_1 \mathcal R_{L} + \alpha_2 \mathcal R_{R}
= \alpha_1 \Vert{\bf w}\Vert_1 + \frac{\alpha_2}{2} \Vert{\bf w}\Vert_2^2.
$$
Esta regularización causa un doble reducción de los parámetros: la que provoca Lasso y la de Ridge. Para tener más control y además una interpretación de $\alpha_1$ y $\alpha_2$ es más frecuente utilizar una combinación lineal convexa de ambos términos de regularización. Es decir:
$$
\mathcal R_{E} = \alpha \big( r \cdot \mathcal R_{L} + (1-r)  \cdot \mathcal R_{R} \big),
$$
donde $r$ es el porcentaje o ratio de regularización Lasso que aplicamos; y por tanto el porcentaje complementario hasta llegar a 100% es de regularización Ridge; y $\alpha$ es un multiplicador para dar más o menos peso al termino.

Comentarios:
>- La clase `sklearn.linear_model.ElasticNet` implementa la pérdida con esta regularización para el modelo lineal.
<BR> También se puede usar en otros modelos que ofrecen la opción `penalty = "elasticnet"`.

**Posible pregunta de examen:**
¿Qué condiciones hacen falta para inventarse una regularización ?
Un termino de regularización es una expresión matemática que solo puede incorporar w
Si tienes algo nuevo ponle un hiper-parámetro

Hasta aquí es el ultimo examen. Del bloque i y ii


## Repaso general 
1. El preprocesado de datos es el conjunto de operaciones que se realizan sobre los datos para transformarles, limpiarlos, normalizarlos, escalarlos o codificarlos
2. Las métricas de rendimiento son medidas que cuantifican el grado de generalización de un modelos a nuevos datos y que se buscan maximizar durante la validación
3. Un atributo categórico puede tomar un número finitos de valores
4. ``PolynomialFeatures`` es un técnica de aumento de dimensiones que consiste en crear nuevas características que son multiplicaciones de las características que ya tenemos elevadas a cualquier número no negativo 
5. La estandarización es el preprocesador de los atributos para que estos pasen a tener media cero y desviación unidad
6. El filtrado consiste en eliminar aquellos atributos o características que no aportan información para la tarea a desarrollar
7. La reducción de dimensionalidad es el proceso de proyectar los datos de un espacio de alta dimensión a uno de menor dimensión, preservando la mayor cantidad de información posible
8. La regularización elastic net entre otras cosas, consiste en modificar la función de pérdida añadiendo términos que dependen de los parámetros 
9. El aumento de dimensionalidad es el proceso de añadir nuevas variables o características a un conjunto de datos dervadas de las existentes o de fuentes externas.
10. Log-loss es la pérdida natural para aprender modelos de regresión log ´sitica
11. PCA es una técnica de reducción de dimensionalidad que consiste en encontrar las combinaciones lineales de las variables originales que capturan la mayor cantidad de varianza de los datos.
12. Una regresión lineal es una técnica de regresión que consiste en ajustar una función lineal que relaciona la etiqueta con las características
13. La regresión log ´sitica es una técnica de clasificación que consiste en ajustar una función log ´sitica que estima la probabilidad de pertenencia una clase binaria o múltiple y asigna la clase con mayor probabilidad
14. El descenso de gradiente es un método de optimización que consiste en actualizar los parámetros de un modelo de forma iterativa, siguiendo la dirección opuesta a l gradiente de la función de pérdida, hasta alcanzar un mínimo local o global
15. La función de pérdida es una medida que cuantifica la diferencia entre el valor real y el valor predicho por un modelo, y que se busca minimizar durante el entrenamiento del modelo
16. La regularización consiste en añadir un término a la función de pérdida que penaliza los valores altos de los parámetros de un modelo, evitando el sobreajuste o la complejidad excesiva

## Resolución de ejercicios prácticos
### 1. Datos
[[#^7ed2e5|Home]]
Este cuaderno plantea una serie de ejercicios relacionados con un conjunto de datos de compras por internet de ropa.

1. Crear un diccionario y su diccionario inverso para convertir cada uno de los siguientes atributos a un entero y viceversa.

- _Género_ ['H', 'M', 'U'] , indica si es una prenda para hombre, mujer o unisex.
- _Material_ ['Algodón', 'Lana', 'Pana', 'Cuero', 'Microfibra', 'Poliester'] , indica el tejido mayoritario de la prenda.
- _Talla_ ['-XS', 'XS', 'S', 'M', 'L', 'XL', '+XL']
- _Estilo_ ['Deporte', 'Fiesta', 'Joven', 'Trabajo', 'Etiqueta']

```python
import pandas as pd  
import random  
  
genero = ['H', 'M', 'U']  
material = ['Algodón', 'Lana', 'Pana', 'Cuero', 'Microfibra', 'Poliester']  
talla = ['-XS', 'XS', 'S', 'M', 'L', 'XL', '+XL']  
estilo = ['Deporte', 'Fiesta', 'Joven', 'Trabajo', 'Etiqueta']  
  
  
#Cremos los diccionarios de conversión  
categ_to_code_genero= {string: i for i, string in enumerate(genero)}  
categ_to_code_material = {string: i for i, string in enumerate(material)}  
categ_to_code_talla = {string: i for i, string in enumerate(talla)}  
categ_to_code_estilo = {string: i for i, string in enumerate(estilo)}  
  
#Cremaos los diccionarios inversos  
code_to_categ_genero = {i: string for string, i in categ_to_code_genero.items()}  
code_to_categ_material = {i: string for string, i in categ_to_code_material.items()}  
code_to_categ_talla = {i: string for string, i in categ_to_code_talla.items()}  
code_to_categ_estilo = {i: string for string, i in categ_to_code_estilo.items()}  
  
#Imprimimos los diccionarios  
print("Diccionario género->código:", categ_to_code_genero)  
print("Diccionario material->código:", categ_to_code_material)  
print("Diccionario talla->codigo:", categ_to_code_talla)  
print("Diccionario estilo->codigo:", categ_to_code_estilo)
```

3. Crear una tabla aleatoria de 10 ejemplos donde cada ejemplo viene representado por estos 4 atributos.

```python
N = 10  
df_ejemplo = pd.DataFrame({  
    'Genero': [random.choice(genero) for _ in range(N)],  
    'Material': [random.choice(material) for _ in range(N)],  
    'Talla': [random.choice(talla) for _ in range(N)],  
    'Estilo': [random.choice(estilo) for _ in range(N)],  
})  
print(df_ejemplo)
```

4. Convertir la tabla en otra numérica llamada ![](http://latex.codecogs.com/gif.latex?X) utilizando los diccionarios.

```python
x = df_ejemplo.copy()  
x['Genero'] = x['Genero'].map(categ_to_code_genero)  
x['Material'] = x['Material'].map(categ_to_code_material)  
x['Talla'] = x['Talla'].map(categ_to_code_talla)  
x['Estilo'] = x['Estilo'].map(categ_to_code_estilo)  
print(x)
```

5. Responde a las siguientes preguntas:

- ¿Cuál es el vector ![](http://latex.codecogs.com/gif.latex?x_1) ?
	- La columna de `genero`.
- ¿Cuál es el vector ![](http://latex.codecogs.com/gif.latex?x^{(1)}) ?
	- la fila 1
- ¿Cuál es el valor ![](http://latex.codecogs.com/gif.latex?x_3^{(2)})?
	- la fila 3 de talla

7. Pasar el atributo _Genero_ a una representación one-hot con ceros y unos, no con valores lógicos (True, False)
```python
#hacemos una codificacion one-hot del atributo genero
one_hot = pd.get_dummies(x['Genero'], dtype=int)
one_hot_df = pd.concat([x, one_hot], axis=1)

one_hot_df = one_hot_df.drop('Genero', axis=1)
print(one_hot_df)
```

### 2. Separar e imputar
[[#^7ed2e5|Home]]
Descargar y descomprimir el fichero ZIP de la siguiente dirección  
[https://archive.ics.uci.edu/static/public/27/credit+approval.zip](https://archive.ics.uci.edu/static/public/27/credit+approval.zip)  

1. Crear un dataframe credit_df a partir del fichero crx.data que se habrá descargado.
2. Separar credit_df en dos dataframes: train_df, que contendrá el 80% de las filas, y test_df, que contendrá el 20% restante.
3. Sustituir los valores perdidos de train_df por el valor medio de cada atributo.  
    Guarda en una lista ese valor que has utilizado porque a continuación debes utilizarlos para sustituir los valores perdidos de test_df.  
    Te darás cuenta de que si no pasas antes los datos a una representación numérica no es posible.

```python
import urllib.request
import zipfile
import os
import pandas as pd

# 1. Definimos la URL y el nombre local del ZIP
url = 'https://archive.ics.uci.edu/static/public/27/credit+approval.zip'
zip_path = 'credit_approval.zip'   # evitamos el '+' en el nombre de fichero

# 2. Lo descargamos
urllib.request.urlretrieve(url, zip_path)

# 3. Creamos un directorio donde extraer (por ejemplo 'sample_data')
extract_dir = 'sample_data'
os.makedirs(extract_dir, exist_ok=True)

# 4. Abrimos el ZIP y extraemos sólo crx.data al directorio
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extract('crx.data', extract_dir)

# 5. (Opcional) Borramos el ZIP para no ensuciar el workspace
os.remove(zip_path)

# 6. Cargamos el fichero en un DataFrame
file_path = os.path.join(extract_dir, 'crx.data')
credit_df = pd.read_csv(
    file_path,
    header=None,     # no tiene cabecera en el propio fichero
    sep=',',
    na_values='?'    # tratamos los '?' como valores faltantes
)

# 7. Vistazo rápido
print(credit_df.shape)
print(credit_df.head())



from sklearn.model_selection import train_test_split
credit_train_df, credit_test_df = train_test_split(credit_df, test_size=0.2, random_state=42)




# este bloque de código es innecesario, lo hacemos para saber cuantos datos faltan
#como esta tabla tiene '?' cuando hay una celda vacia lo remplazamos por NA
#EN ESTE CASO CONCRETO AL HACER EL READ CSV YA HEMOS GESTIONADO LOS VALORES FALTANTES 
credit_train_df = credit_train_df.replace('?', pd.NA)
credit_test_df = credit_test_df.replace('?', pd.NA)

#one-hot encoding con columna de na en cada categorico
credit_train_df_enc = pd.get_dummies(credit_train_df, dummy_na = True)
credit_test_df_enc = pd.get_dummies(credit_test_df, dummy_na = True)

#alinear las columnas de test a las de train
credit_test_df_enc = credit_test_df_enc.reindex(columns=credit_train_df_enc.columns, fill_value=0)

#calcular las medias de cada columna
means = credit_train_df_enc.mean()

#imputar nan en train y test usando esas mismas medias
train_filled = credit_train_df_enc.fillna(means)
test_filled = credit_test_df_enc.fillna(means)

#guardamos las medias en un dict para referecnia posterior
lista_medias = means.to_dict()

print("Shape train: ", train_filled.shape)
print("Shape test: ", test_filled.shape)
print("Primeras filas tras imputación univariada:", train_filled.head())

#podemos ver ahora cuantos NaN hay 
print("NaNs en train:", train_filled.isna().sum().sum())
print("NaNs en test:", test_filled.isna().sum().sum())
```

### 3. Ingeniería de características
[[#^7ed2e5|Home]]
Crea un dataframe "de juguete" con los siguientes datos  

       N = 10
       data = {
           'x1': np.random.rand(N)*10000,
           'x2': np.random.rand(N)*100,
           'x3': np.random.randint(1, 100, N),
           'x4': np.random.randint(0, 5, N),
           'x5': np.random.randn(N),
           'x6': np.random.randint(0, 2, N),
       }

1. Añade dimensiones polinómicas de grado 2 pero SOLO de las columnas x1, x2 y x3; manteniendo todas las demás.
2. Añade características ![](http://latex.codecogs.com/gif.latex?\log(\cdot)) de cada una de las características del dataframe resultante del ejercicio anterior.  
    **¿Qué problemas puede introducir esta transformación?**  
    _AYUDA. Al tomar el logaritmo estamos convirtiendo en números negativos los valores menores que uno, y en positivos más pequeños, cercanos al orden de magnitud, los valores mayores que uno._  
    
3. Añade características ![](http://latex.codecogs.com/gif.latex?\exp(\cdot)) pero SOLO a las características originales del dataframe.  
    **¿Qué problemas puede introducir esta transformación?**  
    _AYUDA. Al hacer ![](http://latex.codecogs.com/gif.latex?\exp(\cdot)) estamos convirtiendo todos los valores en positivos. Además todos los que tengan un valor absoluto mayor que uno son agrandados, y los que tiene valor absoluto menor que uno empequeñecidos._

```python
import numpy as np
N = 10
data = {
   'x1': np.random.rand(N)*10000,
   'x2': np.random.rand(N)*100,
   'x3': np.random.randint(1, 100, N),
   'x4': np.random.randint(0, 5, N),
   'x5': np.random.randn(N),
   'x6': np.random.randint(0, 2, N),
}

df = pd.DataFrame(data)
print(df)


degree = 2
interaction_only = False
polyf = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

actuation = ['x1', 'x2', 'x3']

# Ajustar y transformar solo las columnas seleccionadas
polyf.set_output(transform="pandas")
polyf.fit(df[actuation])  # Para datos de entrenamiento
df_poly = polyf.transform(df[actuation])

#para obtener dforiginal + dfnuevo
df_poly = df_poly.drop(columns = actuation)
df_poly = pd.concat([df, df_poly], axis=1)
print(df_poly)

#añadimos carcacterísitcas log -> problema log 0 y negativos
log_df = df_poly.copy()
for col in log_df.columns:
    log_df[f'log_{col}'] = np.log(log_df[col]+ 1e-6)

#añadimos características exp() ->problema overflow
exp_df = log_df.copy()
for col in exp_df.columns:
    exp_df[f'exp_{col}'] = np.exp(exp_df[col])

print(exp_df)
```


### 4. Reducción de características
[[#^7ed2e5|Home]]
Descarga de internet el conjunto de datos Student Performance (https://archive.ics.uci.edu/static/public/320/student+performance.zip).
Aquí(https://archive.ics.uci.edu/dataset/320/student+performance) puedes encontrar información sobre los datos.

1. Separa el conjunto de datos en dos (entrenamiento y test)
2. Sobre el conjunto de entrenamiento evalúa la posibilidad de hacer una selección de características:
	1. Hacer un filtrado por varianza.
	Para ello escala los atributos que NO sean categóricos al intervalo [0,1] y luego elimina aquellas columnas con varianza < 0.03 (por ejemplo)
	2. De las columnas no eliminadas, elimina aquellas que estén correlacionadas con otra, con un coeficiente > 0.95.
AYUDA. Obviamente este punto SOLO tiene sentido para atributos continuos. La varianza para atributos discretos o categóricos puede no ser representativa
3. Para las columnas categóricas, realiza una codificación con enteros.
4. Aplicar las transformaciones al conjunto de entrenamiento
5. Transforma del conjunto de test de la misma manera que transformaste el conjunto de entrenamiento.
```python
import urllib.request
import zipfile
import os
import pandas as pd
from os import remove
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
url = 'https://archive.ics.uci.edu/static/public/320/student+performance.zip'
urllib.request.urlretrieve(url, 'student+performance.zip')
  
extract_dir = 'sample_data/'
with zipfile.ZipFile('student+performance.zip', 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
remove('student+performance.zip') #<- al terminar borramos el ZIP

extract_dir = 'sample_data/'
with zipfile.ZipFile(extract_dir+'student.zip', 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
remove(extract_dir+'student.zip') #<- al terminar borramos el ZIP

  
df1 = pd.read_csv(extract_dir+'student-por.csv', sep=';')
df2 = pd.read_csv(extract_dir+'student-mat.csv', sep=';')

df = pd.merge(df1, df2, on=["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"])



#separamos en train y test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#selección de características
#identificamos las continuas y las categoricas
cat_cols = train_df.select_dtypes(include=[object]).columns.tolist()
cont_cols = [c for c in train_df.columns if c not in cat_cols]

# escalado [0,1] de continuas en el entrenamiento
scaler = MinMaxScaler()
scaler.set_output(transform = "pandas")
scaler.fit(train_df[cont_cols])
df_scaled = scaler.transform(train_df[cont_cols])
print(df_scaled)

#flitrado de varianza
threshold = 0.03
selector = VarianceThreshold(threshold=threshold)
selector.set_output(transform = "pandas")
selector.fit(df_scaled)
df_filtered = selector.transform(df_scaled)
print(df_filtered)

#filtrado por correlación
threshold = 0.95
corr = df_filtered.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
train_df_final = df_filtered.drop(columns=to_drop)
print(train_df_final)

#codificación entera de atributos categoricos
label_encoders = {}
train_cat_enc = pd.DataFrame(index=train_df.index)

for col in cat_cols:
    le = LabelEncoder()
    le.fit(train_df[col].astype(str))
    label_encoders[col] = le
    train_cat_enc[col] = le.transform(train_df[col].astype(str))

print(train_cat_enc)
#concatenamos el train final 
x_train = pd.concat([train_df_final, train_cat_enc], axis =1)


#aplicamos las transformaciones al conjunto de test
test_scaled = scaler.transform(test_df[cont_cols])
test_filtered = selector.transform(test_scaled)
test_cat_enc = pd.DataFrame(index=test_df.index)

#codificamos categoricas en test
for col in cat_cols:
    le = label_encoders[col]
    # Si aparece una categoría no vista, asignamos -1
    test_cat_enc[col] = test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
#concatenamos test final 
y_test = test_df['G3']
x_test = pd.concat([test_filtered, test_cat_enc], axis =1)
print(x_test)
print(y_test)

```



### 5. Estandarización, Filtrado por varianza y PCA
[[#^7ed2e5|Home]]
Carga la tabla de datos del conjunto llamado "Iris".
Puedes hacerlo usando sklearn con el siguiente código:

  from sklearn.datasets import load_iris
  iris = load_iris()
  X = pd.DataFrame(iris.data, columns=iris.feature_names)

A continuación responde a estas preguntas:

- **¿Qué se debe hacer antes: una estandarización, un filtrado por varianza, o da lo mismo?**
Conviene hacer primero el filtrado por varianza puesto que este elimina las características con variación baja en sus unidades originales, si primero estandarizamos, todas las varianzas serían 1 y no podríamos descartar nada con un umbral >0

- **¿Qué se debe hacer antes: una estandarización, PCA, o da lo mismo?**
Conviene hacer primero una estandarización puesto que PCA es muy sensible a la escala de las variables, sin estandarizar, el primer componente queda monopolizado por la variable de mayor varianza 

- **¿Qué se debe hacer antes: PCA, filtrado por varianza, o da lo mismo?**
Conviene hacer primero el filtrado por varianza puesto que eliminar características inútiles o casi constantes antes del PCA hace que los componentes extraídos representen mejor la estructura de los datos relevantes y reducimos el coste computacional

### 6. Rectas , planos e hiperplanos
[[#^7ed2e5|Home]]
- Dada la recta $x2 = 0.5x1-2$, responder a las siguientes preguntas:
    1. ¿Cuál es su ecuación implícita?
    2. ¿Cuál de los siguientes puntos está sobre esta recta?  
        A=(4, 0) , B=(3, 0.5) , C=(2, -1) , D=(-2, 3)
- Dado el plano ![](http://latex.codecogs.com/gif.latex?1+x_2+2x_1-x_0), responder a las siguientes preguntas:
    1. ¿Cuál es su vector característico?
    2. ¿Cuál de los siguientes puntos está sobre el plano?  
        A = (4, -3, 6) , B = (3, -2.5, 4) , C=(-2, 1, -2), D=(-1, -1, 2)
- Dado el hiperplano ![](http://latex.codecogs.com/gif.latex?1-x_5-3x_4+5x_3-2x_2+x_1=0), responder a las siguientes preguntas:
    1. ¿Cuál es su vector característico?
    2. ¿Cuál de los siguientes puntos está sobre el plano?  
        A = (-2,-3,3,6,2))  
        B = (1,-1,1,2,3)  
        C = (1,-2,-3,5,-4)  
        D = (1,0,0,2,-1)

Dada la recta  
\[
x_2 = 0.5\,x_1 - 2
\]

1. **Ecuación implícita**  
   \[
   0.5\,x_1 - x_2 - 2 = 0
   \quad\Longleftrightarrow\quad
   x_1 - 2\,x_2 - 4 = 0
   \]

2. **Puntos sobre la recta**  
   | Punto      | Cálculo                      | ¿En recta? |
   |------------|------------------------------|:----------:|
   | A =(4, 0)  | 0.5·4 − 2 = 0                | ✔          |
   | B =(3, 0.5)| 0.5·3 − 2 = −0.5 ≠ 0.5       | ✗          |
   | C =(2, −1) | 0.5·2 − 2 = −1               | ✔          |
   | D =(−2, 3) | 0.5·(−2) − 2 = −3 ≠ 3        | ✗          |

> **Respuesta:** A y C.

---
Dado el plano  
\[
1 + x_2 + 2\,x_1 - x_0 = 0
\]

1. **Vector normal**  
   \[
   \mathbf{n} = (-1,\,2,\,1)
   \]
   (coeficientes de \(x_0, x_1, x_2\))

2. **Puntos sobre el plano**  
   | Punto          | \(1 + x_2 + 2x_1 - x_0\)       | ¿En plano? |
   |----------------|-------------------------------|:----------:|
   | A =(4, −3, 6)  | 1 + 6 + 2(−3) − 4 = −3        | ✗          |
   | B =(3, −2.5, 4)| 1 + 4 + 2(−2.5) − 3 = −3      | ✗          |
   | C =(−2, 1, −2) | 1 + (−2) + 2·1 − (−2) = 3     | ✗          |
   | D =(−1, −1, 2)| 1 + 2 + 2(−1) − (−1) = 2      | ✗          |

> **Respuesta:** Ninguno.

---
Dado el hiperplano  
\[
1 - x_5 - 3\,x_4 + 5\,x_3 - 2\,x_2 + x_1 = 0
\]

1. **Vector normal**  
   \[
   \mathbf{n} = (1,\,-2,\,5,\,-3,\,-1)
   \]
   (coeficientes de \(x_1, x_2, x_3, x_4, x_5\))

2. **Puntos sobre el hiperplano**  
   | Punto                     | Evaluación                     | ¿En hiperplano? |
   |---------------------------|--------------------------------|:---------------:|
   | A =(−2,−3, 3, 6, 2)       | 1−2−2(−3)+5·3−3·6−1·2 = 0       | ✔               |
   | B =(1,−1, 1, 2, 3)        | 1+1−2(−1)+5·1−3·2−1·3 = 0       | ✔               |
   | C =(1,−2,−3, 5,−4)        | … ≠ 0                           | ✗               |
   | D =(1, 0, 0, 2, −1)       | … ≠ 0                           | ✗               |

> **Respuesta:** A y B.

---

1. **Recta** (\(\mathbb{R}^2\))  
   2. Pasar de forma \(x_2 = m\,x_1 + b\) a \(m\,x_1 - x_2 + b = 0\).  
   3. Sustituir cada punto.

4. **Plano** (\(\mathbb{R}^3\))  
   5. Leer \((a,b,c)\) de \(a\,x_0 + b\,x_1 + c\,x_2 + d = 0\).  
   6. Sustituir cada punto.

7. **Hiperplano** (\(\mathbb{R}^n\))  
   Igual que en \(\mathbb{R}^3\), pero con más coordenadas.



### 7. Regresión lineal
[[#^7ed2e5|Home]]
La siguiente URL contiene un conjunto de datos recogidos de una central de ciclo combinado
https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
El conjunto de datos contiene 9568 ejemplos datos recogidos en una central eléctrica de ciclo combinado durante 6 años (2006-2011), cuando la central estaba funcionando a plena carga.
Las características consisten en las siguientes mediciones ambientales:

-- Temperatura (T),
-- Presión ambiente (AP),
-- Humedad relativa (HR) y
-- Vacío de escape (V).

El objetivo (target) es predecir la producción de energía eléctrica neta horaria (EP) de la central.
No hay valores perdidos.

Utiliza el siguiente código para cargar el fichero de datos

Tareas

  1. Separa un 20% del conjunto de datos para test
  2. Aprende un modelo lineal para estimar el valor de EP
  3. Utiliza el modelo para predecir el valor de EP en el conjunto de test
  4. Dibuja un histograma de las diferencias al cuadrado entre las predicciones y los valores verdaderos
  5. Responde a la siguiente pregunta: ¿Cuál es la ecuación implícita del modelo lineal aprendido?

```python
# Descargar y descomprimir el fichero ZIP

import pandas as pd
import numpy as np
import urllib.request
import zipfile
from os import remove

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip'
urllib.request.urlretrieve(url, 'combined+cycle+power+plant.zip')

extract_dir = 'sample_data/'
with zipfile.ZipFile('combined+cycle+power+plant.zip', 'r') as zip_ref:
  zip_ref.extractall(extract_dir)
remove('combined+cycle+power+plant.zip') #<- al terminar borramos el ZIP
df = pd.read_excel(extract_dir+'CCPP/Folds5x2_pp.xlsx')

##################################3

x = df.drop('PE', axis=1)
y = df['PE']

#separamos en train y test
test_size = 0.2
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(x, y, test_size=test_size, random_state=42)

#entrenamos nuestro modelo lineal
model = linear_model.LinearRegression()
model.fit(df_x_train, df_y_train)

#predecimos
y_hat = model.predict(df_x_test)

#vamos a ver que tal fue la prediccion con una funcion de perdida
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(df_y_test, y_hat)
r2 = r2_score(df_y_test, y_hat)

print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

#Dibuja un histograma de las diferencias al cuadrado entre las predicciones 
#y los valores verdaderos
errores_sq = (y_hat - df_y_test) ** 2
plt.hist(errores_sq, bins=30)
plt.xlabel('Diferencia al cuadrado')
plt.ylabel('Frecuencia')
plt.title('Histograma de las diferencias al cuadrado')
plt.show()

```


### 8. Problema completo de clasificación binaria
Descripción del conjunto de datos

El baseball es uno de los deportes donde más estadísticas se recogen.
El objetivo de esta tarea es predecir si el número de carreras logradas será superior a un cierto valor.

Tareas

1. Responder a las siguientes preguntas:

  2. ¿Cuántos ejemplos y cuántos atributos tiene el conjunto de datos?
  3. ¿Cuántos atributos son categóricos?   ¿Cuántos son numéricos discretos?    ¿Cuántos son continuos?
  4. Imprimir una lista de los atributos que son categóricos junto con el número de elementos diferentes en cada uno de ellos
  5. Imprimir una lista de los atributos junto con el número de valores perdidos (NaN).
  6. Eliminar todas las filas que tengan algún atributo perdido.
  Crear diccionarios para cada atributo categórico.

```python
#cargamos el fichero
from sklearn.datasets import fetch_openml
dataset = fetch_openml(data_id=41021)
#obtenemos la tabla de datos
x = dataset.data
#columna de valores objetivo
y = dataset.target
#descripcion del conjunto
  #print(dataset.DESCR)

# ejemplos y atributos del conjunto de datos
print("Ejemplos: ", x.shape[0])
print("Atributos: ", x.shape[1])

#cuantos son categoricos, cuantos numericos discretos y cuantos numericos continuos
cat_cols = x.select_dtypes(include=['object', 'category']).columns.tolist()
disc_cols = x.select_dtypes(include=['int', 'int32', 'int64']).columns.tolist()
cont_cols = x.select_dtypes(include=['float', 'float32', 'float64']).columns.tolist()

print("Atributos categóricos: ", len(cat_cols))
print("Atributos numéricos continuos: ", len(cont_cols))
print("Atributos numericos discretos: ", len(disc_cols))

#Imprimir una lista de los atributos que son categóricos junto
#con el número de elementos diferentes en cada uno de ellos

print("Atributos categóricos: ",cat_cols)
for cols in cat_cols:
  n_unicos = x[cols].nunique()
  print(f"{cols}: {n_unicos}")

#imprimir una lista de atributos junto con el numero de valores peridos
print()
val_perdidos =  x.columns[x.isna().any()].tolist()
for i in range(len(val_perdidos)):
  print(f"{val_perdidos[i]}: {x[val_perdidos[i]].isna().sum()}")

#eliminar todas las filas que tengan algun atributo perdido
x = x.dropna()
y = y.loc[x.index] #doperamos en x y alinemos y por indices resultantes

print(x.shape) # se han eliminado muchas filas

#creamos un diccionario para cada atributo categorico
cat_mappings = {}
for category_col in cat_cols:
  uniques = x[category_col].unique()
  mapping = {cat: code for code, cat in enumerate(uniques)}
  cat_mappings[category_col] = mapping
  #aplicar la codificacion al df
  # x[category_col] = x[category_col].map(mapping)
print(cat_mappings)
```

7. A partir del conjunto de datos que ha quedado, realizar las tareas de la siguiente lista.
        ¡ IMPORTANTE !
        La lista de tareas no está ordenada; es posible que haya que hacer unas antes que otras.
        Tampoco se especifican detalles para una tarea si estos son imprescindibles para la misma.
        No conocer estos detalles supone fallos graves.

  - Separar un 20% del conjunto de datos para validar el modelo.
  Crear una nueva característica que sea el cociente $W/RA$.
  - Filtrar (eliminar) aquellas características que tengan una varianza $o < 0.02$  .
  - Obtener una representación diferente mediante PCA con una varianza explicada acumulada mayor o igual al 85%.
  - Aprender un modelo de regresión logística que estime la probabilidad de que la variable target $RS>800$.
  - Imprimir métricas de resultados derivadas de la matriz de confusión.

```python
#separamos en conjunto de train y test
from sklearn.model_selection import train_test_split

#IMPORTANTE tenemos que binarizar el target RS>800 ->1 <=800 ->0
y_binary = (y > 800).astype(int)

#creamos una nueva característica W/RA
x['W/RA'] = x['W'] / x['RA']

x_train, x_test, y_train, y_test = train_test_split(
    x, y_binary, test_size=0.2, random_state=42
)

#filtrado por varizanza o<0.02
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

threshold = 0.02
selector = VarianceThreshold(threshold = threshold)
selector.set_output(transform = "pandas")
#escalado
scaler = MaxAbsScaler()
scaler.set_output(transform = "pandas")
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

#eliminamos la columnas que cumplan el threshold
selector.fit(x_train_scaled)
x_train_filtered = selector.transform(x_train_scaled)
x_test_filtered = selector.transform(x_test_scaled)

#pca con varianza explicada >= 0.85
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#antes estandarizamos los datos
scaler = StandardScaler().set_output(transform = "pandas")
scaler.fit(x_train_filtered)
x_train_scaled = scaler.transform(x_train_filtered)
x_test_scaled = scaler.transform(x_test_filtered)

#aplicamos pca
n_components = .85
pca = PCA(n_components=n_components)
pca.set_output(transform = "pandas")
pca.fit(x_train_scaled)
x_train_pca = pca.transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

#aprendemos un modelo de regresión logística que estime la probabilidad de que
# la variable target RS > 800
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

clf = LogisticRegression()
clf.fit(x_train_pca, y_train)

y_pred = clf.predict(x_test_pca)

#imprimimos las metricas de rendimiento
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 9. Regularización
Dado el siguiente contorno de la función de pérdida donde el punto azul representa el vector de parámetros inicial, y suponiendo que w0 = 0, se pide:

![Sin título.png](data:image/png;base64,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3XeL++67T3z4wx8WQgjx2GOPCUB87GMfEy0tLaKhoUHce++94l3vepfweDwCEDt37hTJZHLW6/2DP/gDAQhFUcT1118vHnzwQbF9+3YBCJPJJP75n/+5oOMeiUSyx8rlcom77rpL3HfffaKhoUHU1NRk35PHHnts1nP/8i//UqiqKgCxfft28YEPfEDs2rVLWK1WAYivfe1rBa3loYceEoD413/912m3v/zyywIQdXV1YteuXeL+++8Xt99+u6ipqRGAWLNmjZiYmFjQPk6cOCEeeughcf311wtAdHV1TXv/M8h8dubCTTfdJADx/PPP5739f/yP/yGsVqvYuHGjeOCBB8RNN90kTCaTAMTnP//5vNt89NFHs/vdvn27eOCBB8Q73/lOsXr16ln7ev755wUgbrrpplnbeeKJJ7Kf0fXr14sHHnhA3Hrrrdn9f+xjH5v1nMV8JufDN77xjexruuqqq8SDDz4orrzySgGIL33pSwIQ7e3ts5537Ngx0draKgDR1NQk7rzzTvGe97xHNDQ0ZI/P1NTUgteR+V7efffdoqurS3i9XvG+971P3HPPPaKqqip7rMbGxmY998UXX8w+pqOjQ9x9993ijjvuyN52++23zzoumc/yJz7xCWGz2URHR4f44Ac/KN7znveIP//zPxdCCDEyMiLWrl0rAFFVVSXuvfde8b73vU94vV7R1dUl7r777rzfh8x7le87+bnPfU4AQlVVceONN4oHHnhAbNq0SaiqKj7/+c/P+ZkxmUzC6XSKK6+8Utx7773i7rvvzn7uXC6XePXVVxd8rBeKMoHm4CMf+YgAxK233lrU87/zne8IQNTW1oqDBw9mb9d1PfuB8Xq9sz7gGQK1WCziF7/4xbT7Ml8aj8cjotHotPvm+rHOIB6PZ7f927/929O+IOfOnRMdHR0CEP/f//f/5d0nIHbv3i0CgcCsbWdeDyAefvhhEY/Hs/f19fWJVatWCUB8//vfn/a8p556SgDCbreLX//619Pu+6d/+qfscTh69Gje15QPX/7ylwUgNmzYIAYHB7O3RyIR8d73vje7zplf1l/+8pdCURRRW1srXnzxxWn3HTlyRLS0tAhAvPDCCwtey1zvSX9/v3jmmWeEpmnTbo9EIlmC//SnP73g/Qhx4X3KJc1cLJZAAfF3f/d30+579tlnhaIowmQyif7+/mn3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b5LPNVLYtWuX+NCHPiQuv/xyAcUZKYTDYXHVVVcJQFRUVIj3vOc94r777hONjY0XNVL41re+JcxmswBpZvDud79bfOhDHxLveMc7RH19/byS+nyY67365je/mf08XX311eLBBx8U119/vVAURXzkIx+56GcgHy72Pp0/f154vV4B0kTj/e9/v7jxxhuFw+EQt912W7ZvcK42lpm3ZzDXe6/ruvjt3/7t7OvcsWOHeOCBB8S73vWugo0UfvGLX2Q/oxs2bBAPPvig2L17d/a9ytc/uZjP5Hz4sz/7s2nv3Yc+9CGxc+dOAYgvfvGLedtYhBDi6NGj2V5rr9crbrzxRvGhD31IvO997xObNm0SiqKIhoaGBa8j10hh9erVwuv1invuuUfce++92e/42rVrxejo6Kznvvjii6K2tlYAor6+Xtx6663iwx/+sLjrrruy7XdXX331tOcspCVreHg4+/zq6mpx7733invuuWdRRgqf+cxnBEgjhZtvvlk8+OCDYvPmzfMaKezduzdrfrJmzRrxwQ9+UNx5553C4XCIzZs3i3vuuafg9rKFoByB5kF9fT3PP/88Tz31FB/96EcxmUw8++yzPP744xw/fpxrr72Wv/zLv6S7u5urrroq+zxFUfje977H97//fXbt2sX+/ft5/PHHiUajPPzwwxw8eDCbTikFtm7dyo9//GOuvfZaXn/9db773e/yz//8z9PO0nbu3MmhQ4f4zGc+g8lk4ic/+Qkvv/wyO3bs4Dvf+Q5PPPFENhW8XPijP/ojnnrqKd75zndy4sQJfvjDHzI0NMR9993Hnj17+NjHPlbQ9lwuF88//zx/+Id/SENDA7/61a946aWX2L17N2+++ea8KsHPfe5zHDx4kE9+8pMoisKzzz7LT3/6U86dO8eOHTv49re/zec+97nFvmS+8IUv8NOf/pTrrruOU6dO8Ytf/IJEIsHf/M3fzJmqXCw6OzvZs2cP9957L6FQiCeeeILR0VF+//d/nyeffDIrYCoVFEXhO9/5Dk899RTvfe97GRoa4sc//jH79u2jtraWr33tawtua7jrrrs4cOAADz30EOFwmMcff5z9+/dzww038IMf/CCbxl0O/O7v/i4/+9nP2LVrF0ePHs1mSR5//PF5PxubN2/myJEj/Nmf/RkbN27kyJEj/OhHP+L111/H5XLx5S9/eVZUuxBUVVWxd+9e7r//fvbt28cTTzyBy+Xic5/7HHv37qW+vn7Wc2688UaOHTvGH/7hH9LS0sK+ffv40Y9+xKFDh2hoaOCxxx4rKhpubGzk9ddf59FHH8XpdPLEE0+wb98+HnjgAfbu3VuUovyv//qvsxm+vXv38uSTT9LU1MSzzz7L+973vrzPufrqq3nzzTe5++67iUQi/PznP+fcuXM8+uijvPbaa3N2RiwWihDLKPsro4wyyiijKHz3u9/lkUce4aGHHiqPMFwhKEegZZRRRhlllFEEygRaRhlllFFGGUWgTKBllFFGGWWUUQTeljXQr371q3zta1+bdtv69evz9iKVUUYZZZRRxlLgbetEtHnzZp555pnsdbP5bftSyiijjDLKeBvibcs6ZrOZxsbGS72MMsooo4wyfkPxtq2BnjlzhubmZlavXs2HP/xh+vr6LvWSyiijjDLK+A3C27IG+tRTTxEOh1m/fj3Dw8N87WtfY3BwkKNHj+Y1zU4kEtOszHRdZ3JykpqamkWPXiqjjDLKKOPtCyEEoVCI5ubmwodKlNTX6BLB7/cLt9st/umf/inv/RnbqPKlfClfypfypXzJd+nv7y+Ye96WEWg+7Ny5k9tuu43/+T//56z7ZkaggUCAtrY2+vv7l8zi6WIQCKIkCRIjRJwgMaLMNnk2Y8KNHTdO468ddYVk3jU0QoQIECBIgNSM9VdQiQcPHrzYWR6z+gzihAgyRohx0jnrsuDAQz2V1GNl6eYcCnRijBOljzgT2dtN2HHRhotWTJTWQlEgSDFEnFOkuTBJyEozdtZjZvasx6L3JfxoHEEXPdnbVKUVlW2oSgn2I5KQOALJEyAMI3BzC9ivAFP14radmoTAPkgMyeuqA9xXgHMNFJuR0pIwcQgmjwMCVAvUbofqTaAU8X3VdRg9DoMH5etXTNC8DRovg2JG7030w8k9kIjK19i+BTp2QKEzj2MROPgqjBsD7+tWwY5d4HAWtBkxNIjY8zIkYqCaUC7fibJ+Y0HbSA4OEnnpJUQ8DhYLruuuw7Z69YKfryUSjL/0Egmnk63vehdTU1N4PJ6C1vD/BIGGw2Ha2tr46le/uiDf0mAwiMfjIRAIXDICzYc0GkFiTBEjSIwAMXSmDz1WUajEgRcnXhx4cGJaIYQaIUKAKaaYIkpk2n12HHjx4qUKFy4Ulid1LtCJMEWAEcKMTzueTrx4aMRNHeoS6ulShInQR4Q+dFIAKKg4WEUlnVgp7Eu7sH2Ok+AESYayt5mpxc5GLDSX7PhLIj2ELs5lb1OVDkxcjqIskugA9Cgk9kPyJDJQUMC6HmxXgDr/cOiLItYLU6+BZkwRsdSC9zqwLUKcGPfB0CsQG5XXbdXQtAtcRW4zHoLulyFgTLFxVEPXTVBRV/i2UglJosPGbMyKKth8M3gK3JYQcP44HNkLugYWG1x+A7QsnLwARCyGeOUFGDTG6rV1oFx/E4rNtuBt6NEo4WefJWVMYbJv3ozzmmtQFnhiIIQgGAzi9XqL4oO3JYF++ctf5j3veQ/t7e0MDQ3x2GOPcejQIY4fP05d3cU/DCuVQGdCRxAmToAYU0QJECWFNu0xCgqV2PHipAonbpyYVwChJkkaVOonRBDBhY+ZBatBpdVUULFsEbVOmhATBBghgj97u4pKJfV4aMSJd8nIXUcjxhAhukkRyN5upYoKOnHSiEKBEcFFoBEkzgmS9GTfAxMe7GzEShtKiY69EFNoHJxBpKsNIvUufgdaABL7IGUM2VbMYL0MbNtAWUQkL3QIH4XgARn1AjjWgOdqMBdJ0ELA1GkYfR00Yw6vdx00XAPmIjMxE2ehZw+k44ACTVuh5QowFXHiN9YDx1+GZExGo53bofPywqPR0BS88RxMGRmW9nWw7TqwLPz9EELA8aOI/a/LqNtVgXLjrSgNCz/hELpObP9+YgcPAmCuq6Pittsw5dHD5MNi+OBtSaAPPPAAL730Ej6fj7q6Onbt2sUf//EfZwdaXwxvFwLNhyhJpohmLwkjorkABXeWUF14cFzyCDVNmiAB/EwRYAo95yTAjAUPXqqowo172cg0RZwAIwQYIcmFeaEWbHhoxEPTkqZ4E/gJ00OUITCiYhUbFbRTQTumEqe8dWLEOUWCswhkStSECxsbsNGJUqIIXEakB9BFt3GLgqp0YWIHilKCSDs9ConXIT1ibN4uo1HrxuJSpRloMZnWjRpmLIoZKndA5VaZPi1qrXEYewP8xjZNNmi4Grzri0sVp+KSRH1n5XWbG1bfCJ7mwreVjMPJV2HEOOGprIYtt0Blgel3XYcT++HUIXni4KyEq26FmoaCNiN8E4gXnoFQEBQFZfuVsHV7QSLPZH8/4eeeQyQSKFYrFbfcgjXPgO2Z+I0j0MUic8BeDHTT5K6mEgtuLDgwLVtqsVSIzSDU+AxCVVDw4KAKF1U4qcSBeglfo45OiCB+/EwxRTpnvSomIzKtwo0HU4mjsbkQJUCAEUKMoXFh8K6LKjw0UUkt6hKtRSNBmF4i9KJhRCuoOGmigk5sFD4Oaj7oJElwhgSn0UkYe7NhYz021qCWqC6rCx86B9BFr3GLgqqsNYh0YZHBvEj1QPwN0KfkddUN9qvAUlgacRaSEzD1KiSNFKzJDd5rwNFR/DajYzD8skzvAjgboekGsBf53k72Qs8rkDTKJPUboe1qMBfx3o12w4mXJaGqKqy+Ajq2FV5nnRiBfc9BNCxPDjZcDht2FLQdkUrJumi3cYLQ2CyjUefC66taOEz4mWdIj40BYN+6FedVV6HMs44ygRaIzAH7QeAYTveFL7MJBTcWKrDgMUjVjRXLCkiJLhRxUviJMEUUf54I1YSKFyfVuIxq5MLrDaWGjk6YsEGm/mkiJBUVD1VUU40Hz7JEpnI940wxQiRHhGPCjJt6vDRjpwQ//nkgRUcjhOkmMU0AVEUlnThoKlm6Ve4vTYLzxDmJThQABQt21mJjPWqJPhe6mEBnP7ow6lyoqMp6g0gLE57MgtAhdQrib4IwsgimBrBfC+bZMzELQvQsTO2VNVgAW4usj1q8xa/VdxTG3jREQSrUbIW6y0EtIvpPJ6H/DSk0ArA4ofMGqL54xDULybgk0VEja+Cpl9Goq8CMQSoJB1+BfoMAq+th561QURgpibOnEa+9AloabHaUG25BaWld+PN1nejrrxN/6y0AzA0NVN52G6orf0q+TKAFInPADgT6EW4HIVKESc2Q61yAExNurLgNYvVgxfE2MXGKksRPxLhESc+oodqwUI0rS6iWZYr6ZkIgiBDBzyR+/CS5oJqWkWkV1UZkuhxkmiLOFMMEGCaVsxY7FXhpxk0DpiX6DCQJEKKbKINk0rsmHFTQQQVtJYsSQRJ3kj7iHEdDimkUTNhYg52NqCVKJetiDI39CGGoNzFjUjajshVFWSRZi5Sh2D18QbFrWSMjUrViEYtOQegghI4g3wdVpnQriyQ9gGQYRvZAqEdet7qlyKiipbjtBYfh/EsQN2rqtWuh4zowF3FMh87ItG46KWura6+Gts2Fb6fvLBx6RRKq2QLbd0H72vmfEzfqsTb5eRNTU4gXnwW/EbVv2SaVugVEtMmeHsIvvIBIJlHtdipuuw1L8+x0d5lAC0S+A6YjiJImSIoQKYKkCJIkOoNwMrCg4sGCFyserHix4lzhpCoQhElkCXWKKDq5b7+sn1bjooYKKrFfspS2jExnk6kJM16qqKGaStxLvj7ZbuRnimFCjGeFOCoqbhrw0oyDpamja8QJ00uYHnQjOlcw4aKNSlZjZpERXA5kC8wgcY6RNgRWCmoOkZamHqyLITTeRIgx4xYrJmUbKptRlEV+f/SoFBolT8nrigmsW8G2HRRL8dtNB2FqD8QNtzNTBXiuBWdn8dsM9sDwq5A20rCetdB4bXEiIy0NA2/C8FuAWFw0GgvD8RfBZ5zo1LTAlpvBVuBnLRqWAiPfCNidcPv9YJnjPUgl4ZWnjPaaddC5AQChaYh9e+HkMfm4+kaUm3ajzBFJ5oMWDBJ++mnSPh8oCs6rrsKxbdu0x5QJtEAUcsCS6IRIEiRFgCSBeaJVC6pBqJJYvSs8UtXQmSLKpEGokRyiArBgopoKaqig+hJFp5nIdBIf/hlpXgtWqqiihlpcLLKlYQFIkyTIKH6GSBppT7gQlXpoWJJ2GIFGhEHCdJMyokRQcNBEJatLXidNMmQQqc/Yk4qV1djZiKlEx1kXvUZEmklXOzEpO1BZj7IYMRCANgHx1yAtWxtQnWDbCZZ1xfd5gtH2sge0kLxuawHv9WApUhylpWBsH0welddNdkmi3otEa3MhNArnXoT4lLxebDQqBPQfg9OvX2hT2bgLGhcg0kwlkSRuk9s5eRBqGqH+IkKn0QGIhKQgqblD9pZmltPTjXj1RbntYlK66TSRV14hcfo0ANbOTlw33YRqlZmcMoEWiMWqcDUEYVJMkSRAkimShOYgVTsmvFipMgjVs4Jrqpn6qc8g1OnpXilGyo1Olxsygg4xafzLFfzYsFNNNdXU4FhC9WwGUabwM0SIsZyo1ISbBqqWsFYaZ5wQ54gznr3NRjWVdGGnoaQReYphYhwjbRhBKCgGkW4qCZEKIdA5iy4OIAgZ+3BjUq5EVRYpBgJDaLQXdOOkw1QD9uvA3FT8NvU0hA7JSzatu11eik3rRsdg6CVIGCcTrhZovgGsRXyGstHoEXl9MdFoZAreeg6CRpvKhuugbctF9n0CfAPQsBpWrb/4PoSYflIT9MPrz8p2mBzSFcGgVOlOGmvZugNl+xUFpXTjx48T2bMHdB2T10vFO96BuaqqTKCFYinaWHJJdSqHVPMd3EosVBmkWo0NF+YVp/7VEQSJ4SOMj/Cs6NSGhZpsdOpcdnckHZ0gQSbxMYV/hkGCixpqqaYaC4tI3S0AaZIEGGGKoWntME48eFlFJXVLcmySBAlxflqd1EwFlazGRUtJ+0lTjBLnGClk2lUSaVfJIlIhdHROoomDYBxDRanDxE5UpYgWjWkb1yB5DBI5fZ6W1WC/ZnH10VRAqnUThsGBqVJGo462Itepw8RhGD8g16yYoeEqqN5cXNQcGoVzLyy+NqrrcP4ADJ6Ea94PtoucnE70S+I9vRfat8K6qy++jwyJ6rpU7T79uDRl2Hj59IdpGuKN1+CUIZwqQqWbHhsj9PTT6JEIitmM6+abSdTWlgm0ECxXH6iGToAUfpJMkcBPkliemqoFlWojQq3GhhfrijBDyEWcVJZM/USnEZaKmo1Ma6jAtsxpaw2NKaaYxEeQQI5pg4IHDzXU4MG75G0xEfxMMUiIiewazFjx0oSXZixLELWniRGmmzB9CENxbcJOBZ1U0I5awhOIFOPEeWsWkTrYhFqCeqwQKXSOookjkHFsUlowc9XiXY30OCTelNaACKM+us2ojy7i8xrthsAe0Ixapr1DqnXNRZJzYgqGXoaokX521EPzjWAv4vXni0a7bgLvwtOfWaRTUhC0EPgG4MBTUoTUurkwg4ZwEH79Q9h5C7TmTxmL82cRr74kX5/dIeuiTQs/0dJjMeleNCRdupJdXTTfdluZQBeKS2mkEEdjiiR+g1CnSKLNiFMVwI2FamzUYKMaG7ZLpI7NBx0dP1EmDEKd2SrjxkEtldRSsextMilS+JnEh48I4eztJsxGvVQmoJcy4k+TYIph/AxmfXgVwEUNVbTgoqrk+9dJE6GPEOey/aQKFioMwVEpjRlSjBHn6DQivSA2KgWRxgxXo5NkTSaUtZi4AkVZRNQIoPkgvienPuqS0ahlYSYseaGnpJNR+C25XsUM7iuhYktx5g5CSPOF0ddBT8pt1G6H2h2gFvE7MDMard8I7deAqYTZmUwUqWnw8n9CbRusuwas83zukgnphRsNQzgge0nHBqGuWVoDWo3fjkRcttrYnVmXI6nSfQb8k9J4YdsVsG3Hgo0XhK4Te/NNYocOEVZVOj7xiTKBLhQryYlIpkpT+EkwaRBrvijVhTlLpjXYVpTiN0QcH2EmCBHKmgFIOLFSSyV1VOJehtpkLmLEDCr1TVPyWrFRSy011GJbQoIX6ITw4WeAKFM5+3dQRQteGksuOhLoRBkkxDlSRl0RVFy0UEkXFhZJQDmQqd2jpIx67AXV7qaStL8IEURjX46rkYqqbMHEdpTF2PeBtASM7wXdOMkyN8r6qKl2EducBP8rkDRckszVUHUj2IrsSU1FpFI32/LihVU3g7OI7Wlp2Tc6YgiWbJXQdTO4F1EPzof9T0IyCltvA5f3wu3plExTmyxweI+0/0unpT8vQIUHKr3groauTReed2QvjPZLYhY6XHZN1nNXpNOIva/CWUN1vapVCozsC//sJXt6COs6NV1dZQJdKFYSgeZDjDSTBqFOkiA4y64PHJioxUYNdmqxrRi1b4I0E4SYIISf6DQPXBsWaqmgjkq8OJfRUF6Kj3z4mGRympVgJW5qqcO7xCneBBGmGCLASFb8ZMKMh0aqaCm5baBAEGeUEOemGTM4aMbNWqwlbL1JMUqMt3LERibsrC+ZIYMuxtF4AyGMqBE7JuVyVDYsTrEr0kb/6KEL/aPWTWDfCcX2pgoB0dPShEEY5ODaBJ6rQC2S9IPdMPwKpGOAAjWXQf2VxYmWAkMyGk0aJw5NW6F1Z3GRbQaZ6PP8Qeg+CNtvh+pVF2q3o91wfr90SkqnoGdAtsXccBd4jROWmS0usQgc3w99Z6SgqNILkSAcfUNaBdavurD7s6elg5GuSS/dW96BUrtwg/yyiKhArHQCnYkkOn4S+EgwSYIAyVmK30yEmiFV+wpI+abR8BFm3Ej15tZNrZizkakX56LtBQViQYScqZdOME4o2xIiyayaamqpW9KWGJ00U4zgZ2Ca6KiS2mx6t9RIMEmQs8QZzd5mpwE3a0vaApNixCDSTPuLBTsbsLMOpQS1WF30oYk3EEY0r+DFpFyFqhQp3MluOCKj0ZThC6vYwX714tpetDgE9koyBdlK470OnEWqi7UEDO+BgDFJxeoxotHCPGcBaZTQ+xqMG5GboxrW3AKuAn1wk3F5fCw2CIzDGz+dXfcMjMO+n8nbPPVynNq5NyGig2qTtc4MGWaIOJWCM0dkC8yud11Q42oavPQLaFs3PUoFxKQP8fzT0ktXVVGu2YWybsOCXkaZQAvE241AZyKNjp8kE8TxkWCK5Cy1bwVm6rBTi50abJe8dUZDx0+EcUJMEJ7WImPBRF2WTF0Fk6mOzghBoiTw4qJ2gWnKBAl8TDDBxLQUrwMnddRRTQ3mJYrsZX/rJJMMTLMNtOGimlbcNJRcvSuVu2cMA3sJO3W4WYeNEowey+5ngBhvoRkTZ1Rs2NmMja5Fq4OlYvcUmtgPmVqv0oyJqxc/hzQ9BPFXQTMm9ZjqwbFrcWnd+BBMvQxpo/5obwPvruJFRqE+2fKSNvqQa7YWH41O9kL3S5CKyTpry5Vy5ujFThrSSZlO7TsGI2dhy61w5GnwNMCG66fXPY88C/EwXPXeC7ed2w/xiFQuH38T1u+AzVdeuH98GPb8EjZeAeu2XlDnAjz7f2Vf6fbrZi1LJJOIl5+HfsN/ec16lGuuRzHPf2zKBFogMgfse4EemtxVeDHhxYwXE25UrCtMAXsxpNCZNCLUCeIEZhnKgxcrddipw44X6yU2lBdMEWHMSPWmZpBpLZU04F5wmjdBikGmSJBihCANuFlH44KVzAJBiCATTODHjzAiZQWVKqqpo47KJerrlOuP4GeAACPZKN2MlSpW4aUZc4kHb6cIE+QsUQbAOPWyUYObtdgpYs5kHggESXqJcxTNEHOpOHGwBSsdi/b1FSKJzmE0YQh3AFVZh4krF+exK3RIHpUzSIXxPVp0WleD4MELvaOKRaZ0XZuKi3C1BIy8JkemgbQDbL65uJmjqbi0AvT3yOutV8Gq7fPsOw1jJ2CqH2pWw2AvTPTJ2ubOu+VUl1ycek2S5bbbZARpMsHASeg+ANfdD5NjcPKArG16jROgl5+U6dib3iOvZwh0uA/eeFZGpTUNs3tIMcajHTmEOPSmvL+6VqZ05xltVibQApE5YP8c6MaZ54C5MOHFRFXOX9cKSIkuFEl0fMSZIME4cSI5hgMAZhRqsFGPgzrsuC5h/VSSaZRxgozPIFMrZupx04D7ogIkHYGO4DXO0k4NLVRnTxI0dFSUBZFxmjQ+fEwwTmya25CDWuqooWbJeks1UoZ6dyDrv6ug4qGRalqwlTi1nCZKkLNE6CdDQlaqcLMWB0WkBvNAeu2eJ8YxdCNlbaISB9uwUqT/a+72RcgQGhlzQrFgUrajsgWl2DFkIG0B43shZRijK3Zj2kuRo8gAUn7wv3Rh0ou1XoqMLEVG/6F+Ixo1WmhqLoP6ncVFo+OnYegIbL774lNdpvohNgW9e2UN1VYLR5+X5gkbdk2fBzp4SvrrXv5OqGqSNdCDT4HDLS0CM/QjhCTJcBBefkL65za1XSDdWEQaLLgqYcvV4Jj/JEkMDUov3UQcrDbZLzqHe1GZQAtE5oD1BHwIt4spNAJoTKERm8NS3oqaJVN5MeM2fpZXOqKkmSDOuBGhzqygujBTb0Sntdgu2fzQDJmOGWSaSfM242U9c6sFdQQqCuMEOcEwl9NORY4KdJQA44Qxo9KIB+8CWy3ChJlgnEl82chQQcFLNfWGrngpINAJMs4kfcRzWnEqqKGa1pLXSdPECHGeCL0I45hb8eJmXQmJNE2Cs8Q5nvX1NVODg21YWOTkFEAXo2jsRYiMIrjSqI8uwq8W8qR1G8BxA5iKJD0hIHICAq8bEa7hZOTeUdzcUS1pRKNGPdPqgVW3FKfUzRPRzYmpfjj1KzlGrWETxEJw9AVo2SgdiOIRsBsnfL1vwdl9srUlPAmJCKy7Flry1ChTKXjhp7B2K3TkOBm9/qwk0XXboHlhrkoiHJZ1UZ/xmdh+Zd5WlzKBFoj5DlgCPUumk8bfANo0NWkGJhSqMFGNmWpM1GCmYoWTqjDaZsaIM04cP4lpdKoC1UZ02oCdiiV28pkLOoJJIowRpBnvvKSXERAdoBcrZjbQlE3f6uiGk1LSaLNJcDWrcRaQFk2Txs8k44wTJZK93Y6DOuqpWcJaaZQpJuknZChc5X4rqKGNSupKOuJMI0GIc4TpWTIi1UkS5wQJTmf3YaEJB9sw413UtjPWgJrYB5kRbUojJq5dXH10VlpXAesWsF9ZvEl9OiKdjOI98rrZa7S8FJGGBSMafdGojSpQuw3qrlicunYmsr2eaTj4fajqgParp7sbaRpE/NKRqHXzhWhU0+DsGzA5KKPPzTdJ8dFMpNMyTQtSfZuIw+nDEPBJ8uycRxiUTMjHp1NQJevWQtMQr++B0yfkY9o6UHbdjGK98P0vE2iBKPSAaYgsqfrR8JPGjzbLAAHAgkKVQai1mKnBjGMF11RTRrp3zLjM7EHNRKcNOKjBdklrp/MhSZpXOcsWmqmbIzI8yiAaOpfRUvTriBBhnLFpUamKShU11FO/ZAreJFEm6Z9WJ7Vgp5pWvDSVdOB3PiK14MHDOhwU+QM/AzoxYhwlwbnsbTbasbN10faAQqSN+ugRMNYvZ5DuRFEW0Z+qR6RJfcpIF6tO2Tu6mCHe0W5JpJm5o65N4Lka1CKIeaZS11YtlbqORYig8uH4f0E6DmtvA0eOkb6uy3pvOABvPQPeRqnKFbrs9zz6PFgdsPpyqGrMH/FGAjLte+QNcBlRvqLC5TfKuqfJlP95g91w5i0ITUkDBpcbrr8z+zhx+iRi7ytyjW4Pyq13oHi9QJlAC0YpVLgyktOZNMjUZ/zV85CqE5UazFlCrcKEaYUSUdiITkeJMTkjOjWjUGeQaQN2rCugLpyJPs8zzhhBttOGPU/UnEJjD2fZQCMNFDk9Iwdp0kwyyThj02qlTlzU00A11UvigZsmiZ9B/AyiZa37LFTTUnLBkUaSEGdnRaQeNpRMbKQRIsYRksiB29KMYb3harS41yJEGI03cuqjVkzKFahsXFz/aKpfpnUzJvXmNnBcD2qRQjM9KftGoyfldVMFVN0E9lXzP28uBHtkbVSLS/Kpu1w6GS2qZ9YgrcGD8rL+DvDkrC/ik7enMurgddBzXD4vZUSFdW0yKvXOk82Y6JeipMA4TPngijvBUze9rjoTQz2w9xno2ixdjDzVsO95cFbINpkMiU6MI577NUQjYLZI04X2jjKBFoqlamORrkIaPoNQfaQJ5HEVUlCyEWotZuowY1+BUWoanXHijBJnjNg0OlWAKmw04qARxyUVIgHs4SwNuOmkNi9xDeKnm3GuYjXWOdaaIeMYSRKkqcC+ICVviBDjjOFnMscD10IdddRRj7XEKloAHY0AI/joI2W0c6ioeGmmmjYsJXRYkkR6jjDdWSK1UY2bDdhZZOuIgTQ+YhzKuhrJ1pctRuvL4r4buhgx6qOG0YNSbaR1F+HCI9KQOCQvGfs+2+Vy/mixRBUflCKjzLg05wbwXlOcAUM6DsMvSxMGAEedrI3avIVtJ2X0epptEBqDYz+Tdc/GzRfSw0LAyV/Kx7mbAAUG9kPL5WCrkY9TTbIPNIP56q2aJltlzu+HqVHYdjs4jZOT0JQ0VchgchxefQpa18C2ay9ss+cUdJ+QZg05bSwiFpNTXUalKYey/QqCnWvwer1lAl0olrMPNIVgkjQTpLPEmsgjVKowCLXOIFXPCojuciEQBAzPmVFis1plKrHQgJ1GHHixLsug60H8mDFhxsQxBtlOK545aqVv0k0l9nnFSBmcYIg4KaIkcWGjg9oFCY9SpJhgnDHGcuaWKlRRRT0NS9IKI+0Cx/HlCI4UFDw0UkN7QQ5HKeKYsM4ZOWskDNVuT7bVR/aRri+ZIUOSQWIcQjNsCKVidztWiozGDMj66Ek08SYYCmdV6cLE1Ytre9GmIP6KFBuBFBfZbwRzkcIoPQWBNyBiDJE2uWRt1F6EATzA1BlpB6gnJck3XgPVmy7+PC0hSW7kGEycha5b4MwzUNkop7pYZqTCzzwn07ktV8jrvvNywPfa3WCb0fOaSs4fUWYQj0jF7tqrobYVXv4vWee8/p1gNz7XLz0h07I33y2vZ4i555S0AXz3f5tlZi90XQ7qPiFtDYPVtVS99/1F8cHK8H/7fxgWFBqw0JCTVgwZRDpuEGsAjbBx6TG+3DZUajHTYJCqF9MlFSdJ9amcGLMeDzHSjBBjxEj1hkgRIsVZQtgx0YiDJhxUL2HdVENwlmEEUvyUJJ3XkShMnBAJ1i5QCNNCNQqyQ3KUAGcYZRPNFzXGt2ChiWYaaGSKKcYYJUwIP5P4mcSBM5veLZVtoIJqNPo0EGYSH71EmWKKYQIMU0kDNbRhv4i5RCY1HGUKF1XU5unVNGGjis1UstpI7fYZUrRx7DTgYT3WRabHrazCQhMJzhk9pCHCvIyFehxsx1yk4YOiKJjYiEonGvvRxQl0cQ6dPkzsMNpeiogcTV5w3QXJ07LtRZuEyE8NkdHOwkVGqgWqrpeORZMvghaEiafAuQ6810r3nkLgXQuuZhh8ASKD0hLQVj1/z6ieBv8pCA+AuxMcVTLyNFlkj2gueQaHIR6Udc546MLtZhtExiGdmE6gkYAcdda6SZJiLjKxXCaKtLsgGoSk4dhVXQ+Oigvk239O9pHe/N4Lz1cUKSTqOQWrOvPWTBVVRbn6OkRNrbQAjFxQuheKcgS6ApyIkuhMoDFhkOok6VkCJQsKtZipx0K9UUddKYKeFDpjBpmOESeds3YrKg0GmdZiX5La7xRR+plkgjD1VLKeRuPoyH2dYRQ/Ea6ko+C6pI7gAD24sLGRwmdTRokyxig+fNmobanTu1EC+OglbFjqVVBDK1vnfY5OmjA+NNKMchovq6ina15xkuwjPU0kx5DBSTNu1pfEtP6CYvdU9tjZaMfB5Rf12E3jJ8UgZqqx5HnfdDGBxqs5bS9eTMo1qMoielP1uCEyMkQ8agXYrwdLEcOsQRJZcJ8x5QUpWqq6sbiZo0LA5DFIBqDp+os/PtQPySkY2Svdjsy1cO55qO6Ezl2SIKf6Zeq2qk2OSvP3gtUpe0MHDkjD+s7rwZ7zGzvWK9W4HduheW3+fadTcqbouTdly8tlu6Gi6sLrUBT599g+WSe99vbpUebx/TA6AGu2zDkSLXtYJsYJxhN4W1vLKdyFYqUR6EzIFg6NMVKMG6SankGoUtBjptEg1EsdoWagIfARZ9gg1NyeUzNKlkzrsZe831RDJ0QcL05GCTBKkDoqOc84bdTQWmD0kolmD9KLAysbFpD+nQtp0tn0bjJrkqBQRQ0NNCyJejdOCB99VNGCc4GRYRgfgxxlFVuoWGB9UzobnTaGewMouGjDzVrMJTDJ14gYQqNeTLhw8655LQEFghSDpBggxSgm3Li4fpYoSaZ1zxhtLzLKUZVOTFyDoizi/UgPQOyVCyIjy2qp1lWLTBUnRsH/IqSn5HXneiMaLf3J1zSE+qD/19BwDVRthERYGtHXr4f6DXD4cahsgNU3GOsMw+mnwVULFock23z+urGwjCQBghMyVRvxS5/ckHFdVWXEu3EXuOvy10uP75fjzzLpW4DzJ6D3NFTVTa+JzoOyiKhArHQCnQkdgR+NcdKMkWacFKkZhGpDpc5I+TZgwb0CaqjyRCCRJdN4jqDKZJDpKsMNqdRkmiDFWcaYJEIanXoqaaWaSuzznmjkpoAFglGCnGKEjTRRXwLjBB19Wno3gwoqaaARL95LciJ0Qc38BjZcNLIek1HhSRIjyChWHLjnSYMnCRLgZNa0XkGlgk7crFm0ohak0EiQxlJAT2qIF9EJU8EuTHOcRAiRNNK6x5GRtNlQ624uXq0r0pA4AInDcpuK1TCo31Cck5GehuCbEDaGY5tchlJ38W5O05Dt9UzBme/LFG7D1WDK7fVMSXI7+lOobJK9oBmcf1mmbdfddvF9TQ7B/v+S/2/ski0uNqcUGzm90hFpvmHcQ71w8GXYeq0UFk0My1aW+lWw8XKpwl0AygRaIDIHbGpqCo9n8S0Nyw1hEOoYaUZJMZYn5etApRGLcTFju8QqX7nmJCPEGCI6rd80E5k246AOR8nTvBOE6WYcDZ11NFKNixQafiJU48I842RDrjVCLz6SpKnDzeoStW3kIkyYMUanqXet2GigkVpql3S8Wi4y5BlinEGO0cZ2nDmmBj76iOInTRKNFPWswT2Pe1CCSQKcyI5RU7DgposKVpe0X3UuCHQUVNL4CfIrXFyDlfaLnpjowmekdY1B4Uo1Jq5DVRbR+6r5IPYSaDJVjLkZHDeCWuTJWGIEJl+QtVEwlLrXFtc3OhO5tcKeJ6TLUctusOX8RvqOge8IrL4H+vdDIgTr3iFVtromp7zouoxKF3Ki0HMEzrwO66+Fti2Fr/n8CWlIb7NLp6L1O+S8UNfCBXtlAi0QmQP2t2PjNHs8VCsKNYpCtapQrShYi/W6vETIpHxHSTFqCJNm9qNWG+neRkPle6nrp34jMp1JphYUGnGwClfJjRvS6IDAjIkBJunFxxoaaDAiyzQa44QYJUiIOM14acJbkGtRMUiSZIxRxhmfNiu0nnrqaVgy792ZOMdeHHhoZO2sQd8aaRQgwCh+BmliA46LROQxRglwkpQxNs6EHTfrcNFaUgeluRDgKUxU4OSqWTXTucbfybTuaSOta7QHKeswcR2KMrfmUggNwQBQMdv1SAjDyWifjEwVM9h2SqFR0dFoTm3UVAnVN4NtkcOxMwQ6fgAmDkPr7VAxQ/2sp2HgOYgMQcO1MHRCRqR162GqT5JuTZec6nKx/WT+P9EPbz0nzRW23JLfoWg+xKLSHclkAkfhqfcygRaIzAH79ug4jjwHrEJRqFEVag1irVUVnG8jUk0jGCfNCClGSM3qRTUbyuAmLDRhvuRG+X4SDBFlaEaa14ZKM05W4aSqhH2NADGS+AhTQwUOrAziZ4BJrJipwkUTXmzLLFLX0PDhY5QREplRXajUUEsjjdixX2QLhSNDJAFGGeEkrWyfVi+dSTQ6afo4jJ1KGlm3oO1HGSLASTTDcMJMBV42lszVaPr+ZPQZ5zQxjlDJzZipzbk/N0WvoeGfdn/2cSJumNSfQlHqMHP3LA/VXOhiDJ3jCDGBojRgVm7I86CgjEazLS/14LgJTEW2ACWGjWjUKAVUXCanvBTqqZs2ej1NNoiOQvfPZbtL1aa5rQAnj8PIHvkYzQbRSUmGVe2yRgqzVbW5iIcgEYSxUxD1QeNl0HtKOhZd/q4LNdJSIZ2e1g+aizKBFojMAev1T5GurGRSCHy6wC8E4TkOh0PBIFOVGkWhTlWoeJuQagw9S6YjefpQPZgMMrVQdwmj00yad5AoQ0SnCZBcmFllkOlS+PMeZZBxQlRgYzV11JRARVos5HHwM8oIkRwzeS/VNNJIRYnWFmUKFRN2KjnLHiqooZ41edOsucTTywGsOGliw4IHmQt0wvQS5HTWTN5GDV42YV2kB+7MNQrSTPETY5j3plmCI4EgwUk0giToxkwtFdyQV9mri1EULCjKxQVoQkSzXrxm5WZUZQ4FaPKkVOtmzOTtV4B128UNGLQpGb2qOe+/noKp1y64GJmroPoWsC7Avk+LG+rc4xA4K40WBp4BZyM0XgfmPCdsudFjeBB6nwT3ami8Xra3ZIdix2f3isampGPRVD9EJsDbKltdklFwVsGa3TIaragCZwm1KbGoNKjv2iLni85AmUALxLxm8kIwKQSTumBCCHxC4NfzWcmDXYE6RaXOiFbrVQX7CifVTP10mBRDpPDlGXXWiIVm43KpHJJ0BOPEGSTKCLFpNV4PFlpwsQonthJGzwlS9OJj1Eg5NuKhnZo5nYs0dMYJUU/lktj2AYQIMsIIAaayt1XippEmPIvouRQIxjnPJH3YqCBJjHZ2YM8xe8hEcxkkieFnAD8DdHJVUePVdFIEOUuY89nWFCfNeNiIeYFTcvJBGi8omKggzGto+KngBkw5ryeNnwTn0AmjYMHCKuIcxUIzdjZftDVmLgghstFpSvxfFDwXV/LqEYi9DOk+WQ+t+IAkx/ken3wL0sNgbgf75dPvj/VJpa4eA+daSaLzQU9Ls4b4ADhWw2Q/hHplLbXzbrBf5ITBfwpCPRAdkab1ni4wG4rrWEDWQhs2yRYXIeS4tJ490LBRtrd4W8FklY9LJ6D1SqnevRhyh2svFKePwFt75f9b10hf3ZxotEygBaLQA5Y2SNUnBBO6/DupixlxnESFEZ3WG3/rFAXTCibVBDojpBkmxTCpWdFpDWaasbAKC95L5LuhoTNCjEGijBPPrlAB6rGzCheNJVbyThBijCB1uKmbw0FomClOMowFE014acaLY4nqpTFijDDCJBNZwZETF400UUVV0crdOCEm6CHMBJU0UEfnNPeiNEnCTBBigjghbLjw0IiHxnmjzzRJZL05PymliRHgFFHDAxdUKotU7AoEMY4Q5wR21pLgLC6uxUIrCgoaIeKcQieMiUqsdGHGS5QDpPHhZEfeNO6C928QqCb2o4njmJU7UJULIitdSFWyquRRDyfPyIjSvID6ZXoU9CmZBrZfCdbt09OjWlzWRj1XLcxwId4PqQAEXoOKraDUweBzUNkBzTdKgpuJ2DgEzoHvLWnSULVp9ug0fx/0vSFNF2rXSAIdOgT9b0pxUb0xUaX3dQgNQ8NmqJujJzQXqSTs+5n0021dgJtSLs4dg8N75Fq8tXDdHdl6aZlAC0Qp2lg0g1DHdcG4QaxTeQ6lCtQYhFqvqtSrCpUrlFCFIUYaNrro/DOiUyeqQaZWGi5RqjeJxiBRBoniz9rlSfFRE05acFKNbVlaQUYIcJ5xEjm2hjVU0EIV1UuUAk6SZIQRJhjLTmWxYaeRJmqoKToSjhJglDPZFpYpBokRIEYQExaceKmk/qL9pBopJuknbEyrseKggTVY54gukwQIcIJ41gPXgpv1VNBesNAowVliHEUnTgXXY0U63aQYJsweLNRRwY3GbUNEeAM7mxflt5shTyGmSIkfY1KuRmVTtv1F1lJfQRe9qEoLJm5FKXYEGhhDvs+B49aFke7FEOsB3zPgvQ5c6yEVka5FnjXS8i/b1pKQ5gpjb8j+09ptMuqcK+2ciIAtJwLPjUKrO8DdLK9XNkDbVQtba89hOP26/H/zOtkjairgpH5sCF5/BpJxsDngmndAbWOZQAvFUvWBJoVM+47rgjEhGNN1YnmOrlOBelWlUVFoUKVQSV2BpBpDZ9BI9Y6SmpZGlYQlybQJM9ZLkOoNk2KAKANEpil5nZhoxUUrLhxLHDXrCHyEGcSPP2dWqBMrLVTTiGdJBpSnSDHOGKOMZpW7Fqw00EgddUW3wKRJkiBCP4dQUGnn8mlp3YthkgGCjOLEi5t6JhkgRoBG1s87CDzOOFMczyp2zbjwsqlgoZEULb1JgnM42YHNEDqlGSHKQQQJnFxJnFOYcGFny7Q0b7FIiZ8ADszchKJciOA1cQSdflRaEYyii0FMyrWYlPVzb2zWizJILD0CkV+AYxdY1hdvWp/Znp6E4e+Dcw14dk6PWjO9ngBxP4zvl9Z+VRvAu+7iKd65EA/BySdl3bOyATa+a/qaLoaew3DmDfl4dy1sewc4Cnj/IiF47ddyvqiiwo5dBGuaywRaCJbTSCEkBKO6YFzXGTPESjNTv2YF6hWFRlWl0YhWzSuMUNMIxozIdJBUTiJVOuo0YqbFcDJd7rqpQOAjwQBRholOc22qxUYbFTQuQX/pTERIMMQUw0yhZW37TDTjZRVVecesLRYaGhOMM8JI1sDehJlGGqmjvuhB32F8jHEOnTTVtFHFqgVF9VMM46OXTq7KRsN9HMKGiwbmT9MJdCIMEOAketYTutYQGhVW79UMh+lcE3qdJEl6iHEYUHCyHRtrCtpudq0iDCgoigtNHEITRzArt0/rGdXFOBqvAQKLIv1aNXESXRxCVa7ApCwgbZlLLKH/BFOTHJ2m5EnRitTFvXdztzf2C9lWU7MbzHl+B4WAqVMw9DK4VsmUrXcBa55vv6kY9LwGoRFIRmDTe8BdoBrbNwhvPSsjSYsNtt4GNQUMG0in4c0XYFCOuQs2dOC54Y4ygS4Ul9KJKG1EqaO6YEQXjAqd5Ix3QAVqVYUmVaVJVWhcYYSaSfUOkGSAFKEZbTJ1WGjFQivWZR8mrqEzTIx+IkwYP8IgI+ZVuGjDhWeJ+zrTaIwQYAA/sZypLHVU0krVnBNjFgMdnUl8DDOcbYExYaaBBuppKJpIA4wwznk00rSybd4Uroz+/EzQSx2dOPEi0BnkGComGlm/ICMFnRQhzhHinCE0ktaAHtZjKkLok6kZKyjoxAnyaxTM6IRwc+fcDkU5NV6dJGlGUXFgEjXovIkmTmBSrkATbxjORdMN6XUxjmDQaIWpNXpJHQiRkquZTzQ0E9HnQBsD5x3T217Sw5AehHQ/KHYpSHLM43WbIbLgfgi9BTW3gz3HK1iLScvAeL8kY5NHWu/VXDa9dzPf71HWp1aX7TCTx0E1yyHbtdvAXiNTuOExqDfERHbP9FTvQhELw5Gn5dxQRYG1V0HHPL2n+XDyIBzbR1AoeO77ZJlAF4qVZOUnhMAvYFToDOuSWCMz3pIMoTYbEepKI9QgGv0Gmc6sm9Zipg0rLVhxLjOZRknTT4T+GSleDxbaqGAVTixLuCZhpHf7mWQqZ+i2GwdtVFNLZclrtTo6fvwMM0Q84++KKUukxZoyTDGMDRd2Y82TDOCiKqvEjRNmgm7ihLDgIEYAO5WYsJAinh34XQjSRAlwgiiyb1LBjId1VNBZdM0yzEsIUjjYDqiYLzKGTSdOjKMI4iQZMOwJb8ZCPZp4E00ckStT7kJVZrtVCaEDESMSdWLi+nn7SS/sOAoooDpkzTP6HDh3g7nzAnmlhyH2LJiawbxKqngTb4GIzyZazej1VG3SW3f85+C5Bio2Xegb1RNy8guqQZ4uiJ2XKd6qXRdfcwZDL0PCL1W5zmZI+GDqLDTshHhaeuZm/HMXA02Dk6/A4Cl5vWkNbLqxsLroUA/BlMDTsbpMoAvFSiLQfAgJwbAuCXVYn92bqgJ1qsIqVaXZSPmulBpqBI0BUvSRnNUiU5MlU8uymjcIBBMk6CPCCNFs8tmEwiqctBlUsJQIE2cAPyMEshGRAystVNGId0GDuwtBppd0mCFiBnmrqNTRQCONi3I3ihOihzepYzXVtKGg0GfUTOvoxE4lccKMcIpK6nDTgHkRM2ITTOLnKCkCQKY+ugXHPHaC+SBIEWEvZuqxsW7e9aQYI8EZBClMVGKhhTjHUKnAwRZUI4sgRIg0zyJE2Oj9bJnW1pKBJk6giyOYlbsublQvBCT2Q+ok2G+A2AtgWSeVt5kUrR6ByBNAGqybwdwCJkNJHHtJ/t+66cL2gvshela2t0w+A9YmKRwyGb2aehp8T0PaD+4r5Pg0RYGUX97uXAvuHRe2pyiQjl1oXcnc5jtmGCxsAGcTeI0UeeCcnAbTdANY3dKgYaF1z4uh/zic2iNbXNy1cgB3AUYMZRFRgVjpBDoTISEY0nVG5iBUiwKNisoqI0qtVlcGmUbRGSBJH0km5ohM27Aua800icYAUXoJE85ZkxsLbbhowbWkUWmSNAP4GcJPyoiKM3XSFqpL7n4kEMZ00CGihshJRaWeBhoWQaRBxrHiwE4FaRL0cZhK6qijM/uYUc6QIEobBabW5ngdEfqn1UftNOBlU8Gj0wQpFON16yRIMYyNDuN6jBiH0QhgowsLrajYiHGUFCPY2YCV2QbumjgGWFBpQecUKltRFFOWTHUxQVr8HItyH4oyt+hF1lZtUqkbewmSp2Rk6XqvHNadQfQ5GZmaasDUAOleMLdJgZEek1GlMqNUMfFriPdKEq6/Gyw52wu8AaHDMkK1NkA6AO6d4OyExJhU3lq8Fx4f90lLv8br5LxRRZG3nf+JJMbKdilSSkWh7Q7ppxv3gcV9QZw0F1IxOc2lEPiH4fDTsi5qtUsSrVpYbbVMoAXi7UagMxEUgkFdZ0iXkWp8xjvoUGCVKgm1RVVxrIDoNIZO/xxk2oAlG5kup+n9JAn6CDNILOsdbEKhGScdVOBdwlqp7G0N0M9ktk6qoNCAmzZqLjq8uxhM4WdoGpGaDLfdxUWkAMOcQiNJC5dlb5tkgADDtLEjO9llsZBGDGcI0Q3oXOgfXTfLv3chkKYLE7i4HjPV6MSIsJc0PirYhYVG0viI8iZmGnFw2bzpY12cJy32YlauRVUunEykxC8BDTO3z9nGIkSANK8COiqrMSmbpPI2+rTsFXXuljXO9DBEn5TuRbZtkhDTI7LFxXHjdKKdiWg3TD4nZ4pW3SgJM+mDsZ/IVG3FFjBVQPS0TN9WvwPMrtnRYrAbJo5Aw1XgMtpp+p+BxBTU7wR3u2x9Gdkr+0kbr11YxJmMwls/hqoO6LhubivBfIiF4fCv5Yg0VYX11y2oX7RMoAUic8Ae7w3QVu2mxgw1ZnCbSpNRWE4Iox91SJekOioE6RnvaK2R7m1RFRpWQLo3mkOmuWnejJq33aiZmpepzzSJziAR+ogQzOnp9GChw6iVLkUrCmTSy7JOGsipk9ZQQRs1eJdAcJSPSBuMiLRYsVEEPyOcwoKDaloI4yNGAAduGlm/YMu/hSJFmCmOEUdOTjFhx8MGnLQUtJ80fjT8xrQW+WOtkyDBKeKcwmykiRXM2NmMeQG2g5o4gSZeR1XaAReCcYTwYVZ2oSqr53yeEGl0ugENXRwBxY2Zm1GwXkjL2rZA8oS0A3TcAiZjPSIFof+Qc0etF2mRSYekh66zS9ZAQ0ckWXp3XbAA1GIw+iNjgHfHHC80cWHMmZaStn6uJkmqGYzuk45Fq+9dGBmOn5GDuwFcdXLSi62ADIOWhmMvwsg5eX3VBtkvOo97UZlAC0TmgP3l2QCOygsHzAxUG2Raa4Fas7xuehuRqiYEo0IwqAsGdB2fPjvd26SotJoUWlX1kvv5hgwBUi/Jaab3JhRasNKOlcZlNG2YJEEvYYZyaqUWFFpw0UHFkvjwZhAkRh8+xnPmhErBUQ21VJRccOTHzxCD2RppZgJMsUSqkWacc8QIomLGiZdqWvNGnzNtAotFjFGmOEbaOBmwUkUVlxXc9pIPOlFjnmgIM/VUcvOCnytEFI0DQALQULkMhcZsbTRfnXT68+Ok+TUqazEpG40bddm7mDgKySNQ8cCFXtD0IMTfANsOsHQs8AWmpHVf4E2InYPG+y/clxyXkar3hukq3bmQjsO5x2X0WWUQuJ6GiUMQHYPW3dNnis6HqQE486wkaLMd1u4GTwFtKjDddMHbIPtFbflPRssEWiAyB+y14QBJpxtfGibTzGjGkFCRJFprhrq3IalGjXTvgBGhzkz3VquSSNsMMdKCFIJLhAAafQaZhmdMZWnHSgdWqpfJTjCJRp8RlUZyouRabHRSQQOOJXM7ipKkD980wZETK23U0ICnpCcTskYqI9JcIm2iiTrqizJk0NEAMWdKNUmMXg5Qk+0vXRyRCjRCdBPkNAINUKigHQ8bUIs44clEyzpRIrwGqKSZwMW1eeuf827rIkSZ/zk6iqKSFs8jiGHmndO3kR6D+ItguxosbaBNSFLVA+C8FdQ5aqx6DNIDcioMygU/3Wi3HNRdczuYHFKxGzokp71U3zS9VioXeKFdRU+AFpUm9sMvyfubb5KtK6E+GNsnHYvqryzoGBAPwZmnpek8inQrmm9EWj5M9MORZyGdBLsLtt8hRUYzUCbQApHvgOkCghr40vIykYaJFLMIBy6Qar0F6o2/nrdB+jeT7h3QBX26zvgMk3y7UTttU1Va1Us7F9VHmh4jzZvrz1uJiU6sdGBblraYjIK3mxBjxLPHy4mJDippxbVkLkwJ0gwwyRBTpI0TChsW2qimCW9J08oZ1e4Qg9n2FwtWmmimltqSmuWPcQ4ffQBYcVBPF5UlGFgu/XWPZ9teVGx42YjLsPQrFBHeQGMKB5dhprEkJ0xp8ToKFlS2oRjtI5mf4JlEm9J/iKK0YVaumb2hxGFIHJDGCukBqcK1rgdL5+x5m4oiB3rH90myNTeBHpJpX9c7ASf4fiVTu84uiJ6T6lzHGqicY8i10GXqN3IcVLs0ZHBuhdFDYHbKuaAIsFVB623FHSwtDT2vwrjRplLTBatvKqxNJRKAQ7+CyJR83pZboKFz2kPKBFogCjlgYQ3GUzB+EVK1KpJI6wxCbbCA/dIMMlkw4jlkOqBPN3RQgUZVoUNVaTNdulSvjmCEFN0kGSQ1bVB4IxY6sdGCZcldhgBipOkhTB+R7Kg1mWp20kkllUuU3k2jMcQU/UySNKJhK2ZaqaaZqpK2wMjeVR9DDJI01K5WbKxiFdXUlIREBCJr0JA2BFROvDSwpiDbwLkQZxw/R0kbo+BsVOPlMqwXGQA+E0l60QhhZ/Ocr7uQ2q4QCdLiFyhKMyZ2zhITCRFHMInOAIJBEBpm5XYUZY51awHQJwAzmBunuxPlzuIUAsI/lH2h1sskgYqUFCeZW8FmCL9Cb8loEiEFRZl6aD7xT2oKJp+VQ7xdm2UNNfwWeK+HRJosedqqpep2MS0rI8ehd48kbWcNrLsd7AV8TlJJOPIM+Abk9TU7YfWO7N1lAi0Qi1XhhgxSHUvBmEGq6TyP85gkkWYuVSs4StWN2mm/rtOnzTbGr1UV2lSVdlWhptBxQiVCEp1+UnSTmKbktaDQjpVObNQsQ4pXQ2eQKN2Ep4mOZHq3kgbsS5Le1dEZJkAvvqyBvQUTLVTTQhXmEvbW6uhMMMEwQ1mLQDsOVtFC1UXMBxa+jzQ++vDRnx1t5qWJOjrnnOKyUAh0QpyfkdbtxMP6gtS6F2aMzibKNH4i7MXJlVgWGEELoQFRFKUSISLo9COYRDAKIgGYQbGi0onKmmm+ugtCegRUr4wKM4g+JyNQ561gylln5FeAANedc29Pi8m07qzbozD+JNS+U6p0Aab2ghaGmiIjzvkQHIbTT8taq9kOa28DTwHGHLoOp1+DvmPyetMa2HQTmExlAi0UpW5j0QX405JMxwxi9ecpqFqVC2TaaJGRqnmFEmrAiEx7dJ3RGUKkCkWhQ1XoMKk0XKK6aQiNHpJ0k8iR+8gU72psdC5Tf6mPON2EGSGWjY1dmOmkglZcJTdIABmVjxpEmmmBMWNiFVW0UDXn/NJioKExzhjDDGdN611U0EILlQVGdHMhRZwxzhPEGPuFiVo6qKJl0anjNDGmOEaMYUCqdb1chrNAk/p8CPEiKWO7dtbiYGu2v3RBaxO/QhcDqEoLCqtRqQMqpln8FVRD1QIQfUr2g9p3yvaWVD/EnpfqXEvXhTN4PQ6x56Sy1z7HNJRUAKZeAddGcBgOSKkAJIYgMQipSbC3g/dq+fipPZCcgNo7Lj5OLTo2ewzaxZAIw+lfX6iLtl8LTXOkmOdC/3E4+aqMiD31sP12gol0mUALwXL0gSZ0SaSjKRgxSHVmlKoiU75NVkmojRawrsC0b1wIenWdPkOIlNsm41SgXVVpN6k0X4IWGYFglDTdJBkgmZ0Yo6DQjIUurDRiWXIVbya920uElEHoFhTaqKCTiiWZCiOHjgfpxUfESLeqqLRQRSvVJSXSNGlGGWGUkewYNQ9eVtGCs0StNpmRanFDhSzro2uoXMSszgxijDHF0axa104DVVyGmQIjvBzoJIhxmATSlFzFiYudWFjYmDEh0mi8hi7OY1KuyU5oKUZ4ZDwREgdlqjZTC42/BiIsCVTNaQdJdUPiTdlLal2Xf3uJYRlVVmwB11pjJNkT0tjBUiUN6IMHpOWfyQVaSKZ0vdfNv87AeRh4RroVNV5fWK+nlobzL4HvrLxetx46dxW2Dd+gTOmmEuCoILj6OjwtnWUCXSguhZGCLqTSN0OoIymI5JnIXWOGJotxsa68OmraqJv26Dp9M+qmNgXaVJUOo+d0uQeJZ1K850lM6y91oGaj0oolthDU0OknynlCWfWuAjThZDUVS2IZmOkl7WGCsGEkr6KyCi9t1JSUSJMkGWaIccbBOFmpoZZmVmErwWvLVx91UU0Da7K+u8VCRyPEWYKcBXQUTLhZRyWrF6UETjFClDfRsjXXThzsWPBgcF2cIy32oChVsu9TKeEs2civpCuRMyetqk1K0wU0cL1n/udr8Qt2f/FBGZG6r5IORSAjUt/TULFRKnGt9YaB/Dw1z4kjMPo6IMBeC623g7XA1zx0BPqMbVTUw9p3FGZKHwnAoV9CJEAwnsLz3s+WCXShWClORCENhpMwbBBqIE/at8YMzRZotkpSXUkRqi4EQ0LQo+n0zph9ajXItPMSkWkAjXMk6J2h4m3AQpchPFrKqFQgGCPOeULTpsJUY6NrCdtgJJGOE1piIo0TZ5AB/EwCMuKXHaRNi3Y1AlkfnaCXSfqN+iNU0UItHZgWuf0UYfwcIYEPAAtuqrgMG0XOuAQEaWIcIc5pAFTsOLkK6wIN9IXQSfNrFKyG4XyJTrRirwEp6VAEsoUlcURa/znfLY0YFirwSfllBFp9K9iNvkw9IW+r3C4VvAtFeEBaAWpx2R/ashsqCmsRmtYvanFKcVFlAWnhVAKOPENQdeK5/NYygS4UK4VAZyKqSSIdTsFQcnYdVUH2oa6yykujZeX0owohGDHItGfGRBmrkebtugRpXg3BoBGVjuQIfmyorMZKF7Ylj0oDJOkmxGBOtbYCM11UsgrXkiiIl4tII0QYoJ+QMQhb9pA2U099SVpfksQY4ywhJoztW6inC08J2koi9DPFcfRspNuOl41F9Y5mkGKcKG+gGWloG+04uBx1gdG5ELE5hUPSqeig0QKzQJvJVD/EngHLWpnCTZ6WHrnWLWBdU5g6VotLcwVTBVQZI9O0CIz/QhJoxeaFbSeDZBgGnobYuLxefxXUbS9sG/EgnPo1xCZlpN11E9QWMONV1wmGQni83jKBLhQrlUBnIqZLIh1KwlCeCNWETPO2GJfq5fEYuCiEoejt1nS6dZ1ozifMrkCHQaaNyyxACqPRTZJzJKYNBG/CwlpsNGFZMnMEgDga3YToJUzKSH/aMdFJBe1ULImJvc9I7Qazo81UWqmmlWosJTxxCBBggP6sGYMNO6tooXoRUV0uIkwywhmSxvalReC6Rbe9aCQJcIKI0ZcqRUZbcC6wjpkPMho9SpyTQOHR6FxIiz3o4jgKlZiUm1GVhoU9UfPJlK1iAcUJ1o3ShL6oRYRgwkgLmxyQNkwZGu8rbnu6BiOvgl8eK9yrYdXNMg28UGgpOPs8+Hvk9ebt0LpzwScGZRVugXi7EOhMRDRJpINJGEhCdEYN1anCKgu02CShOlZAujcTmZ43yDS3h7ZCUVhtUuhS1WVtjdGNqPQsCUZzolKnUSvtwrakg8DT6PQS4Twh4tmJLArtVLCaSuxLEBH7CNOdE5GaMdFKNS1Ul0wpLHtIJxhkMNv64qKCVtqoKHBiSv7t60wywAQ96GgogJdm6li96LRuHB9+DmdFRg4aqeIyTNgv8sy5kcZHhL050WiHEY0WN6RAFyNo4gUEYUDBpOxAZfu0Id7zImMFWAqE3pLORqpNjjozORfX6zl5QhKp0GXvaNsdYC3g5EgI6N8HQ4fk9aoOWHPLxSe/UCbQgpE5YIMTAZpr3j4EOhP+tCTSgaSspc5U+daaoc0KrTbpmHSpe1B1IRgWgnOabI/JFSBVqZJIu0wqlcu40BAaZ0nQTTJrjqCg0IaFtdipXcK+UknkUc4RImQQuQq04GINblxLsO9xQnQznlXtWjDRRg2rqCqZs5GGxigjjDCcVexWUUMLLSURGqVIMMa5bNtLqdK6Ao0gZwhyDikyshhORm1Fb1dGo28RR7rpqDhwcTWWIttohEii8Sq6kGbpilJvCI9K8DumTYBSMb2HdOELW/wPTGREpnTTMaMuehtUFOiBO34Gzr94wXRh/R0XNaMvE2iByJrJHwjg9bipsUGtDWqsUGMDrxVWyEjNBUMTUuGbIdSJGWxqU2RU2moQ6qWOTjUh6NMF53SdPk0nN5huUBXWmFS6VHXZ7AQ1BP0kOTNDwVuFmbXYaFvC6TAZwdFZQkwaxKYAzThZQyXuEo9Vk/uTRJrpI7VipoNamvCWTFyVJMkgg/iQNS45rq2RRpqKnvqSiwh+Y+ZoJmosTVo3SRA/h0kyBYCNGqrYWvDc0VykmTCi0YxSdw1OtqMUeRx0cZa02AMkAQsm5VpMyhztKAvaYBTCP5apWcduMC8wPTwX0kFpwmArcDupCPT/2qiLKtBwNdRuLWwboVHZL5qZK7rudqicex1lAi0QmQP27QMBbJWzD5hJgWor1Nmgzi7/vt1INapJIu03CDUx412uM0O7TUaotUs3YGRBSApBt65zTpMzTjMwGUretYaSd7nER5OkOWMoeDPWgVZDdLRmiUVHkyQ4Q5AxI9UK0ICdtbhL3gKTMWToZiLrbGTHQid1NOAuWT04SpR++rJCIzMWVtFCLbWL3ke+tK5U63YuagapQBCmmwAnEWgoqEbLS1fRLS8yGj1MnDMAmKjAydULdjGatT0RJs3zCGEYUCidmLhh4QKjXGhTEP2VNKRHleYKtgKJKwM9DeM/k6pd77WFi4t0DYZfhimpaMazBppvLKwumgjDqV9B1CfT1qtvgrq1eR9aJtACkTlgk/4Amt2NLwkTCfAZl1SeI2JWZHRaZ0Sr9XbwWC59WnQh0IX08u1PQF+e6NSlSiJtt8l2mUvpjhQ1UrxndJ3JHDK1K9Clqqw1qdQuU700gc55kpwhPs3taBVW1mOjfglHmwVIcpYgQ4b4B6RV4Brc1C2iLpcPOoJhpuhhIuu168LGauqoLYE3bQZTTNFPHwnj5MCJi1ZaS+JolCbBKGcJGvNBzVhpYA1uFhdJpYni5whxI4q24Kaa7Ysal5ZihAhvoBuCKDsbjEHdhZ+YCSHQOYwmDiDTzhWYlFsWLjCatrGUnDuaMmZpmtvAcXPhKV09Bf4XpT8uSFP6qgIJEMB3FEZeo+h+0Vnioh3QeuWsH+0ygRaI+Q6YEBBMSUIdTxh/4/lJ1apCvUGmmYt9aTsiSoKoJom0LyGj01w+NSNJtMMmL5fSyMGn65zRdM7N6DGtVhXWqiprTCqOZTiDEQiGSHFmRiuMFzPrjfTuUpnZh0lxlhCDRLIUXoWVdbipX4SLTj5o6Azgpw9fdvqLGwdd1JdssLeOzjhjDDGUtQaUU0NbS1IfDTPJKKdJGiceLqppZB3WRR6rCANMcRSdFKDipgs3a4siPQCdJDEOkKAHABNuXFyDuUjVsi7GDIGRVMWalMsNgVERn8vkCYjvkSPJ1IriU7qhIxAwzA7M1VB7u3QvKgSRIeh/RrbQmB2SRJ0FrGWmuKh6NXTdPG2iS5lAC0ShB0wICKQukGmGWNN5jpzHAg12eWl0gHeFR6makG0yvQahhnOKkQqy17TDBp02qLhEJwe64X50VpeGDZpx3FWgzaSyzhi/thwtMUE0ThOnO8c20IbKWmyswbZk/rsx0pwlRB+RbFq5CitrcdNQYiJNodGHjwH8WRFQDRWspo6KEkW/KVIMMZh1NFJQaTTqo8XMIM2Fjo6PXnz0GiYMKrW0U03bonpTNRL4OUrMGJdmodKIRr1FbzPJIFHeQCeBgoKdy7CzsajUthQY7UEX0uZOURoxcwuKUoSDk+aD6DM5Kd2rL0xtKQSJEfA9I2usihVqdoO9wPFyyRD0/QoSkzId23wjeAus946flhaAQpfORevvkPVRygRaMErRxqILmEzAWALG4vIylZr9OJt6gUwbjXqqaQW0l8yFyTT0JORlZqq31nyBTKsuUc9pUkjh0WlNzjPNwKUorDWprDOpuJeBSBPonCPBGRLEDJJRjakwG7DjWaI6aRyN84ToIZwlcK8RkZaaSBOk6WGcIQJkbPua8NJJLbYSpa9n1kctWGmlrST9o0mijHCGiOGWZMVJE+txLoLwAKIM4+ctdBKAQiVdeFi3iGg0QZR9JJHjtizU4eQaTEVaF2riDJrYA6QAO2blRlSlrfANzUzpWjrBcZMkwoIWFJV2f0lZq8V9Fbi3F7iNFAw+D6Eeeb1mGzRcVVh0EhyWpgtaAmyVsP5OcFaVCbRQLFUfaFyTEepoHEYMUp0ZparIVG+jHZqdklwtK5RQw5ok0u6EdEjKfSleE6y2wWr7pTNwmNQFpzSNs7o+TSTVrCpsMJnoWAbhkY6gjySnSTCZkwxvwsIG7DQsUZ00gca5ZSLSKEnOM864QXIZM4Y2akrWQ+rHT78xPh2gEjdttOMowWsJMMoYZ7Peul6aqadrUSIjjSRTHCXKIABmKqhmO7ZFjHpLcJ4oBxCkUbDg5ApsdBS1LSEChsBIOjipymZMXJUd4l3Ywo5JU3p0UD3gvF0a1he0IE1Oa4mckNcdq6HqJlAL+H4IAeP7YfyAvF7RJi0AF9DrmUUsACefgkQQTFZY9w6CSuVvLoH+6Z/+Kb/3e7/H5z//ef7yL/9yQc9ZLiMFXUhR0kgcRmLybyyPPV+dDZoc8tJoB+sKrKPGdeg1yHQgybS2E69JRqWrbVBzCRS9mhD0GmQ6OEN4tM6kssFkWpaodJwUJ0kwaPxQA1RjZgP2JfPezUekVVhZj6fkYqMgMc4yRsAQv5S69SXTPzrMMAI966/bRPOi2140Uoxxjilj/JgUGa3DXaQCNoMoI/g5UrJoVCNEhL2kDZ9eK+04uaIo8wUhNDT2oYujAChKLWZuLa5nND0GsadBj8hpLPYbpRVgoQifgKlXAb34umjgHAy+IEm5GNOFVFy2uYRGQFEJ1m7Hs2bnbx6B7tu3jw9+8IO43W5uueWWFUegefedkmQ6HIOhGIRmpEkVpMq32QGrnJJQzSssQk3qUoR0Pi7bZHLPCTxGZNp1iSLTsBCc0nROado0C8HljEpDaJwizvmcNhgXJtZjYzW2JeknTaJxdgaRVmNjPW5qS0yk44Q4x1i2h9SJlS7qS6bYTZCgnz6m8AOy7aWVVmpKMNYsyhTDnMyKjCqppYF1WBYhYNJJ4s+JRher1BXoxDlODEl8ckzaNVgocH5mZn2ij7R4CYgDVszKLlRldREbikPsWUjL14l1M9ivLdzdaFpd1GbURQs0ko+Ny7poOiqnxbS+A1wF2C/qGpx7EXxnCUaTeHY/+ptFoOFwmMsvv5y//du/5etf/zrbt29/WxDoTIRTkkyH4zAUheAMQlWRad5VTkmq9faV1Y86H5lWmSSRdtnAs8xkqhtGDSc1jYEZUel6k4kNy+B4FEfnrFEnzUyEsRqCo3XYsC2B4CgTkXYTzpJ3LTbW46G6hH2kOoIh/PQwQcp41704WUMDlSUi7ABT9OW0vVRQSRvti54/KkVGPfjoQyBQMVFPF16aF9WXKqPRw4Y5vYqbtbhZU3TfqLQCfC1rvmBno9HuUvj2hIiQ5rmcntGNmLim8JSuEJDYDwkjjWqql6PS1AJNJtIRmHwakmOAAp6roHJbYdtIRSSJxickiTfdAFXrC9vGwH6CaTOezu2/WQT60EMPUV1dzTe/+U1uvvnmty2BzkQkLQl1MAqDMQjPIFSzIlO9LU55qSqtSc2ikDLI9Fxc/s1N89aaYY1dRqfLreadKyptVVU2maRJw1IqeNMIuklwigRhg2xMKHRhYz02XEsgOIqjcYYgfYSz70MddtaX2JAhjUYvPgaYzBJ2Ix5WU1cSoZGObqR1hwxFsEI99TSzatFp3ThhRjhFzKjtOnDTxIZFzR3VSOLnCDEjVWzBQw07sBQZnQtSRDmYHdptphoX12IqYntC6OjsRxOHAVCUGiOlW0SknOqD2HMgkqDYwbkbzAXa7gkN/K9C1DCSL6ZfVE/LdG7Q6Dmt3Q71CzeSh99AFe4PfvAD/viP/5h9+/Zht9svSqCJRIJE4sJMxmAwSGtr64ok0JkIpi6Q6VBU1iJz4TJJIm11wSoH2FZI/TSpSwHS2YQ0v8/9kDVaYI2R5rUtY3o6E5WemFErrVQUNhoKXvsSEqlA0E+KE8TxZ4dtS+XuJuy4l4BIY6Q5Q5D+nD7SRhysx11Si8A4Kc4zzigBQAqN2gyhUSk8dhMkGKA/O39UqnVbqabIqSIGBAI/g4xz3nAyUqilgxraFjVkO8IgU7yV7Rv1sN5wMSru85Wknyj70EmiYDYERp1FbUsXA6TFC8iUrsVI6RYwyzO7oRBEn5Yeuihg3wm27YVvJ3xcCozQwVILNXeAuYCTmJniosoOaLl1wUT8G0Wg/f39XHnllTz99NNs3Sqtpi5GoF/96lf52te+Nuv2twOB5kIImExKQh2IyrSvlvPuKcgUb4sT2pyylroSelDjukzxnkvIWacZqEj3o7V26YS0nKnpgEGkp3NM7U2G29GmZXA7GjGINHcaTItBpNVLYCIfJc1pAgwQzZ7MtOJkPR4cJdzfTKGRDQtd1NGwCOeeXAQI0EdvNq3rxkMb7dgXmTZOEWeE04QN8Y6dCprYsChfXY04kxwhbpje26immh2Yi0xBa0SIspeU4Ypkox0nV6IUEenLlO7zCDECLCalm4b4q5CUZvlYVhutLgWuKT4Ek8/IOqvqhJrbwVZgzXfqLAy9IHs97bXQdqcctH0R/EYR6E9/+lPuueceTKYLb7SmaSiKgqqqJBKJaffB3BHo2aEA7Q3uFSfSWSjSulT29kdgIAb+5PT7HSZodUKbC1ocK0PdG9EkkZ6OwWROwdSmyIh0rR0allHJmzb6Sk9oOhM5UWmDqrDFZKJ9iUVHPtIcJz5NuduEhc04lmQSTJgUJwkwnJ0PCu1UsA431hJGwGMEOctY1mPXjYO1NOAuQVuKjm5odTNqXZUmmmikadFDvAOMMsppNNIoKNTQRg0di9pumD6mOGa0p5ipYgsuCjQTMCAQhsDoLUD66bq4FnMRkbhM6R5AE4eAjEp3N4pSxElD8gTEDHWtqQqcd4BaYHCSmTWangRMUH0TOAtU+kZHZV1Ui8sotv2dYJ+/p/g3ikBDoRC9vb3TbnvkkUfYsGEDX/nKV9iyZctFt5GdxvJCAGelmyo71DrlpcYBNc6VQTaFIpySkWm/EaHm2g+qSDOHNoNQvSugdjqZlkR6NjF9tqnbBOsNMl3OeumYrnNM0+nOmQ5ToShsMqmsN6nYlpBIA2gcJ04vF070GrCwGfuSeO76SXCSABPG/sworKaSLipL1tupodPPJL34so5GTXhZTR3WEpwcxInTRy9BI21sw047HbgX6a2bJskIpwllIz0XTWzAsYjtpokyyUESRgraQRNVbMVUZBo9zQRh9qATRUHBwTZsrC8qRayLfiOlmwBsmJWbijNeSI8arS6G65BjN1gKPFHQUzD5HMSN3/jK7eAurKZJMgi9v4TklOwzbdkNlXO/nt8oAs2HYkVEf/9qAGz5D5jHDnVOqHfJvzXOlddOMh90Idtl+qLQF5ntkuSxQLsLOlxS5XspU71CyDrpmbjsM83VTbVYJZm225bP5D4qBCc0nROalh0AblZgraqy2WTCu4S55hAaJwyrQGEkW+sMIm1cAiIdJ84JpggYkaIVlXW4aaeiZH2rCVKcy6mPmjHRQS2rqCrJPibx0U9/doh3DXW00IJlkccryBgjnEYjhQJU00YtnUVHowJBiHMEOAXomLBTzXbsRfaiznYwasLFNahFiMTkZJdnEUKeNMhh3ZcXLq7To0Zd1HAdsu8E245CFwPBfRA6ZGyjHapvLcx0QUtA/9PSSxcFGq+DmvxTYcoEuggVrtnhZiLKtEs4Ofs5qgLVDkmmdS5JrFWXmHgKQTAlibQvIvtPc7VIdlVGpR0uWT+9lCcKKV2S6Kn49HqpVZEq3nV2qF+mFK8mpP/uMW36ZJgWVeEys4lVS1gnDaNxkjjncoi0BjObcdBcYiIVCIaJcZIAkew0FjMb8dBUIhN5gABRTjNK2KhfurCxhgaqF6F6zSBNmkEGGM9OY7HQShs1ixQZpUkaU14kIVhx0szGRUWjSQL4OEDaaE+poBMPG1GLTKEnOGs4GOmo2HFxXVE9o9J4YS+6kG5BirLK8NItsL4sNGlGnzRchyydcqpLoXXR6FmYfBHQDNOFO8BcQHpZ6DD0MkwZ9dmarXK+aHkay+JwsQMWT0siHYvAeATGohDL43NrMUG9ExoqoMEgVdslsrUrBElN1kx7DUJN5LCpSZFq3s4KGaFeyukywTScjstLrsl9lQk2OGSKd7mmxQwZRNqrXVhItaqwxaSyRlWXrE4aReckcc6SyLaJ1GBmCw6aSkyk0pYwwikCJI1TrCqsbMJbsh5SSdYBzjOW7R+to5Iu6nGUQBUcJkwP3cSNGm+pREYhxhnhNGmSKEAN7YuqjepoBDhO2JjGIo3pdxRtvpBmigivohECwMEW7GwuMqV7lrR4BUij4MKk7EZVijBxSJ6E2Cssqi6aGAPfr2VkqzoMcVGBk2HGD8LYPvl/92pYdfM0hW6ZQAtEMQcsnDTINALjBrmm9dmP89olmTZWyIuntCYwJUcm1dsbgZ7IdGckBdlz2umCjgpwXaKTAyFgKAWnYjI6zWiPVKSF4AYHNC/T1JugEBxLa5zS9azPsVOBTSYTG5ewThrLIVKtCCLVEKQRhNGpuUj9MY3OOUKcI5TdVyMONuKhokSknUKjhwkG8AMCFYU2Q7Kz2LaXTO/oEENGVKbSRDMNNC5KDCSj0TPZmaM2XDSzcVFK3Rhj+DmMRhxQ8bKBClYXN42FNFHezI5Is1CPi2tRixBuCeEnLZ5FMAWomJSrMSkFDsaGGXVRm9EvWqDrUDoCvl9CykfR4qJcha6jQdr/meWPc5lAC0SpprH4YzAagdGw/BtMzH6cwwKNLmiqlIRa41jZad/JhCTS7jD4ZqSyG+ySTDsroPISeN6C7C89G4eT8enTYipVSaTr7eBchqg5IQQnNZ1jOeYMZgXWqSqXmU1L5nIUQ+fEjIi0FjM7cc07AeZ1IvhIY0Ihgc4WHKy+SFQZR+M0AfqIIJAnVO1UsL6Eit0wcc4wylR2uLSFtTRSS4HONnkQJ04vPdlJLw6cdNCJa5Ep4+m1UYUa2qmlvei+UWm+cJgYsqXETh3VbMdUZNScoJsobyLQULEZNoAF2NwZECKFxkvoohsAVenCxA0oSoFn0noUor8GzXAdsl9T+Gg0PQWTz0O8R16vvBw8Vxa2jciwVOjqSbC6of1dYHWXCbRQLNk0lvQFMh0Jy0hVmxGlWk0yQm2qhFWVUpy0kqz5chFMQU8YuiNywkwu6m3QVQmrL2FkOpGSRHo2TraXUwHarbDJAausS3+yohttMEc1HZ9RJ1WA1SaVbSYT1Uv05uYSKcB78OCY4wf8NHGOEGMHThowM06aY8S5HOeC6qlhUpwgwIiRFrWgsBYPHVSUbJD4zLaXWipZSwP2EkS8E0wwQD9pUoBiTB1tXtTc0ZlKXTsVNLNpUS5GYXqNdhcNFSvVbMdBEYOsAY0gYV5FM4RbDjZjZ0tRka0mjqKJNwDdcC+6rfBWF6FB7GVInZbXrevBvgsK6TsVAgJvQFg6KeFYI6PRQraRmILepyAVkh66bXcSTNvLBFoIlsvKT9MliY6EYTgEIxFIzZjGYjXJyLS5Ul5WaoQaSV8g0+HYDGchO3RVyMjUeQnINC2kUcPJuBy7loHHBBuNqHQ5HI+GdJ3D6ekuR62qyjazSuMSCY5i6PhI0zJP/fApAjRiYUeOIOhNIsQR7Cog0vMR51iOYteFmU14aCyR0CiNTi8T9DNpeNSqdFBLK9WLVuumSNFPH5OGUYINOx10UrlIA/zpfaMq9aymipaiHYdShPGxn5QRNVewGi8bipruIlO6B0kg53laaDBSuoVHtroYIS2eBWLIVpdbUJUCU7EAibcgvhcQYG4ExztkbbMQRE6B/2VAB2uDdC4yFfCa0jFJovEJUMwE3TvxtG4tE+hCcam8cIUAX0yS6XAYhkJS0JMLmxmaKmR02uJemTXUaFqmeM+FpZFDBpmaaVeFjEwvha3gZBpOxKTwKNMHa0KaNGxyLI+Cd0LXOWL0k2a+XA2qwjaTidYl9t2dCR3BHiLYUNiZEx29SYQkgusKTJUKBANEOUmAuFGNrsXGJrx4SmQNGCbOaUazbkYubKyjEW8JiHoKP730Zlte6mighZZFRqMJhjlJ2OjxdFFFExuLnvAi0JjiBGFk6tSCmxquwFJkWjtBD1H2GZGtw1DpFt46I92LnjFaXRRMyk5MytbCF5QegOgz0kdXrZDiIlOBaun4kBQXiSSYKqH2nWDxLvz5WgoGnoVwH8FIAs/VnysT6EKxUszkM4Q6FJKX4fDsCLXCeoFMV7nBvsJUvpE0nA/DuRCM5dSAVaQL0lq3NG9Y7taYlC4NGo7HwJdTK601w2aHJNSl7isNCsGRtMbpHGOGKoNIVy/DWLUMThHnKDF24qINK4MkOUKMGszsxFlUtJRG56whNMqcJrThYj0e7CWqjw4zxbkctW4jHrqoX7QJQ5o0/fTjM9KvVmx00IF7kXaDfgYZ4yw6OibMNBrjzYtFjFEmOWT435rwsoUKijA4ADQChHkFjZBhvLAdOwVOLiHT6vIqupCpWFXpxMRNhddFtSmI/gr0gJwv6rhFtrsUgtQUTPwStKA0bqi5HezNC3++0GH4VYJJG57Oq8sEulCsFAKdCV3I9pnBIAyGZOpXn/Hu1DolobZ6ZOp3JdVPQylJpmdD0wVIFkWmd9dUypFsy73m0RQcj0oLwQyR2RUpOtrkWHq3o6gQHDOMGTK1WreisNWssm4JW2BycZYEbxGjApUEgjAa11JBO1YEIi+JJtGxXkQYEyPNcQIMGdGiGYV1eOgskRFDCo3zjDHElLF9E2uopwnvorcdIEAvPSSNOnItdbTQuqgpL0miDHKcuNFO4qaBRtZiKrKWqxHHx0ESTADgZBVVbEUtYo2CFBH2kaQPACstOLmqqGHdmjiBJl5D1kWrjbpogb+lIgHRZ2VECsWZLmhx8P0KkqOAClU3gWttQZsoi4gKxEol0JlI6zLdOxiCgSBMxqbfbzFBc4Uk01Y3VJZuStWi4U9KIj0bmt4a4zDJFO+6Sqhd5vR0XJetMMdiF/pKM6KjLU5oXmJ7w6QQHNd0juY4HFUoCpcZVoHmEhKpQDBKGg+maeKibhIMkCKBzm1zmAGMkOI8CZIIkgg6sLLuInWzSRIcY4opIzXqwsxmvDSUwPsWpEn9KUayJgxenKyjEdci+1M1NAYYYNwwSrBgpYNOPIuIRgU6E/TiowcBWLDRxAZczO/JOvf2Mg5GJwGBGRc1XFF0z2ic08Q4iEBgohIX12Mu4oREF6NGXTSKHNR9a+F1UaHLmmhSDg/HshYcNxYoLtKkQjdmjDRzXwnuyxf89DKBFoi3C4HORDQlU70DQegPzjZ38NolkbZ6ZB3VtEKsB0dikkjPh6ePY6uySiJdW7m84iMhoDcJR6OyvzSDahNsdkqDhqVM76aNFpgjOS0wDgUuM3pJLSUg0jg6R4ihAlsNEjOj8AoR4uhsxs6qPNHnaeKcJE4rVqoxk0JwnBgeTFyFa06lL2TGtUU4SSA7QLwBO5vwlqR/VEcwwCTdTKCjoxi9ox3ULNpEPkSIHrqzU15KEY3GCDLEcZKGermaFuroKnqtCSbxcQCNGLJndBOVRY40S+MjzKuGl64JJ1cWNR5NiKhRF5UtKkXXRZMnDNMFAaYGcN5emLhopv2fc52cLapc/FiXCbRAZA7Y1FQAj+ftQ6C5yNRP+wOSTEcj8rYMzKqsm7Z7oM0j+1EvNXQhTe5Ph6RxQ2YUm4J0P1rvlu5Hy1kv9adlRHo6dsGD16ZI9e4Wx9L2lGpCcFrXOZzWCRtvns0g0s0lIFIfafYRJYFOBSaCaNhR2IqDVUbaLmMTqKAwSJKjxHGhGoO8VGyoxNHpI0kTFioXUN9MoXOGIN2E0DEML6hkHe6SGNXHSXGaEXyGFZ4DK+tppGqR/Z0aGoMMMFbCaFRHY4yz+BkCpPnCKjYX3e6ik8THoeyINAeNVLOtqDSsToIIr5Ey+k9tdOHk8oIVv7IuugddSMu8ovtF04PSRzcrLroTTAVG7eETMGUQsW0V1LwD1PmPTZlAC0TmgP31EwFaG93UVkJdJdS5weNcmW0kF0NSk7XT/iD0BWS0mot6lyTTdq/09L3USGpSxXsmNF3Ja1VlineDG+qWMcWb1KX/7rEYBA0hl4r0373MATVLeAKiG567h9M6AePraDeIdFMJiHSIFGE03JiowoQNlTQCc07kqRlq3UGSuDFRgQkfaS7DwRpsJNCxFUh+YVIcY4oxI6qzY2ITHlaVwPsWYJwgpxklaZz6NOGli3osixQxhQjSQ082Gq2jnhZaF6XUDTPBMKdIk0RFpZ41VLFqEWvsZorjSFN6J7VcgbWINKwcj3aMGDKFaqaGCnYV5V6kieNoYi+yLlpn1EULfK+1KYj+EnRDGOS8rXDnong/+J4BkTI8dN8574DuMoEWiOw4s58FcLimHzCziSyh1nug3g2VK4BwCsVEFHqnoDcg/5+LSht0eKDDK4VIl/qEIZiC00EZmYZz6qXVVkmkayuXryUmk949Ep3eU9psga1OaF1CcwYhBOd0wcG0No1ItxpEWqoaaQKdQ8TwYmIdNhQUhkixjwj1mLkCJyYU+klyigQ3UjFv6vZiGCXGMaayRvW12NhCFZUlSOum0TjHOEP4AbBiZh0N1C1ypNnM2qgVG538/+z9Z2xka3rdi/925UgWizlnstk5x3P6zEmT5KuZq6v/1djWSDIkGzZgawTDhmF/MCwZtjDQB+s6aGBdG0q27KuxZAVPnpND58hmbDZzzpXT3vv9f3h3kUV2YiWSx8PVKDA0uau4Way1n+dZz1qtePOKNEsyywARY93FSwU1dGPJcf0nSYBlbqMRJd+WbopZwlxDkDLci17JadVFF7PGXDQBuLAob2Xvo6vHpf2fOgco4LwCtsPZHSO5LBW6ehTMbmPN5dnV7AGBZon0CRufCZBUSlgKwbJxU7Wnv95pk0SaJtTKErDts3WSFyGSlFXpREAKkjLdkRwWSaQtPqnu3cu5qRAyKWYoKA0b0i1esyItBLtLpIp3twh/KSWJdDSxaRzhM0si7XTIx1UMpN2N7qk6wQwiPWHMSPMl0nVUbhKlFivHjEpjlAT3ifE23o02bRCN9wlxHCcteYt1BKOEGCaIjih4WzdAlEHmiG7sd3rppAZ7nisvQYKMM7ah1K2mhnoa8oo0W2WKJUYRCENg1IObspyOp5NilQfEmAPASR1+jmPK4eJEI2SsugSMVZdTOOjK+jhCBFH5EUKsAWYsyquYlCy9a4UOsQ8znIuOSQvAbJ77ahiWvwvq+gvXXA4INEs874QJAesRWArBUhAWg7ASBv0ZpvFlbqguhRqffFtauNSnokLVpQhpbE0SaqaRg9UsRUgtPtnute5hEktCk8KjweDWlRivRVal3SW7JzwKa1JwNJhhGegyyRnpYadsOxcD6dbu/QwidSpw0mzmkNmEOQ8iVREosGHF94gYYyT5PyjdEBbF0PlfBLiMe2Nmmi+iqDxijYWMtu5RfAWJTdPRGWeFSVYQiIKtvKioTDPFsrE36sRFK2248njMcULM0E/SWP+poJkKWnL2081s6UqV7llsOVTLApUINzdWXey04OIsSpYXIkIk0XgfXcjjmJUTmDibvYlI4h7EjSQVaws435B7ozuFnoDlH0ByHjDJXFFX25YvOSDQLJHNCdN0WZkuBiShLgYhFHv665w2SaRpUq3Y42puJ9CFXJMZX4fxgKxU0zCbJJm2lUkRkm0PyXQ5Lol0JCxnlSDnk81u6CmVAqTdqEqTuiTR3ihEjMdhVaDHAcdc4C7SOdKF4LFBpCHjz9WjKJws4B7pCio3iXAB94b6tpcY0yT5a5QWLFw7jXli9LFGdCPSzMHRAql1Q8QZZG5j5cWPmy5q8o5LW2eNccZRDRP5OuqpoTZn2z4djQUes75RPZZQx2FsOa7+SJXuHTTiKJgp4xhuGnM6VpxBotwHwEIZbl7BnOXsWgiBzm00Ib1rTUoTZl5HyTYXNDUKsffkuoq5UoqLslLoarD6LsSksxOll8F7dOO/Dwg0S+S7xhJLwkJA3ubXZcW6vUo1mzbJtK5Mtn4te0hCL4MQ0rd3fB3G1iGQIewxm6DBK8m02bd3ZKrqchVmILjV3L7EAodKodsLzl2oSnUhDewfRGFtm+DohAvKivQYdCEYMog0YvzZligKpy1m2vO0CEwhuEmERVRasLGESgSdC7ioK1D1uR2a4WY0QnBDrdtBCR2U5G1Sv7nysmS0jE20U0k9ZTkTHkhP3QnGWTdmrh68tNCaV96oTHcZQkM1HIwOUZLD/BFksssqd4kb1bKbJso4mpOXbop5IlxDJ2HMRS9jzcFZSeaLfgRohunC51GULG0J1XmZ6CLiYPKC60tg9u38+4WA9U8h0ic/9pyA0vOgKAcEmi0KvQearlLn1zdJNb5NBWsySWFSrQ9qy6CmFKz7eI66GoPRNXlbzyArkyLXYzr8e9vmXU1IIn0c2lqVtnjgcAnU7VJLfTIhiXQu4/fdbINT7uL57mpCMKDp3M8wZCgzKZwxm2nJs+0xTZI5Ungx48dMVYFDu5+FMCkesc6SUTG6sXCcMiryDMEGiJJkkLkNX91SXByiFleeFwXLLDHJJDoaJsw00kRljqQHkCTGLP3EDBP5MuqoogNTTibygiCPCSLXSqyUUMFZLDmonzUiRPgY1bhgcHE6p7moLhZRxY+QZvROQ1yUJRlrAYh+L0Oh+3mwZGHdBxC8D8Gb8n1XJ5S9RjAUPiDQbLAbRgrrEZhbN25rENmWFaoosiqt90N9mRQo7deW7/PI1GySJNrhl+3evXj8qi7XYQYCW714y2xwpFQqeK278LgWU5JIxzIeQ50VTruL53CUMiwCH2ZYBFaaFM5ZzNQVKf0ljQHimIBO7AVr784SpY/1DZP6BlwcwZd39qhAMGv46mromFBoo4qGPKvRBAnGGCVs2PaVUkYLLVhzvOgQ6CwxxsrG/DG/ndE4S6xw1/DStVLOSZzU5PC4VKLcIsGE8bhajblotvuiYUNcJIOxcxIX6XGI/VBWpJjA+RrYsrPuIzIMax8gd0UbCFovUFpWcUCgO8VeOBEFozAfkIQ6u/b0HNVillVpmlDLvXu/XvIsrMXgyZq8ZbZ5bWbZ4m0vk7Fse/HYVxKSSIdDMuIMpMCnyyvJtLTIVn0A66ok0sfxTd/daousSJuKZLWYFIJeTadX0zZ+7gaTwnmLpSh5pBE0vmOoab2YOY2L2gJVqil0BgkwbpgkWDFxmFKadpBEIhCskcSG6Zmz1DgpBpljjQggq9EeavOajUrLxHlmmDZUtTZaac3LmD7MKnMMbOyM1tBNaQ7EB6ASY5W7JIzVmRK6KKErpwuHrXPR3PZFZUj3++hCkrFZOY1Z2bntnjyIBrH3ISVj2nLy0I1PwcqPQKgE425KO3/+gEB3iv1g5ReKwcwqzK7Lt7Hk1v93WKGhHBr88ubaRz63aSxHYWRVkmmmAMlllVVpV/nemDYkNUmij9YhmLFX2uiSRNq4C2YZYU0S6WAM0kLnCgucckGLvTj3HxNyh3QwI/2lw2zirMWMp8A+u6NGokvasq8eGydx7sipaCdYI8FD1gga2aPdlND1ElIKkWKUEOskqTQsBJ+FGdYYYREdvWCz0ShRRnlC3LDty3fdRSXBDP1EDRN9H7VU05ljS1dnnf6NeDQH1ZRzMif3ohRzhPnU2Bd14uFVLFl6/Aoh0LiJLnqBtHPRVZRsw7UTNyFhhGvbesBxZUfWfRtILMLK9wlGTZR2fv2AQHeK/UCg27EalkQ6syZbvttjzfweSaSN5VKYtJ/avULI5JiRVRhdh0SmGYJTEmmHXxLrbj+u6Sj0BWAyw0yixALHfNBVUvz2blSTu6T9GVaBfrNs7bYWiUiDQnBL1RgzFn5NwBGLiRNmM44C3mESnUfEGTZmlyYUenDQg2OLy1Gu0I3d0THCvEr1jmLSwqRYIUEvaxzBRwueZxJjzJiNrhuzUZ8xG82nGpXmC1MssQiACzettOHMUVUrECwzzjLjQP4t3QjTrPEAkeeqi9wX/dCIRjPj5jw2mrM/jhhEE58inYuqDHFRlnPvRB/EP5HvW5rB9WZ2ay6pdYLhBKX+mgMC3Sn2I4FmQtfluszUCkyvyp3UTFjMUtnbVA5NFeDZR6HbupCmDY9X5J5pOo5NUaRRQ1e53DPd7XzQYAr6A3IdJi06spnkTunRUvAUmdzjutwlfRTb3CUtM4i0rUhEuqTr3FQ15oxfgs0wYzia5w7pdgTQuEuUBaNadGPmFE4aCqTe1RAvVeVmmuLfZIkkOifwv9DtSCCYYY0nLBkZniY6qKYuz73RNdYYZwwNFRMmGmnOS2AUYY1Z+o2WrplaunPOGc10L1Iw4+cErhwsBXWSho9uegXnMA6OZV3F62LGcC5KolCCRfkCipJl+zs1DrF3jDWXKmPNZecvigcq3Cyx3wl0O+JJSaTpW3SbIKnMLYm0qVyuzhRZP7JjJFQpPBpehYXw5uetZjkr7SqXVoK7iZQubQMfBSBgKGcVZF7psVKoLnLLOalDb0zukqaJ1GcQaXuRiHRK07mlaazqmzukZwuw+rIdkyS5T5So0datxcppXAVr674IaQKdJEwva5ylYsdRajGSDGQodcvx0E1tXi5GSZKMMUrIUNX68NNCS87pLttbulKl25lTi3j7qouXdko5lLWJg0AQ4wFxBgGwUY+biyhZzsOFWEcVP0AQAuxYlLcxKVnOfNV5GdAtEmDygftLct1lBzgg0CzxWSPQ7VgNw+QyTK7ItZnM36DNIlu9zRWSVO37IIUFIJiQVenjVfl+GqUO6C6HTj+4d0Hkk4YQMBWF3nWYyRB0VdrhuE8SajGDv5O6rEYf7hKRCsOM4U7GDmmlSeG8xUxtAa+4VAR9xBgksUFqhwvY1n0W0veTROfHzNKKh84Mi0CZp5kiaBgg1D/DQUhGsa0yypIhBjLTTU1enrrPEhi10Y6Xnb2wP3083WjpSgGOAy/1HMnJeEEgCDBIiBEA7FRQzmnMOVg2Jhgnyg3D/cmHm1dzMF2IofJDhFgCTFiUq9krdLU1Y80lDCaXsSta/tJvOyDQLJE+YQ8GArQ1leD5jNjwPQuJlKxKJ5dlyzdz/1RR5N5pc4W8leyTn3MuBEMrsjpV08HWijRr6K6QqzG7OeNdTciK9HFo03/XY8xJDxV5Tpom0t4oJDKI9GyRZqSqEDzSdB5oGinj/prNJs6bzZQW8IohaLR154vU1n0WrrGIiuAM5bgyKr0HrLLC5lWbGYXzVOB8RjUYJs5AhotRNaV0UY0ljyo6QoRRnhjpLgp11FFLXc6ipTDLzDKwYbxQSw9eKnI6VpQ5VrmPQMWMkwrO5RTUrbJMmI82TBc8XMXCy8krE0KoaLyXodA9i1k5md0D0SOSRLVVUKzGruiLW9QHBJolNtJY/iiA01WCww5VZVCZvvnAuY/mijuFEHJeOrEsb6vhrf/v90BLJbRUQMU+KLxTmiTRoRUpQkrDYZHt3Z4KWaHuFuKanJM+Wt8M/raZpDHDUV9xvXeTuoxSe5hBpOUWSaTNRVBgx4TgrqHYFUihUY/ZxClLYYVGUyS5t62texYX7gK0deNoKIAdM+OE6WOd81RQaRgw6AiGjVzSE/jxY8eKiZssUYrtuSpdHWHUeSuAwI6VHmrzyhvV0JhgglWWAfBSQitt2HK8oEgRZ4a+DeOFcpqopDUnL90UYZa5iUoEBRNlnMSdw1xUI2KIiwIomHBxHjstWR3jaYVuN2auoGSjrhVJ6VqkziJ3RT8HtudXswcEmiXSJ+wP/zJAQit5plm8xwXVfnmr8kOFb//MFneKUEwS6fiS3D/N/E17HNBaCW3V0tBhr3dOA3FJpI9Xt67E1Hklkbb4dq8qVXVpZP9wHdaNit6EVO0e94GviK3mZ81Iqyxw3lMcQ4Z1XXBD1Zgy/gjsCpwxzOoL4bELT7d1zSgcw0lXHiYMAsEQQSYIcwQfvazRhpcOvJgNEpknxh2WOYGfhgzye0yQWaK8QtXG1z4LAaIMMEfMSHhpxE8blTmvpgCssMwEE+hoWLDSSlvOgd0CnUWesMo0AC581HMkp3g0nRQr3NsI6vbSYcxFs/v9CFJEuEbSCBB3chQnR1/yXU9DZoteAwSK0oCFN7Pz0BWa9M9NjcqPHZfB/uzHcUCgWSLzhLlcJawGYXEVlgPy7Xro6e8xmyWJVvtltVpTAa7PUJWaSMk277jR6s2MbXPZJZm2VsmW716SqS5gKgADyzIcPP3s3IuqVAiYiMCD9a3euy1uOFVW3MDvuC6r0UfRzfWXOiuc80B1EebaM7rODXVTaFRmUrhoMVNfwKvGIBo3ibBs/EQ+LJzDRXkeYp1HrDFOGBMKV6ii1CAPgeBDFvBi5fS2VuJ9VoiicZmX51Sq6DxhgVlDvOPBwWHqcOcR7xYnzhNGiBmipRpqqaM+Z2IOssAcg+joWLFTz1GcOSWxbJ2Lyn3RU1lHo0lx0X3ihpWgjWbcnM/auUgXE6jiPUBFUSqMNZcs5lBCQPwaJGVYOPbT4Dj71JcdEGiWeNkJS6ZgeR0WVmFhBRbXIJ54+jglHqgp37z5ctMG7DpUTc5NRxclqSYz9jYdVkmkbVVyVWYvyTSchKFlGFx5uio9UimN7Ysp9MnEfExWpOORzc81OOFkWXF9d6Ma3Df2SNONkhYbXPaC5zmvR0LAkgoLKXCYZHbpTpA2q7+jbnrsNptNXLCYKSnQE0EgGCPJfWIkjZ+oEwfHcGDLmUCS3GYFgNOU48PGNBEescZVanBh2RAarZOklzVqcdKOd8cV1jIhBpkjhYYJhQ6qqc8xwxNk9NoUkxs7ox68tNKGPUdiThBhml6SxFBQqKELH1n6xBqIMMMa9419UQ8VnMO6Ayeopx/TCFHuGOKicjxcxZTlz6eLJVTxQyCGgtdYc/Fl+UAyItFshw3Dhc3f+wGBZolcTlggvEmmC6uwuk39CuCwb5JpbQWU76OVkudB06WBw9iSbPUmMkRITpusTNurpXnDXpHp86pSlxUOV8Khit0zaVhLwoM1KThK//qr7HDKD01FdDgKa3AnAsNxsCjw18slOT4LgzG4F5EEG9elE9JrXqjdYWcvYcxH+zPmo8csZk6aTVgL9AMm0LlHjHFD3OPAxGlcNOUhMhokgB0TrXgZYJ0gKc5SsbFDqiHoY40IKt2U4s/yxTyByiCzrBpWgOV4OEQttjwq6FVWmWB8QxDUShu+HPdQNVTmGCBkzFl91BnuRdm/CCVZZ5lbRjSalQrO4MhhlzXFPGE+QZDCjAcPVzFnWR0LEUQV30cQBBxYlM9jUl7ePdiCZD/EPpbvW9vB+fqGa9EBgWaJQqyxJFOSUOeN2+IaaNvcg2xWSaZ1lfJWXrr3s8YXQdelT+/ooiTUTDJ122VV2l4tje/3CuEkDCzJqjSWnk8qckZ6uFJWp7uBUEpWpIPBTeWu3yYr0rYirsCsq7CmQutzqkpNwPfWNwVIJkWS6VwK3iwBVxZdtDVdcF1VmTHaui4FzlrMdJpMBdsfXSDFbaKEDMPDfEVG6UrzCSGmifBahofsY4IsEKMGJx05rqcIBNOs8YRFBAIbFnqoxZ9DhZZGnDijPCFqEHM+LV2BYIUJlgzbPicl1HMEaw7JNhpxlrlNkjVAwccRvLTmcJwgYT5AI4IJG25ewbqD9nkmtq65WLAor2NSsnQ/So1C9F1AB0uDVOgqlgMCzRbF2APVNNn2nV+BuWX5Nrkt0sxmlZVpbQXUV4J/HxOqrktbwScLsjLNbPN6ndBeBZ01ULbLRghpaLrMLe1f2qrgLXPK9m6nf3ei1mKq3CXtC7CxFlJikRVpp3f3WsyZ+DQETpM0sAdYSsF31uGLpVCTQ4E3ocn5aNB4qagyKVy2mKkoUHtFQ9BPnH7iRrtP4ThOOrHnvOqxQpx7rNKCh1JsLBNnkgjNeOjAiwXTFveiNDI/pyNIoj/TQjBMnH5miRgVdAN+2vMQGMmW7hRLhojHg5c22nNW6YZZZZY+o7K1Us8R3Dm0nAUaq/QSZQpI54sey1rtqxMnzEeorKCg4OJCDgpdFY130MUUoGBWLmNWerI6Buq0kSuqbrgWBcPJAwLNBrthpCAErARgdknenkWoDrsk0voqaKhi3+6jajpMr8CTRUmmmQKkci90VMube49EVasx6FuUCt70XqnNLA0ajlRByS4Y8Sc1SaK965srMF6LrEi7S3aHSCMaDMZhLinj1S544LATboRhPgVn3NCY47nQjP3Re0biiwIcMozq7QW6CgwYIqMVQ2RUjoXzuCnNsRpdIk4va5hR0BB04KUGF7bnkGcaGoJ5YiwSY5E41Tg4gf+pr9fQecIiM0ZWphs7R6jPS2C0ygrjjG+odNtoyznZJUmMGR4RJ4wCVNJOOU05HSvEE9bpB8BOOeWcxZwluQtUItwgaZBxLgpdIXQ0PkYXwwCYlVOYlTNZHQN1Ue6KigSYywiqr1BaVndAoDvFXjgR6bok1Lllg1SXQVW3fk2pZ5NM6yplxbrfoGpSePR4HqZW2bICVFcGXbVy19S2B2HhSQ2GV6BvaWvUWosPjlVB7S60d1Vd7pI+WIeYcaHhSROpt3irOELAf1+RCt06m2zl3o6A2yRnpW12OOTMn8gjQq69jBpG9Q4FzlnMdBWorSsQjJDgATFUg+SO4OAwjpxXXmKoWDFtOBM9Cyl0Iqg8IUQSDRcWTCgESOLEzDHKnptLukyYQWYNgZGJLqqpzcNPd7tKt54GaqjNqRrX0ZhnmADzAJRSTQ2HcqqUYyywwl0EKhbcVHA+a3HRdvs/Oy24OJ91RauJO2jiHpDjrqi2BtHvgB4lGDZT2vArBwS6U+wHKz9dlyszM0swvSjfz/xNmExyftpYLW/+PZw7Pg/xpJyVPp6H+fXNz1vMUnzUVbs3Sl4hYDoIvYvybRrlLkmk7WXF3ylVdTkfvb8mlbQAbrNs7RaDSKcTcC0MX/bJ+wGYT8InYdm6de+giEvq0jhiJ5jVda6pGmv6pi3glQK2daPo3CbKrLGDWYqZ87jzWnlJY4wQLiwbXrnrJLnPqkGRCq14cWPhPqsk0WjFu2HO8DwkUBlgdiNrtJoSuqh9IWm/CBoaU0yybPjVluKjjXbMO6jG0zFtmVhlmkUeI5AWgA0cw5pDpZwixBI3DTP63MVFCUaIcBsAKzW4uZx1vJomBow0F4FJacbMG9lFoukhiHyHYNhCad3/74BAd4r9QKDbkUzJynR6Ud6C21yE3M5NMq2v2n/VaSgGIwuSTNczVj08DkmkXTV7YyW4HodHi7IyTbd3nVY4Wgk9lXK/tJjQMog0klGRni6TxgyFau0Ox+BWBH7KBz7jZwqo8MMAXPJAw7bXSiHkhY0m5LrLw6hsPXvMcMUrZ6gvgy4EfZrOXcMWUAEOm02csZixFeiqaZIkd4hu5I524+AYzpx9dSOovM88rXjopgQzJuaJcpsV2vHSY1SOo4SYIkITblp36F0rEEywwhjLgMCJjSPU481BwJPGMktMMIEPH+283Bs2RowA63jw4tlWHUZYY4ZHaKhYsNHAsZz2RTWSLHNzQ1xUxjE8OcSZJZklwicINMyU4uU1TM/wKX4RdDFu7IpqKEqNsSuaBRHrMYKhGKW+8gMC3Sn2I4FuRzAMk/MwtSDbvZkK33R12lwLzTVyH3U/YTEAw3OSUDPFRzU+6K6Val7rLrd4E6pcg+lb2twptZjknPRYdfHnpGkivZdRkZZY4LQfOgogNgqo8G4Qmuxw0gWqgMmknH++XgL1z3lNuROBkTiUmKHDAaNxiOjw+dLn75lux/a2rkuBixYLbQUqsxPo3CXGhCHY8WLmAm4qcqxG10hgRqEkw3RhiTiPWMeMQicljBPGi5UOvM/0zH0R1onSzywJw7y+gyoasgydzkSUKDZsO0pyCRBghRXChKiiihpqt/x/khjT9JIgYuyLduPb9jU7gRQXPSDKDJBOdOnJus2sskqYD9GJY8KFl9cwZznz1cUsqvgRkDIMF76AouzcYP9AhZslPgsEmglVlbPTqQV5C2yrTn3eTTKt8u+f3VNVk1aCQ7PSuCENi1muwxyqk/FruwldSP/dBwuwkhGy3eqD49VQXeSLEVWHAaMiTc9IfVZJpO2e/Nrdc0n4MCQrQbsCYR0abXDVeIqnq8702/EE3ApDi106HAEkdPjTVWkd2JFl4TSj63ya0ggYLykNJoXLVkvBTBhmSXKTKPECVaPboaLTxzpTRHBj4QzlGySbLVJoDDDLCvKPtQIvh6jFugvRbjo6A/SjodFFN45tFbCOymzGvqifBqpoz8lHN8AwQcNxyEkNfk5jyvJn1Agbay4hFKx4uIo1y7awLpZRxfeBOAqlWJQvoigHcWZFwWeNQLcjGIaJeZiYk+reTCGP3QZNNZJMG2t2v9J7HiJx2d4dmoNABnGVuSWRdtaAYxfjzABmgvBwQZozpFHtgRPVMhGmmLNbVZeq3Qdrm6rdMhuc9csotXzwJC6PWWGVyS52k2zVmrf9PH++KivPs25ZDYPMS/2LNTjigp4cslE1IXig6dxXNXTkfZ4ymzlWoBDv7QYMXsycx0VllnZzz8MoIcYJE0PFh50rVL1QsfsyTLPKiLEzasfKUeopySF+bCdIP8555phlxvDZ9T1TMCQQRjTaOABuyqjnCOYczmOUGSPRRcdKKZWcx5xl21onQZgPjTUXE24uYaMxq2MIEUAV30MQBlxYlS+hKC9f3Tkg0CyRPmGDQwFaW0qw7fILdyGRTMmqdGJOvk1kWN6ZzVLR21IrK1THLqxz7ATz6zA4Kw0b0isxJpMUHh2q233h0VpMEunjVVmhAvgckkg7y4ucC6rJKLWH61LEAzKT9Fw5NBRoZpzUpR1glRWabfLcDhvJL2fcW00ZVlLwThCOOuFwHvcf0AWfqCqzxgn1KQpXrIXLHp0lyS2ixIxqtAsHx/OsRldJ0MsalThowIUO+AoQvxYiTh8zxEiioNBOFY15tHSfhU2rwnVGeEwDjVRSuUV0pKE9JUIKssQc/ejo2HDSwDHsOaTOJFhlmVvoJI1YtPPYsnUcQjWM6GVb2MVpHHRldwwRQeX7CLGGDOf+PCal+oXfc0CgWWIjzux3AjidJfhKoaoSKsvl2/J91AbNBroubQYn5mB8bqsQSVGkgUNLLbTU7Y+d06QKI/OSTJczDPy9Tuipk2S6m1VpNCUFR/1LkthAhnwfq5Im9sU0ZkhqkkR71zcNGeqccL4cqvLcr51PwntB2ZI955Et3BthWNWkxV9aoSsEPIhKsv16xdMVay4Y0XRuqCox42fqNJu4WKDd0aRRjY4Z1agHMxfyqEZXiDNKmDY8lOch/HkWVDQGmWfJiB+rwEsPtXnljKaRJs84cfp5RDkV1NOwZWY6xyxRYpTjx7fNUCFOmGkekiKBGQt1HMGTA8GrRFjiJiphFCxUcDZrha5AEOUOiQ1D+x5cnMjuGCJhkGjatehNTMrzq9kDAs0S6RP2//5BAF1/+oSZzVBRDtWVUFMt3zqL03UpKlYDkkjHZmFlfev/VfmhvQFa9wmZLgclkWYKj0wmKTg60rC7s9KkJu0CexclqQLYLVK5e6SquMrdmCqFRv2BTfP4VresSPOJUVtXwWXaXFN5NyBzR7/k2/yalRR8d11Wnmfcm7PSfJEQgtuqxoAhMnIqcKmAIqNZUtwislGNHjJmo+YcqlHNiFsrFjJbuvmqdDXD/tCMGYGgl4c4cNBMy1Om9PPMo6GyxBIVVNCwrT2qkmSGR0QJoABVdOKnIevHpJNkmdskWEEqdI/jycG8IUYfMWQmaC67okKoqPwYIaYBExblc5iUtmd+7QGBZonME2a1lrC0AotLsLQMi8uQeFbyiheqq6CmSr4t8+1fG75nIRQxyHRGzk0zUV0O7fXQWi/XZfYSqibtA/tnZDh4Gn6PJNKO6t2b62q6bOven4eg8ZywmKTn7olquQ5TLIRTcGcVhg3TegXpaHTGD+4C/Pz3ItKd6AulskW9psLdiExx+Vr5y78/FyzoOh9n7I42mUxcsZpxF6EaLcXMRdyUFWBvNBOjhGjA9VxThZ0gSIxHzJAghQmFTmqoy9J4QSCYZ44gQTro5AkjJEnSRjuuF6yCRIgwxijVVFO5zY9WR2eeoQ3ThTLqqaYja3GRVOg+JGrklJbQRSndWR0DIMEYEW4AYKUWD1dQsvh9Stei99HFKNL67xXMytOP44BAs8TLTlggKAl1fhEWFmF17elj2GySTGtroLZaVqyflbZvNC6J9Mn002RaUw5t9dDWsPd5p0tBSaQj85LMQLZRu2olmfqyH9XkBGEod+9nKHfNJtnWPVEt27zFwloSbq1sxqiZFThWKp2NbHl0/1ZS8P2ANJ2vssJADLxmOOeWqS2Fqj63QxeC+xkiI6sC5y1mDhXIyWjamI0m0FFQOIaDHhw5i4AyMUmYB6zhwMwp/FTk0ebdrtKtoZQual4Y8L0dGhoD9JMgjoKJTjrx7mDuOMoTzJhpfo4X7QqTLPIEADd+Q1yU/YVIgCGCSMs9Fw34OZE1GWfuiuYSiSaEMKz/pFLYrFzCrBzZ8jUHBJolsj1hyWQGoS5JUt1uw2exyHZvrXGrrJCt4P2OSEyS6ei2ylRRpGFDR4Ocme6lcUMiJfdK+2e2Kngb/HC0ERrLd68bMBmAu3OwaBCaSZFxaidrwFNEIl2IwY0VmDcsCh0mWY32lOYucorrcD0shVNes7T68z7nObuQKmyQ95ou+EhVWTSq0WqTwqsWC74CKLbi6NwiyozhYlSOhYu48eY5bwyS5A4rhA2v3i5K6KQkZ4tBgWCSVUZZAgRu7ByjAWeWwqVJJllkgVZaKadiYyaqGxcR2y8eRhhBQ6WbQ889ZoglZg1xkR03DRzDloN6OMwkazwEBHYqqOBs1gHdKsuE+MCIRCvFy+cwZflYVHEdXchgbbNyDrOyOVc9INAske8ai67DyirMLcD8gny7ve1rNktCra+FuprPRoUaiUkifTItrQXTMJvlakxHg3y7VxcGQsjs0r5puV+aRolTEmlX7e558E4HJZGmk2BMCnSVSyItpinDeBhursC6MZsttcKFcmjJY/VFFy8m4ZmkTHNpsMFlz6bTUb4QQtCv6dwyDOpNwGmLmeNmE6YCXBGNkeAOUVRjrnkKFx15GL2DNJDvNXZFAcqxcwp/1mYLmVgjQj+zJFGxYKaHWip26H6UxiwzLLJIG+2UGFWoMBJrdXSSJIkRJUiQZZbpoosSSl+4ohMnxDS9hrjISgNHceXg8RtniWVuIdCwUkIlF7Jec1FZJ8z76MQx48bD5zBneY4y/XPNygnMyjnggECzRqH3QIWAtXWYm5dkOrcAsdjWr7HZZGVaXwt1teDPPcx+VxAMw8g0jEzBeoZC1maVwqPOJqnq3as5cDAqK9LB2U3RkcUsnY52s707G5JEOmucI0WBLj+crgVvkYhUF9LV6PbK5g5pjQMuVuSv2H0W+qPSZ1dDktwJl4xKsxTodx8Wgo9TKtNGNVphUrhqseAvQDUaQeMGURaRVxy1WLmAG0eOHrVpzBDhIWuoCKyYOImfmjz2OxOkeMQMQeQLRzMVtFKRVes5ShQzZlZYJkIEDY0UqY1KFMCJi0oqdxzarZJgiodGoouJWg5RyovXQp6FJAGWuIFOAjNOKrmYtRG9NFx4H40wJux4eA1LlmphTfSiCTlXNSlHMHORUCh0QKDZYDeMFNbWYXYOZuZgdl62gTPhckkybaiTt/2s8l1Zl2T6ZBrCGS1Ujws6GyWZ+nYpyHo7Uqo0aOibhrUMD96mCjjeJHdKdwMLYUmkaVOGdEV6qqZ4RJrUZOpL77q07gMZ5n2hHLwFbrkHVfg0LO0BAbwm6ZnbVMCf7bGmcV3VSBjV6EmLmZMFqEYFgiEj4UUaGpi4gIu6PHc8I6jcYZmAQc5teOjBl3NLV0fwhEWmke0fP256qMOWRXWrodHHI5IkaTHUuGYsG6rcnZjRP/24NGbp33AuqqSViiyzPAFUoixxHZUIJqxUcB57lgQoc0U/QGXNcC16NetwbmlC/wkAJqWLSPAEPp/vgEB3it12IhICllckkc7Mybbv9hlquV8SaWO9VPnux/mpEHJO+nhStnoz800ry6CrSa7G7JVhw8wqPJra2t71eySRdlTvTgt9MQK3ZzdTYEyK9Ns9VVu8GWlEldXokFEFm4BjPjiVp9DoWRhPwCch6ZcL0GKDy96d++a+DFEh+ETVmDBUY36TwtUCpbyso3KNCAFj/aMDOydx5WW+oCEYZJ1RQwzkw8YZynHl0dJdIMAg88b80coxGrJadREIHjNMihRttOM0KuNsHJW2J7oIBIs8YdXI8iylhlq6s48hyzCiVzDh5zSuLL14dZJE+IgUS4Zr0WVsWa7caOIxmvgQEIRDVZT7vnJAoDvFXlv5aZoUIk3PytvyNiWsxSKr06YGeXPvUjsyG2iaNGwYnpQOSOlnkckkXY+6m2VyzF60eANRSaRDc5tORy47HG2Annqw74IgaiEsiXQmTWqKVO2eqgVXke5/NQHXlmHGGB84zXDOL9dfCvl7SOnShL43JldsrIq0AzxSgLzRNJ5oOtdUlbiR8nLSYuZUAapRDcFDYgwh1VhezFzCjT/PdZd5YtxnlRQ6VhROUp5XSzdMnEeGe5EJhS5qss4YHWeMNdZooZUynt+KSRNrggTrrBEkhAkT5ZQ/1epdY4YFhhGACx8NHMtaoaujscId4iwAGGkuLVkdQ6AR4dMN1yI3F7DTmt3jEGOo4j2CQY3Ksr9zQKA7xV4T6HbEYrIynZ6FqZmn56fl/k0yrarcf/unsbhs7w5PwvL65ufdTkmk3c3g3YOLgEQKBmbg0TRE03ucZulwdLxJRq0VG/MGkaZnpGaTNGQ4WSPNGYqByYgk0oDRIaiww+UKqCnwmGBVhY+CsGB0Uyos8KoXKgt0gRA3qtGxjGr0NYuZ8gJUo/OkuE6EuDEfPIGTbux5rbvEULnDCmuG+rcdL4cozbmlq6LRn7HqUk8ZHVRndbx55phnnh4ObzFX2F6NJkgwxKCR+mLFiYMF5mmk6al90TArzNCHjoYdN40cx5qlKEigs8YjIkwAUM1VbC9JYdFIoKBs5IYKdKLcJGH4+bo4g4POrB6HLqYJBVV8vtYDAt0p9huBZkIIqfCdmoGJKbk+kwmHQ7Z5mxvlW+s+ywVdDcDQhCTTTF/euko41CIFSLvdntZ16XD0cBJW06pZk2zrnmiCsl2Ig5sNwa1ZWZmCbK2eqIajVcWxCNQFPFqHu2ubHrttHrhYDp4CPmeEgKG4XIdJGtXiUae0DCyUyGhU0/nUqEYLqdRNoHMzY92lxhAYOfMQGOkI+llnzCC9cuycphxHjis0AmFYvsu5hA8XR6jPai6aJInZ+KcZ/2zYNkg0RYpBBrBho5GmDSOGWWaIEKWDjqcuLOKEmOIhKkks2GjkOI4sVbEgd0UVzJS8JOtUI0GYcZIEKKV7g2wFghj3iG/sm57AQU9Wj+FAhZsl9jOBbkc8vkmm07NbxUgmk5ybNjfKm2sfWPKlkW7xDk7A9MLm5+02KTw61AL+XY4yA5hegfsTMJthjtFUASebZV5psTEZgJszsJpus1ql0KinQlanhUZck0YMg0HZbjUrcjZ63CddlQqFmA7XQjBiVPoek6xGGws0D48JwccZs9FKk8LnLBZKC9AzHiHBPaJohsDoIm5q80x3mSXKA1ZRjWOepjwv44UlQgwwi5bjXDSNAfqxYaOJZqzGz/iYYZIk6aBzS5W6xBIzTHOUY8/MIk0RZ4qHJIhgwkQ9R/FQJBsrIMYicRYJM041r26pWGM8JEY/AE6O4OTYjo97QKBZ4rNEoJnQdSlAmpyG8UkIhrb+f1UltDTJm28PyOl5CEdlVTo0sVXFW1MOPa3S+Wi3q9KloCTSscWMx+OTFWlTkddzhIAna7K1m7YI9NrhTC10+otz3ysJ+HQJ5gwjBq8FLlXktz/6LEwl4KOQzCIF6LBLkZGjQGT9WNO4pmokhaxwz5nNHDbn72IUROMTwhsCo0NGukuu7VeAMCnusEKQlLRipJQOvDm3iSMk6GXamIuaOEQt1VkmnkwxiQ0bVVSjoLDEEtNM0UHHFhcjHZ0nPMGEQvsLqkMNlRkeEWENBajhUE4B3S9CZrt5jV7CTFLOKZzUbjmXMfqJ8RAAB124OL2j4x8QaJb4rBLodqytSyJ9VqvXVwptLdDaLGeo+wFCwMwiDIzD+Oym8MhukwrentbdX4cJROHBBAzPb+aq+j1wqkUa2ReTSHUBg8ty/SVtWl/mhAv10FSkC6AnIbi+DBFDXNXghMuV+RnVb0dKh9sReGSIjBwKXPRAV4FmsGEh+CilMmPsjdaaFF6zWvAUQGB0nxiPDYGRHwuX8nQw2m68UI2Dk5Rjy7FNnEKjnxlWjeM1UU4blVmRclphq6MzxRQaGs00b1lxWWaZJRbxU071S/Y+BTpzDBIwREG5rrk8+9ib5LnOAGFGKeMETmqfGdwdZ5godwGw04aLcy89NwcEmiXSJ+xP/mSNlpYSqqoUqqrA6dxn6pwsEI1ukunM3NaQ7RKvJNLWZlml7gdE4zA4Lm+ZVWldpSTS1rrddW6KJqB3CvqnIWWQS6lLtnY7a4r7WFRdxqg9WICEIcip9cCFBqgqgvhK1WXiy4M1mfiSXns57QdrAX/OpRR8EJSxaSCdjF7xbIZ354sBTeOGKl2MbApctJjpKkArY5okN4mSRMeCwllctOTpYDRJmF7W0RG4sXCGckpz3EMVCEZZYhIp3/fj4Qh1OUWjPWYYMxba2EwqCbDOPPOYMNFG+453Rxd5wgqTAPiopYaurNdcMpFJntJXd4QyjuKmASXjMemkttgDZprQ22jCzcUXPo4DAs0SG3mgv72G07l5wrxeqKqC6mpJqBUVYCpmmnKRkEzKNu/ouJyfatrm/3k80NoE7a37g0yFkGswA2MwOb9Zlbockkh7WnfX1D6RkqYMvVPyfZBq3RPNUr1bjDllGklNJr/0Lm6a57eVwfn64tgDBlOyrTtpXMC4zNLNqKOAXQBdyODuOxHpZGQBznvkykshqvuALvggw1O32WziVYsZR54Hj6BxjQjLhu9tK3bO5LkzGiDJbZaJomFC4ThlNOYQXp2G3BedQzei0Y7RgDtLop9hmjBh2unAgoU11lhiER2dZlpw4sxqfzRzzcVDOfUceWal+CIII5YuTXpBRggwiI8juGnacrwkAcJM4KAKFzUZn58kwjUEAht1uLmyhXQzcUCgWSJ9wm7eXCca9bK4CGvPSFyxWCSh1tRATY1CdTVYrZ8tQk2lJImOTcjqNNPAweuF9hboaNsf1oLh6GZVGjVmdSaTnJEeaZOxa7uFlAoDs1K5m16BcdnljLSnXq7DFAvhpJyPDhv7wSZFRqidri1OFulkRBJp0Hhu1DnhlQK3dQMqfBiCOeOipMYqw7xLC/DzCCF4qOncMRJeXAq8ZrVQn2fbQEfQR5w+w17Pi5lX8FCaR0tXxq6tsGi0iVvwcCQP96IQcXqZJkEKC2aOUIc/C4s8DY0hBtHQMGMmRhQ/5VRShQdPVuS5+ZiWmKEfgY6TEho4hiWLajvMBCmClHGMEGOs04ePHtw0Y9omZkoRJs4iQR5TQhfejF1QmeTyMQIdKzV4eOWZcWgHBJolnnXCkknB0hIsLsLCgmDhGQbxigLl5ZJQa2sVamvB4fjsEKqqSiXv6Lhs92aSaZlPVqUdrbDXY2FdlyHgfU+2JsRU+OBou3Q72i3RkaZLv90HExA2SN1pk63dYhPpagxuTG/aA9rMkkSPVBa+EtZ0eLgu1160tJVemVTsFuq+hJCxaTcikBKyGj3nkWsvhahGl3Wd91Ma68ZL2jGLibNmM+Y8D75AimvGzqgZhXN5tnQFgmGCDCN/sWXYOEtFzqsuSVQeMUOAKKDQRTX1LzBOeBaWWQIUbNhw487J8i8TUQJM8xANFRsumjix413RFGHmeR8rJaQIUUo3HlqeIs9MxFlihbv4OIqb+oxjLRDmQ8PIvgoPV58i0Z84Av3Wt77Ft771LcbHxwE4cuQI//yf/3O+9KUv7ej7d3LChBCsr8P8PMzPC+bnIRR6+uv8fqirg7o6Sah2+2eDUFVVVqRPxmS7N3NmWlkBnW2SUPfao3d5XRLpyPRmK9rpgMOt8ubcpfaurstItXsTEDJWUFz2TSItZmt3OgjXpzdXX0rscLUZ6nbQao2r2VWtwRR8sgRTRlu3xAKvVEFDAVekQhp8GIQZoxqttsBrJfmnvGi64NrYCtfWYwRcVpqay6i0mPicxUJZnqOYODrXiLCw4Xtr53SeLd0FYtxjhRQCB2bOUI4/R2LW0RlinnkCQG6mC4VGgghTPCBFwtgVPYFjh9WxSowlowVbxWUsz3B1ShHeYki/Rh+gU7ZthSXFEmE+QKBipdIg0c2Z6U8cgf7VX/0VZrOZzs5OhBD8wR/8Ab/1W7/FvXv3OHLkyEu/P9cTFolIIp2bE8zNPbvtW14OtbVQXy8J1Wbb/4SaTMoW75NxmMlQxyqKNGvobJd7ppZdigp7FuIJ2drtH9sUHZlMcqf0WMfu7ZSmifTu+GZF6rbDyZbizkiFkC3dW7NSsfvVQ88XGCVUGFqBsXW592lS4FKDVPjuFKNh2daNGhct7R659uIq4HNgICYNGFICzMhq9FgO1WgoFOJH/fP81ruTzAXiG58vKbHzuVfrOdRTxetlPnrybFsIo6X7yGjp+rBwJU+VbpgUt1khRAoTcJQymrNMKcnEBCuMInez/Lg5Qn1O4qLtSLd4s0WKBFM8IEEEMxYaOI7rJY5Dm/eZYJ73sVOBn5NbZp8CnXUGSBGgissArHAflRBVXHlKNJSZKWqhwgjmlm3lnzgCfRb8fj+/9Vu/xS//8i+/9GsLtcYSi0kinZ19NqEqipyh1tdDQ4MUJu13UVIsJqvSx6OwlGHKbrXKtZjONqit2Ts7wXR7t3dka2ZpfZUk0t3y39WN1u69cYgYrX63Xa6/HCqigjilyZZu23M6dELA90ak0rarHNxWSaQjq3I95kgWwRVJDe6swqOAXEexmaS37uHSwp3jsCZno9OGQUi1BV4vyU6p+2c3HvONP7yGYnNism+WyiIRRUvG+JmvneP4uXYaTSZes+YvMJozbAAThkr3Am4a80h2UdF5wCqzG1Fmbo5SlnP1uESIfmbR0XFh4ziNWYd0ZyJFigH6qaCCuoz26E6hkWKKh8QIZm24oJNknQFK6UYnhRnXBpFqxFnkGgoKZhzEWcbPCdw0PvNYKiuEeN8g0XI8vIYJ2082gWqaxre//W1+8Rd/kXv37nH48OGXfk+x9kAzCXV6GoLBrf9vtcrqtKFBoaEBfL79TabrAXj8RJJpOLz5ebcbutqhqwNK93BeurAiiXQso2r2eeWctKtpdypmTYehWVmRpsVGXiecaZXrL7t9ofFgHh4uSmejhhKoMYqZJ6uSVDv82VfJy3H4eAkWjZ+vyg5Xq8BfQGXwYExmjqZnoxc8cHgH1aimC1755rvMLK6iJ2OYDBLVE1H0ZAyzzUlllZ+/92uvIBSlYAKjKDqfEt5Q6Xbh4GSexgsjBBkwWrB+7JzJwwJwu7joGA34yK0Pv8QiE4bfbAWVNNG8JallJ9DRmOERYVZRUKilJ6tcUY04K9yjlG7s+DfETRGmiDCNlzYULNjxv1D0pLJqBHMnseDHw2uEg4mfPALt7e3l0qVLxONxPB4Pf/zHf8yXv/zlZ35tIpEgkaEICgaDNDY2Ft1IIRQSzMzAzIx8G49v/X+PBxobobFRob6+MApfXRdoWmHVwkLI9JjhJ1KAlGknWFMN3R2yOt0rX95wFB49kS3edMSawy6J9HDr7sSrabo0rr8/sUmkPjeca4PW7OIKc0YsBf+lV+5yNpXCWly2cN9slVFqCS13Fa8QMBCEmyvSW9eEtAM87S+cJWBYg/eDMGv8Duutcjb6oqi0a09W+Ov/73WADdJEUUCIDTIF+N1fuUCoqZQ1Y93lhMXMmTz9dHUEvcQYMBS15Vi4bEhwcsX2ueg5KvDlWD0mUOllihBxFBS6c0h0SUOS6AQgKKGUdjqybukKdGYZJGgYLlTTiX+HMWQCwTK3ACjnzEYVGmGaAAPU8PoLRUaZUFkzSDSBhTL04GnKSqt+sgg0mUwyOTlJIBDgf/yP/8F/+k//iQ8++OCZFei/+Bf/gl//9V9/6vO76UQkhGBlBWZmYHpaVqqZwh2TSap7m5oUGhuhrCy3P+yFBcGdOwJVhdZWhe7uws5hNU2Kj4YeS0Vv+tljsUgSPdQpSXUvkFJhaBwejmzOSS0WmQZzvGN3EmFUTe6R3p/Y3COt8ML5dmgo8hrOu2NSaHSlEWq9UkD00QRUuGSMWiEQUeVsdMwILy+xyGq0rkAiIyGgPwY3wqAiDRIuv8DF6C/uz/CN/35/42M1vCoPoihYPJsWXP/P107yUyfquK5qDGb46b5hteDNs00wS5LrhvGCHROXcFOTh5dumBS3WCaMigmFE5TRkOO+qIbOIHMsGorfXJyL0lhnjVGeGK1hN510bfjp7hQyV3SEVaYBqKCZygwTh5dhgY9RMOOhGYEgxAg2yvBzPKvHobJOmPfQSRAJWmkq/dmfLALdjrfeeov29nb+43/8j0/9315VoC+CqgpmZ2FqSjA19XS71+uF5mZobpZipJ3OTldWBNEorK7C2JgglYLXXlOoqip8LzESke3doccQyHj8pSVwqEtWpo5dNEFIQ9dl4PfDx5vxaooi119Odu2O4Cipyh3S3slNZ6NaH5zvgOoi3H9Sg+88huZSueqSxvvjcq/0y52Fy+oEGA9LtW7aErDbK00Y7AVa6wmoshpNR6W12ODVEnBuq3Z3WoH+t799kUvt8gpmTNP5SFVJGg5Gr1ostOap/gqj8QkR1oyW7nGcHM4jDzRl7IsuGNVtO156KM2J+ASCMZaZMBJdqiihh9qs27AAYcKM8BiVFHYcdNKFIwdT+2XGWWIMgDLqqaZzxz/bKg9QiZJgFRe1uGnEwdOuMC/bYdUIEuJdwkGNxp90An3jjTdoamri93//91/6tfvRCzcQkEQ6NSWJNdM9yGaTrd7mZlmd7nRVRtMEH34oiMfhi19U8jbcfhHmF2BoRAqQ0vulJpO0D+zpgrrC+kvvGDOLcH9Yvk2jqQZOde+OMUM8KYVG/TOb7kItlbIi9RWwIo6r8N8ewSuN0Gn8XElt07D+7bbCK4STmmzp9hsXT06zNGBoLZBBvRDwICp9dXWkp+5rJdCc0ZJPz0BnF1fRnjMDravy8/E/eQNzxhVEWAjeTW06GPWYTVy05LczqiK4Q5Qx5MV6AzbO48rZ93b7vmgVDk5TjjXH482xzhDzCASluDiaZSxaGnHiDDNEkgQWrHTShTuHCnmNGeaNGLISqqnj0I6t/3RShqjIsfE9mYSZfl8jQYJV4ixixoEJ2xazBY0QoWD8J6uF+0//6T/lS1/6Ek1NTYRCIf74j/+Yb37zm/zgBz/g7bfffun370cCzYSqShHSxIRgcnJrwHZNDfz0Tz//SSaE2EKUd+4IxsYEX/mK8tRcVAhJroX0AE6lJIkODG9V8ZaWQE+3FB/tRVW6vA73h7YKjuoqJZHW78KMMhyHu2MwNLfRYaSnDk63yn3SfKEL+GBczjgvNcr7GF2Tay9HKuFY9eb9ZiLzc0LAchTmwnA8izb8fAw+XIR1o2Xd5oErFeAskIhrJQXvZXjq9jjgknczbzStwjXZnCjbVLh6Msb/8wuX+JkLTwct60JwR9N5oMoD+00KbxYgIm2EBHeIIhB4MfMqHkrymIvOEuU+q2gIvFg5RwXuHIgPYJUIfcygouHExokcFbpJkjxmmBhRTJhpp4PSHa6nZCLIArMMGOeqgjqO5FQZq8Q2dkU3yTPOMrdRMGPCihknCVYwYaWKS5uP4SdNhfvLv/zLvPPOO8zNzVFaWsrx48f5J//kn+yIPGH/E2gmhJAOSePjgokJ6OxUOHny+X/gKyuCH/5QbMxRr18XHD8Op04pWLYlHM/MCAYGBIGAbBn39Cg0NhaOTJdXJJGOjEpiBcOarwUOd+/NrDQQhgfDMvA7PYOu8ksibd6FKnktDDefwIRxcWExS3vA401gzZNwZoLw7jiU2qV4KJiEGje8blxwP4tAQbaYHy1K8lyJQSgJhyvgStPO71vTpYvR/TUjgcUkU14K5aurCbgVhofGxaTPDG+UQIX1+XugtaUO/vEbTbx9uAav9/kPZFrX+SClEhNgVeAVi4X2PMv1FVQ+JkysQKsuAZLcZJk4GjZMnKWc8hzzRSMkeMAUCVJYDYVuaQ4KXRWVJ4wQIoiCQgttlOeQBxpmmWn6EOi4KaOBY1n55wYYJM4y5ZzZIFGNJIt8gkoEDy04qcVBOToqS1zHjJ0KzgE/gQSaLz5LBLod2yvM7dA0wegoDAzI6vLKFYW6Op75PXNz0m3J4ZDio/FxOHZM4ciRwrZ6n1eVlvslkXa07b6CNxyFB4+lcjfdLveXSiJtqy/++sncGlwfkbmkIO0Bz7Tmv0MqhCRDkwJ+J5S7pAXgs7AWg74lCMSlkra7ApYiMBGAV5ug2vN80n0eluPwwSKsGErtZhe8WlU4A4aZpKxGo4YS+KwbTrjkY9R0wc2xVRZDcaq8Ds63+re0bV+EqNHSnS9gSzeOzidEWDLci3qMjNFc80DjaNximXWSMkGHMppyNF1IoPKQKcLEMaFwmDoqs8wWBemANMYYa0YyTBPNVGWxnpJGhDWmeYhu+Oc2cgLzDqvsAINoJCihCwtOdDSWuUGCNez4seIhzjJOqvFxmCQBEqzipgET1gMCzRafZQLdKWIxwXe/K+joUDhxYmd/sP39gtu3BV/9qkJJSXEYZHkF+odkVZqeldpssrV7+NDuB4HH4lK12z8qVbwgd0lPH5Kio2IT6eiCrEiD6crKDRfaobmASTmaLtu5tV65zgIwsAR35mR7t7sCXFZYCMONGah0yTZwrtCFrETvrsrZpc0Elyugq0B/anFdhnaPGbrAWqs0X3jRustOoAvBXU3nvtHSrTBUuiV5rro8IMaQIQaqwcol3NhznGNq6NzPMF3IR1ykodPHDCuEjWNV0ZRDBSkQTDLJkrGeUksd9TtcT8lEjCBTPEBDxYGHRk7s2IReJ7nhLBRlhgDDeGjCSzsAcVZYpw8/J7HiRZA6cCLKFekT9u1vT9Ha6qe62kJVlQWbbRcDKAsIIWT12Nq69Y/o/fd1FAVeeUXBbH76DyxdzabfhkKCb39bzkvLy4vLHImE3CvtG4BghsdwfR0cOSStA3fThCCRlLukj57I92H3iFTXpcjo7hjEjVZ3XRlc6oTyArRAJ9bho8nNRBeQ4qJ789Jw4ZUmSXofT8o27k91gt2SffW5HasJWY0uGUTX6IJXK8FToG7DcAw+McwXbAq84oWOAszXp3Wd91Mq8QKqdCdIcpMIGgKPMRfNNdVlu7ioGgenchQXCQSPWWAGaaPWgJ8OqnIi5FlmmGUGgEqqaKI56+PECTHJAzRS2HDRzEksWfoDr9FLglVqeG3jcypRFvgIP6dwslX0cECgWWIzD3QMp3PzFcrvN1NdbaW62kJNjYWSkl2K/MgTwaDgvfcEXV0Kra0yqWR+Hm7fFpSXw9WrhkptW/s382NdF3z8sWB5Gd58U6G0dHfYSwjpv9s3KE3t089GrxeO9shVGFsBY7VehmQK+kblCkyaSEs9cKan+ESaVOH+uMwiTSt2u+ukGUO+QqOBJWmu4DbOpRAwG4LrMxBJQosPFiOSYNvK8ifPNHQhU17urMo5ZqGr0aAK7wZh0egedDngiif/YPCI0dJdMFq6R8wmLljMeRkvrKHyEWGiRqrLJdw05DEXnTHERTqCEqycpwJnjuKiSVZ4YnjoVlLC4RzXXBZZZNJwLSrDTyttWR8n04TehpNGTmDLYiUowBAaMfyc3PhcjHkCDFHGMez4t3z9AYFmifQJu359gUjExeKiSjCoPfV1LpeJ2loLtbVWamstlJXtoZv6C6BpgqEhePRIEItJQ/twWDodXb368nbswoLg1i0pJnrtNYWGhr2xGAyF5Jx0YHgzSs5qleYMRw7tbsxaSpUpMA9HpJE9yIr0TE/xZ6ShmGzrPpEdMaxm6bF7rKk4ZvXXp6F/Sf5MP9sD3iI4N60n4f2FTTvAFresRguh1NUF3IvC3YgUMJWa4U1DYJTfcbeqdKtMCm9aLbjz+OUnjLnoojEXPYqTIzhynouukeA2K8TR8nYuWiDIALMIBD5cHKMhJyP6VVYZ4wkiD9eiJDGmeECSGFbsNHES2w6FTlFmWKWXCs5gwWOEbsud00rOPxWsfUCgWeJZJywW01lYUJmfT7GwoLK0pG5xCgJwOEzU1FiorbVQV2fF7zcXdbcyF4RCco/U75eEY7cr3Lihk0rB6dMKLtfm402rcJeW5HrMhQtb/3+voKrSg7e3X/rxgnxxb26EY4elmf1uIaXKtm5mReovhbM90FJX3PteCMC1YVg0hEZep5yPthVQvazp0LsIg8tyDup3StN5axGaL+lq9PaKsddpki5GLQXaG51PwjtBiBgCo4seOFoAh6QJTecDw3jBqcAbVgu1eSi9dAT3iTFszEUbsHEBN9YcSTSGyk2WCZLCjMJpyqnJ0cRhjQi9TKOh48bOCRqx5+CqFCDAEx6jo+PGQyddWLKsjlMkmOQ+SaJYsNHESew73DcNMkKEKQQ6CibMOCjnNGbsTxksHBBoltjJCVNVweKiytxcirk5lcVFFVXdeqocDhN1dZJM6+utlJbuv5avrsv2rqbJ1qzZrDAzIxgclMTZ1CRXYyor9544n4XpGXjYL9+mUVEOJ45Kk4ZipZ5sRzK1SaRpv90qP5w7XNw9UiFgZF5WpOnUl7oyuNwF/gIQz2QArk1DSylcaJDmCM9T7RYKKwl4bwFWjQuSLq9s6xbifuO6zBodT6uAbdJ8wZHn8yQoBO+kVFZ0gQKct5g5lmea+igJbhn7oqXGXNST41w0hc4dVlgijgIcwUcruQ3QQ8R5yBRJVOxYOUEj7hxySsOEecwwGipOXHTRnbX1n0qSSe6TIJI1iaYIoZHEhBULLkxYnulOdECgWSKXE6brgqUllbm5TVLdTqhut4n6eusGobrd+0eUlEgI7HYpFPr//j+BELJd29W1P4lzO9bW4dEADI9srp14vXCsB7o7d28NJpGU6y+PnmyqiOsqJZEW09lI1aS/7oMJWTUqChxpkKsv9jx+9lhKqnHP1kkThmfNPiNJGYd2rLpwdoCaLueiD9Zl29VjgdeqoL5AnrqPojJrVAfcJtnSrclzlq4KwceqxogxoG41m7hqMWPNowu1bMxFE4aP7iu4qczRR1ea268xiTQqbsPDYXw5tYdjJHnAFDGSWDBzPMdd0ShRhhnasP7roht7lmSskmSKB8QJY8ZKEyd3HMy9ExwQaJYoxBpLmlBnZlLMzsrW7/aWb1mZmcZGKw0NVmpqrE8ZGew20qKhpSXB0JBgeFgST1MTHD784ipU1wWPH0Nr696GhMfjUnDUN7iZbmO3y33SI4fAVaAX4JchGpfORv1jm4YMTTWSSMt9xbvfUEzuj44Z1oQOK5xrl/uj+U4Tnicc+uETGF+X7d3XmqGygBaE8zE5Gw0aFyNHS+F8eWESXpZTsqUb0EABzrjhlCv/89SvalxXNXTApyi8ZbXgy+PKIoLGx4aProLCOVy05VDxpZEZi1aDk9P4MecgCEqi0ss0QWKYMHGEeipyIK5M6z8bdrrozto/VyPFJPcNErUYJFoYl44DAs0SxdgDVVXBwkKaUFMspmWBBsxmhdpaCw0NklD9/v0hSJqeFjx8KNNbLl9WqKh49gvB5KTg+98XmM2SRA8dkib3ezUDVlVZjT7s21yDMZlkRunxI7u3TxqOwt1BGJrYVBB3NMoZaUnhLpKfwswqfDoMa0YqSoUXrnQXx6j+8Yps88ZVST7HqmTFWqgYs5QON5Y3PXXLbPBGNZQXQMyU0uWqy7BxsVVvhTdKnzalzxYLus47KZWo4V70msVCSx4KLxXBDSJMIXvP3Ua+aK7iokyFrg8b56nAnkN7eOuuqMKhHCPREiQYZogEcazY6KQLV5YVrYbKFA+IEcSMhUZO4MzB/CENgQaYCAVDBwSaDXbDSCGR0JmZSTE9nWJqKkUksrU8dbtNNDXZaGqS7d69rk7hxS5HExOCmzcFa2ubnyspge5uGZm2V+IjIWB8Eh48gsWlzc+3NsOp43JeuhsIhOF2PzyRKU2YTDKL9FQ3OIvk/ZveH709KldgQIZ4X+gojL9uJuIqfDolW7kAJXZZjdYWyKoPYDIi90ZjmhQBnSuXmaOFuEZ7HIePgjIizWWSNoB1ebZ0t7sXnbCYOWs25XxRKRA8Ik6fYZJQa5gu5GpGv0qCWyyTRMeJmYtU4smhPawjGGKOeaOqzdVwIUWKYYaIEcWMhS66szahLxSJSreim1jxYA42HxBoNtgLJ6K1NZXpaUmo2+enZrNCfb2VpiZ58+RrqVJEpNu/IyObwdqKIqPXDh2SaTF7VZXOL0ginZja/FxDPZw+vnu+u8vrcLMPptMrKBY40SXzSC1FajrEknDrCQzOyo9tFtnWPVyEdZvJgDRliBi/+yOVcL6Aqt24Jo3px43Kus4Jn6sqjPnCugo/DkhT+kK1dHUhuKVp9KryArnecC+y53HQCZLcIIJuiIuu4sk5pDtMipssE0HFisJ5KvHn0B4WCJ6wyBTyCqqJctrJXj2novKYYSKEMWGmky68WbZidVSmeEiUACbMNHICV5ZG9jEWWeaGPF6wgpbSywcEulPstZWfqgrm5lJMTqaYnEwSCm2tTv1+My0tNlpabFRU7I9W73aoqvTcHRwUzM9vft7jkVVpT8/eVaWra5JIR0Y326o11ZJIG+p35zHMLMKNR5t5pC4HnD0sw72LdX2xFISPhzb9dStL4JVu+baQSGpyd3TQ8DX2GtVoXQGr0cEAfLoMqgC7SfrpthWgJa4K+CQEQ0ZLt9EmbQDzVek+MTJGVQGlisLbec5FVwxxUdwQF13FQ3mOJglJNG6yzBpJTCicwU9NDoIg2Gq4UIePLmqybjNraIzwmBBBTJjooIuSLKtIHc0g0fWcSTTCFKvcJxyMc7j0/z4g0J1irwl0O9bWVCYnU0xMJFlYUMn8jXg8pg0yramx7DhYezextibXYoYzDBAURc5KDx9WqKvbm8ccDMKDPhn4nRb6VJTD6RPQkkXSSK4QQgZ73+yDkFFRlZXAxaPQWKRdViFgYEauvaTbukcaZEVqK/C12EwQPpiQgd0grQILuUMaSMK7C5tWgN1emfCSr8sQwFAMPg6BBnhM8FYpVOVZ5S7rOj9OaYSFwKrA6xYLTXnMRSNofESEdVRMKFzETVOOJgmaseayYKy5HKWMlhyVrLNGrigIw7WoDlMOJPqEEYIEUDDRTge+LGer20m0iZNZt3PDTBINqlSXth8Q6E6x3wg0E/G4ztRUivHxJFNTqS2tXrtdoblZkmlDw/6Ym2ZC0wRjY9KUPrMq9fkkkXZ17Y2CNxKRYqOB4c3Vk3I/nDm5O567mibVuncHN80YGqolkfqLJHaKJqRad8T4PTht0lu3o8DEndKkAX2/MX/22uFzBZyN6kKuu9wzZu+lVnizGioKMFdeVeFHAanSLZTxQkwIfpxhAXjOYuZEHvuiKQSfEmbOcC46jpPDOZokbF9z6aKE7hwyPAEWCdJvuBb5cXOUhqyVvjo6T3hCgDUUFNrooIyyLI+xSaK5zkQPVLhZYj8TaCZUVTAzI8l0YiJFPL7Z6rVYFJqarLS12Whqsu07Ml1dFfT3y9WXdBaoxQIdHXDkSPHN6p+FeFy6Gz0a2HxM/jJJpC1NxSfSRBLuDckdUt3Y5exphTOHiic0mlmVbd1AVH5c74dXu6GkwOs+syF4f3yzGj1WBefqC6fUnYvBu/MQSZNdBRz15X/cpA4fhmDUqHLb7XDVm1+VqwvBNVVjwNgXbTP2RS15iIvuZTgXtWHnLK6sq740hghsGNE34uY4ZTkda5Uwvcygo1OCkxM0Zm39J+PQRllj1SDRdsq2edW+/Bj5kegBgWaJzwqBZkLX5ZrM+HiSsbEk4fBng0yTSSk46uvbquCtrYWjRxWam9n1tvTziPT0CaneLTaRBsNwow/GDHclq0WqdY91yCCAQkPTpQHDvXH5vtkEp1tlkHchnZxSmlx3Sc9GfQ74XAtUFWhvdLvAqMkFn6sGRwHOWabxQpkZ3i4FX54t7wFN41pK7oseMZu49ILEdE0IVOT9O5/zBBwmzl3klVANVi7nodCdIEwvawhkmssZynPaFQ0Q5SHTqGh4cHCCRmxZzmp1dMYZY5UVQKGNNvxZqnwzhUXZ7okeEGiW+CwS6HYsLamMjiYZHU1sESFZLAqNjVba2/cfmc7NCfr6ZJs3/azzeGR7t6dHtqh3E4nEJpGmFcW7SaRzy3Dt4abQyOOCC0dl6ksxEIzCR0OyKgVpBfjqocLvjk4G4MMJiKbkOTxZDWfqCudi1LcO11dkuovLLHdG6wpQUc8n4cdGWLdVkeKiljzXgeZ1nVuqxudfoMyNCsEnqsa6LrApMsy76zlXUjMk+dSIRctXoTtPlDvGrmiZsStqy+FYYeI8MKz/XNg4SVPW/rkCwRhjrLIMKLTSRvkukegBgWaJ/x0INBPLy2kyTW5JlbFa5cy0o0POTPeLACkSke3dgYFNN6F0e/foUQW/f++JtNwPZ0/JGWkxIQSMTEmhUcQI1a4ph8snoMJXnPt8PAfXHm9mjx6uh/MdhRUZJVT4JGNv1O+E11ugvECt45UEvDMP68bPcLoMTvvzJ+moJt2L5ozjnnDBOXfhyH87hEGeEQFNZgULcE3VOGk2c/w5s9NVVD40FLoOQ6Hrz1Ghu0qCmyyTQseDhYtU5hSJFiXJfSZJkMKBlZM04cxS8CQQjDPOCnKg3kIbFVRkdYyte6I7s/07INAs8b8bgWYiTaZPnmytTB0OE21tkkyrqy37IkVG0wRPnkBvr2BlZfPz9fVw4sTux6olk1Js1Nu/2dqtqoTzp6Gutrj3rarSY/fBY/n+kXa4cqJ49xdPSpHR8Jz82GWXKy8tlYW9n9E1GdQdVyUJnauD49WFqe5VHT5ZgiHDiarGAW/WgDvPCwFdwM0wPDQuaOqt8GZp/qsuz8O1lIpVUThrEOaEpnNf03jdaqHkOScqgsaHhAmgYUbhCm7qclTohklxjaWNSLRLORouxElxn0liJLFh4SRNWZvQCwQTTLBsrMq00EoF2T0pt5NoM6deaEB/QKBZ4n9nAs3EwkKKJ0+SPHmSJBbbJFOPx0Rnp52uLvu+SZCZnxc8erS1vev3w/HjCh0duzsnjcflHmnf4KZqt74Ozp2ShFpMpK0BLxwF+w5eDzUNVoNyXcblkHPUbDC7Bh8OQNAgi7YqaQnoLGCIeSwlzRfG1+XHdV45G/UU6D5GQvDRIqSEJLk3aqChAJXuaBzeN9yLPCb4fGn+GaNpCCNrdEnXSQFW4Es2efBxY6f0p61WSl/wvE+i8ykR5g2F7jnctOfooRtD5TpLhFGxYeIilZTmQMgJVB4wSYQENiycoBFPlr63ABNMsIR0I2mmlcocSHSSe8QJY8FGM6eemyd6QKBZ4ieFQNPQdcHsbIqRESlASqU2f+XV1RY6O+20t9uw2/c+PSYUkkQ6MLBJXm63bO329OzuGkw0CvceyvWX9B5pSxOcOw1lvl17GM9EOAqLq7JidTnkPNVshrfOQ212XS9UDe6OwYNJefHisMq4tEKvvAwty7auqsvosqvN0Jbd1sJzEUzBj+ZgxWjBn/HLtm6+le6qCj8MQFADMzIaraMAiumPUioBIWg2mfAoCnc1DQXoMpkY1XWswJUXVKBp6AhuEmUcKSM+ipOjOa65JNG4zhIBUlhQOE8F5TmQXxKVB0wRJo4FMydpwpvDcSaZYDEvEk0xwT0SRIxQ7lPYnnFuDgg0S/ykEWgmVFUwMZFkeDjB9HRqo9ozmxWam610ddn3xbw0kZAk+uiRIGqsYFit0NMjydTj2b3HFwrBnQcy5DudWNLVAWdOSBHUbiIQln67I1NQWQZt9WAxy6q1phxOH8pdybschA8GYcVoiTZXSJFRIX11A3F4bxwWDSVtVzlcbixMFqiqw7VlGDCcmOqdUmDkzLOlm9Th3SBMGuR83AkXPLmTc1AIvpNUuWAx02YYLUSE4AcplQaTggWFapNCfRYS6V5iGx66+ay5pNC5zTLLJAzXotzCuVNoPGSKILG84tAySTSXdq5KkgnukSSKDSdNnMK6rUo/INAs8ZNMoJmIRnVGRhIMDydYXd0UHzmdJrq67HR32/H59rbFq2lyDebhw801GJMJOjvlnNTn2z0iXQ/ArbswNiE/NpvhaA+cPCYj1YqNQBi+/yl43XD1lFTtCgHv3JRV5JkeSar5QNdl7ujdcfm+zSINGLrrCvIjyPsQcHcO7s3Lx++1wxstUF2gi5HHQfhoSdr2uczwVg3U5FaUbUAIuB2Be8bFXINNZozm0rSJCMH3kirnLGaaDQJVheDPkyrHLc9X4L4Mj4lzx1hzqcPGZdxYciBRDcEdljdci07ipyFL03cAFZ1eplgnigkTJ2jElzeJZi8sUkkYJBrDhotmTmHJaE8fEGiWOCDQp7G8rDI8nGBkJLnFsKGmxkJ3t532dvuersQIIZiakkQ6O7v5+dZWOHnyxVmmhcbiEty4A3OGy4/NJpNfjhwqnmE8QDIl1bqD43C8E84fgaFx6BuFQy1wuK1w97Uahg8GNn11G/xwtQc8BTR8mA/Du2PSfEFR4HQNnKotjOJ1LSlbuuspaRx/vkDJLplz0RIzfKEUyrL8nWtC8L6qkRSCN6wWLMCqEHyqai9cYdkJpo01Fx1BORau4sGew36njuAhq0wZhHwUH6055G9q6PQyzRoRTJg4RgP+HMh4gnGWDGGRXHHJjkSTxJjkHikSOPDQxEnMhlDqgECzxAGBPh+6LpicTDE4mGBqKrnR4rVaFdrbbRw65KCqam8N7hcXBffvC8bHNz9XXw+nTu2u7+7kNNy8I83rQc5qz56U7d1iiZyFgMl5aVSfTEmhUWO1DPJ+3utuSoX1kPz6+iwCNHQdeqdkXJqmS4/bi53QU0BD/qQGn0zCY2PdpdoDb7YWRmCU0qW4aCQsP25ywevVYM+zqbKSgh8EIGzsi75RAs05dCC+m0yRACxACkn0P2W1YHvOk0cXgjCwpkuv3brntHiXSPEREZLoeDHzGh48Oex3CgR9rDOGPIHdlNCVg/Wfjk4vM6wSxoRikGh27Yat6tzc9kSTRJngHipJnJTQxAlMWA4INFscEOjOEI3qDA8nGBxMbNkvLSsz09PjoKvLhs22d8KjtTVJpCMjm8rdqipZkba07A6RCiFno7fvQ9h4ofaXwcWzxU1+0TT4iw9gPQwlbvjZNzfns5lYWZdCo6kF2fJVVbh6Ojuh0XpEVqMLMg6SBj+81gPuAlajI6ty3SWpFV5gNGAku2gCvBZ4uyZ/L924Ln100/uiZ91wOgfHpUFNIybABrSYTbgV5bm5vKOazl1V21DtlpkU3nyOu1EQjfcJETV2RV/DQ1mOu6LDBBgyrP/a8XI4h0BtHZ1HRjC3gsJRGqjIiUTHWWYJDNs/f5a2f3HCTHIPDRUXPho5TjgYOSDQbHBAoNljbk5WpWNjyQ2De4tFoaPDxuHDjj2NXQuFBA8fCgYHJbEAlJfDmTO7R6SaJtde7j3cTKRpaoCL58BXQKefNEmOzUhirCyDtSC8dlrORrdjcFx67545JKvPvlEYmoBXTkhD+2zu99GUTHnRjNnolS7oLOB+bCgB74xtCox6KuBSY2H8dJfj8KN5CKlgVuDVSujK809fF3AtDH3pFSA7fK4E8pl0pITAapDndiL9UUolIQSft1qIAx+kVFyKwhsW8zMJN4rOB4QIoGFF4VU8VOWw3wkwSog+1gFowcNRfFnHmOkI+phhmVBeJLpptqDQnoMBfYwgk9xHR8ODn5JgM77SsgMC3SkOCDR3JJM6jx8nGRiIbxEeVVVZOHzYQVvb3tkHxmKC3l5BX9+mEUJ5OZw+rdDSsjtB34kE3H0gyTRtGH+4WxrWOwpUsakqfOcTsFvh8nEoecFr0MScVOj+n6/Lj4WAv/wQGqqk6ChbrEfg/X5YNGajrVXSnN5RoJ1OXcCdWSkwAulg9Fab9NXNFwkN3luASUMIdKQULlXkP3MdNKLRdKDSIueirhzaxKoQ3FI13IrCMbNp4/mqCUFIwKiu41Sgx+jVz+g6N1SNt60WvM95bifR+YgIS6QwGYYL9TkaLkwS5gFyXtGIixP4cyLRfmZZIpgXiaZt/xQU2unMOgotyjpTPEBHRwm66Cm9eECgO8UBgRYG8/Mp+vsTjI4mNvYk7XaF7m47hw87KCnZGwVvPC6J9NGjDLN4f7oi3R0iDQTh+i2YmJIf22ySRI8cKoyB+8o6ROLQVCOJemwWWmo356CRGCyswMySJNETXdJkYTUghUguB7x6KrdZra7LndE7Y/J9pw2uHoLmAppMTAflukssJSvQK43QneV+67MgBNxdkxFpIN2L3qoBV54NlPmk3BeNC3CbJIlma7oQFYJ3Uio+ReFVozU7qGlMaIIloeNVFAJC8BXDYGFUkwT6ptVM1QueVBqCT4gwi9zDuYiblhwNF2aIcI9VBFCHi1P4s16XeZpE66nIUqC0NcXFRCedlGQ5nw2zwjS9RIIxzpZ++YBAd4oDAi0sYjGdoaEE/f3xLSkxzc02jh51UF9fIPuWLLEfiHR2Dj69uSk0Ki2BS+dle7dQ6B2BB8PSvajTCAr/zseSLHxe8Lrg4YisWK0WeTvZJVu4z5qb7hTLQXivH9aMlmt3nVx5KZSnbjQF743BjLGX2uGHV5sKE9g9EZHVaFIv3KpLUIXvB2Bdk8KgN0qzN6NPCIEVMCkKISH4OKXiURQOm004FIWHRkxavclEGIET+LLt5X9f2w0XTuOiKwdzA4A5otxlBR2owclpyjHvEYmO8oR11jBhopNuvFkeI8QS0WCCmtLGAwLdKT6rBCqEIB7XiMU0SkutmPNIuy8G5KpJir6+OFNTqY3P+3xmjh510Nlpx2rd/fZuIrFJpGmz+IoKOHtWoamp+I9HCBh6DLfuQcyYlzXWSyItxHw0FofhSUmeLgfMLMKnD+FsD7QaQqZAGH50Ay4cgepy2MFr7o6g6XDrCTyclB97nfD6YajxFeb4QsD9ebg9J98vdcDbbbK1my8CSfjhvFx5MSHbuUd8+R0zqctEl2njeXbeDSdzjHML6IL/mUrxisVCh/G3nhKC76ZUuswmyhSFckXB+gLhUSa254rm41q0QIzbrKAjqMLB2Rzi0ArRztXRGeExQSOBpYtu3FmuyexLFW4qlaK3txeLxcKxY8ee+8t9+PAh9+/f5xd+4ReK8TCeif1MoEIIZmZirK8niUQ0IhGVcFglEpG3dKvUZjPR2uqmvd1DXZ1zz52DtmN9XaOvL87wcGLDOtBmk+3dI0f2pr2bJtLe3gyz+CpJpLthXJ9KSZHRwz7Z+jSZpBHD6ROyxVsoLK/D//oI3jgnW7wgV1m+94mMSjvSXrj7SmNuDd4fgFDMiDBrhjOthcsbnQ9LgVEkWdiWbkqHDxZh1FBQd3vhlUqZmZortouLuhwypDvbP9GwELybUjlq3nQsAvhOMkWrycRhw3x+J+SZiUfEeGS4FnXh4BTOrGeZAEvEucUyGoIK7JyjAksOJDrALIt5kKiGxgiPCRHEjIVuDuHKwrBh3xHot7/9bf7e3/t7rBnWMXV1dXzzm9/kb/yNv/HU1/76r/86v/Ebv4GmaU/9X7GwXwk0GlV5//1FpqdjL/w6q1XZ4mcrk1Y8tLd7qKlx7IuklTSSSZ3h4SSPHsW3rMK0tNg4ftxBTc3ut3fjcanaffRo02+3tlYSaW3t7sxHr92Ue6QATqdMfCnU/mg8Ae/ehjKvNFswmSAUgf/1MRxuhZPd+d/Hs5BU4dPhzYSXyhJZjfoKFaitypbulCFg6iqHV5oKo9J9sAY3V0AAVXb4fG3+c9G+KHwalsestcqQ7mwTXe6qGv2axmsWC04FAgLeT6m8bbXQlAfLZ4ZzN2PnQo7Wf6skuMESKgI/di7kSKLpSjTXPVENjWGGiBDGgpVD9ODYYYt6XxHozZs3uXz5Mmazmddffx2r1cqPf/xjkskkf+fv/B2+9a1vbfn6AwKVmJ2N8aMfzZNI6JjNCk1NLtxuCx6PBbfbgtttxuOx4HJZUBSYn4/z5EmY0dHwFucgl8tMZ6eX48dLceZrAlpApNu7jx7FmZ7ebO9WVVk4dsxBa6tt16voaFTukQ4MbK6/NDTAuXO742w0NS3nowGDECor4MqFwiS+LK/Du7ckeZZ55ceKAv/32/kf+2UYXZDB3YmU9Om9VEDzBSHgwQLcmpXvlzllS7cQKt3pKPx4fnMu+tMNUPKS67v03urzMJWQ+aJJIZ2LvlgKviz/LB8Zc0+ACIJjZjNnnpMVmg3GSXCDKAJBLVau4MnJ+m/NINGUEcx9gUqsOZBoesUlVxJVURlikBhRrNg4RA/2HYil9hWB/uzP/ix/+Zd/yXvvvceVK1cAmJyc5Otf/zoff/wxX//61/m93/u9jSrpgEAlvve9OaamothsJr761Xp8vp319GTSSownT8KMjUVIJuUfmqJAfb2Tjg4vLS0ubIVw6y4Q1tZUenvjPH6cRNPk08/rNXHsmJPu7t2fk0Yignv35B5pukXe3i6JtKSkuI9F12WQ9537m23lQ12yIi3E2kvfEzn/LPXIPVCfNz/h0E4RicuW7oyhdm2plErdQq27zIVkSzdqqHRfa4b27Hbqn4lgCn4wBx4LfLH2xecpmIJvT8Lna6DxBVX2mgrfX4eQDnZFKnRrsjwP67pARaDDCxW32WKWJB8b1n9VWHkVD9YcSDRAkmsskULPi0QfMc0K4Zy9c1OkGGKQODHsOOjmELaXrO3sKwKtra3l1Vdf5U/+5E+2fF5VVX7xF3+R//bf/ht/82/+Tf7wD/8QRVEOCNTAzEyU7353DiHgwgU/J05kb8Oi64Lx8QgPHqyztJTY+LzZrNDY6KKjw0NTkwtLIXpeBUAsptPfH6evL7FRRdtsCkeOODh61IHTubuPMxQS3L4tePxYfmwyyfSX06cVnM7iMk40CjfvwvCI/NjhgAtnCtPWfRlhTs5LFe9rp1+8U5rL/fYa5gu6LlNd3jgCdQVyGIqlJInOGirdI5XSeCHfRkZKl499J9ecfevS5ejySwRIMR1+sA6LqoxFe70E2gro5JQPFknxIWFUBBWGf64tB//cwpDopmORCRMnacw6xSVJkkEGSJLAiYtuDmF5gQvTviJQu93OP/pH/4h/9a/+1VP/J4Tgb/2tv8Uf/uEf8tf/+l/nj/7oj/iX//JfHhCogb6+AJ98sgzAF79YQ1NT7sOjQCDJkydhRkbCrK9vtkytVoWWFjddXV7q6pz7Yl6qqoLHjxM8fBgnEJDPA7NZCo5OnHDg9e5u9byyIrh5U5rXg4xRO3FC4dgxil4dzy/Ax9c3115qquGVi9IesBgQAv7kR7JKtVjg/GEpMirk02I5CO/2SxMGgNOtcLqlMAIjIeB2hvFCtQfeagV3AUVZL8NkRLZ+e0rg0gva76qAdwMwbih0L3rgeAGCv0Guv4SEoCLHk7qCyvuESCHwY+G1HE3oC0WiDw0DegtmTtBISZZq4ThxhhgkRRI3HrroxvwcP+B9RaDNzc184Qtf4Hd/93ef+f9CCH7pl36JP/qjP+Lnfu7n6Ojo4F//6399QKAGPvpoiYGBIBaLwk//dD0VFfnnZK2sJDbINBxWNz7v8Vjo7vbS3e3F49mbXc1MCCGYmEhx/36MxUX5OBUF2tvtnDzpwO/f3Znu7Kzg+nXBsrymwemUO6SHDlHUeW26rXv7nhQ5KQocOyyNGKxF+DUFw/DhPZhdkh/XVcLnzkjv3EJB1aTAaNBI0qnxyWq0UOkukwGZ7JLUwGmVJFqbfXhIVsis7Bfj8BfTcLT0xSQqDIXuI0MneMQJl/PIFgXpYPS9lMqqELxttTzXZP5lWEXlA8Ik0CnFzOt4ceRIotdZIomODxsXcyBRDZ2HRhRarqHcUaIMMYiGipcSOunC9IzHsa8I9Atf+AJjY2MMDw8/92sySdTr9RIOhw8I1ICuC77//Tmmp2O4XGa+8pV6vN7CvWouLMR5/DjEyEh4Y14Kcl7a3e2lpcW9L1q8s7OSSDMFR01NNk6dclBdvXtkL4RgbAxu3hQEDbGPzwcXLig0Nxe3Gg2H4dqtzfxRtxsunYO2lsLflxAwMAbXDWWy1QKXjsuYtELiyQJ8OAApTRo7vNYj56OFQDABP3wCq8YqzcV6OJaF3+9Osb0l/mAN5mIwH4cOD5z2v1zB2xuVRArQbIM3S3P30E0JwQ9TKnO6wKzA6xYLLTkqdANovEeIuJHk8jpeXHtEoio6D5gkSAwrZk7RjDtLB6UwYYYZREenDD9ttD+1srOvCPS3f/u3+Yf/8B/ywQcf8Oqrrz736zLbuYqiHBBoBpJJjb/8y1lWV5P4fFa+8pV67PlmMG2DquqMj0cYGgoxM7O5NmOzmejo8NDTU0J5+S6kRL8Ey8sq9+/HGB1NbnyuttbK6dPOXXU40nWp1r1zRxCXe+jU1cHFiwoVFcUl0qlp+OQGBI1ZX1ODbOt6CjivTCMYhvfuSBtAkHukV09Lg4aC3UcU3unbzBo90iBj0grhC6Lq8OGETHcB6V50tbkwqy6ZEELujvYFIJCCWiec9cMOtX+AzBZ9Lwga0kP3iz7IdeyvCcG7qsaEpqMAr1rNOeeKhgwSjaLjwczreHDnEIcWNNq5+ZGoxj0mCRPHhoXTNOPM0ss3SIDHDCMQVFBJC61b/38/Eejs7Cz/7t/9Oy5cuMBXv/rVF36tEIJf//VfZ2Jigt/7vd8r5MN4IfY7gQJEIip//ufTRCIatbUOvvzl2qI5D4VCKYaHQwwNhba0eKuq7PT0lNDe7tnzqjQQ0HjwIMbw8KbvblWVhdOnnTQ17d7AK5mUqy+9vZurL11dUrHrdhePSFUV7vfKm67LeeXZU9KIoYCiTECSw8PHcKtf3pfdBq+clCYMhYKuw61ReGBU134PvHW0cDujjxbh+rQ0NfA74fPtUFKA68GULp2Lri9DXJOV5pVKqdq1mrJXOC+kpLgoLsBrgi/7oDTHSYUuBB+rGsPGyssFi5ljOa67hNF4jzARNFyYeB0v3jxJNNeZaBKV+0wSIYEDK6dpxp5lqswaqzzhCSCooZYGGjcf414Q6G/8xm/wd//u36WqKouE3n2CzwKBAqyuJviLv5ghlRK0t3t4442qoop+0i5Ig4NBxscjW1yPOjo8HD5cgt+/81ehREJjcTGB32/D7S7M/DIS0XnwIMbAQGJjBaaiwsKZM06am3ePSMNhKTQaMVSzFgscPy7FRsUUGq0H4MNPpdgIoKIcrl6WbwuN1QC8f0fukAK01UsDensBT/PUCrzXB3FjZ/SVbugqUETafBh+PCpXXWxmeKMVmnK0ThRCzjnvrUkCdVngjB+qHZI4X4SU/uKvCajwvQAENXAoshKtyqO5ckNV6VXlH+9Ji5mzOZJoFJ33CBFCw4GJN/BSkiOJfpohLLpIZdZmCwlU7jFBjCRObJymGVuW+aZLLDHBGAANNFGDtOnaEwI1mUzY7Xa+9rWv8au/+qucOnUql8PsCT4rBAowPR3l+9+fQ9fh8OESXnmlgJEXL0AspjI8HGJgIEQwuDmHrKlx0NNTQlub+6UV8dhYmMHBEGtrScrKbLz2WiWufO1dDESjOg8fxujvT2zkk1ZUyIq0pWX3iHRxUQqN5g0VqMsl56MdHcUzq097616/Lb190yKjs6ckkRcSui7j0O4Nyft1O+H1s1JoVChEE/BuH8wayuNDdXCluzAt3WgKfjQKC8a88VwdnMqBoIWA783KIO0jpdC9w5eNgQCsJKDCDodeQN4xXe6KLqnSiP7tUmjMo2J+oGrcUmWL5LDZxKXnZIa+DHGDRANo2A0SLc2BRDPVuX7sXKQia+/cOCnuMkGCFB4cnKIJS5aPZY5ZZpAWYK20UU7F3hBoWZkMIE3/Ul555RW+8Y1v8NWvfhVToXtKBcZniUABRkZCvPvuIgCnTvk4d64I5cZzIIQ0aujvl1Vp+tly7Fgply692Iw0kdBIJHSGhoLcv7/OX/trddTWFsAFPAPxuM7Dh3H6+uIb9oYVFRbOnt3d1u7YmODGjU2hUXU1XLlS3PloLCadjJ7Ii2pKvPDqJaivK/x9La3Be7dh3ZjDnuiCc4cL1z4WAu6Nw+1R+XG5F94+CiUFUALrAj6dgn5DZdxeBq+15DYXzWzR7qRdOxGB1QQMBaHKAW/UPP9rUzr8yDCiNyHDuTvymD0PaBqfpCSJdppNvGoxY8qBRBPovEeYddQCkOgiKcM793wOJBolyT0mSKJSiosTNGZ9jEkmWWQeBYUOOlGCpt0n0Fgsxn/9r/+V3/md3+H+/fvyYIpCY2Mjf//v/31+5Vd+BZ/Pl8uhi47PGoEC9PcH+PhjuU9x8WI5x4/7dv0xRKMqg4NBBgaCfOlLtTtu5/7P/zmNz2flypWKojkixeM6vb1xHj3aJNKqKgvnzrl2TWykaXI2eveu2PDYPXQIzp9XcDiKR6ST0/DRNYgYe5aHuuDi2cIa1IOcw17rlWpdgAqfNKz3FXBdZHpFVqPxlIxF+9zhwql0B5bgkylJqBUuORf15HiOspl1CgFhFX40D3VOuPiC605dwPtBGDF8UC554FgeFxGPNY0PUxoCaDWbeH2PSTTTO7caB2eoyDoKLUyce0yiouHHzTEanrme8jzIQO5RVlnBhImaYB31pfV7JyK6du0a//7f/3v+9E//lGQyiaIoOJ1OfuEXfoF/8A/+AT09PfneRUHxWSRQgPv317h5U8oLX321kp6evXnsO4pOMr5mcTHOX/zFDF/6Ui0NDQVcLHwO4nGdBw9kRZpu7dbWWjl3zrlrxvWRiKxG0/NRm00a1R8+XLz90WRSOhn1D8qP3W5ZjRYydzSN8Vm5NxpPyJbxxaNwuK1wx4/E4cePYCEgPz7eBOfbC1PtzoVkSzeuyn3Rz7dJ84VcMRKSlWPPC9qzabLtXZd+u1+ofbFbkhBwPQy9hjj+hAsu5PEYxzWdd1MqOtBoMvGW1Yw5RxJ9nzBreZLoCnFuGCkuNTg5Q3nWRvYBotxnCh2dSko4Ql1WiTKZMWjxYIJXS6/uvQp3aWmJ3/3d3+V3f/d3mZqa2niRffvtt/nGN77Bl770pULdVV74rBIowI0bKzx4sA7A1auVHDq0Px9/mkC/9705dF3w1lvVBV/FeRGiUZ3792P098c3xFANDVbOnXNRWbk7hgzz84JPPhGsGCshZWWyrVtXV7xqdG4ePvhkc+Wlsx0unwd7gTeSonEpMJo2xEzNtdIK0FGg+9F1aQGYzhmt8cGbR8BdgHWaUAJ+YOyLmhQZ0p1LNFpCg+/PgVWB16ohrZN7XnV6YxmehOHnmnY2370fgZtGVyHXSLQ0pjSdH6sqmoA6k8LnrRYse0iiS8S5yTI6gjqcnMqBRFcJ85BpwwzfxyGyG25raNI3NxjnTOnZvSfQNHRd5y//8i/5D//hP/Duu+9uvJh2dHTwq7/6q/zSL/0SbneB9Oo54LNMoACffrrMo0fy8nw/kmj69x0Kpfjv/32SN9+spq3txZfQqqozORlleTlBIqFz6JCXysr8Xy3DYY179+IMDW0SaUuLjXPnnJSVFZ9IhZAm9bdube6PtrXBpUvFW3tRVeli9LBPfux0ymq0pamw9yMEPHoCNx4ZXrcO6WDUUEDzgrFF+GBARqU5rPDWscJ46aq6jEYbW5cfH62Ciw3ZE1QoJduzmaN9IWQrNqmDDqwnpcnCwzVpspCNzfVQDD4MyUi0Jhu8lYfhwqyu88OUiiqg2qTwBasF2x6SqAzlXpaVMS5O4M86l3SJII+YBQRNlNNOdlshKirhYJiy0rL9Q6CZGB4e5nd+53f4gz/4gw3RUWlpKaurq8W82xfis06gsJVEX3utku6dygJ3Ee+/v8jqapIvfKHmhWss6+tJrl1bIRJRN6wLR0fDNDW5uHy5oiDK3WBQ4+7dGI8fJzYqhK4uO2fOOPF4il8ZJxLSqL6/X77AWq3SFvDo0eK1dReX4P2P5eoLSAejVy4WJuUlE6sBeOcWrBkCqkILjIJR+NEjWAnJ39v5djjRnP9xhZAeurcNe8F6L7zVBvYcn24fL8J6SgqAAinpfWs3ba6xHPNBixuyTRmcTMCPAtJwocYqI9FsOZ7bBV3nBymVpIBKk8IXrRbsBSDR1/Hgy3KtBGCeKLdZQQAteDhG9ldHc6wziAyhbaeKJrITWe6JCjcajbK+vs7a2trG7UUfz8zMMD4+vlGd7Kbz0Hb870CgsP8qUellKyPZystt/Jf/MsHlyxUcOuR97sw0kdD49NNlgkGVz32uktJSqeoIBJKMjkZobnZltXv6Mqyva9y6FWVsTDobmc0KR47YOXXKid1efPX4yorg448FC0brs6wMXn1VoaamOCSqaTIq7cEjSRgOh9wbLXQ1qqrSBrDfUNFWlsGb5wqX7qJq8PHQZlh3axV8rkdaDuaLsTV4b1xWpaUO+GK7fJstfjgnVbeXK+TaiknZzBO1KPmt5cwn4fsBmStabpGGC7m6Fi3rOt9LqSQElJsUvmS14NjDSnSGCHeRBVU7Xg7jy/oYk6zwBLmpcIhaarM4xp7tgb5ISPKiwx4QaOHwySdL9PXJS/9Ll8o5dsy3Z49FCMGdO2vcu7eG1dgc/+mfrnuKANPPDUVRGBoKcuPGCk6nBZ/PSjCY4uxZP83NbmIxFbvdXJQKbXFR5ebNKLOzcsfVZlM4edLJ0aMOLLn2yHYIIQTDw3DjxmZbt7tb7o8WS627vCKr0XTKS1eHnI0WWqk7Pgsf3IVEEmxWuHoK2gooZOqfhk8fy5ZxmRs+fxxKC6BNW43B90cgnJQV6OfbcjOj/3BRkujna6B6W1s337XglRR8dx1iAnxmSaK5Nk9WdcH3UiliAvwGiToLQKJv5ehYNEmYB8gn5yFK6ST71+UnLDLJChiB3BU7DOTeUwI1mUx0d3dTUlJCaWnpjt92dXXlcrcFQfqEDQ4u0929ezuVxcL168s8fCgr0bNnyzh9ugDJwnkgkdDo6wvw8GEAq1Xh/PlyOju9JBLaFiGRpun8+McLTE5GOXWqjLo6J8vLCUZGwrz6akVBZqAvw/R0khs3YqysyL0Tt9vEuXMuOjttRY96SySkm9HAgPzYbpck2t1dHBOG7dWoxwOvXS783mg4Cu/egnlDPHW4DS4dgxytWZ/CQgB+1CsNGGwWmerSlIMIaDuiKWlGvxiR1ePVZujK4eXhzircW4XXqqCzwNfnARW+sw5hHTwm+Clf7tZ/67rgu6kUUQE+ReHLNguuHEn0XcNswWlUormQ6Cgh+lgH4Cg+Wsn+CmaAWeYJZJUluicEarVa0TQNRVE4c+YMv/Zrv8bP/dzPYS7UX0kRkT5h//E/jvPzP9+Iy7W/jR92grt3V7l9W17BHT9eysUXLZvtIh48WCMW07h4sYLr15eN98txOi2sriZ4991FqqrsXL0qh//JpMZ3vjNHba2DCxfKdyWvVAjByEiSW7eihMNSaVRRYeHiRRd1dcVffVlYEHz0kSAtC6itlW1dn684P/vCIrz30aZS92gPnD9TWBcjXYfbA3B/SH5c7pMt3ULtjEYTctVlfl1+fLYNTrXkX+WpOrw/DqNGpX6qBs7WZX/coaAk0TdroNDXgWFNkmhAA6cCP1UGuSb9BXTBd1MqESHyJtF3CBE0vHPfxJuTAf0QAYaRHbXT+KknO7GpjuAR06wQxoKZ0ztIcMmHQHNmjvHxcf7xP/7H+Hw+bt++zde//nVaWlr4zd/8zT0VCGWDWEzwzjthdL2oOqpdwenTfi5flpfLDx8GeP/9xX3xc504UcbFixVomk4qJUilBE5DRWGzmQgGUxuzWyEENpuZsjIba2upXQv7VhSFzk47P/dzPi5ccGGzKSwvq/yv/xXkBz8Isb5e3HFDdbXCz/yMwsWLChYLzM3B//gfgrt3xYbfb0Hvrwr+r5+Gw4fkx48G4M/+CpaWC3cfJhOcPwJfviJXW1bW4c/eg5GpwhzfZYe/dgoO18uPb4/KqjSpvvj7XgaLCd5slcQJUmT0zpgk1mzQXQL/R8PmDLSQ8Jjhp8vAb5bt3L9ag8XUy7/vWSg1KfyUzYJHUVgXgu8mVaI51FT2DMP5qGG6ECPLkwZ0U0qr0Xq9xyrzxF7yHVthQuEI9ZTgREXjAVPEyfHk7AB5q3Cj0Si///u/z7/9t/+W4eFhFEXB4XDw8z//83zjG9/g8OHDhXqsBUP6iuM//IcxLBYvJ044uXCh+Ev+u4GhoSAffriEENDQ4OTtt2s25pH7AcmktuFGFAql+PM/n+HSpXI6OmRpEo9r/MmfTHLmjJ8jR3J0/s4T8bjOnTtyh1QISQaHDzs4fdqJw1HccxkKSZHRlEE0Ph9cvVo8kdHUNHzwKUSjsso6fQJOHS9swks0Llu66cDunla4fLxwLd2hWfhoaHMu+oXjhbEAHF6R0Wi6gCo3fKFdmi8UCotxae+XKxI6fG8dFlW5i/qFUqjLcaYdNMgzLASliiTVXCrRCBrvGikuXsy8mUMot0Bwn1WmiWJC4QIVVGQZpp00zOejJHFj5xTNWJ9TEe+bOLPvfOc7/PZv/zbvvPOOPLii8NZbb/Frv/Zr+8ZEATZP2P37i9y4IX+5n/+8d1dNyIuJyckIP/7xAqoqqKiw88Uv1hTMxL3QuHlzhenpGFeuVBAOqzx+HGJlJcHXvtZUtPi2nWJ9XeP69SiTk1Kxa7MpnD4thUbFWj1J48kTwaefCmLGBfjhw9IS0GYr/P0mEvDx9U1P3apKeONVKKS+Tgi4MyCN6UHaAL51vnAq3cUA/NCYi9qt0nShoQDyhlnDuSihStu/L3VAWQHsnMfC0tqvywtXq3I3SEjp8IMAzKbADHw+DxP6kBB8pwAkGkbjHULE0CnFzBt4sWdJojqCO6wwTwwLCpeowpdlDmim+fyLfHP3DYGm0dfXx7/5N/+GP/7jPyYejxstsk6+8Y1v8Iu/+Iu4XHtb7WWesL4+M729cWw2hZ/5mVJKSvb/DHcnWFqK873vzRGP63g8Fr785Vp82aT97hJSKZ1bt1YZH4/gcpnx+WwcOuSlpsb5lGVgMqmxvp6iKp/L9hwwM5Pi2rUIq6uylVtaaubSJVfRzeoTCWkJOGiQjsslnYxaW4tD3iOjkkiTSTkPvXIBujsLex/TC/DubWkDaLNK44WWAomYogn44UNYDMpq+mIHHCvAuk4gDt8bgWBCxqJ9vh3q8pzlDgWlYlcATS54qyb30G9NyD3RScOE/q1SaNljEg0ZJBpHpwwLr+PBliWJaghussQyCayYuEIV3ixzQDN9cyvwcpT6p8wa9h2BprG8vMy3vvUtvvWtbzE/P4+iKPh8Pn7lV36Fb37zm8W625ci84R5PF7+6q+CLCyolJdb+MpXSoq+xrBbCAZTfO97cwQCKex2E2+/XUNdXWHTUAoFIQThsIrX+/w/kLQXcEODk7Nn/btKpEIIhoYS3LoVIxaTs53GRiuXL7spLS3uRdfsrBQZBQxDhNZWeOUVBaez8M/TcBje+1g4swcAAIbHSURBVFhaAgK0Nsu90UJaAUZi8OObsGCodI93ynlpIdrGmg4fDW7ui3bXyYzRfJsZcVXa/y2EZbX4uRboyFPsPhGBH89LAqx2wBdrIVe3S13Au0EYTYACvFkCbTn+eRSKRANovEuIBDrlWPgcXqxZOg2p6FxniTWSODDzClU4szRsWCfKfSYRCOopo4utkTj7gkCj0SihUGjjFg6HN95fXl7mP//n/7wltWU/7YFGIjp/+qcB4nGd9nYbb75ZwHiJPUY8rvH978+xuJjAZIIrV/bOhD5f3LixQm/v+oYlX3Ozi3Pn/AU1WngZkkmd+/fjPHwYQ9fli/6xY05On3YWNUhb0wT37gnu35ezPodDVqPt7YW/TyHkqsvte/K+3G7Z0q19QRRXttB1uNkHDx/Lj2sr4M3z0g6wEOidhOsj8mep8cHbx8CZZ8NA06XhQlqhe6EeTuR5TuZj0k83qYPfBl+uk2HduUAX8EEQHhsk+rkS6CwAieajzl1D5T3CJNGpxspVPFmnryTR+IRFwqh4sHCF6qyr2UWC9DEDPO1WtCcEeujQoQ2SjEQiLzROyEQhnIh+8zd/kz/7sz9jcHAQp9PJ5cuX+eY3v0l3d/eOvv9ZJ2xuLsV3vhNE1+H8eRcnT+7PSi0XqKrOhx8uMTIiU4WPHi3l4sXyos/xioFgMMXdu2s8fhzayCZtb/dw5kzZrraoAwGNa9c256Mul4mLF110dBSXzJeXBR98sGlQ39Iiq1GXq/C/y6VlePdDCBgWfaeOw5mThRUYjc1IU/qUKsnzrfNQU6ANrOkVueqSVMHjgC+eAH+eM1ch4Po09ErTG45UwuXG/NZnVhPw3VmIalBigb9WD54cxUpCwEchGDQMOj7nha4cX8q2k+hP2XIzW1hB5T1CqAjqsXEFd9bG8TFUPmaROBpl2LhEZQ45oJtuRUeop8owa9gzI4WdwOFwUFZWhs/n2/L2j/7oj3K5WwC++MUv8rWvfY1z586hqir/7J/9Mx49ekR/f/+OTOqfd8L6++N8/HHEuA/vrgYy7wYyd0Xr6py8+WbVxkrJZw3r60nu3FnjyRN5UaAo0Nnp5fTpMkqKsTvwHExOJvn00yjBoLwgrKmxcOWKm/Ly4p1XXRfcuwf37gl0XbZXr1xR6OgoPImmUjK0e8ioFKsq4c2r4C1gk2Y9BD+6Ib10TSZpunCkvUDHjsD3H0AwBhYzvHW0MKYLvQtwbVq+3+KDN1pzn2ECBFPwnRkIqeA2w0/VQ67Xg0LAJ2HoNwRoV71wKEcSDRokGhECv0nhyzna/i2Q4gPC6AhasHMBV9bG8UGSfGoEclfj4CwVWRPxMPPMsIYJhRM04cO1NwT6t//2334mMW5/ayu0V9gzsLS0RFVVFR988AFXr1596de/6IR99FGEgQEpKvrKV0p2JbFjNzE2Fua99xZRVYHbbeatt2qort5dUU4hsbqa4PbtNcbH5YWPyQSHDpVw+nTZrimPNU3w8GGce/diqKr4/7f3n8GNpNmdPvokHB3ovWeRLO+999W2qtrNaCVd7WqkK23srmYkTUzsh1XExs7MJyliIzbmL2lkrrSh0b2zsxrbvqtdVVdXl+vy3rAci94TBAEQNvN+OABBlkcCLLr3ichAAk1mZ4FA/vKc95zfQdNgyZJ0Vq/OwGHW9fsZ6O83OHz4+USjd5vhyHEpMHI4YPtmWR9NFeGwWADeiYrSvFrYsjw15g6BkESi7QOpLS66OwiH7knqtNQpbS7pSZyvNywi6gpBukVEtDCJhMbxYbgSFdEtTlhksnZzSDf4MOpYVBQVUTNTXNoIchS54Z1HOquewSXoQQYIcIJedAxqyGI5iS1EGxhcjhot2LGyijrCbv/kr4FOJrdv32bu3LlcvnyZJUuWPPXnnySgum7w4YfDdHaGcDotvPlmLhlmXZunKIODQT77rAuXKyR3/BuLJq3nMlX09vo5c2aQ1lYfADabxpIluaxYkTfadzrReDzS9nL3bjytu3FjJg0NE5fW1XVZFz13Lh6NbtkyMWujw8Nw8IhMeQFYvAA2rE1dPyfImujXUavBojx4cQM4U1C0r+twrAmuyzIYiyph07zk09Gdw1JcFIxAXjq8OlfaXczij0g6ty8gE1deKR/vqZsoJ4fhUlRENzlhicn3cjAqon4DSqLeuXYTInqPAF8jN7tLyGAJif/jxk5wmUcO80ns2hVB5zz3GcZPBg4a3fkU5xbOTgHVdZ3XXnsNl8vF0aNHH/kzgUCAQCAw+tztdlNdXf3YN8zv13nnHTdud4SSEhv79s2cytwYoZDOl1/2cPeufJgbG51s21aMLZk81BSgq2uEr78eoLtbFoHS0iysXJnPokU5z+3f1tYW5OjReFq3stLO5s1Z5OU9WWlGRnSamgJUVNgTHvo9MCDRaF/UTai+XoQ01eb0ug6nz0mREUBhAezeDnkpvP/q6JUqXX9AXIx2r4XKxMY8PpZLLXAymo6uKpD5oo4ko9zBEWlz8QQh0y4iWpCE6AWjg7q7/DLF5cVyqEriJuKUBy7IfWVSItqv63wUneJSFh2FZmYodxN+ziEntIpM5iVokgBwHw+XoubzS8mn7hmN42MECXOWZvyEsLjD7MhdNjsF9L/8l//CgQMHOHr0KFVVjx778IMf/IAf/vCHD73+pDfM5Yrw7rtDBAIGDQ0Odu1yPjdruefJpUsuvv66H8OAggIHL7xQOjpSbDpz/76XU6cGGByUaDAry8rq1QXMm5f9XIqnIhGDixdHOH/eTyRikJNj5bd/O/eJn6GTJ714PDpdXWGqq+1s357YRSG2NnrunIFhyCDtbds0amtT/+9tbZN2F79f0qxbNsiEl1Th8cGnJ6HPFZ0BuljmjKaC5l44dFVGpOVnSXFRdpI1g94gfHRbxNRhhZcboSyJgqWwLkYLrb5ob2cZ1CVxvNMeOB8V0Y1OWGpSRHujo9CCBlRaNF6027CauC5eYYQrUZu+9WQx5yl+tY9irG/uagqpSDAl7CXAOe4z4vbyUu6q2Seg3/nOd3j33Xc5cuQIc+bMeezPJRqBxujoCPHRR1KZu2JFBuvWzQy7vwfp7Bzh4MFufL4IdrvGtm0lNDSkyCJmEjEMg1u3PJw5M4DHIyapeXl2NmwopKYmMZNqs7jdEY4d8zJ/fhr19U+/SIRCBm63VPhmZlrYvj0LqzWxC1Rfn8EXXxgMRtstFiyAjRu1lLfZ+Hxw6CvoiPZczmsUIU2VKX04DEcvQtN9ed5QBdtXpeb4fW745BJ4A5BuFxEtSTKKDkR7Rbs80ne6Zw7U5pk/nm7AoW6465G2lJ2l0JhE8dYZD5xLQSTaHRXRsAHVFgsv2K1YTIjoOXw0IZmirTipTNBpCOASA9zHiwXYSAkFCQqxmxFG3F7Kcotnj4AahsGf/umf8vbbb3P48GHmzk3MLiWRqqumpgCHD8vC96ZNWSxZMn0Lbp6Ezxfm88+76eqSD3Rjo5PNm4vGjR+brkQiOteuuTl/fhC/X5pIKysz2LChkMJkqjRSyIOuSy0tQU6e9PHyy9mm3LEiEYPTpw0uXZLnOTmwY0fqPXUNA85fkjFphgH5efDCztSmdK/egROXJX1clAcvbYSsFHSZef0ion3DIni7Fsug7mQI61JY1OySyHm7yZFoMXRDHIuaopNzdpTAvCTauMeK6GYnLDYpoh26ziehMBED6q0WdtqsCWfoDAy+xkczASxo7MRJcYJOQwYGZ6KWf3YsbKEEZ4LHmBJGCs+TP/mTP+FnP/sZ77777rjez9zcXDIynv7NSvQNO39+hNOn5VO3e7dzQotCJhNdNzh3TgZiG4akPbdvL6EqmQWYKUQwGOHCBReXLw+NTjlZsCCbNWsKpoRXsK4bo+lltzvCz3/u4s03cykqMn9unZ0SjXo8ckFfvhzWrNFSnsbu7JICI58P7Hap0q2vS+Hx+6TVxR+QftEXN0BJCsbehsJw8Cq0RNeON85NvkJXN8SEvilaHb2hCpaVmj+eYcCxXrgW7cfdVgwLkrhBGbsmmkx1bmtE57NQGB2Yb7Ww1Z7451TH4CheOghiR2MPOeQmOAYtgs6JqFtRFjY2U0JaAseYdQL6uDudf/mXf+EP/uAPnvr7Zt6wY8e8XL3qx2KBV1/NeS5zIieLnh4/X3zRw9CQjAFatCiH9esLp9RUl2QYHg5x6tTAaA+pzaaxYkUey5blTYkiqs7OEF9/7cNq1Xj55eykU6/BoBjTNzXJ8+Ji2LVLIzc39Sndg0fiNoBLF8H6NakzXhj2wicnYWBIKn+3rYS5KWhHMQw43gRXoy00S6thw9zkZ4uebINL3bK/ogzWVSZ3vOO9cCVq57ilGBYlIaJjq3OT6RO9F9E5FApjAEttFtabyK+HMfiCYfoJk4GFPSZmiQaibkVewgkbLcw6AU0WM2+YYcjs0Lt3g9jtGvv35yQVGUx1wmGdU6cGuBL9xubk2Nmxo5iyspnj0NTd7efEiT56emR93Om0sW5dAQ0Nz6dgTNcNAgGDgYEIAwNhenrCDAxEiEQgK8vCrl1OsrJSJ+h374qnbiAQNYvfrDF//sRW6ZaVivHCM/ibPBOhsIxGux9dd10xH9YuSl7sAC7eh69vy359CexcnLyH7oUuOBVtnVlYBFtqkjvXk31wySX7m4pgSZ75Y50YhstREd2eDfNNfrWbIhGOhKTifLXNykpb4ksOAXQ+Z5jh6Bi0PSYmuHgIcZQeQuiUkcEaCp/JrEEJaIKYfcMiEYMDB4bp6AiRkWHhtddyJtxIfLJpb/fx5Ze9o0U4y5blsnZtwaSPGksld+54+Prr/tF/Y0lJGps2FU2oWb1hGLz3nhuPRycjw4LDoZGTY6G62kFJiY30dC3h4qFnweORdpeODnleXw9bt2qkpaX2/9XcAoePivFCRoaIaEV5ao5tGHD6Gly4Kc9ry2HXWjCRQXyI211w+LrcCJTlyWzRtCSTTTf64KsWOe+GfNg5x/z4MoBTfXDBJfsbCmFZvvljHRuGq1ER3ZmEd+7VcIQTYRHRTTYri0yIqJcIn0fHoBVFzedtCToNidFCDzpQj5PFPP3NUQKaIMm8YcGgzvvvD9PfH8bpFBF1Ome2iAaDEU6c6OfmTalkyM93sG1b8bR2MHqQcFjnypUhzp8fJBSSr8S8edmsWzcx66OBQLzXeN++HMrLx1+lHywqehTP8jOP+72LF+HMGSNuFr9Lo7w8tSLqdsNnh6E/6gC0dhUsX5KaaBHgdqu4F0UiUJALL22A7BREuh2DMhYtGIa8LHh1hXjpJsNY16KaXNhTn5z135l+OBetsl5bACuTWA8+Oiy2f8lOcTkXjnAuKqLb7VbmmnDYcBHmIMOEMKjAwRYTvrnt+DiHLEAvIY85PLl0WQlogiTzhoE0vL//vhuXK0JurpX9+3PIzJw5EdnjuH/fy5EjvYyMyJdk0aIc1q0reG5OP88Dny/MqVMDNEXLHu12jVWr8lm6NG9C+kfPnPFx/vwIixens2FDZkL/D1038Hh0mpuDeDw6GzdmJiSovb0Ghw7Fx6StWJH6AqNwGI59HffSra2GnVvFDjAV9AxIv6jPL6YLL65PjRn9gAcOXJA2l8w0eHkZFCU5xKh1SIZzh3Uod8JLjdIzapZzA3BmQPZXF8hmBsOAI8Nw0y89py/kQq3JOskToTBXIzoasMduo9ZEpqqXEF9EfXPrSWMdid8V3cbNdeSDvZZCyp7QI6oENEGSFVAQ27ZYCi4/38q+fTkzzvLvUfj9EU6e7B8VmIwMK+vXFzJ37swymujp8XPsWB+9vbI+mptrZ/PmogmpSB4YCPPZZx4iEYP9+3PIzn7yVdXtjtDXF+bKFT/p6Rb6+8N4PDobNmSydGliC1mhkMGJE/Gh3SUlsHu3RnZ2av+WN5pESCMRaXF5cVfqWl28I/DJCTFdsFhkSHdjdQqO64cDF0VM7VZ4cRlUJln52+WBj2+L21BJFrzSCGlJJDguDMKpaLXvqnxYY7JlxjDgCzfcDoiIvpwLVSZE1DAMjoQj3IroWDV42W6j3EQV2VjfXLOWf7EeUSsamygh7zF9pkpAEyQVAirHERH1+WaXiAJ0dIxw9GgvLpdU6paWprNlS9GU6atMBTEjhq+/7h+Nuuvqsti4sfCJg7/NcvOmn/x8GyUlckV1uyPjekAHB8NcuxagrS1EUZGV2loHubnW0dmkmzZlmS46undPxqTFzOJ37NCoq0utiPb2SUrX45FWl13bJCJNBeEwfHFWxqMBrFkEqxYkf9xgWNK5HYMizrsWQX0SLSkAfT746JYM6S7MhFcbISOJj9OlQTgZFdE1BbDKpMjrBhx0w70A2IBX86DMRKZANww+D0Vo0XUcGuy12yg0IaK38HM2avm3jizqEzRJ0DE4TR89+J84jFsJaIKkSkBB5kK+/76IaEGBlb17Z4+I6rrB5csuzp4dHJ1CsnBhDmvXFswIA4YYwWCEs2cHuXJlCMOQtpdVq/JZtmxi0roAly+PcPt2kA0bMikvtxMI6Hz00TAjIzqvv547KpSnT/vo6AixbFkGc+Yklxf1eAw+/9ygJzrrcskS2LAhtSndkRH4/Mt4q8ualTJnNBXJC8MQI/rYkO55tdLqkmwbTUSHL67C3ej7snUBLEyyJWVgBD68BSMhMaHfOxeykvjzjRXRZNZEdQM+GYLWIDg02JcHRSbEPWwYHAiF6dYNMjTY77CTY+KPfIkRrjGChsY2nJQnaJIQQuc4PbgJkYOdzZRge6C6VwlogqRSQOFhEd23L4f09NkhogBeb5iTJ/tH+yrT0y2sWpXPokW503Jo9+MYHAxy9GgvnZ3i1pSXZ2fLlmIqKlLf2nP3boC7d4OsX59JdrYVwzC4cyfI11/7cDotvPJKNj09Yc6dG6G01M769alJLeu6walTcQej4mJJ6ebkpO7vqOtw4hRcjaaN62pkXdSeoqD+2l04dlEEtaIYXlgPaUmuuRoGHL0Zn+aytgFW1iV3zCE/fHBLfHRz0kREs5NI4IxN564vhOUmq3PDBhxwQWcI0jXYnw9mpjoGDYMPQmEGdIMcTWO/yYHcJ/BynwA2NPaQTd4josgn4SPMUboJPKa9RQlogqRaQEGJKEha99ixvlED95wcO+vWFVBfP/19dcfS1DTMyZN9o7aAc+c62bChMOXDySMR46FWluHhCKdPj3DvXpDcXAu5uVZ27nRis2mmq3IfRUuLOBgFAtH5n9s15sxJ7c3QzVvw1QkR1Pw8eGmXWA6mgtYumegSCkNeNryyKTUVumfuwrl7sr+8FtYnaaA/HJBI1B2QMWj75omYmuX8AJyOFhYl0+IS1OFDF/SGIdMCr+VBjomPt88weD8YZtgwKLJo7DUxBi2CwZd46CFEBhZeIIfMBHtEx84RbSCbReSN/jcloAkyEQIK40U0P1/SubOhOncsum5w8+YwZ84MjK4blpSksX59IeXlM8eEIRCIcPr0ANei/moOh4V16wpYuDBnwoupDMPg0089tLeHsNk0/v2/n5hUssdjcPCgQXfUTWfxYknpprI/tacXPv1CXIwcDtizHaqSTI/G6HfBxyekyCg9DV7emBr7v8stcPI27F789PVQl1cKkLKe0BriDYqIuvySxt03F3KTaJs5OyAbJGe24Nfhg0EYiEC2BV7LhywTKzNDusH70VmilRaNl+y2hM3nxxot5GFjN9nYE25v8XIOeWNWUEB1tLpXCWiCTJSAgoxB++ADEdGcHCv79mXP+D7RRxEK6Vy65OLiRRfhsHzEamoyWb++kPz86T8uLUZPj5+jR/vo65Nq3ZKSNLZuLZ7QYqo7dwJcvx4gN9eKrhssXpw+Ya5Yui6m9BcvyvOiInjhhdRW6fp8UlzU3SNroRvWig1gKhhboWu1ymzRuorkjzvkg9ynZM1dXkn79g3D3pVQ/IRLzUhI0rmDIzJTdN88WRs1y9g+0c1FsDjP3HF8EXjPBe4I5FlFRM0k1np1nQ+jE1warBZ2mDCf9xDhM4YJoFOOna04E+4RjY1AswAbKKaQdCWgiTKRAirHFxH1eHScTgv79uWYmqgxE/D5wpw7N8j1624MQy6Q8+dns2JFPjk5M8NP2DAMrl4d4syZQYJBHU2DZcvyWL06P+XeusGgzpEjXrxenT17slNq9fckWlslpev3S6S4c2dq54xGInD0ZLxfdP5cGY1mohf/IUJhOHgKWqKFS5uXw+KG5I/7JFxeOHVHqnid6VKAtHcllD6hdccfhg+apMAo3SYimsxg7rGORVuLYaHJtiFPBN4dBK8OxTbYmwcOEx+7sebzZn1z+wlziGEiJntEDQzO0U9HdHrLVkqJuH1KQBNhogUUpE/0ww+HGRqKkJEhRR8z2Tv3abhcQU6dGqC52QuIkDY0OFmxIo+CgpnR+uLzhTl+vI+7d+XfmJNjZ9u21BcZjYzouFyRh9yLYqRyLXQsD1bprlwpxgup/H9dvgYnT0vRTlkpvLgT0lNgeKXrUlh0Pbp+uWK+DOmeCPxBuHBfzOn3rhRLwLN34fx92LsCyp+wLukPS4tLny81IjrWOzeZUWiuMLw3CH4DKuzwSh6YyeTfikT4Muqbu8FmZYkJy792gnwV7RFdTiYLSewDEkHnOL24COLExjJ3BkW5+UpAn5XnIaAgF7qPPhLbP7td48UXs6msnBlRl1m6u/2cOzdIa6tv9LWamkxWrMibMUb1LS1evvqqF69XLhQLFmSzfn3hc2ntMQyDTz7xkJ9vZe3ajJSvjeq6GC9cvSrPq6rEBjA9PXX/n7Z2aXUJBiE7G17eLUVGqeD8TTgdPfdUtbk8iK5Dj1tENBwRP127Tczqh3ywbeGTfz8Qho9uQ683NSIam+KiAXvKYI7Jmr7eEHzggpAB9Wli+2fm3uliOMLpqOXfLruNehNuRU34ORftEd2Mk+oEh3H7ifAV3fiJkOUOsTu3QQnos/K8BBQk5fbppx46OkLSiL3LSX39zIi4kqG/P8CFCy7u3vUQ+wSWlaWzYkUeNTUpGt0xiQSDEU6dihcZZWZa2by5iDlmr17PSHt7iA8/lP9ncbGNXbucEzLw4PZtgyNHDMJhcDplXbS4OHUiOuiCTw6Cezj1xUU3m+HIeYlyq0ulzcVENvGpBMPw+WXIz0p8PFowIoVFqRLRIz1wwy0uQy+VQ7XJr1h7UFpcdGBRBmx5ss3sYzkWCnM96lb0it1GmYm7mLP4uIUfKxq7yKYwwfYWF0GO0YPP7ebf5S5WAvqsPE8BBWlH+OILGYUGsHlzFosXzxwj9mRwu0NcuDBIU9MwunSFUFDgYPHiXObOdU6J+ZzJ0NU1wpEjccemurostmwpmtAB3s3NQb780kMgYGCzaWzZksW8eam/aRsYMPj0UwO3W9YqN2/WWLAgdSLq90uFble3iM+mdbD4KdHbs9ISbXMJh6Uy9+WNUqlrll43uEegIVqVG1vvP3kLXD54eXnix3xQRPfOFeciMxgGHOqGOx5Jvb5aAWaL4u/64fPocO/VWbIlfj4Gn4cj3I/opGmw324nL8FsiY7BV3joJEQ6Fl400d7SiY+g20tdbokS0GfleQsoyAfm+HEfV69KE/6KFRmsXZsxo/xjk8HnC3Ppkovr192j01AcDgvz52ezeHHutC44ikR0zp0b5OJFF7oOaWkWNm0qYu5ck7fvz4DXq3PokIfOThHuxsY0tmzJxGGm+uMJBINSXHT/vjxfsECENFWtLpGI9Io2Red0Ll4AG9elJu3aMwAHjkMgCLlOeHWzuV5RXYdPL0NWmrgUjeXIdegdhm+sM3eOwYisifakIBLVDfi0E1p8YNdgbyWYndh3zQdHZRmSLU5YZELYw4bBh6EwvbpBtqbxmgmjhSA6BxlmiAj50faWREegqSrcBJkMAY1x7pyPM2dkAF9Dg4MdO5wTMvdxuhIMRrh5c5irV9243aHR12tqMlm8OJeqqul70zEwEODw4d7Rlpfa2ky2bi2esGjUMAwuXPBz5owPw4DsbBnSXVqa2psR+f/A6dNyKSkpgRdf1MjMTN3f6cJlOHVW9muqYPf21DgXuYbho2Pg8UFmuhguFOYlfpwhH7x7BuZXwNJqmeByvV3WPZfVwKIq8+f4YCS6fx7kmxTRsA4fd0LHCKRZYH8lmK3hO+uVDWCPyTFofsPgvWAYt2FQHDVasCXR3lKFg81kPdMg7RhKQBNkMgUUoKkpwJEjHnQdSkttvPRS9qxzLXoahmHQ1jbClStD4wqOcnLsLFqUQ2Ojc0LToBOFrhtcvOji7NkBdF2i7E2bipg3b+Ki0e7uEIcOeRgelhabdesyWbYsPeU3Iq2tYrwQDEJmpqyLlpam7v9xtxm++Eqi0sICKS7KSsFyuc8vIjowBA67pHPNjERzeeHzK2C1gC8AoQjMLRMBzR4jeLH0biIEwiKifT4xnt+fRJ9oSIcP26EnABlWeK0Sck22ZsdmiVqQytxKE8cZa7RQa7Wwx0SPaC8hDuHBwGARGSxLYHqLEtAEmWwBBejoCPHpp8MEgwbZ2RZeeSWHvLzZ2Sv6NNzuEFevDnHz5jDBoCyUahpUV2cyb142NTWZ026t9MFotKYmk927S7HbJ+bfEQzqfPWVlzt3ZB2+utrOzp3OlN+4ud0Gn3xiMBidXrJ5s8bChal1LvrkkJjSZ2WJiBamwF0oGIKPj0NXv6zn7lkHteWJH2ckCN1DMOiF8jwxW8hwxEVzrHh2uaRat29YCo1Kcp48Li0QFrOFfp+YLeyfZ96xKBCB99thIAhOm4io00REb0QnuNwNSFr4tTwoNHGcLl3nQChMxIAlVgsb7InfHDcT4CQSEm8gi7pnnN6iBDRBpoKAgrgWffzxMG53BLtdY/duJzU1M8elJ9WEQjq3b3u4edNNT09g9HW7XaO+3kljo5OKiumT4tV1g0uXXJw5I9Ho8uV5rF9vcqDjM3Ljhp9jx3xEIgZZWRZ273ZSVpbalG4oJKPR7t6V54sWwaZNqZvqMjwMBz4H15CkcV/YkZoK3XBYCotaukTkdq5Jfq7o8IisjVos48Xz3D0ZkRaKQGU++ILQ0icm9U+a9DLWbCHLAa/NM29APxKG99phKAS5dni9CtJN3MNHoubzHSHxzX0jH8yYr92J6HwRCgOw2W5loQkXjYv4uI4fLWo8/yyVuUpAE2SqCCiA36/z2WfxYo916zJZsWJm9ENOJC5XkKamYW7f9uDxhEdfz8y0Ul/vpK4ui7Ky9GkxDaa52cunn3Zhs2n8zu/UTHhqemAgzOefe3C5ImgarF+fybJlqf/MnT9vjK6LVlTAnj2p6xcNBqVCt6NTxGnbJpiXpLE7SEHQl+fgVos837ICFtWbO9agBw5ehTnFMrklVvh0vllM6QucUFcEq6PHv9cDh6/BnqVQ/YT7qJEQvN8k3rnZaSKiZkeheUIiop4wFKfBvkowkwQJ6mK0MBCB/KjlX5qJ48R6RDXgJYeNqgSrxQwMjuKlneAzV+YqAU2QqSSgIJHI8eM+rl2TCt3GxjS2bcvCZpv6F//JxjAMurv93Lrl4e5dD4GAPvrf0tIs1NZmUVeXRVVVxpRO877zThs9PQGWLMll0yYTC3AJEgoZfPWVl9u3YwVNDnbsyCLNzFXvCdy/b3DokEEoJKYIL72kUVCQugrdL4/B7Wiku3qFbMliGHD8Ely9I8/XLRbnokQJhcULt6oA5kbTwU2dYvFntUjkGdahcxBeXQH5TrjZISnf6sInr5P6QvDeTZnikpsu6dxMk4kEVxDeaxPz+MoMeLlczi9RvBF4J2r5V26Xgdxm6iO/DIW5FZFh3PvsdgoSvAkOYfA5boaIUICNXU+pzFUCmiBTTUBjXLvm5/hxL7oORUU29uxxzloPXTNEIjqtrSM0N3tpafGOjhsDsFo1qqoyqKvLoro6c8oVILW1+fjoo06sVo0VK/KYPz8bp5lFqQS5ft3PsWPymXM6LbzwQjbFxal9bwYHZV3U7ZaU6+7dGjU1qRFRw4DT56RKF2DBPPHQTUWby+mr4lwE5q3/xqZtDQMOXpF07db5Ipi6DiduydDurQvEfMFqgWdxuPMERUQ9QanK3T9PqnTN0OuXNdGwAQ1O2FVqzmVoIGr5FzSgIQ12mXAr0g2Dj0JhunQDp6bxuon2lrGVubU42MjjDUyUgCbIVBVQgM7OEJ995sHv13E4NHbtUuuiZtB1iUybm700N3sZHg6P++/5+Q4qKzOoqsqgvDxjwop3EuHjjztpaYlXHFdXZ7JwYQ41NZkTmoru65OUrtsdwWKBTZuyWLQotUYfgYDBZ58ZdHTI8w0bNJYtS92/6doNOPa1iFRttbS5pMJd6GITfH1F9hfVixG92SX24RH4+UkZgzanJP76yVtSTPTa6sSP6Q6IiPpCUJQpfaIOk/fcbT74uENchhbnwuZic8cZ61a0PBPWmzDf8kfniA4l0d7STYjD0crcZWSw6DGVuUpAE2QqCyhIE/znnw/T3S0X/ZUrM1izZvoUx0xFBgYCUTH1jVa+xrBYoKQkncrKDCorMyguTsNqJoeVJJGIzr17Xm7cGKajY2T09YwMK/PnZ7NgQc6EGUoEgzqHD3tpbpYq3Xnz0tiyJbXLCLpucOyYwfXr8nzBAtiyJXXFRc0tcPBLSe2WFEuFbiqM6G80w5Fzsj+3BravMhfhjgThk4syhDtmKB8KSwTqD8Guxc8WeT6Iyy8i6g9DmRNenQtmVyvuDMPB6PzXNQWwymSFc9MIHB6W/c1OWGzCaGFIN3gvFCJgQL3Vwi4Tlbm38HM26pm7DScVj/DMVQKaIFNdQEEuNidP+rhyRdZFKyrs7N7tJCNj8iOl6Y7fH6GjY4T2dtnGGjaAXByLitIoKUmntDSdkpI0srOfrxPS0FCQmzeHuXlzeHQwOUBVVQZbtxZP2PlcvDjCqVNivFBQYOXFF7NTvoxw5YoY0hsGlJdLv2iqiou6uqXNJRCAvFx49QXx6k2WO21wKDolpr4Sdq1NXEQNAz66ABYNti+EYT+0D8DZe/DCUqgzGfGBtLa83ySmC1U58FKDuXVMgCsuON4n+8mMQTvnhTNeMbF/MRdqTVQLd+g6HwdlBNoqm5VVJu4wTuPlDgHsaLxIDtmMP4YS0ASZDgIa486dAF9+6SUcNsjMtLBnT+rbDmY7w8OhUTFtb/eNWzuNkZlppaQkneLiNAoKHOTnO8jOtk14VkDXDe7fl6g0ZijhcFh46aUyys2amT6Fjo4QBw96GBmRZYSdO53U1qZ2GaG1VUajhUKQkwMvv6yRl5ea93LQJW0uHo8YOrz6AhQ8YYTYs9LcIW0uug41ZWJCn2inhWHAB+ekeMgbkEhxZZ04GCVLt0fMFsI61OXBnnoRazPEBnInO8HliBtu+MEG7MuHEhOXrhuRCEejI9B22m00JHhnoGNwiGH6CJONlRfIxjGmMlcJaIJMJwEF6Rf99NPh0baD1aszWLlSpXQnCrc7RE+Pn+5uPz09Afr7A6NG92Ox2TTy8hzk59vJzxdRzcuz43TaJiQF7HaHeOedNvx+naVLc9m4ceKqdR9cRli9OoNVq1L7mRscNPj4Y4Ph6MSVF17QqKxMzfG9XhHRgUE59su7Zb5osrR1w6dfS89oVSm8aHKSS2/UjN1hE7OFVNHuho/vSFHSvELYXmt+zfarHrjulkravRVgZtqgbsAnQ9AahAwN3iww1yP6dTjM5bCOTYN9dhtFCYb/fnQ+wc0IOpU42DLG7k8JaIJMNwEFaTs4dsxLU5Os35WX29m5MwunmU+jIiHCYZ2+vgA9PQH6+gIMDgZxuUJEIo//6mRmWsnOtpOdbcPplC07W8Q1Pd1CWpo14bW/rq4R3nuvA02D3/qtavLyJra4TGZ/xgcg1NQ42LUrK6WG9H6/VOh2d0tKdOtWjfnzU9cr+vFBSetarbBnhxQYJUtnn5jQh8NQUSzWfxMxDs0szS747K5Eu8tKYYNJH17DgM+6oNkLDosYLeSb+MiN7REtsMLr+Yn3mhqGwSehMG26QZam8YaJytx+wnzOMAYGS8lgcbSoSAlogkxHAY3R1BTg6FFJ6drtGps3T8yoKsWT0XWD4eEQg4MhBgeDo9vQUIhw+Nm+UmlpFtLTrdFN9q1WjVBIJxiULRCI74dCOoYBCxfmsHVrEgtmCdLUFOCrr7xEIga5uVZeeik7pbaTkYg4F92OTlxZtQrWrEmNSIfD8PlhaGmTSGz75tQYLnRFRTQUhrJCeGWzDM2eKjT1w+Fm2V9bAStN2BKCpIM/7IBuv1j+vVEFZjrAPBF4ewBGDKhxwEu5iUfGwajxvMswKIlW5loTPMgdApyO2v1tI5sK7EpAE2U6CyiA2x3hiy88o+m1OXMcbN2apQzppwgjI2E8njDDw7HH0Ohzrzc8zuwhUTIzrbz1VtVz72Pt6wvz6afDeDwT1151+rTO+fOyP28ebNuWmgpdXYcjx+Mj0TatgyWLkj4sPQNiQh8MQWmhTHJxpKg84UY7lOSKW5FZLnfDiTbZ31IDi0zec/kj8G6bWP4VpckEFzNdXz0heH8QIsDSDNhoYn6CK1qZGzRgvtXCVhN3LQ8WFRlurxLQRJjuAgqS0rh4UUZV6TpkZFjYvj1L9YxOA3TdIBjU8fsj0S2+Hw4bOByWx27p6YmnflPFyIjOZ58N09UlN24TYTt5/brB0aNSoVtZKeuiDkfy/17DgBOn4Mp1yM+Dt/YnXgD0KPpc8OFRmSlanA97tyQvore74NBVcSR6bXVya6Sn2+F8l+zvqYd6k8VU7hC80ypuRdWZ8FK5uQKlscO4zc4RbY3ofBoKYwAbbVYWJ1iZG4kWFfUTJhcr691QmJunBPRZiQnohQsuli83WaM9RejrC/PFFx4GB6VKbeHCdDZsyMRuVwVGitQjvZw+rl+XddGGBgfbtztT2i/a2iqmC+EwFBTAK69oZGWl5viXr0FDnVTnpop+F3x4DPwBEdFXN0NaEvexgRB8cB76h8WI/rXV48ehJcrRFrjWK4L3SiNUmowZevzwQdStaEEObCt5+u88irHtLa/kQpWJFajL4QhfRz1zX3bYqEywqGgEnY9xE0Cn0O3nxdxKUwI6q3N+X35p0NQ0ve8fiopsvPVWLsuWScf49et+fv3rIbq6Qk/5TYUicSwWja1bs9iyJQuLBe7cCfLee248nsjTf/kZqa7WeO01jcxMGBiAd94x6O9Pzfd06aIni+fYcOJZQ4vCPNi3BdLToHdQ0rqBoPlzTLPD3hWQlyWtLh+cB4/f/PE2V0vkqRvw6V0ZzG2GknTYXSbCd8MN5wfMHWdVFsxNAwOJRl3hp/7KQyy1WWm0WjCAQyEZyJ0IGVhGK3FbMf/HmtUCahhw+LDBtWvTW0StVo0NG7LYty8Hp9OC2x3hvffcHD3qHZ2fqVCkkkWL0tm7N4f0dAt9fWF+8xt3Sm/aioo03nhDIz9fWlLee8+gvX3iv6dja1JGRsDng9Y2GaH2JApyx4voh0dlbdQs6Q7YtxJyMsQC8MPzMqTbDJoGO+ugMlvGpx24DUMmBbk2K27xd3oAbrnNHWdbDpTZxTP3gEtSw4my1Wal2KIRMOCzUJhQgiJajJ3VZLLpCT65T2NWp3A//XSQe/ckZF+9WmP16umf9gwGdU6e9HHjhnzbMjMtbNyYSUODqtRVpB6PJ8Inn3jo7w/LWLFtzpRWhQeDBp9+Kh66Fgvs2KHR2Jj676muQ2cXDHugvVNe6x+A3Byp4HU44JuvyRDvJzEwBB8clXRuSYGkc5NZE/X44b2z8ljghP2rJEI1QygibkV9PshJg9fnQ4bJY33dBxddEoHtrQQznh5+Xaa3uCMyvWVvXuLrqj7D4J1gCJ8Bc6wWdpsoKkqmJmZWR6Dr11tGRfPsWYNjx3Sm+/2Ew2Fh2zYn+/blkJtrxefTOXjQw4EDbtzu1KXZFAoAp9PK66/nMGeOA12Hw4c9nD7tS9n3yOHQeOUVjYYGEblDhwyuXEntd9QwpEL3o89gxC9FRovmw2uvQHUlVJTLY/gZvj6xSDTNIVW6sVYXszjTJRLNTIMBD3xy6dnO41HYrbIGmp0mJvQf3xFRNcO6Qqh3imH8p50yEi1R0i3SzmLXoDMExz2JHyNT09htt2EB7kV0Lpt9c0wyqwUUJPLcvFlE9OpV+YLq+vQWURDv3G9+M5fVqzOwWKC1NcQvfznEhQsjM+Lfp5g62Gwae/Y4WblSwpDz50c4eNDzzP2wT8Nq1di1S2PJEnl+/LjB6dOpW5rQNDGfz8wEjxdWLYfyMolEb9yCbCesWCrR6LNQkCvVuGkO6O6HA8eSE9GcTJkX6rBBlws+u8wjnbGehQw7vNooY896vfD5XVkbTRRNgx0lUJIGAV2muPhNaFe+DXZH39drI3DN9+SffxSlFgsb7VKJeyococPsm2OCWZ3CHRuy37lj8MUXBroOVVVSPj9TKlmHhiJ89ZWXjg5ZlMnPt7J5cxYVFcpTV5FampoCHDniQdehpMTGSy9lp3QAwvnzBqdPyyVrwQJxLkqVvWD/gBjRV5SJY9HFK2JIv2yxOS/dPhd88JWshZYXSZ9oMo5F3UOyFhqOQGMZ7Fxk3qavxwsfNIlRwvxC2F5n7jgjYXinDYbDUJYuln9mXCwveOGUN5oSzoNyE1XMsUHc6Rq84bDjfMY3RxkpJMjj3rC2NllvCYehpEQMrlM1JWIqcOtWgBMn4mbpc+Y42LAhk+xsZQeoSB2dnSE+/XSYQMDA6bTw8svZFBSkzvhhbK/onDmwa5eG1Zqa76nHA7/5QPbr66Rq91kjz0cx1myhsgRe2pCciLb2SxpX12FRJWxZYP5YLUPwyR1JYa8qhzUmDe0Hg2K0ENRhXjbsMOk5fHAI7gQgXYO3THjmhg2DD0Jh+nSZIbrvGZ2K1Bpoiqiq0ti3TyMtDXp6pHx+aGjm3F/MnZvGb/92LosXp6NpcO9ekF/8YogzZ3wpS7cpFOXldt54I5fcXCsej86777ppa0uir+MBFi7U2LNHw2KBe/fgwAGDYDA1n9/efshIF4Hq6U1OPEEKiV7ZJDZ/7T3xaS5mqS6UyBPgWjucuWv+WDW5sLVG9s91wvVec8fJd8jEFg1oGoYLg+aOsz0HimzgjxrQJ3pJskXXQ9M06NUNjj2H9VAloA9QUqLx+usa2dngdsO77xp0d88ccUlLs7B5cxbf+EYuFRV2IhGDc+dG+PnPXdy+bbJOXqF4gNxcK2+8kUN5uZ1QyODAgWGuXUuimfEB5szRePVVDbsdOjrggw8MRkaS+542t8DXZ6C0BLZulGKiB4nl68JhaXFxDUm7y5MoLYSXo+nbli44eDo5EW0ohS3zZf/cPbjSav5YC4ok+gQ42ipRqRmqMmFTdDjQqX64Z6IgyKbJ3NAMDfrDcNhEi0y2prHTbhMxj+hcj0ysiKoU7mNC9pERGbXU2yuWXzt2aDQ0zJx0box794KcPOlleFi+0SUlNjZuzKS0VK2PKpJH1w2OHIlPEVq2LJ316zNTtm7Z12fw0UcGfr/MFd23T8PpNHdsr1fWPRvmiIjG0HVpoTEMWXPs7ZOf6+6Nr/dt2yTVuk+irRs+PiHHa6yGnWvMr2GCiGcsAt25COaaNIwHMZ5v6pf5pK/NhyKTTk3HeuHqkIjha5VQlJ74MbqC8IFLKnzXZsHKp7QOPYqL4QinwxEswD6HjZInOBWpNdAEedY3LBw2OHjQ4P59eT5TekUfJBIxuHTJz/nzI6Op3NpaB2vWZFBYOIVGTCimLefPj3D6tJRYNjQ42LHDmbJ1y6EhEdHhYXA6Ye9ejdxcc8eOROIeuXebZR0U4uLZ2QWfHYayEqipEqFtboFLV+H1V6Xo6Enc74TPvo6uYdbDlhWmTnOUE01wuVUE/qVlkuI1g27AgVvQPgyZdnhzAWSZKOTRDfikE1p9kGWFN6vNTW+5MQJHouYVL+dCjYnW4s9CYe5HdJzR8Wfpj7lbUWugE4TNpvHiixrLlsnzs2cNDh7UZ9x6odWqsXJlBr/zO3ksWJCGpsH9+0F+/eshPv9cBnkrFMmwcmUGu3Y5R+3/DhwYTplLVm6uWP/l5UkR0HvvGQwMmPuOxsTz7AU4fgruRW+eNQ26e2RId00VrFsNC+ZJmjdWaDT0DCnH2nKJPAGu3YVTV02d5igb5kpFrq5Le0ufSWcgiwYvNEB+BvhC5ntELRrsLoU8O3gj0iMaMfFnXpABi6LmDIfc4DbRBrTdZiVX0/AYBl+EwhPS468E9ClomsaGDZboaCW4c0fWW3y+mSWiIK5F27Y5+a3fyqOhQW4/794N8stfujh82KOMGBRJ0diYxiuv5GC3a3R0hHjvPTdeb2pENCtLY/9+jYICWZN8/32Dvj7z39GF88RMoTxaUerxwNGTUFcDq5ePjzQDAejrl3XRZ6GhCratkv0LN2Uzi6bBjoVQWSDtLQcugttELyWAwwovN0ivaL8PDt57dj/gB4/zUjmkWaAnAEdMFidtckKpTez+PhmCUIIfFUe0qMimQbtucMGMkj8FJaDPyIIFUrQQq9B9+22D3t6ZJ6IAeXlWdu/O5pvfzKWuziFOLU0BfvELF1995U3ZRU8x+6istPPaazlkZloYGIjw7rtDDA4m4TIwhowMEdGSEhG1Dz4w6Ooy9x3NzBRDhfToGl7/oIjVgrmQPWaOZSgEN29DUSEUJ5A+XVAHG5bK/qmrcPWOqdMEJH37wlIozIaRoIio32TRc3YavNQga7stQ/F5oomSO6Yy99YwXDRRmWvR4IVcyLTAYCSe0k2EAovGpui4s3MTYLKgBDQBKirE4DovL25wPd2nuTyJggIbL76YzZtv5lJdbUfXZdrL//2/g3z1lVdFpApTFBbaeP31HPLy4m0uqTKiT0vT2LtXo7wcgkH46KPUmNB3dUuadGyhUCQiPrm378qaaKKjhZfNhVXRPs5jF+FWi/nzc9jgleVi/Tfkg48vmrf8K8kS83mAKz2ymaFyTGXu1/3QYmIKTKYV9uSIUN0JwGUT0fU8q5V50cktX4TC+FKYylUCmiC5uSKitbXyBTp8WDx0Z7I9XnGxjVdeyeG116QtISakP/+5i0OHPAwMpCaCUMwesrOtvPZaDqWlNoJBgw8/HObevdT0itrt4p9bVSVp1Y8/NmhpSe776cySqHRsG8vtu3DmvKyDblgrryd6bV6zCJY0yP7hs1JkZJbMNLH8S7NDjxsOXjHfLlOfD+sqZf9Em/n2lsV5sDB6Y3Go25xnbpkDNkQHppz0QKeJY2yyWSmwaIwYMv5MT5GIqircRG8boxiGwblzUlgEUFYm9n8ZGTOvSvdBurpCnD8/QmtrPGqorXWwcmUGJSWqalfx7ITDBocOeWhulqvi5s1ZLF5sovfhEUQiUkXf3Cxpzl27NOrrzX0/fT5xKCorEcHs6pG11qJC2LFFfiZWqZsohgFfnoOm+1LE9Opmsf4zS5dLLP8iOiyshK1JuBUduQ83+qS95fX5UGiivUU3ZBB3lx9y7fBmlayTJsqhIbgdkD7RtwqkyjcRXLrBu6EQIQOW26ysjaZ2VRtLgqRCQGPcvy8eusGgjDras0ejtHTmiyhAX1+YCxdGuHs3fktYXm5n2bJ0amrsKev1U8xsDMPg2DHfqNHC6tUZrF5tshHxAXTd4PBhg9u3o3Mxd5ofh+b1wqlz4PWJwXxFGcxtiP0bkuvp1HVpb7nfKePP9m+VQd1mudcjVbkAaxtgZZ3J8xrT3pLlgLcWmBuBNhKGt9vAE4aaTCkySvT9ChvwzgAMRKS4aH9+4uPP7kR0vog6+79kt1FttSgBTZRUCiiAyyUeui6X3OmuX6+xdOnsEQ+XK8LFiyPcuhUYTRllZ1tYtCid+fPTSE9XKwWKp3PunI8zZ8TWZ8mSdDZuTI3hgmEYfPWVwY0b0arVHRpz55o7rmHINrYvP1nxjBGJiG9uZx9kpsPr2yHbhIlAjCutcLxJ9nctlnYXMwQj8M4NcPmh1An75pozjO/zw7vtEDFgVT6sMdGz6g7DbwalMndpBmzMfvrvPMixUJjrUdP5Nx12IsPDSkATIdUCChAKGRw5YnAnWk03Zw5s26aRljZ7hNTr1blyxc+NG34CAflYWa0aDQ0OFi9Op7hYpXcVT+bKFT/Hj0u1ybx5aWzbloUl0TDjEaRSRCeSYAjeOyKDuXOdIqLpScwnP3kLLrWI4O9dAeUmpsoADPnh7RsipvMKYUeduePccsMX0aKkF8ugzpn4MZoD8Gl0TXZPDtQnmPGPGAbvhcL06wZlFo0tIz7y8/KUgD4rEyGgMa5eNThxQsaiOZ2y7lJWNvW+qBNJOGxw506Aq1cD9PXFC4yKi20sXpxOQ4MjZS40ipnHrVsBDh/2YBhQV+dg9+7UuBYZhkxxuX594kR0ZERaX5KJSH1+eOcweHxQnA/7tooZvRkMAz6/IindNDu8sQZyTWbH29xw4LYcc2MVLDU5deV4L1wZkkHab1ZDngnHo689cNEXPUY+5CX4/gzpBu9E10Pn+rzsKCxQAvqsTKSAAvT2SvGC2y1fpNWrNVauZFauCfb0hLlyxc/du/H0blqaRmNjGvPnp1FUpKJSxcPcvx/k8889RCIGFRV2XnopOyXzeceKKMiaaKpE1O2GDz6Fqgrxxk2GIQ+8+yX4A1BVCi9vHJ82ToRwBD44J5W5uZnw+mpINyFaIC0tx1vluvZKI1SZuHzqBnzYDp1+cSx6w0RRkW7Ahy7oDEGBFd4oEP/dRLgd0TkcCjPidvNnpcVKQJ+ViRZQkJTu0aMGt27J84oK+bJmZc0+EQUYGdG5eTPAtWt+PJ54bX1BgZX589OYO1etlSrG09ER4pNPhgmFDEpKbLz8cnZKPiMTJaLNLfDZFxKhLVscb20xS88AfHBUWmaSNZ8fCcLbp8Hjh/I82LvSvCDHKnMdVnhjAeSZKJoeCcNvWsXury4LXihL/N/mi8CvB2DEgLlpsPMpPsSP4qtQmKDbzZ6iQiWgz8rzENAYt27J2ks4DGlpkjaqrZ2dIgpy8WpvD3HzZoDm5hCRiHz8LBaoqXEwf34a1dX2lKx7KaY/vb1hDhwYxu/Xyc+3snevuBgli1T+Gly7Js937NCYNy/5z9zNW/DlMdlfs1LcjJKhtQs+OSlVusvmxt2LzDDogXfPQjAM88phxyJzx4no8OEt6PJAbjq8MR/STCSSevzwXlt06koBrCxI/Bid0cktBrAtWzx0E8EwDIZVEVFiPE8BBZkWcfCgQV+fPF+yRCp1Z/s6YCCgc/t2kJs3x6+VpqdbqK930NDgoKzMNitT34o4g4NhPvxwGJ9PJyfHyt692WRnm2gkfICJEtEr18SIHmDTOlhiUqhi3GqBL87I/sZlsLTR/LHa+sXqzzCSa28ZCUlRkScoadyXGxNvKQG4MRT3yn21QuaKJsoFL5zyghV4swAKEhRz1caSIM9bQEGauk+fNrh0SZ7n58sXtrhYiQPAwECYmzcD3LoVxO+Pp3gzMizMmeNgzhwH5eU2FZnOUoaHI3zwgZvhYZ26Ogcvvmiif+ERPCiiu3aZ7xMdy9kLsoEYLcxLQvRADOdjk1teWA9zKs0f63o7fHVD9vcsgXqTxUD9Pnj3JoR1WFYKG6rMHeerHrjuFvP5b1SDM8E+U8OAj4egNQi5VngrH+wJJClm3TizI0eOsH//fioqKtA0jXfeeWeyT+mpWK0y1eXllzUyMmBwEN55x+DUKX00jTmbKSiwsXFjFv/+3+fx6qvZzJuXhsOhMTKic+2anw8/dPPTn7o4csRDS0twxo2UUzyZrCzLaMamoCD56DOGpmls2WJhUTRK/OILg+bm5D9bq1fImDOQlO791uSOt2K+zA8FOHRG1kfNsrASllbL/hfXoNfkCLTCzLhn7qVuuG3ynDYVQXEaBHT4vEsKhBJB02BnDmRZYCgCxzzmzsMM01JAvV4vy5cv58c//vFkn0rC1NRo/NZvaTQ2yp3ThQvw618b9PQoQQCwWDSqqmTg8u//fj6vvprNggVSYOT369y4EeDjj4f5138d5MABN1ev+hkeVqb2M5n+/jBHj3pxuSKkpWksW5Yaq7+xbN6sMW9etO3jc4PW1uS/jxvWSuRpGPD5YTGkT4ZNy6CmTAwXPj4B7iSEYsNcqCmS9cxPLoHXb+44c/JhZdSg4ch9iUoTxWqRyS2O6Pizk32JHyPdArtzZPpLkx+aRhI/hhmmfQpX0zTefvtt3njjjWf+nclI4T6K5mYpMBoZkbuoZctgzRq1NvoodN2gszPMvXtBWlqC4yp5QUaw1dTYqalxUFpqU+/hNMftjnDnTpDbtwMMDsZvkDZtymLJktQLKEg69+BBg7t3xZP2lVc0KiqS+xzpulTm3m8FhwNeewUKTJoZAITC8P4R6HOJ0cIbOyDNZEtKMAzvnoFBLxRlw2urwWYiuDcM6Q9tc8s4tLcWmCsquu+FT6Jm+nvKoN6EycI5L5zxgg3xy32W/tBZvQY6nQUUIBAwOH483u6SlydroyUlSgCexMBAmNbWEC0tIbq6QuOmYNhsGqWlNioq7FRU2CguVmun04GREX1UNHt64kVlVqtGba2duXPTqK01qRbPiK4bfPaZwf37YLPB3r3Je1uHw/DRZxKBZmbCG6+KyYpZxhotlBXC3i0i+GZw++CdM+APQX0J7DFZ5RsIw29uwHAAqqNFRWZq/071wQWXeZMFI9of2pFAf6gS0KcIaCAQIBAIjD53u91UV1dPCQGNcf++RKM+n3zwliyRaDQVzeMznWBQp61NxLS1NcTIyPjo1GbTKCuLCaqdoiKrEtQpgNer09UVors7TFdXmP7+8OiNkKbJ8O3GxjTq6uw4HM9vtSkSMfjkE4O2Noka9+3TKCpK7vMSCMB7B2DQBXm5EommJxFIDwyJ5V8wlHyPaJcLPjgv0XIylblji4pWlcOaisSPMdZkocAhJgu2BP/0vgj8agD8BizKgC1PqTdTAvoUAf3BD37AD3/4w4den0oCChKNnjhh0BQ1gHY6ZW1mNveNmmFwMExHR5iOjhCdneFxVb0gEU1RkZWSEhulpTZKSmw4nakrTFE8jK4buFwRurrC0S30UBoeoKTERmNjGvX1jpT0e5olHDY4cMCgs1P6t/fv1ygoSO576PXCuwfA44GSYtj3kkS5ZmnvgQPHRfhWzoe1i80f60Y7HIlW5r64DOqKzR3n9gAcuhc9TgPU5SV+DF8Yft0KIxGYlw07TFQJtwXgo6hf7gs5MOcJNytKQGdABDqW1lZxShkeludz5sCmTbPXxSgZDMNgcDAyRlBDo0b3Y8nIsFBSImJaWGiloMCqRNUk4bC85/39Yfr6IvT1henvjzxUba5pUFhoo6xMttJSO1lZU6euMRQy+PBDg54eyMgQEc3LS+476BqCdz+SiLSmCl7cZd4RCOBms8wSBdi+CubXmT/WsZtwtQ3sVnh9DRSYTDMfbxXLP7sV3jTpVNThgw87ogYJxbDAhMvQKQ9c8IFDg28UwONah5WATuM10McRDhucO2dwMdr0bLNJSnfJElT6MQkMw2BoSKenJzy6DQyER316x+JwaOTnW8nPt1JQYKOgQPYzMqbORX4yCYcNhoYiDA1FcLkiuFw6/f1hXK4Ij7qq2O0aJSUxwbRTUmKb8ksUgYDBBx8Y9PfLvN/XX9dwOpM75+4e+OATqaZdMC9539wz1+DcjejElS3mh3HrOnx0AToGIScD3lwrBvQJH8eAD5rEqSg/Q5yK7CbuRS8Mwql+sGoyhLsgwak0ugHvD0J3GEps8Npj5ofOOgH1eDzcvn0bgJUrV/K//tf/YufOnRQUFFBTU/PU358OAhpjYEDWRrujJfD5+ZLWTbY6UBEnHDbo6xMx7e0NMzAgovAoUQUR1uxsKzk5FnJyxj9mZVlm1A2O36/j8cS3sYL5qBRsjPR0C4WFVoqKbBQVyWNOjmVaukr5/QbvvSfzfvPy4LXXNNLTk/t33G+FTw/JzXEqLP8OnoI7bTL67I3tkGMyevQH4e0zMDwCVQXwygpza6u+EPzmujw2FsCuOYkfwzDg405o9Ynp/FvVia+HeqLroUEDVmfJ9iCzTkAPHz7Mzp07H3r9W9/6Fj/5yU+e+vvTSUBBoqamJvj6awN/tF+roUHsAJO9G1Y8mtia3eBghIGB+KPb/eSeU02DzEzL6JaVZSEzUxv3msMhc2IdDm1SBCUSMfD7Dfx+ffRxZEQevV4Rytjj0wwr0tI0cnOt5OXJVlBgpbDQNqVSsanA4zF4910DrxeKi6WwKNno+doNOHpS9pN1KwqH4f2voHcQ8nNkjqjDRPQI0D8snrnhCKyohXUmz6vLI5GobsDmalhckvgx/BH4dYuYzs/Phu0m1kNv++GQW3pEX8uH0gfel1knoMky3QQ0RiAgdoAx2zGrVXpHV6xQ1brPi0jEYHhYx+2O4HaPfxweTtxVyuHQooJqGd23WiVNb7VKwVPs0WKJr5cZxvgt9lokYhAKGYTDBqEQ0Udj9DEQkMdEyMiw4HTKlpNjJTfXMiqYs2mCjsslkajfD5WV8PLLyfdsnzoLFy7L3/Xl3VCVhEWfzw+/OSSPNWXw0kbzlbl3uuHgFdlPxu7vcjecaJPU6WvzoeQREeDT6ByBD9plPXRXKTSacHH8YghuBSDbIuuhY4u6lYAmyHQV0Bj9/VKt29EhzzMzYe1acVKZjimymYJhGPh8Bj6fPm7zevXR10dGdAIBY9KtCDVN0qzp6RoZGfKYni7RstNpHRXMsRZ6CujpkTXRcBjq62H37uSyCIYBX3wFt++C3Q6vv5qc0ULvoESi4XDy01u+vg0X74u5wuurodCk/fDnd+HuIDgd8NZCSDdReXx2QDa7Bt+ogZwEo+ugLqPPhvWHR58pAU2Q2BvW2TlEWdn0E9AYzc0GJ0/K4G6AwkLYuFGtj04HdN0gGJSIMBAwCAb10egwEpFIMvao6/Hnum6gaRqaxkMbSKRqs0lGwmaTwh3Zly0tTdbvJit9PBNob5cWF12HRYtgy5bkovBIBA58Dh2dUqj05l65KTbL3Tb4PDoNZtsqWFBn7jiGAQcuQNsAZGfAm2vMDeIORmQ91B2Amlx4qSHxyNgwJArt9ENRGrxeKRaAidAVhPddEsnuzoGGaHWwEtAEib1h/59/GWL/KzmUmUxPTAUiEYOrV+HcOYNgUF6rq5P10dxcdYFUKCaCu3cNPv9cLp2rVsGaNcmJaCAg7S2uISgqhP0vS0RqlrPXZUu2MjcQgt+cTr6oqN8H79wU7911lbCiLPFjeMOyHurXYWkubDTRq3rGA+eirS3fLACndRZOY0kVvhEpJ798bbLPxDxWq8ayZRq/8zsaixbJh7u5GX7xC4OvvtLxeGbd/ZFCMeHU12ts3SpKcu4cXL2a3PcsLU3WQNPToa8fDn7JY6vAn4VVC6C+MurF+zUMe02elx1eXCpp3LYBOHvP3HEKM6WQCOB0B3QOJ36MLFvcVOHykHjnJsqqLGlpCRpSWJRs+DirBbS+Tj5gJ06J4XMsgpuOpKfLWKZvflOjpkY+GNevw7/9m8Hx47IGp1AoUsfChRpr1oiInj5tkGwyLydHRNRqhZY2OHna/LE0DXashqI88Afgk5NiRG+GwmzYtkD2z92DFhPTUgAWFMG8Qrk2HbwHfhPnU5Ml0SfAl93iWpQIFg125chaalcILpqYHjPueMn9+vRm+2aZGG+xwL378Jv35e5vOpOfr/HyyxZee02jvFxuEK5cESE9eVLaFhQKRWpYuVJMToJBmfGbLCXFsHOr7F+5Lq0uZrHZpBI3I128c788a/5YjWWwODow+4trktI1w5YacSbyheCLe+YiwHWFsg7q1+Fwd+LHyLHBpmif7BkvDIQSP4cYs1pAAZYsEmNnpxPcw7IOcf3mZJ9V8pSVaezfb2HvXo2SEqnKu3QJ/u//NTh9WgmpQpEKNE2+XwA9Pak5Zn0drF0l+8e+hrZ288fKyoAXokHC3Xa42GT+WBvnQkmOrIt+dlnWMxPFZoE99VIA1OqWQdyJYrVIO4tNg7YRuOxK/BjzM6DOATpwxEQ6OcasF1CQu75v7IfaaqmI++oEHDoyvVO6MSorNd54w8LLL2sUFUEoBOfPw89+JhGpSu0qFMkRF9DUfZdWLhszjPtLmeJilrIiGcYNcOoqtJkc7G2xyLizdDv0DcNxk2JckBFfDz3VAd0mBoPnOWBTtDDqVD/0mRgIvjUHMjQYerI3yhNRAholLQ1e2i1T5DVN+rJ+8z709E72maWGmhqNt96y8OKLGoWF4yPSo0d1hoeVkCoUZojN7k1VBBpj60YoK5Ub+Y8PMupCZoZF9dLOYhhw8DS4TYgWgDMddkWnvlxvh1ud5o6zoEgs/mLroQET66ELcqEuS6LIQ90yRi0RMiywOxf2J9F3qwT0AZYtlpRudrakdN87IE4hM6XZp65O4xvfkIi0tFQi7mvXZI308GEdl2uG/EMViudEabQydGBAekRThdUKL+6EnGwYHoZPv0iuMnfzcigpgEAQPv1abqLNUFUIq6Petl/dhAGTYrylBnLSwBOEL++bO8a2Esi0gisEJ00UN1U4oCCJkXJKQB9BaYmkdBvmyAf21Fn48FOZ5zdTqKnReP11C/v3a1RVyQ1CU5O0v3z2mZ7SdJRCMZPJyNBYHI3KvvoqtS5T6enw8h4Z7N3VLctLZrFa4YX1Y4qKzpk/1qo50hcajsh6aNCEGDussh5q0aDZJSPQEiXdCjujNzDX3NDynK/RSkAfg8MBu7eLybPNJi4hv3oPmlsm+8xSS3m5xquvWnjzTY26Onnt3j145x2Dd9/VuXPHQNeVmCoUT2LtWo3MTHC74cKF1H5f8nJh9zZZWrp5K7m+9bFFRXfa4NItc8fRNEnlZqXBkA+OXDd3nKJM2BCt7j3ZJoYLiVKZCcvyZP9wN4yYjKzNoAT0KcxrhG+8Ju4ggYCMIPry2MwoMBpLcbHGiy9a+K3f0pg/X75g3d1w8KDBz34ms0lHRpSQKhQP0t0tIwd90Yt//wS0wlVXSX0GSH9oe4f5Y5UVwcaoR+7XV6DDZJ1HugNeWBqt8O2RYdxmWFICtbkyteXgvcTXMgHWFkCBQ1pbjqR4LfpJzGorv0Ssm3QdzpyX9VCQtpftm6CyYgJPdBLx+QyuX4dr1wxGoj1fFgs0NsLixRrFxcomUDF7CYUMbt+W78dYwWxshA0bNDIzJ+b7cfgoNN2Wosc394r5glm+OAO3WiSl+9ZOiU7NcLkFTtyS68Oba8yZzvvD8Ktr0h+6qFjWRxNlIAC/aZWiou0lMP8Z3xvlhZsgybxhXd3wxVFZ1AdYOB/Wr5aU70wkEjG4dw8uXzboHXOnWlQECxZoNDbKSC6FYjYwMGBw7ZrBrVvSEgayttjQAEuWaBQVTex3IRKB9z+W7oCCfJneYtYzNxyGd76U9dDSQti/NT4uL1E+uQj3+yA3E95aC3YThTltbvgomlJ+uVGM5xPl4iB83S9OQ9+sgexneG+UgCZIsuPMQiH4+mzcJcTphG0bk5vlNx3o6TG4csXg7t14NaDNJhePhQu10XJ+hWImEQ4bNDdLtNnVFX89NxcWLZIxgmlpz++z7/XC2x+CzwdzauGFneaP5fbAb76AYAiWNMCm5eaO4w/Cr0+BNwDzymHHInPHOdEKl3tk5Nk3F0FmgjcHhgHvt0OXH8rTYV/l083vlYAmSKrmgXZ0wpfH49HognmwYc3MjUZj+P0GTU1w44aByxV/vaBAotK5c5/vBUWhmAi6uw1u3pQbxljNg6bJtKNFizQqKyfvM97dI5GorsOalbDKpPAB3O+ET6LVvXvWQX2VueN0ueD9cyJiOxaJkCZKRIe3b8DACFTlwCuNiU9/cYfgVy0QNmBDISx7Sp+nEtAESeVA7VAITp8T30qYPdFojK4ug+vX5SITiTp6WK1QWwuNjWJsb7EoMVVMD7xeSc82NY2/OczOhvnzNRYsYMLWNxPlRhMcOS77L+0WJzWznLoKF25K6vWtXZDrNHecc/fgzF2Z3vLWWsjLSvwYgyPwmxsippuqpcgoUW4MwZFeqZJ9qxoK0h7/s0pAEySVAhqjs0uqc93RaHT+XIlG057wh5tJBAJSVHH9usHAQPz1tDRJ8c6dq1FaOjUuPArFWEIhg/v34dYtg7a2uGmKzQb19TBvngxmmIoDyI+dhKs3ZB30zX3S8mIGXYcPj0JnHxTkwhvb5d+fKIYBH56HjkEocMKbaxMffA1wrReOtkiP6FsLxf4vUT7ugBYfFDrgzWo51qNQApogEyGgIIvyp87Go9H0dNi4FuY2pOx/MS3o7ze4dUsE1TemrysnR6oUGxo08vOn3sVIMXsIhw1aWmQwdkvLeFeesjKJNuvrwW6f2p9TXReTl84uyM2Rljszwgfg88OvD8GIH+bVyjg0U8cJwK++Bn8IFlXClgXmjvPxbWgZgvwMeGtB4kLsC0sq16/DynxYW/jon1MCmiATJaAxYo4hMQPo8jLYsgHy81L+v5rSGIZBe7vc2d+7N/4ilZsLc+aItWBx8dS8u1fMLEKhuGi2to7/PMZu7ubN08jJmV6fRb9fTF58Pti4DpaaLOAB6Qn98Gh0HXO1CKkZWvvhwAXZf2kZ1BYnfoyREPzqujwuLYGNJlLUdz3weRdowJtVUJT+8M8oAU2QiRZQkDvDS1fh3EX5olos8sFetdx82fl0JpYmu31b0mRjPT2zsqQwo65OUmVqzVSRKnw+Ec2WFhHNyJjJGzk5kqKtr5/49pOJJrYempEBv/sN81EowLkbcOaaHOPNHZBv8hJ58hZcapHpLd9YB1mPEK+n0TIkkSjAvnlQYaLH9GAX3PGI0cJbj0jlKgFNkOchoDE8Hjh+Km4B6HTKEO86E43CM4VgUC5m9+7JY6yfDmTNtKYGqqvFozc9fXpf2BTPn/5+uVm7f3987zLMLNEci67DL96WGoz1a2D5EvPHMgz46Bi094h4vrnDnCBHdHj3jIw+q8iHvSsTr6gFOHIfbvSB0yGtLQ5rYr/vj8Av7ksqd1U+rHkglasENEGep4DGuN8qQhpreamulHSL2UX/mUIkImne5mbptRs7sknToLhYjO+rq8W8QaV6FQ8SDBp0dkJrqwjng0MfiouhtlajthYKC2fu56fptjgVpadLFJpMpmvED7+KrocuqINtq8wdx+WF35wW0/n1jbDcREo4FJFU7nBAxqBtM3GMWCrXghQUFY4p7lQCmiCTIaAgqdzzl+DiFbljtFhg8QJYvWLm944+C7pu0N0dT7eNreYFuTBUV0NVlaR6nc6ZezFUPB5dN+jpgbY2ufnq6Rk/btBmg6oqufGqqZk6bScTja7DL9+BIXfyvaEgEeiHR2V/11poNNkmc6MdjtyQ690bq6HIxCW3cxjejw7wNutS9Fkn3PM+XJWrBDRBJktAYwy5xRD6fqs8T0+HtSvFiEEFWHG8XhHS1lZZNx2b6gVJx1VUQEWFRkXF7LlQzjYMQ1qjOjtFNDs7H/4s5OZCZaWIZkUF2Gyz87Nw+y4cOiI35P+vbyZ/Y37mmqyJ2mzil5tnYg0S4NNL0NwrfaFvrZU+0UQ52QaXusWd6JuLxK0oEUbC8MtoVe6aAlhVIK8rAU2QyRbQGG3tktZ1DcnzgnxZH60w4eAx04lFp62tBh0d0Nv78JDz3FwR1LIyGRY+3aopFUI4LBFmV5cYdXR3PyyY6ekimJWVslaushGCYcCv3pUOgFXLJRJN9ngfHpXq3MI86Q+1mhA/fxB+dUpaXBZXweb5iR8josOvr4PLD/X5Mks0UW4Pw6Hu8QYLSkATZKoIKEja5doNOHMhbhdWVyOFALmTe2pTmmBQfEk7OkRQ+/sfFtT0dFn/KinRKCmRfVWUNLUwDHH86euD3l4Ry/7+8VXaIOt5paVxwSwoUOvhj+PSVclw2e3w+79jTvDG4vPDrw6CP5CcX25bP3x0QfZfXg41RYkfo9cL79yU7/ruOdBQkPgxPu2EZi8UpcEbVeAZVgKaEFNJQGP4/SKi16MfDk0TN6PVy6XNQ/FkYoUknZ1yEe7tffgiDJL2LSqCggKNwkIoLFTRy/NC1w0GB0Us+/oM+vpELMf2Y8bIzITycskmlJUpwXwWwuG4rahhSMX/v3sjuZaWGC1d8HHUNvCVTVBdZu44x5vgSitkpsE318lM0UQ52wFnOyWF+1uLICPBYilfNJUb0GF9IcyxKgFNiKkooDEGXfD1GWiJDqe1WqXQaMVSiagUz0YkIutmPT0yRaanB4aGHv2zDodcoEVQNfLyJB2ckaEu2GbQdQO3GwYHY5sIp8v16Jsam01uaoqKJFtQWgrZ2eq9T4S2dvjqZLzKv7FeloNSec04fhGu3JH5od/cJY+JEo5IVa7LC/UlsGdp4sfQDTGc7/dBXR68aMLp7aYbvuwBqwYv5rqpKVYC+sxMZQGN0dUNp8+LRRdIOmb5EjFjmI1GDKkgEJC+wIEB6RXs73/8RR1EWGNimpsbF9bsbDUDVdcNPB5wu+Wi7XaLaLpccqPypPc0JpZFRRpFRfKequjSHH39cPZCvCDR6YStG6Da5ESVJxGJwNuHZX5oTRm8vMnccfrc8M5Z+YzsWgyNJqLZfp+IqG7ArjnQaCKV+2E7tI9AbsjN7yxSAvrMTAcBjdHaJkLaF516n54OK5fBovnJr20oRAhcLkkl9vdL1Do0FL+TfxwOh1ys4ps2up+RIdt0FVldN/D7pZ/S54s9Gni98r4MD4tByJOuHHY75OfLDUh+vkZ+vjx3OpVYpoKBQRHOe/fluabJNWHd6om9wR4YEhGNRGDTMljSaO44saktDhv81npzLkWxVG5aNJWb6OzQ4ZCkcofdbr67SgnoMzOdBBTkQnXvvqxvDLnltcxMWLFEWl9SscahGE84LBHV0JBsLpcxuj/W7OFJWK1xMc3IkJufmLDa7SLCsS323G6XfjmrVTYztoa6bhAO88gtGIRAQDa/3xjdj20+H4yMPFkcx/77cnIkIs/JkarnvDwRTbWuPDG4hkQ479yLv9ZYLxW3z8uU5eodOHZR/v5v7pDpLYmi6/DeWehxQ2UBvLoi8Ra+VKRybwyBb9jN6moloM/MdBPQGLoubiPnLkkEACKky5fAQiWkz41wWNKX8W3885GRh9suksFiiYtq7CJjGHGRi+0bhnxGUvGN1jT5bGVmShGb7EuULWIpNwMqmnw+uN1w9qL0ecb+vvV1YsIyGUMqPj4uhUXJjD4b61K0eb60tyTK2FSu2apc1caSINNVQGNEIiKk5y/HhTQjQ4R00XwlpFOBcFjSoD4f4x5HRgxCIYkEg0Ee2g+FUiOAICJot8vnwWYTAbbbJRJOS5MtPV0b3U9Lk89RZqYSx6nCoAsuXB4vnHU1IpyFJsQiVYy1+kumteVqGxy7mdwA7rFVuf9uceIGC0pAEyT2hv3ykyFe3pKDM3Oyz8gcug43b40X0vR0WLZYhFTZA05PDMMgEpG/r67LDdPY55boXERNi0eksX2LJS6YaqrN9KW7R4QzVhwE4p+9ZiUUm+ifnAhau+BAtLXlrV1QlJf4MQxDxp61DUBJDry+xlwq9zfXYWAEGvJhd4IGC0pAEyT2hv3o/ydv2IalsHDOZJ+VeWKp3fOX48UvdjssmCtVu07n5J6fQqF4OoYBre0inF3d8dfraqRwcKoI51g++EpcipKZHerxywDuYBg2zIVlJiZVjTVYSNQrNxkBndXJvtICcPvhq/Nwtx22rYTsaWhaYLFIMdG8RikuuHBZUj+Xr0lTdX2dRKVT8QuoUMx2IhH53l68It9bkO/0vEb53k7liU3Z0eyd9xkL6x6FMx02NIrh/Ok7UFsEuQlmBYuzZOj2pW442iJVufbn0KUwqyNQl2uIlt4cTl+TCkWbDTYskWh0ui//tLbBpWvQ3hF/rbxMvpA1VdP/36dQTHdGRuDaTdlGRuQ1u12WX5YukrXoqc7pq3D+JixugM1JTn/56LykcsvyYP+qxK9RYR1+eU3Gni0pgU3POD1GRaAm0TRY2gi1ZXD4LHT1w9ELEo1uXzU9o9EY1VWy9Q+IN+ade2LK0Nkld7SLF8DcBrVOqlA8b/r6JTN0+27ccCIzE5YsnH61C1kZ8ugdSf5Y2xbCL09Cl0uKi5YkOD7NZoGtNfDRLbjSI+YKJRN8DZ/VEejYOw7DkP6mU2Oi0TULpcIsVrQxnfF65Ut7vSluWm+3i4guXjA5pfAKxWxB16G5Rb6DY9c3S4ol2pxTOz2vM80d8OlJKCmAN3Ykf7xrbXA0WpX7zXWQYyIKP9wMTf1QkAFvLYzP/XwcKgJNAZomrho1ZXDkvCyMn7wMt1tlGruZCrOpRFaWTHhZtVwKjq7ekKbsazdkqygXIa2tnp5fZIViKuLzwY1bssUq5S0WqUtYslAEdDqTyggUYGEl3O2BjkH48jrsM5HK3VAFLUNSlXuhC1ZN4HhIFYE+4o7DMODmfRHQYCie6l2zcGb1WHZ0yh3x/dZ4j1lWlpgyzG9UU2AUCjMYhpi7X28a/91KT5cU7aL502N981nwjsD/OSDXyD9+IzW1FW6fzA4NR2DLfFhkwmDh9gAcuifR5zcXQd4TrAJVG0uCPOsb5vPDiUtwJzoZxZkpC+W1M2zgtccjX/brTXGbOk2DqgqpBKyrUb67CsXT8Hjg5m3ZYtEmQFmpiOac2pn1PQqG4Mw1mdAC8Pt7IT0tNce+0iqjz2xW+HcbpFI3UQ7cglY3lDth/xMGeCsBTZBE37CWLiku8vjkeW25COl0NWB4HJGIeO5euzl+nSYtDRrmSFSqWmEUijjhsKxtNt2RqDOGwwHzGmDh/JlZX3CrBU5eEScigMZq2LU2dcc3DHj/nBQUVRfCKysSP4YnCL+4KtW522th/mOuXUpAE8TMGxYOw7mbcOmWFATYbLBiHixrnFlp3Rhut9xJN92RAqQY+Xky6LuhTqV4FbMTw5D2sFt35YZz7EDwinIxMKmrmZnXhYEhCSa6otOhcp0STFSVpv7/5fJKKjeZsWeXuuFk25Nt/pSAJkgyb9igWz5AnX3yPDtLekfnVKb+PKcCsYtF0x25WEQi8f9WXgaNcyQ1pYZ9K2Y6/QNw6w7cvifFQTFysqWafW69mOzPRGLp2qtRT16bDVbNl9qQiUxLn28Wc4V0u6Ry0xNs8Rlr87egCLY9wi1JCWiCpMJM/k6bFBnFqs8qS2Q+Xv4M/QKBtL/cvisXkLEpXotF1ksb5sidtxr4rZgpDLrgbrNsMZcgiC9rzK2H0pLJObfngWFA031p74ula+dUwsalz2cJS9dlYsuAB+aVw45FiR+jywPv3ZT91+ZD2QPWpkpAEyRV01jCYbjQBBdvSWSmabC4HlYvhLRp1AxtBo8H7jSLoPYPxF+32cTpaE6tmF9Pp6ZwhQIeL5oWi7R5zW2Qz/hMb/fq7JMiyj6XPM/LliBhItK1T6JnCN45I/t7V8r80EQ5ch9u9D26N1QJaIKkepzZsFcW1O9FiwjS02DtIpg/TZujE8U1JEJ651584DfEI9O6GtlUmlcxVRkYjIumayj+euwzXF8nn+HZcEM47IWvr4gjG4DDDiuj6drJup4db5LK3JwM+OZ6qc5NBH9YCor8YekTXTbmJkAJaIJM1DzQ9h44fknWSUHu2DYsFXOG2UJfv1yE7t0fL6aaJuX8c2rkLj47e9JOUaEgEhFby/utcL9tfNvJbBRNkHXOC03xQklNE1/w1QsgY5JvfkNh+MVJ8AZgRS2sa0z8GDf74Mv7Yvn37xaDM/p3VQKaIBM5UFvX4dpdqdj1B+S18iJYv0TsrmYTgy4R0uYWEdax5OXKRaq6UoqRZmLFomJq4fXKuLCWNmjrGF89a7XGRbO2evaIJsg16/o9OHsjfs2qLJF1zoIpNAnmZoe4E9ms8P/ekfjvGwa83yRronPy4IUGeV0JaIJMpIDGCIZkSsGVO/HK1TmVktrNm4XR1/AwNLeKmHZ1x91ZQC5e5WUiptWVU3t8k2L6EApBZ7dUkbd3Spp2LJmZspZZWw2V5bPvJs4wxKr07A1wRyPwvGy52Z+KZjGfXoLmXqgrhheXmTvGwAj8+rr82/fOhcocJaAJ8zwENIbHB2evQ1OL/NE0DRbUSaFR5ixdEwwG5YLW2i6RwNj0GUh/aUWZbOWlM7c1QJFadB16euWz1d4p+7FpJzFKikU0a6qgqHByznMq0NwBp6/Fl5sy0iVVu6BuatZtxAqJNE1M5vOdT/+dx3G8Vaa15KWLzZ9nWAloQjxPAY0x6IZTV+F+pzy3WmHRHDFjmOz1hcnGNSRi2tou61Jje00BnM7xgqrWTxUgKdjePujqkaxGV49EnWPJzpbosqpCPj+zvZCtvUeEsydaOe+wyzVoScPUjsA/OCcG82ZbWcYSjMDPr8JICNZXwpxMJaAJMRkCGqOrT4Q05uRhs8mHd1lj6nwkpzPhMHT3QEeXpN8eFUVkZUFpsfTflRZLJDEV75oVqSUQgO5eucnq6hHxfPCzkZYmghnbVPZC6BmQ605Hrzy32WBpAyybO/Vb7toH4MPz8h3/7Q2QnZH8MZv6ZeyZzQKvVLupKFYC+sxMpoDGaOuWO8He6LqM3SZl4ksbp/4H+nkSDkt0ERPUR100rVbx6C0pElEtKVI2g9OdcFj6i3v7oLdfHse2l8TIzISykuhWCoUFqZkIMlPoGYBzN8TPG0SEFs2RtpTpkPkKR+C9s9A3LAO2N81LzXENA95rgm4PlNncvL5CCegzMxUENMb9TjhzHfpd8txhl7vCJQ2yrxhPKCQX0+5eiVR7+uITZMaSni6RaVFB/FFFI1OTcFgqtvsH4mI5MPjwjRJAbo4UnMVEU/1NH013vxQHtUUdwzQN5tVI7cV0GYIx4IGDV2DQK5W3v7sJMlIYXPT54O0b4Bt2890dSkCfmakkoCB3Q80d8oEfiN5lO+wioksaVGr3aQy5RUxjojroGl/lG8PhkAilIE9M8fNy5TEjBSkhxdMxDHAPizgOuuSxf2B8v/BYMjKguFCyC7EMw2xfw3wanX0Scbb3yPOYcK6cDzlJFN48TwwDrrfDiVsQ0SEzTczkK/JT//861gKuITf7ls1CAf3xj3/M//yf/5Ouri6WL1/O3/zN37Bu3bqn/t5UE9AYhiHuH2evg2tYXrPZYEGtRKXT5c5xsolFNH390DcgjwODDxcnxUhLi4tpXq6Yg8e2qVxYMVXx+UQoXUPyOOSOb4/7G6Sny83NqGAWSvGY4tlo7ZK2uVhthcUiwrli3vQRTpBh2kduSMEQQFUB7Fyc2shzLIYBw7OxCvfnP/85v//7v88//MM/sH79en70ox/xy1/+kps3b1JS8mR356kqoDEMQ2wBLzTFfSg1TWbuLZ87tZqbpwu6Lhf0vn4R10FX/AL/JDIyREiznXFRzcqUNdbMjNnVcA/y2RwZAY93zOaJ7w+5H66EHYvNJjcpBfnRLU+EU2UBEkfXZajFxVvxzJXFIq0oy+fKpKjpgq7D5VY4c1eiTpsV1tbLuudEr2nPyj7Q9evXs3btWv72b/8WAF3Xqa6u5k//9E/5b//tvz3xd6e6gI6lvUeENJaSAWlyXjZXHI4UyREOy0XfNTReVIc9UvX5NOx2EdTMzKiwZko0lZ425jG6P1XFNhKR3txAUNaTR/wikr6R+H7s0eN99NrkWDRNosfcHMjNjj5Gt+xsVeSTLMGQOAdduROfBmWzwcI6WD5v+vWXD3jEYag3msqvLICt8yHnOWXcktGDaZmgCgaDnD17lr/4i78Yfc1isbBnzx5OnDgxiWeWeipLZOtzwYWbkuK93ylbUZ5U7TbMgskQE4XNJhFQ4SNsFgMBEVL3cFxUhz1iCef1ieiEQiK6j6oQfRCLRUTUYRfhtdul+nrsvs0mVcUWS3yzjtnXtPj6rmE8vK/rcgcfDoswhsPjn4dCIpTBoFyIA4HHp1Ufh6bJDYMzS7ZsZ3TfGY/WJ3JG5GzF4xPRvH5PvGFBxHJJg3jWTrfq/XAEzt6DS1GTGYcNNs6F+RWTfWbPzrQU0L6+PiKRCKWl4+fqlJaWcuPGjYd+PhAIEBgTTrjdj6lamMIU5cGe9TDkEbPnW60iql+ckckJi+qlPF0VHKWOtDTZHudYEwpJlBYTVK9P1v/8gejmj0d1oZCIm9//6KrhqYDDIZ+fjAxpcYg9ZmbE92Ppa3XD9vzoGRDhvNMWv2HKz5He8cbq6Xmz0tIHR2+CJ/pdqCuGLfOlYGg6MS0FNFH+8i//kh/+8IcPvd7UAaunWUop1wlbV4qn7vVmuHoHfH6ZFn/+Jsytlqh0Jg/2nirY7ZBrl9Tk04hE4oIaCkW3sDwGg3I3HgxKpKhHI8nYFonE92PrLRrxz62mxfctlngUa43ux57brHLOsSg4LS3+aLdPr+/BTCcSkWzTlTvxXnGAimJZ36yephOevH44fgvuRZeknOmweR7UFk/ueZllWgpoUVERVquV7u7uca93d3dTVvbwJ+sv/uIv+N73vjf63O12U11dzbEmaBuWO5+iaSY46WlSmr58rnzRLt2SiPRGs2wVxTLcu7ZcRQtTAatVIjdl8KB4Et4RmeZ0vTk+GcVikUhzSYNkoqYjhgFX2+D0HQhF5GZtaTWsqU98tudUYloKqMPhYPXq1Rw8eJA33ngDkCKigwcP8p3vfOehn09LSyMt7eHcgN0KPW54+wwsqpQ/Zto0My+Ifbkaq8Um8PId6Snt6JUtM12q8hbUqTYYhWKq0tErwnmvI56mzcqQpZkFtdPDNehx9AxJurYvWvFekgPbFkLBNGqveRzTUkABvve97/Gtb32LNWvWsG7dOn70ox/h9Xr5wz/8w2c+xptr4VoP3OmWu6M73bCuQRaxp2M6q6xINo9PCg1u3Jf07rkbkt6tLpVig+pSFZUqFJONzw9N9+V7GhsnBlJdv7ge6iqm9/fUF4Cvb8OtqI2gwwbrG2HBNL2+Popp28YC8Ld/+7ejRgorVqzgr//6r1m/fv1Tf+/BsuX2ATjWBC6v/PfiHMnLl0zzfktdlzva6/fiJtIgd7bzamBeraypKhSK54Ouiy/tjWZoHTMX126TLNLi+unf5x3RpbL2fLOs7YNMUVnfOHGGCGaI6HCqHdIjblbNmWV9oMnwqL4fXYcrbXD2ruToARrLJCJ1TuP0SYwhjwhpU0t8bQWgtFDEtKFKee8qFBOFa1hE81ar9NTGKCuU5ZX6ypnhenWvB07ehuFof2pJjhjAT7VgZHAEDt6TAduRETd/slUJ6DPzpMZZXwBO3YGm2NxOCyyvheU1cpc43YlEpIe0qWX8HbDVCnMqREwrS2ZOikWhmCxG/NJ6cqt1fCVtRrp8z+bXQt4MmW3bPwzHm6DTJc+z0iTibJyC1cJXe+Bku0Sg6TZYXeBmSY0S0GfmWZwn+txiZhz7QGSmibXUvPKZIy4+P9xqETEdHNMam5ku6aSGKiieAANnhWKmEg5Dc6d8r9p64jeomgY1ZSKaNWXTe21zLL6A2O/d7JR/ayzgWFE79aprR0Lw5X1oiZqeVOfAjjoIjcxCK79kSMS6qbkXTt4CdzQlUZgtbhkTMRlgMukdFCG93Sq9ijFynNBYJWKqeksViofRdbHavN0mNQfhcPy/lRRIb3ZD1cwyOQmF4WKLrHXG1jnrS2DD3Km55NU6BIfvi4haLbC+EpZELdNnpRduMiT6hkV0qdI9dw+C0S9HTRGsb4D8ZyjC8frhbg/MKZmaH66xRCKS2r3TBve7xl8MCnLjYjqdjKoVilSj6xJh3m2XtrHgGAP9HKeIZmP1zCvS03WJNs/chZHojXZJjghnWd6kntojCUcLha5EjRsKMmDXHHmMoQQ0Qcy+Yf6geDdea5d0habB3DLpH32SMF68L+uqAJX5sjZQOA3WPkJhWS+93SoXi7Em4kV5UmZfVz79qwYVimchHJbvQXPnw6KZkS6FQHOrJeqciTT3ynUs1q2QkyFFlvWlT/69yaLHC4ebwRUt2lpSIpGn9YH0uRLQBEl2GovLC6fvxu2o3lr7eCejkSAcviZNwytqZaH9To/MuXth6cN/zKlKICjpqdutMrR37KcmO0uEtK4CSgtmzvqOQjHil4xMSxe0dI/PyGSmw5xKEc6ywplTG/EgHYNw6raYzgCk22HVHDGfmazvuj8sBUCxQOZBghH42WV5zLTD9lqofsyNvhLQBEnVOLOeIWgfhJV1j/+Zmx1SjLR6DiytkddGgtA9JAbK05ERv1xQmjvljnzsNI/0NCmSqC6FqpLpNyFCoeh3iVje7xQj97E4M+Vmcc4MF02ALpekamPDrW1Wsd9bXiumCJOBPwzvN0GaFV6b/+SfvdYL3R7YVA1pTzhfJaAJ8rzmgeo6tPRDaz+0D0B2hqybTjff3ScxNq11v3N8AZKmSTqrulS2oryZfcFRTE9CYTEaaY2KZmzGZoyiPLkprCmbuenZsXQPiXC2R28eLBZYWCGBwmRPS+nxwqd3JKqcWwBLSx8fhT4rs24e6HTBYpEos6ZQos6vbsr66bYn/I2S/TA8b2y26FpohdwwdPVLdNraLa0x3f2ynbkm60RVJSKmFcXTb/CvYmag61J13t4Lbd3QMzh+fd9mg8riuGhmZTz+WDOJniGp8Wjtl+cWCywohxV1U6f4cTgATgdUZMPNfqjNg5y0ybtuKgF9DlgskJUuawafXpY0yOOqdzVNUid9w7K/cBr5YVosIowVxbBhqXjytnbL1t4jqd9bLbKBNJFXlsjFqrxIpXsVE8eQRz6DbT0SbY4tAAKpnK0qEcGsKJoZrkDPSq9bOgzu98lzTYP55RJxZk+xmwdfCGpzoSYXujxwrlN6OScr6JhFH5PnS/cQFGeLqMTujhw2cDyiuTj23/uHpbeqywVF2eANyOJ9rDF5ughpDGemmNcvnBOPTmNi2ucSezPXsMw0BUmVxQS4tEAJqsIchiHZj86+6NY/3j4PxLayskREs7JYBHS20eUS4WyLpmpjXQWr6iBnik1uil0jdQM8QSjMhMYCWec82wGBCKwok9Tu80QJ6AQQCov5wpp6qCyI3x1dbxdhfPBuSdOkGfnsPXl8ZXk8Qr3bDV1DMrl9qn2oE2FsdAqyVtrZJ2m0jl654PW5ZLt0S34mP0cKNUoL5HE2XuQUTycSgf4huUHr6pfP1di1eBCrypL8uGgW50+vpZJU0tYP55pFQEHeh8ZSqazNnaLXmNjfyuWH/GhUXO6ES91wrgsWFj1/8QQloBOCpkFpLnx6SRyLSnJlQb7HLS5GeVETgrE2X7e6pOCoPE+E1GIR68D6Uvn9qTTFIBWkOeJrpyC2grEZpp19knIbdMt2/Z78TGa6mN8X58kFsChPRamzDcOQz0bvoKxd9g7KTdfYNUyQFGxZoWzlRSKe1ilmLfc8MQzp47xwX1K2INeY+eXi8z3ZN+ehCHR7ocwJtkdk2mIRaKZd1kBv9MHJNnmemzY54glKQCcEm1WcOZbXiolC9xDkZ8HCSmgolSjTZo3fVRmGtLsYhvxcaa6sR3x1A7YvnDoL+BNJzH+3sVqej/ihe0Aiiu4BuVD6/HCvXbYYOU4oyo0LqhLVmUNMLPtcEmH2D0lbyYPrlyDtUyX5IpblRfI5mG5LHhNBOCI355daYMgnr9msUluxrEZqMyabKz1wpkO8abMdkPuIc4pdK7s8cKEbMmxiijAnX8zhr/fJ7xc/Z4c0JaATSIZDhPTBCrEjN8TFIzbhpWMQ/CEpLtowV36mNBfePSN9pvPKJ+f8J5OM9PERajgsF9KYmPYNyRDi2HZ3jKg6M6EgRxySCnJky8tWF9SpTCAo2YYBd1wwB9zje4xjWK0ikCX5cuNUkq/S+w/iC4j96PV2ubYApNlhcRUsqYL0KXKTeb4TbvTDtlopDrI8Jq0eu4auKod7LlnvdEb/DVU5si76uN+dSJSAPgfGiqcvIJaA6fb4eDSLJimMuWNG/6Tbxe6vb3h2CuiD2GxQViRbjEBQLraxNF5MVD0+2Vq64j9rsYgvaUxMc53xTc1BfX54R6RwbNANLk90f/jhIp8YNpv8zYry5LGkQB7VzdCj6R+Gy61wuzue1s7OENFcUDG1RjKGdegYhjXlUJ8vFbbDAUnHZtrFpS0mnLFraGWOtLCMvaaWOmWbDKbQ2zk7yEyDV1dCYEwaymKBQHj8h2J4RFK/9dGJAdOtP/R5kOaItsGUxF8LBCVyiUUzA9FIJhiKr6k+SGb6eEHNzhR7QmeGRMKKZ0fX5ebF7YXh6OPo5pECu8eRlQGFuWO2PMjJUp/7p2EY0rt5qSXuGgRi7r60WnrRp+J76A9D/4gI4vVeONspUeVwEBryH/atDUaki2Eq/VuUgE4SaWOinqJsqCuSBf6Nc6Wh+VaXtL3Eos+p9KGZyqQ54utgY/H44sI6FI18hrwS+fiiW2ffw8ez2URIszMlNZydKYI7dptJY6qehGHASEDeS+9IdPPH9z0j8t+e5G2maSKK+dFMQJ4zvj+VoqPpQCAkk1GutcXHLWqa3HQvrZbiw6lOXjrcGpBxY1trpD2l2SUmCV+3iw0fQOcwXOyG5aVQPoUGcaiP7BTAapGm5aM34e3TIpx5mbBjofz3R0Wfrf2yTjpZnpTTDWdUAGvKxr8eDImgjt2GfTDsFVENh+P9qo/DYoGMtDGC6hAhT3NE9+3x52nR1L3dNnlVoYYhkWAwJNvofhj80SWGkYDsj4x5Hgg+WRxjWK1yo5GTNWZzxl+bzdWwqaDPDVfb4XaXjFoEuQ4srJQ1zulSdBhbw7zcIwVAtXnyfFG01e1ar6R0s9Ok/9MXAndACajiERRmw+trwO2TtYGCMTn9B8VzJAifXJK108ZS+eIUzyB/3eeJwy6FKMWPGJAeiUhkNeyTzeOTKMs3Jmr1ByRtGYvCEkHTREht1rio2qwiyBaL/H3HPVpAA8ZqmGHERS22H47IhTUSGbMffR4TS7NomqRaszKkgnN0P/o8O0tuIlTGJLVEdOkJv9oWn4oCct1YXAmNZfLZmW5sr4VfXJNrXgyLJtW4EUOEE2Tt85UMyJhi9QqzWkCnoo3+s/Rjef2QmwGDXrjRIVtRthQJNJapqDRVWK0SOT2pwlPXxwtqLFLzB+UxEBKRDYTiz2OVpYYRjwInA4tFbiAc9qhLVjRSzkiTyDndIWvAo8/TZF+J4/NjwCMtbre64tW0FoukaRdXSRZqOpObDouLoWUI7rviUWjEkFaV9DHXsqkmnjDLp7H8f08NsXN+DlXTNHrrcok5/d0xw65tVvlyLaiYmhPiFfK3Ckejw1BYtnDsMSJ33boe3aL7kehzA4lCYyI29jH2us0q4m+1yKNtzH4s0nXYZpff63QiEII73XJj3Ddm6cCZLtmmBRUzy1glosNHt6R4qLEArBpc64MlxbDyOXQgqHFmCRJ7w350eIgMpwjo+kpZwJ6O+INyh3q9Iz4tHqTXdG6ZFCJNNVNohUIRxzDErexmpzgGxdY2LRaoLRLHoKppOqxeN+BmnxQL7Zv36H7NQBiu9sLAiKx1Li0Rk4TngRLQBIm9YZ9dHaLZnzOaZ59bAGsr44vb05Eul9y53uuR3tIY5XkipHNKVIpXoZgqDHikGOh2t/hdxyhwRpdkSqeO6YEZml1wugMGo/UBW2tgYfHjf34y2vWUgCbI2DeMtBzOdMDt6EQCq0VSByvKnjzFfKoTjoiINnXFB+OCpPPqiuWLOV3vaBWK6YzXL4J5u1uMD2I4bJIxml8ORdN0WSlG5zCc6oBujzxPs8HqcqmwnQzHoMfR7IGIz01jqRLQZ+ZRdxy9Xuk76oh+oB1WEdElJY82N55OePxyl9vUNT7Fm2aX9dKGUolQVXGIQjExBMNSRXu7e7zZgcUCNYVyQ1tbPN44YDrS74NT7dAarRS2WSQdu7zs0aMcJwt/BI71wh0POPxu/nCpEtBn5kkhe8uQCGks5ZBhh5VlMi5nun+4IW7ScLdH2mFiZKZBQ1RMp0MDtkIx1QmGZT3zbo/M3Bw7MaYsT6LN+pLxpirTFXeAcZk8iwYLisS7drImpTyOO8Minn5dCu8arG521ysBfWaelvM2DPkgnO2UDwbIuui6SqkSmwnoOnS6pNrvbo982WNkZ8CcYnEzmQrTGhSK6URzr9QhPCia+Vkimo1l08fs4GkEIxJx3uiL92w2FsCaCsiZYg5dgQh82QPN0SxcgQN2lIAjaH4NdBqv8pknds/gdj/CGDVKqR1eroZb/TI+p9cDHWlQMoPesWwbrKiEZeXyZb/XCy190OOFnj6ZjhAJPv04CoUizp02uNkq+3lZYtNZVwz50X5iPQjuGfK90qOjGL0BmYqyqjzazRCIBx9ThbAO7f3SJrQsH5bngiUY1wEzseQMkoNnZ3hYFjqrq6sn+UymNt+d7BNQKBSK50R/fz+5uYmtX83KFK6u63R0dJCdnY02yZUzbreb6upqWltbE04fzHTUe/No1PvyeNR782jU+/J4hoaGqKmpYXBwkLy8vIR+d1ZGoBaLhaqqqsk+jXHk5OSoD/ZjUO/No1Hvy+NR782jUe/L47GY6OmbAXWlCoVCoVA8f5SAKhQKhUJhAiWgk0xaWhrf//73SUubYjXfUwD13jwa9b48HvXePBr1vjyeZN6bWVlEpFAoFApFsqgIVKFQKBQKEygBVSgUCoXCBEpAFQqFQqEwgRJQhUKhUChMoAR0kvnxj39MXV0d6enprF+/nlOnTk32KU06R44cYf/+/VRUVKBpGu+8885kn9KU4C//8i9Zu3Yt2dnZlJSU8MYbb3Dz5s3JPq1J5+///u9ZtmzZqEnAxo0bOXDgwGSf1pTkr/7qr9A0je9+97uTfSqTyg9+8AM0TRu3LViwIOHjKAGdRH7+85/zve99j+9///ucO3eO5cuX89JLL9HT0zPZpzapeL1eli9fzo9//OPJPpUpxZdffsm3v/1tTp48yWeffUYoFOLFF1/E6/U+/ZdnMFVVVfzVX/0VZ8+e5cyZM+zatYvXX3+dq1evTvapTSlOnz7NP/7jP7Js2bLJPpUpweLFi+ns7Bzdjh49mvhBDMWksW7dOuPb3/726PNIJGJUVFQYf/mXfzmJZzW1AIy33357sk9jStLT02MAxpdffjnZpzLlyM/PN/75n/95sk9jyjA8PGzMnTvX+Oyzz4zt27cbf/7nfz7ZpzSpfP/73zeWL1+e9HFUBDpJBINBzp49y549e0Zfs1gs7NmzhxMnTkzimSmmC0NDQwAUFMyQIbUpIBKJ8G//9m94vV42btw42aczZfj2t7/N3r17x11vZju3bt2ioqKC+vp6fu/3fo+WlpaEjzErzeSnAn19fUQiEUpLS8e9Xlpayo0bNybprBTTBV3X+e53v8vmzZtZsmTJZJ/OpHP58mU2btyI3+/H6XTy9ttvs2jRosk+rSnBv/3bv3Hu3DlOnz492acyZVi/fj0/+clPmD9/Pp2dnfzwhz9k69atXLlyhezs7Gc+jhJQhWIa8u1vf5srV66YW7eZgcyfP58LFy4wNDTEr371K771rW/x5ZdfznoRbW1t5c///M/57LPPSE9Pn+zTmTK88soro/vLli1j/fr11NbW8otf/II/+qM/eubjKAGdJIqKirBarXR3d497vbu7m7Kyskk6K8V04Dvf+Q4ffPABR44cmXJj+SYLh8NBY2MjAKtXr+b06dP8P//P/8M//uM/TvKZTS5nz56lp6eHVatWjb4WiUQ4cuQIf/u3f0sgEMBqtU7iGU4N8vLymDdvHrdv307o99Qa6CThcDhYvXo1Bw8eHH1N13UOHjyo1m4Uj8QwDL7zne/w9ttvc+jQIebMmTPZpzRl0XWdQCAw2acx6ezevZvLly9z4cKF0W3NmjX83u/9HhcuXFDiGcXj8XDnzh3Ky8sT+j0VgU4i3/ve9/jWt77FmjVrWLduHT/60Y/wer384R/+4WSf2qTi8XjG3Qneu3ePCxcuUFBQQE1NzSSe2eTy7W9/m5/97Ge8++67ZGdn09XVBUBubi4ZGRmTfHaTx1/8xV/wyiuvUFNTw/DwMD/72c84fPgwn3zyyWSf2qSTnZ390Bp5VlYWhYWFs3rt/L/+1//K/v37qa2tpaOjg+9///tYrVZ+93d/N6HjKAGdRH77t3+b3t5e/sf/+B90dXWxYsUKPv7444cKi2YbZ86cYefOnaPPv/e97wHwrW99i5/85CeTdFaTz9///d8DsGPHjnGv/8u//At/8Ad/8PxPaIrQ09PD7//+79PZ2Ulubi7Lli3jk08+4YUXXpjsU1NMUdra2vjd3/1d+vv7KS4uZsuWLZw8eZLi4uKEjqPGmSkUCoVCYQK1BqpQKBQKhQmUgCoUCoVCYQIloAqFQqFQmEAJqEKhUCgUJlACqlAoFAqFCZSAKhQKhUJhAiWgCoVCoVCYQAmoQqFQKBQmUAKqUCgUCoUJlIAqFAqFQmECJaAKhUKhUJhACahCoVAoFCZQAqpQTHP++I//GE3TmDdv3iP/e1tbG3a7HU3T0DQNj8fz0M/4fD7y8/PRNI0f/OAHE3zGCsXMQAmoQjHNyc/PB3ikMAL8zd/8DeFwePS5y+V66Gd+9rOf4XK5cDgc/Of//J8n5DwVipmGElCFYpqTl5cHwPDw8EP/zev18k//9E8AWK1WAAYHBx/6uR//+MeAzKgtKyuboDNVKGYWSkAVimlOLAL1er08ON73X//1XxkcHGT16tXMnz8feFhAjx8/zoULFwD48z//84k/YYVihqAEVKGY5sQiUMMw8Hq9o68bhsFf//VfA/Dd736XnJwc4GEBjUWfmzdvZvXq1c/hjBWKmYESUIVimhOLQGF8Gvejjz7i5s2blJeX89u//dvk5uYC49dAe3p6+NWvfgWo6FOhSBQloArFNCcWgcJ4Af3Rj34EwJ/8yZ9gt9sfGYH+0z/9E8FgkOrqat58881xx/3pT3/Kf/pP/4k1a9aQlpaGpmn85Cc/mbB/h0Ix3bBN9gkoFIrkeFQEeuXKFT7//HPS09NHq2pjEWhMQCORCP/4j/8IwLe//W1stvGXg//+3/879+/fp6ioiPLycu7fvz/h/xaFYjqhIlCFYpozNgKNtbLEos/f+73fo6ioCOChCPT999+ntbWVjIwM/uN//I8PHfef//mfaW5upre3V7W2KBSPQEWgCsU058EItLe3l//zf/4PIMVDMR5cA40VD/2H//AfKCgoeOi4e/bsmaAzVihmBioCVSimOWlpaaSnpwMioP/wD/+A3+9n9+7dLFmyZPTnxkagN2/e5ODBgwD82Z/92fM/aYViBqAEVKGYAcSi0P7+fv7u7/4OGB99wvg10L/7u7/DMAz27NnD4sWLn+u5KhQzBZXCVShmAHl5eXR2dvK///f/pquri7lz57J3795xPxOLQNvb27ly5QqgWlcUimRQAqpQzABiEWjMUejP/uzP0DRt3M/EItDm5mYAGhsbefXVV5/bOSoUMw2VwlUoZgBjK3Hz8vL4gz/4g4d+JhaBxvjTP/1TLBZ1CVAozKK+PQrFDGBsJe4f/dEf4XQ6H/qZWAQKIqZ/+Id/+FzOTaGYqagUrkIxA/jpT3/KT3/60yf+zPz58x8ym1coFOZREahCoVAoFCbQDHVLqlAoHsE///M/c/ToUQAuX77MuXPn2Lx5M42NjQBs2bKFP/7jP57MU1QoJhWVwlUoFI/k6NGj/Ou//uu4144dO8axY8dGnysBVcxmVASqUCgUCoUJ1BqoQqFQKBQmUAKqUCgUCoUJlIAqFAqFQmECJaAKhUKhUJhACahCoVAoFCZQAqpQKBQKhQmUgCoUCoVCYQIloAqFQqFQmEAJqEKhUCgUJlACqlAoFAqFCZSAKhQKhUJhAiWgCoVCoVCY4P8Pq39fyA6QWVAAAAAASUVORK5CYII=)

1. **¿Cuál es el modelo lineal si hacemos descenso de gradiente?**
Con $wo=0$ nuestro modelo es: $\hat y = w_1x_1+w_2x_2$ 
El descenso de gradiente iteraría luego: $(w1​w2​​)←(w1​w2​​)−η∇w​L(w1​,w2​)$
2. **¿En caso de haber un término de regularización, cuál sería? ¿Por qué?**
El termino de regularización seria $L_2$ $Ridge$ porque el punto óptimo se mueve hacia el origen de forma suave y rotacionalmente simétrica
3. **¿Cuál es la estimación del modelo aprendido para el ejemplo (1,1)?**
Por el gráfico podemos apreciar que $(w_1*, w_2*) = (1,2)$ 
Por lo que el predictor final el $\hat y = x_1 + 2x_2$ 
La estimación del modelo aprendido para $(x_1, x_2) = (1, 1)$ la predicción es: $\hat y =1*1 + 2*1 = 3$ 