

1. [[#Tipos de variables|Tipos de variables]]
2. [[#Codificación de variables categóricas|Codificación de variables categóricas]]
3. [[#Recursos:|Recursos:]]
4. [[#Separación de datos:|Separación de datos:]]
5. [[#Exploración de los datos:|Exploración de los datos:]]
6. [[#Preprocesado de datos:|Preprocesado de datos:]]
	1. [[#Preprocesado de datos:#Tratamiento de valores Perdidos:|Tratamiento de valores Perdidos:]]
7. [[#Tipos de imputación de datos:|Tipos de imputación de datos:]]
	2. [[#Tipos de imputación de datos:#Imputación Univariada:|Imputación Univariada:]]

## Tipos de variables
- Discretas: (3, 4, 5) son ints
- Continuas: (3.5, 5.8) son floats
- Categórico: (Verde, Rojo, Azul) no se le puede asignar ningún orden

## Codificación de variables categóricas
Los datos categóricos son elementos de un conjunto y normalmente vienen descritos por palabras.
Existe una relación biyectiva entre un conjunto de categorías y el conjunto de los números naturales.

Esto significa que a cada elemento del conjunto de categorías se le puede asignar un número natural. Si lo hacemos estaremos **codificando** el atributo categórico.

Hay dos maneras de hacer esto en Pandas:
- Utilizando una codificación *One-Hot* usando vectores (Ej: codificación RGB)
 ![[Pasted image 20250129175507.png]]
- Asignando un entero a cada elemento del conjunto:
	- Se puede hacer de forma automática pero no sabremos que se le asigna a cada variable.
	- Para saber a qué categoría se corresponde cada entero lo mejor es ir creando un diccionario al mismo tiempo. Para ello utilizaremos primero `cat.categories` y luego juntaremos códigos y categorías en la estructura de datos `dict` de Python.

## Recursos: 
Bibliotecas de Python que vamos a usar durante el curso

| Biblioteca   | Modo de importar                | Utilidad                                                         |
| ------------ | ------------------------------- | ---------------------------------------------------------------- |
| Pandas       | `import pandas`                 | Manejo de datos tabulados                                        |
| Numpy        | `import numpy`                  | Manejo de arrays n-dimensionales y funciones matemáticas         |
| Scipy        | `import scipy`                  | Manejo de funciones matemáticas y distribuciones de probabilidad |
| Scikit-learn | `import sklearn`                | Biblioteca de ML                                                 |
| Random       | `import random`                 | Generación aleatoria                                             |
| Matplotlib   | `from matplotlib import pyplot` | Generación de gráficos                                           |
## Separación de datos:
NUNCA podemos utilizar todos los ejemplos que nos den para aprender la tarea.
Se debe entrenar con una parte de los ejemplos, para tomar el resto como prueba a la que nuestra máquina nunca se ha enfrentado. 

- Conjunto de datos: 
	- Train
		- Preprocesado (80 - 90% del trabajo)
			- limpieza
			- Imputación
			- reducción de dimensiones
			- etc...
	- Test


## Exploración de los datos:
Debemos de hacer una descripción estadística básica de los datos del conjunto de ejemplos que hemos recibido para entrenar nuestro modelo.
Este informe debe de contener al menos: 
  

Este informe debería contener al menos:

- Tamaño del conjunto de datos, es decir de la tabla<br>

  $N=?, D=?$

- Localizar aquellos atributos que NO son numéricos y convertirlos.

- Media y desviación estandard de cada atributo<br>

  $\mu(x_i)=?$, $\sigma_i(x_i)=?$ , para $i=1\ldots D$.

- Mediana de cada atributo<br>

  $Q_{50}(x_i)=?$

- Moda de cada atributo<br>

  $\text{Moda}(x_i) = ?$

- Máximo y mínimo de cada atributo<br>

  $\max(x_i)= ?$  , $\min(x_i)= ?$
## Preprocesado de datos:

Entre las tareas que podemos automatizar están:
- Localización y tratamiento de datos perdidos
$~$(en este cuaderno veremos varias maneras).
- Eliminar atributos innecesarios o seleccionar los atributos que finalmente vamos a utilizar
$~$(estudiaremos estas técnicas más adelante).

### Tratamiento de valores Perdidos:
Los valores perdidos (_missing_) son más frecuentes de lo que se puede pensar.  
Por ejemplo, en registros médicos, algunas veces no se tomó la temperatura porque había pocos enfermeros debido a una saturación de las emergencias. O en datos recogidos mediante encuesta por internet el consultado no rellenó varios campos que no eran obligatorios. También puede ocurrir en datos recogidos por diferentes sensores donde por algún motivo uno o varios se han apagado durante un tiempo.


Para tratar los valores perdidos debemos:
1. Averiguar dónde están
2. Valorar si rellenamos los huecos (imputación de valores perdidos) o tomamos otra decisión como eliminar el ejemplo el atributo.
3. Opcionalmente se puede añadir un nueva columna a la derecha de aquella donde se hayan imputado valores, marcando las celdas donde ha habido imputación como muestra la figura de abajo.

## Tipos de imputación de datos:
### Imputación Univariada:
Es la técnica mas sencilla, consiste en asignar un estadístico de la columna a todos los NaN que hay en ella.
Este estadístico puede ser:
- media
- mediana
- moda

$x_n$ n columna
$x^n$ n ejemplo/fila