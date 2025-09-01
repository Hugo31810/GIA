```insta-toc
---
title:
  name: ""
  level: 1
  center: false
exclude: ""
style:
  listType: dash
omit: []
levels:
  min: 1
  max: 6
---

# 

- ¿Qué es un árbol de decisión?
    - Funcionamiento
- Ganancia de información
    - Entropía e incertidumbre
    - La entropía como criterio de selección de características
- Índice de Gini - impureza
    - El índice de gini criterio de selección de características
- ¿Cómo se construye un árbol de decisión?
- Generalización a características de cualquier tipo
```
## ¿Qué es un árbol de decisión?
Un **árbol de decisión** en _Machine Learning_ es un modelo predictivo que se usa para clasificación y regresión. Se representa como un árbol donde:
- Cada **nodo interno** representa una pregunta basada en una característica del conjunto de datos.
- Cada **rama** representa una posible respuesta a la pregunta.
- Cada **hoja** representa una clase (para clasificación) o un valor numérico (para regresión).
### Funcionamiento
1. **División recursiva**: El algoritmo divide los datos en subconjuntos en función de la característica que mejor separa las clases o reduce el error.
2. **Criterios de división**: Para clasificación, usa medidas como _Gini_ o _Entropía_. Para regresión, usa _MSE (Mean Squared Error)_.
3. **Poda**: Se eliminan ramas poco útiles para evitar el sobreajuste (_overfitting_).

Los árboles de decisión son fáciles de interpretar, pero pueden ser propensos al sobreajuste si no se regulan bien. Modelos como **Random Forest** o **Gradient Boosting** mejoran su rendimiento combinando múltiples árboles
## Ganancia de información
La idea es elegir la característica que más información nos aporte sobre la etiqueta.
### Entropía e incertidumbre
La entropía es la medida de la incertidumbre que tenemos sobre un suceso que va a ocurrir.
- Si sabemos que tiene probabilidad 100% de ocurrir, entonces no hay incertidumbre y la entropía de esa variable es 0
- En caso de que haya una cierta probabilidad $p(x)$ entonces su entropía es:
$$
E = - \sum\limits_{\forall x} p(x) \log_2 p(x)
$$
### La entropía como criterio de selección de características
Asumimos que para la siguiente explicación estamos en un problema de clasificación binaria:
1. Comenzamos calculando la entropía de la tabla: 
![[Pasted image 20250319180329.png]]
2. Observamos la característica de la tabla de datos de entrenamiento
Podemos calcular dos entropías de $y$ para dicha columna $i$:
  - $E_{x_i = 0}$ es la Entropía de $~y~$ cuando $x_i = 0$
  - $E_{x_i = 1}$ es la Entropía de $~y~$ cuando $x_i = 1$
  Así tenemos dos medidas de incertidumbre sobre la etiqueta, nos quedamos con una suma ponderada de ambos, ahora la vemos.
  
  Para calcular los pesos de cada una recurrimos al número de ejemplos de cada caso respecto del número total de ejemplos.
 - $N_{x_i = 0}$ el número de ejemplos con $x_i = 0$
- $N_{x_i = 1}$ el número de ejemplos con $x_i = 1$

La suma ponderada de las dos entropías es:
$$\frac{N_{x_i = 0}}{N} E_{x_i = 0} + \frac{N_{x_i = 1}}{N} E_{x_i = 1} $$

3. La **ganancia de información** es la reducción en la incertidumbre debida a dividir la tabla en dos partes: aquellas con filas $x_i = 0$ y aquellas con $x_i = 1,$ es decir:
$$
G_{x_i} = E_T - \frac{N_{x_i = 0}}{N} E_{x_i = 0} + \frac{N_{x_i = 1}}{N} E_{x_i = 1}
$$
4. Hay que repetir los pasos 2 y 3 para cada $i = 1, 2, ... N$

**Como resultado tendremos una $G$ mayor que todas las demás.**
**La característica que ha dado lugar a esa $G$ es la que mayor información nos da sobre la etiqueta.**
![[Pasted image 20250319182227.png]]
![[Pasted image 20250402233132.png]]

## Índice de Gini - impureza
El **índice de Gini** mide qué tan mezcladas están las clases en un grupo de datos.
- Si todas las muestras en un nodo son de la misma clase → **Gini = 0 (puro)**
- Si hay una mezcla de clases → **Gini es mayor (impuro)**

Ejemplo rápido:
Imagina una bolsa con 10 bolas:
- 10 rojas → **Gini = 0** (todas son iguales, nodo puro).
- 5 rojas y 5 azules → **Gini alto** (mezcla, nodo impuro).
- 9 rojas y 1 azul → **Gini más bajo**, pero no puro.

Los árboles de decisión buscan dividir los datos de manera que el **índice de Gini sea lo más bajo posible**, separando mejor las clases. 

Dada una tabla de ejemplos asociados al vector $y$ que contienen las etiquetas de cada ejemplo, el índice de impureza de Gini se calcula como: 
$$
I = 1- \sum\limits_{k=1}^{n_c}p(y=k)^2,
$$
donde $n_c$ representa el número de clases distintas; y la probabilidad de una clase se aproxima por el número de veces que aparece esa clase entre el total de ejemplos; es decir $~p(y=k) = \frac{N_{y=k}}{N}$.

En el caso particular de clasificación binaria, donde $y\in\{0,1\}$ el índice de impureza de Gini queda:

$$
I = 1 - \left( \frac{N_{y=0}}{N} \right)^2 - \left( \frac{N_{y=1}}{N} \right)^2
$$



### El índice de gini criterio de selección de características
- El proceso es exactamente el mismo que con la entropía, con la diferencia que se utiliza $I$ en vez de $E$.

- Por cada característica se calcula el índice de gini para cada clase y después se calcula el índice de gini promedio para esa característica y se compara con el índice gini promedio de la tabla completa.

- Con la entropía, $G$ mide la ganancia de información, que es la reducción de incertidumbre.
- Del mismo modo, con el índice de impureza de gini, $G$ mide la ganancia de pureza que es el disminución de impureza.

## ¿Cómo se construye un árbol de decisión?
- Los árboles son estructuras de datos recursivas:
	- Un árbol consta de un nodo y ramas
	- Cada rama da lugar a un árbol o a un nodo hoja

- Proceso de construcción de un árbol de decisión: 
	1. Dada una tabla de datos, se elige aquella característica que maximiza $G$ según cierto criterio, por ejemplo la entropía ($G$) o el índice de Gini ($I$). Dichas características dan lugar a dos ramas (si se trata de un problema binario)
	2. Repetimos el proceso 1 por la rama de la izquierda con la tabla resultante de quedarse con aquellos ejemplos que tienen la característica elegida **= "0"**
	3. Repetimos el proceso 1 por la rama de la derecha con la tabla resultante de quedarse con aquellos ejemplos que tienen la característica elegida **="1"**

- Este proceso se detiene cuando ya no queda tabla, peor eso implica memorizar los datos en la estructura del árbol $\rightarrow$ poco eficiente/sobreajuste
	- Para evitar el sobre ajuste es necesario "podar" el árbol, es decir detener el proceso con algún criterio como: 
		- haber alcanzado un cierto número de nodos
		- o haber alcanzado una cierta profundidad
- Si elegimos un número muy bajo de nodos o de profundidad estaremos construyendo un árbol demasiado sencillo, es decir subajustando lo cual es igual de malo
## Generalización a características de cualquier tipo
- En el ejemplo que hemos visto arriba utilizaba características binarias.
- Qué pasa si una característica es categórica o continua:
	- Si es categórica podemos utilizar una codificación one-hot y convertirla en varias características binarias
	- También se puede reordenar la tabla según esa característica de menor a mayor y hacer una separación por cada categoría.

- PREGUNTA EXAMEN: Si tengo un dataset que tiene 4 columnas 2 binarias y 2 categóricas con 5 categorías
	- Tendría un árbol con 22 hijos cada raíz

![[Pasted image 20250324153127.png]]

- A partir de esa división, el resto sería igual que el ejemplo, pero ahora cada columna tendría varios candidatos.
	- Comenzaríamos contando el número de veces que y = 1 para los ejemplos azules  y lo mismo para los ejemplos no azules. De esta división obtendríamos una entropía promedio.
	- Luego repetiríamos el proceso con los ejemplos grises y no grises etc...

- Si una columna es continua podríamos dividirla en varios cubos. Por ejemplo una característica continua en el intervalo $[0,1]$ se pude dividir en intervalos de $0.1$.