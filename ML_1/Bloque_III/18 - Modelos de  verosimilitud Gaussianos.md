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

- Modelos de verosimilitud gaussianos
    - Procedimiento general
        - Resumen del procedimiento
    - Interpretación
        - Ejemplo
- Variantes
    - LDA
    - QDA
- Resumen:
```

## Modelos de verosimilitud gaussianos
- En este caso asumimos que la verosimilitud está modelada en conjunto por una **distribución normal multivariada (MNV)**, es decir, una distribución normal pero en $D$ dimensiones.
- Los parámetros de una MVN son su centro, que se corresponde con el punto de mayor densidad probabilidad y con la media y su anchura que viene determinada por la matriz de covarianza
- En problemas de clasificación binaria, si asumimos que todas las clases son igualmente probables a priori, entonces: 
$$
y^* = \mathop{\arg\max}\limits_{y} p(y|{\bf x}) = \mathop{\arg\max}\limits_{y} p({\bf x}|y).
$$

Si sustituimos la $p$ por la MVN entonces la etiqueta asignada a un ejemplo  $x$ será

$$
\left\lbrace
\begin{array}{llll}
y^* = 0 & {\rm si} & {\rm MVN}({\bf x}|y=0) > {\rm MVN}({\bf x}|y=1)   \\
y^* = 1 & {\rm si} & {\rm MVN}({\bf x}|y=0) < {\rm MVN}({\bf x}|y=1)   \\
\end{array}
\right.
$$

**IMPORTANTE**
- En la celda anterior hemos asumido que las clases son igualmente probables a priori, pero igualmente podríamos haber asumido que existe una probabilidad a priori sobre la clase. 
	- En ese caso, como ya sabemos,
$$
y^* = \mathop{\arg\max}\limits_{y} p(y|{\bf x}) = \mathop{\arg\max}\limits_{y} \big( p(y) \cdot p({\bf x}|y)   \big).
$$

**Recuerda también que:**

- Es preferible trabajar con logaritmos para evitar problemas de precisión en los ordenadores. <br>
- Por tanto la expresión anterior se reescribe como
$$
y^* = \mathop{\arg\max}\limits_{y} \log p(y|{\bf x}) = \mathop{\arg\max}\limits_{y} \big( \log p(y) + \log p({\bf x}|y)   \big).
$$
### Procedimiento general

1. Aprender los parámetros de cada modelo de verosimilitud a partir de los datos del conjunto de entrenamiento; es decir:
$$
\left\lbrace
\begin{array}{llllllll}
\{\mu_0, {\bf \Sigma}_0\} & {\rm para~~la} & {\rm MVN}({\bf x}|y=0;~ \mu_0, {\bf \Sigma}_0) & \leftarrow \text{modelo de verosimilitud de la clase 0}\\[.3em]
\{\mu_1, {\bf \Sigma}_1\} & {\rm para~~la} & {\rm MVN}({\bf x}|y=1;~ \mu_1, {\bf \Sigma}_1) & \leftarrow \text{modelo de verosimilitud de la clase 1}
\end{array}
\right.
$$
Para ello:
2. . Se separan los ejemplos de entrenamiento de cada clase
3. . Se estima la media y la covarianza de los ejemplos de cada clase


$\rightarrow~$ Nuestros modelos de verosimilitud gaussiano <b>ya están aprendidos</b> $~\rightarrow~$ ¡ Ya podemos hacer inferencia !
2. Cuando llegan ejemplos nuevos, sólo hay que evaluarlos en cada modelo de verosimilitud. 
$\rightarrow$ La etiqueta será la que se corresponde con el modelo donde es más probable.

#### Resumen del procedimiento
1. Filtrar el conjunto de entrenamiento para $y=0$
	1. Calcular el vector media $(\mu)$ 
	2. Calcular la matriz de covarianza $(\Sigma)$
		1. $p(x|y=0) = MNV(x;\mu, \Sigma)$
2. Repetimos con la clase $y=1$
3. Al terminar tendremos un modelo de verosimilitud para cada clase
![[Pasted image 20250404113530.png]]
### Interpretación
Para entender lo que sucede, vamos a visualizarlo con ejemplos de solo 2 dimensiones.

Cada MVN es una _campana_ centrada en su media y con una forma que dependerá de su matriz de covarianza.

Si pintamos _lonchas_ de esas _campanas_ a diferentes alturas, allá donde se crucen serán puntos $\bf x$ donde ${\rm MVN}_0({\bf x}) = {\rm MVN}_1({\bf x})$.
Esos puntos están sobre la superficie de decisión.
![[Pasted image 20250404122024.png]]

#### Ejemplo
![[Pasted image 20250404122533.png]]
![[Pasted image 20250404122549.png]]

## Variantes

El paso más costoso computacionalmente es calcular la matriz de covarianza, ya que calcular la media es rápido y sencillo para un ordenador.

Una manera de simplificar el aprendizaje de los modelos de verosimilitud es evitando calcular la covarianza.

De este modo se pueden dar 3 posibilidades:
- No calculamos la covarianza sino la varianza; es decir dejamos solo la diagonal de la matriz de covarianza.<br>
$~\rightarrow$ Recuperamos el clasificador Naive Bayes (NB) con modelos de verosimilitud gaussianos.
- Asumimos que todas las matrices de covarianza son identicas <br>
$~\rightarrow$ Obtenemos un clasificador LINEAL denominado _Linear Discriminant Analysis_ (LDA).
- No asumimos nada acerca de las matrices de covarianza y calculamos todas<br>
$~\rightarrow$ Obtenemos la expresión de un clasificador NO lineal denominado _Quadratic Discriminant Analysis_ (QDA).
### LDA
Resultado de **asumir** que todos los modelos de verosimilitud tienen **la misma matriz de covarianza.**

Si esto ocurre entonces las MVN de cada clase intersecan en un hiperplano, y por tanto se obtiene un clasificador lineal.

**¿Para qué queremos LDA si ya conocemos muchas maneras de obtener modelos lineales?**

LDA construye un modelo lineal **bayesiano**, es decir que podemos incorporar la probabilidad a priori sobre la clase si queremos. <br>
Esto, en otros métodos para obtener modelos lineales, NO ES POSIBLE.

El problema de LDA es elegir la matriz de covarianza que van a compartir todos los modelos de verosimilitud.
### QDA
Es el más **general** de todos, puesto que cada modelo de verosimilitud tiene **su propia matriz de covarianza.**

La ventaja de QDA es que se puede demostrar, mediante mucha manipulación algebraica, que es posible llegar a la expresión matemática de un clasificador NO lineal.
<br>
Esto no había ocurrido hasta ahora. El único modelo del que conocemos su expresión es el modelo lineal.


## Resumen:

![[Pasted image 20250404122250.png]]