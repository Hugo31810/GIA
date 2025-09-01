
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

- Formulación probabilística de un problema de clasificación
- Pérdida en la regresión logística
- La entropía cruzada como función de pérdida
```
## Formulación probabilística de un problema de clasificación
![[Pasted image 20250324155345.png]]
- Cómo lo hacemos?
	- Tenemos $p(x|y)$
	- Nos dan x
	- Introduzco $x$ en $p(x|y)$
	- Calculamos: 
		- $p(y=0|x)$
		- $p(y=1|x)$
			- vemos cual de las dos es mayor
## Pérdida en la regresión logística
- Cuando explicamos la regresión logística dejamos pendiente una función de pérdida con la que se podía aplicar descenso de gradientes
- Una manera de guiar el ajuste de los parámetros es haciendo que estos maximicen la probabilidad con la que el modelo clasificador toma una decisión para el conjunto de datos de entrenamiento dado.
- Para un ejemplo dado x queremos que nuestro clasificador: 
	- maximice le valor de $p(y=1|x)$ (probabilidad con la que el clasificador acertaría al predecir que la etiqueta $y = 1$) cuando la etiqueta asociada a este ejemplo $y=1$
	- maximizar el valor de $1-p(y=1|x)$ (probabilidad con la que el clasificador acertaría al predecir que la etiqueta $y=0$) cuando la etiqueta asociada sea $y = 0$
- En la clasificación binaria **no hay más posibilidades**,  con lo cual es o 0 o 1.
![[Drawing 2025-04-03 12.16.42.excalidraw|555]]
- Lo ideal sería que las probabilidades fuesen **lo más altas posibles**
	- Por tanto, queremos modificar los parámetros en el sentido de **maximizar** estas probabilidades, que surgen de la expresión dada arriba.

- Pero **no** podemos maximizar $N$ probabilidades distintas (una por cada ejemplo). **Queremos maximizar la probabilidad de todo el conjunto de ejemplos de train**.

- Como hemos asumido que los ejemplos son idd (independientes entre sí) la probabilidad conjunta de todo ellos es el producto de cada uno, vamos a tener que maximizar: 
$$
\max \prod\limits_{i=1}^{N} \left( p(y=1|{\bf x}^{(i)}) ^ {y^{(i)}} \big(1- p(y=1|{\bf x}^{(i)}) ^ {1-y^{(i)}}\big) \right)
$$
- Hacemos operaciones matemáticas hasta llegar a $\mathcal{L}_{\rm log}$. 
- Como la regresión logística devuelve directamente $~p(y=1|{\bf x}^{(i)}; {\bf w})~$, podemos utilizar la siguiente función de pérdida

$$
\mathcal{L}_{\rm log} = - \sum\limits_{i=1}^{N} \left[ y^{(i)} \log p(y=1 \mid \mathbf{x}^{(i)}) + (1 - y^{(i)}) \log \big(1 - p(y=1 \mid \mathbf{x}^{(i)}) \big) \right]
$$
Esta el la función llamada **Log Loss**, a la que se le pueden añadir términos de regularización L1, L2 o ElasticNet.
**Esta función de pérdida busca maximizar la probabilidad de las etiquetas dados los datos.**
## La entropía cruzada como función de pérdida
- En un problema de clasificación binaria podemos interpretar la etiqueta verdadera (0 o 1) como la probabilidad de que el ejemplo pertenezca a la clase "positiva" es decir:
	- si $~y^{(i)}=0~$ quiere decir que la probabilidad de que el ejemplo $~{\bf x}^{(i)}~$ sea _positivo_ es **NULA**
	- si $~y^{(i)}=1~$ quiere decir que la probabilidad de que el ejemplo $~{\bf x}^{(i)}~$ sea _positivo_ es **TOTAL**
- Un modelo que estima $p(y|x)$ **NO** es tan taxativo, eso quiere decir que seguramente devuelve un valor en el intervalo (0,1).

- Dado un problema de clasificación binaria y un conjunto de entrenamiento $X; y=\{0,1\}$ La etiqueta en representación one-hot es una PMF de dos sucesos.
![[Drawing 2025-04-03 12.41.17.excalidraw||779]]

- **¿Existe alguna función de pérdida que compare ambas distribuciones y guíe el descenso de gradiente? SÍ $\rightarrow$ LA ENTROPÍA CRUZADA**

- La entropía cruzada entre la distribución verdades $p$ y la distribución estimada $\hat p$ es el valor esperado respecto a la distribución $p$ del $- log \hat p$, es decir:
$$
\begin{align}
{\rm CE}(p,\hat p) &=  -\mathbb{E}_p \left[ \log \hat p \right] \\
&= -\sum\limits_{\forall y} p(y|x) \log \hat p(y|x)
\end{align}
$$

Para problemas de clasificación binaria $y=\{0,1\}$, y eliminando la notación de probabilidad condicionada para tener una expresión más sencilla, la entropía cruzada "binaria" (_Binary Cross Entropy_, BCE), se reduce a
$$
\begin{align}
{\rm BCE}(p,\hat p)
&=
p(y=1)(-\log \hat p(y=1)) +
p(y=0)(-\log \hat p(y=0)) \\
&=
-p(y=1)(\log \hat p(y=1))
-(1-p(y=1))(\log \hat p(y=0))
\\  
\end{align}
$$
que coincide con la log-loss para un único ejemplo. La extensión a $N$ es simplemente la suma de todas ellas.


## Resumen:


Se puede dar una interpretación probabilística desde varios puntos de vista.
1. La etiqueta estimada es la más probable
$\rightarrow$ Planteamos la pérdida Log-loss.

2. La distribución de probabilidad estimada sobre las etiquetas debe ser lo más parecida posible a la distribución de probabilidad real
$\rightarrow$ Planteamos la pérdida _Entropía cruzada_

$\fbox{Para problemas de clasificación binaria ambas coinciden}$

3. La etiqueta estimada es la más probable "a posteriori" (MAP)
$\rightarrow$ Este es el planteamiento **Bayesiano**, que veremos en el siguiente cuaderno