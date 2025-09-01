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

- Modelos de mezclas de gaussianas
    - Definición de mezcla de gaussianas
- Aprendizaje
    - Nota:
    - Importante:
    - Definición de factor de pertenencia
    - Algoritmo EM
        - Paso "E"
        - Paso "M"
        - Inicialización
    - Inferencia
- Ejemplo
```
## Modelos de mezclas de gaussianas
- Hasta ahora hemos asumido que todas las características o al menos un grupo de dos o más de ellas tienen una distribución conjunta MVN (distribución normal multivariada)
- Si fueran así todos los ejemplos se deberían de concentrar en torno a un único valor, que es la media.
- Lo que suele pasar es que **no todos los ejemplos giran en torno a la media**, en la siguiente imagen podemos ver que hay tres agrupaciones de datos, con lo cual si asumimos que tienen una única media y un modelo de verosimilitud MVN entonces estaremos creando una máquina **con muy poco poder de predicción**.

![[Pasted image 20250409225411.png|275]]
- ¿Qué se  puede hacer entonces? -> **Modelar la verosimilitud "mezclando" varias distribuciones MVN.**
- En el caso de la imagen de arriba tendríamos 3 MVN, cada una con sus parámetros (media y matriz de covarianza)
### Definición de mezcla de gaussianas
- Cualquier distribución de densidad de probabilidad debe cumplir que el área bajo la curva es igual a $1$.
- Esto sólo significa que la probabilidad de que ocurra un suceso posible, sea el que sea, es del 100%.
- EJEMPLO: 
	- Si decimos que la altura de todos los españoles tiene una densidad de probabilidad normal o gaussiana, centrada en 175cm con desviación típica estándar de 20cm. ¿Cuál es la posibilidad de encontrar un español cuya altura esté entre 0 cm y 300cm?
		- Evidentemente 100%
		- Con un poco de código obtenemos la siguiente imagen:
		![[Pasted image 20250409230109.png|350]]
	- Por lo tanto, si simplemente sumáramos N densidades normales, el área bajo la curva sería N.

- Una **mezcla de gaussianas** es una **combinación convexa** de **componentes-MVN**
$$
p({\bf x} \mid y; {\bf w}) = \sum_{i=1}^K \pi_i \cdot \text{MVN}({\bf x} \mid y; \mu_i, \Sigma_i), \quad \text{tal que } \sum_{i=1}^K \pi_i = 1.
$$

	- Combinación convexa significa que los coeficientes de la combinación **son no negativos y suman 1.**
	- Componentes MVN son **distribuciones MVN** (multivariada normal)
- **Un modelo de verosimilitud construido mediante mezcla de gaussianas se denomina GMM (Gaussian Mixture Models)**
## Aprendizaje
- Sabemos que para aprender un modelo de verosimilitud tenemos que aprender los parámetros de la distribución que hemos elegido o estamos construyendo.

- **¿Cuántos parámetros tenemos que aprender en una mezcla de K componentes-MVN?**
	- $\mu_1, ~ \Sigma_1~$: media y matriz de covarianza de la 1ª componente-MVN.
	- $\mu_2, ~ \Sigma_2~$: media y matriz de covarianza de la 2ª componente-MVN.
	 $\vdots$
	- $\mu_K, ~ \Sigma_K~$: media y matriz de covarianza de la Kª componente-MVN.
	- $\pi_1,~ \pi_2,\ldots, \pi_K$ : pesos de cada componente-MVN.

- En definitiva, hay $2K + K = 3K$ parámetros.

- Pero los parámetros propios de cada componente-MVN y los pesos con los que cada una contribuye a la mezcla tienen una _naturaleza_ diferente, y no se pueden aprender al mismo tiempo
### Nota:
- Vamos a construir un modelo de verosimilitud, es decir la $p(x|y)$.
- En todo momento estaremos suponiendo que los ejemplos pertenecen todos a la misma clase.

### Importante:
- Un error muy común cuando se quiere aprender un GMM es pensar que tenemos que averiguar a cual de las componentes-MVN pertenece un ejemplo $x$.
- En un GMM, un ejemplo $X$, pertenece a todas y cada una de las componentes MVN
![[Pasted image 20250409234256.png]]
![[Pasted image 20250409234311.png]]

- Para aprender los parámetros de una GMM necesitaremos calcular el **factor de pertenencia**, y ese nombre puede llevarnos a pensar erróneamente que los puntos pertenecen a una componente o a otra.

### Definición de factor de pertenencia
Definimos **factor de pertenencia** del ejemplo $i$-ésimo $~{\bf x}^{(i)}~$ a la $k$-ésima componente-MVN al valor

$$
\gamma_{i,k} = \frac{\pi_k\cdot\mathrm{MVN}_k({\bf x}^{(i)})}{\sum\limits_{i=1}^K \pi_k\cdot\mathrm{MVN}_k({\bf x}^{(i)})}
$$
![[Pasted image 20250409234917.png|375]]

- El numerador es simplemente evaluar el ejemplo en la componente-MVN
- El denominador es evaluar el ejemplo en la GMM.

> _En el ejemplo de arriba, el factor de pertenencia del punto negro a la componente azul se calcularía como la altura de la estrella azul divida por la altura de la estrella roja._
### Algoritmo EM
- Este algoritmo consiste en alternar:
	- Un paso E (Esperanza/Expectation): se calcula el factor de pertenencia para cada ejemplo
	- Un paso M (Maximización/Maximization): se actualizan los parámetros de la GMM
#### Paso "E"
- Calcular el factor de pertenencia de cada uno de los ejemplos
- Obtener una tabla de valores${\rm\Gamma} = \{\gamma_{i,k}\}$ para $i=1\ldots N$, $k = 1\ldots K$; es decir una tabla de valores de pertenencia donde los ejemplos van en filas y las componentes-MVN en columnas.
(Se verifica que  cada fila suma 1 ) 
#### Paso "M"
1. Calcular $M_k = \sum_{i=1}^{N} \gamma_{i,k}$, es decir la seuma de la k-esima columna de la tabla $\Gamma$ para $k=1...K$
2. Actualizamos los $K$ pesos con la regla $$ \pi_k^{\rm nuevo} = M_k / N $$
3. Actualizamos la media y covarianza de cada MVN$_k$.

$$
\begin{align}\mu^{\rm nuevo}_k &= \frac{1}{M_k}\sum\nolimits_{i=1}^{N} \left( \gamma_{i,k} \cdot{\bf x}^{(i)} \right)\\\Sigma^{\rm nuevo}_k  &= \frac{1}{M_k}\sum\nolimits_{i=1}^{N} \left( \gamma_{i,k}\cdot({\bf x}^{(i)}-\mu^{\rm nuevo}_k)({\bf x}^{(i)}-\mu^{\rm nuevo}_k)^\top \right)\end{align}
$$

#### Inicialización
- Para hacer los cálculos del paso E necesitamos tener las $K$ MVN-componentes (que actualizan el paso M)
- Para hacer los cálculos del paso M necesitamos tener el factor de pertenencia (que se actualiza en el paso E)

- **Posibles formas de inicializar el algoritmo**: 
	- Inicializar de manera aleatoria un **vector de $K$ pesos**, así como una **media** y una **matriz de covarianza** por cada componente-MVN, esto significa **empezar por el paso E**
	- Inicializar de manera aleatoria una **tabla de valores $\Gamma$,** esto significa **empezar por el paso M**
### Inferencia
- Con GMM simplemente estamos aprendiendo un nuevo modelo de verosimilitud.
- Para hacer inferencia debemos proceder exactamente igual que con Naive Bayes o con los modelos de Gaussianas. A si que: 
	1. Tenemos que empezar modelando la distribución a priori
	2. A continuación aprendemos el modelo de verosimilitud mediante GMM
- Una vez tenemos $p(y)$ y $p(x|y)$ les aplicamos logaritmos y probamos con cada $y$ posible en la expresión:$$\arg\max \log p(y|x) = \arg\max \big( \log p(y) + \log p(x|y) \big). $$
## Ejemplo
- Vamos a construir un modelo de verosimilitud con mezcla de gaussianas para un problema de clasificación binario.
- El conjunto de datos será muy sencillo, sólo con dos características para poder visualizar los ejemplos y el modelo.
![[Pasted image 20250410002428.png|400]]
- Para este modelo asumiremos que el modelo a priori sobre la etiqueta es $p(y=0) = p(y=1)=0.5$ 
- Por tanto solo queda aprender el modelo de verosimilitud, que serán GMM.

- Una vez aprendidos los modelos a priori y de verosimilitud podemos hacer inferencia.
- En vez de utilizar el conjunto de test, vamos a probar con puntos dentro del intervalo $[-2,2]x[-2,2]$ así podemos visualizar la superficie de separación
- Recordamos que la etiqueta predicha es: 
$$
\hat{y} = \arg\max_y \left( \log p(y) + \log p({\bf x} \mid y) \right)
$$

- Visualizamos el modelo: 
	- Zonas asignadas a cada clase. La superficie de decisión es la frontera entre ambas
	 ![[Pasted image 20250410002856.png|350]]
	- Curvas de nivel de la $log-verosimilitud(p(x|y=0))$
	 ![[Pasted image 20250410002942.png|375]]
	- Curvas de nivel de la  $log-verosimilitud(p(x|y=1))$
	 ![[Pasted image 20250410003015.png|350]]
	- Muestra de la superficie de decisión como la unión de los puntos donde las curvas de nivel de cada modelo tienen la misma altura.
	 ![[Pasted image 20250410003320.png|375]]
- Parámetros de cada modelo:
- ![[Pasted image 20250410003404.png|344]]