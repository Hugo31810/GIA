
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

- Naive Bayes
    - ¿Qué es?
    - Ejemplos de iid
    - Características independientes
- Algoritmo para construir el modelo de verosimilitud NB
- Ejemplo de un NB-Bernoulli
- Ejemplo de una NB-Gaussiano
- Resumen
    - ¿Qué es?
    - Teorema de Bayes
    - ¿Cómo funciona Naïve Bayes?
        - Entrenamiento
        - Predicción
    - Tipos de Naïve Bayes
        - Naïve Bayes Gaussiano
        - Naïve Bayes Multinomial
        - Naïve Bayes de Bernoulli
    - Ventajas de Naïve Bayes
    - Desventajas de Naïve Bayes
    - Ejemplos de Aplicación
```


![[Implementing-Naive-Bayes-Classification-using-Python-1-1.webp]]
## Naive Bayes
### ¿Qué es?
- Se llama Naive o ingenuo porque sume que todas las características son independientes entre sí, lo cual no siempre es cierto en la vida real, pero funciona realmente bien en muchos casos
- Es el resultado de imponer dos condiciones a la hora de construir el modelo de verosimilitud: 
	- Todos los ejemplos provienen de la misma distribución y son independientes entre sí. A esta condición la llamamos idd (independencias e identicamente distribuidos)
	- Las características son variables aleatorias independientes
### Ejemplos de iid
- La primera condición es muy **leve**
- Prácticamente siempre que se trabaja con datos tabulados y sin orden se cumple

### Características independientes
- La segunda condición es **muy fuerte**
- Es muy frecuente que dos características tengan una relación entre ellas de modo que al considerarlas independientes estamos simplificando su modelo.
- Cuando dos variables aleatorias $X$  e $Y$ se consideran independientes entonces su probabilidad conjunta es el producto de la probabilidad de cada una, es decir, $p(X,Y) = p(X)p(Y)$ 
- Si consideramos que cada característica es una variable aleatoria independiente, entonces el modelo de verosimilitud para los ejemplos de la clase $k$ es: $p({\bf X}|y=k) = \prod\nolimits_{j=1}^D p(x_j|y=k).$

Tomando logaritmos para evitar los problemas numéricos en el ordenador, tenemos
$$

\log p({\bf X}|y=k) = \sum\limits_{j=1}^D \log p(x_j|y=k)

$$
## Algoritmo para construir el modelo de verosimilitud NB
- Separar los ejemplos según su clase
- Para cada clase diferente $k$:
	- Para cada característica de la tabla de ejemplos $x_j$:
		1. Elegir la distribución de probabilidad $p(x_j|y=k; ~ \xi_{[j,k]}),~$ donde $\xi_{[j,k]}$ son los parámetros de dicha distribución.
		2. Estimar los parámetros $\xi_{[j,k]}^*$ a partir de los valores de la característica $x_j$ para la clase $k$.
  
 - El modelo de verosimilitud de la clase $k$ es:
 $$
\log p(\mathbf{X} \mid y = k) = \sum_{j=1}^{D} \log p(x_j \mid y = k; ~ \xi_{[j,k]}^*).
$$

## Ejemplo de un NB-Bernoulli
Si una característica $j$ solo puede tomar dos valores, es decir $~x_j\in\{0,1\},~$ entonces su distribución de probabilidad es Bernoulli, y por tanto
$$
\log p(x_j|y=k) =
\log \mathrm{Ber}\left( x_j |  y=k; ~\xi_{[j,k]} \right).
$$

Los parámetros de una distribución Bernoulli son el recuento de veces que aperece cada posibilidad.
![[Pasted image 20250402011933.png]]

## Ejemplo de una NB-Gaussiano
Si una característica $j$ tiene una distribución normal $~x_j\sim\mathcal{N}(x_j; \mu, \sigma),~$ los parámetros son la media y la desviación.

![[Drawing 2025-04-02 01.21.41.excalidraw|1400]]

## Resumen 
### ¿Qué es?

Naïve Bayes es un **algoritmo de clasificación basado en probabilidad** que utiliza el **Teorema de Bayes**. Se llama "Naïve" (ingenuo) porque **asume que todas las características son independientes** entre sí, lo cual no siempre es cierto, pero aún así funciona bien en muchos casos.

### Teorema de Bayes

La base de este algoritmo es la siguiente fórmula:

P(C∣X)=P(X∣C)P(C)P(X)P(C | X) = \frac{P(X | C) P(C)}{P(X)}

Donde:

- **P(C∣X)P(C | X)** → Probabilidad de que XX pertenezca a la clase CC (**posterior**).
    
- **P(X∣C)P(X | C)** → Probabilidad de observar XX si la clase es CC (**verosimilitud**).
    
- **P(C)P(C)** → Probabilidad previa de la clase CC (**prior**).
    
- **P(X)P(X)** → Probabilidad de observar XX (**evidencia**, constante para todas las clases).
    

El algoritmo **predice la clase con la mayor probabilidad posterior**.

---

### ¿Cómo funciona Naïve Bayes?

El algoritmo sigue estos pasos:

#### Entrenamiento

- Se parte de un conjunto de datos etiquetado con varias clases.
    
- Se calculan **las probabilidades previas** de cada clase P(C)P(C).
    
- Se calculan **las probabilidades condicionales** de cada característica dado cada clase P(X∣C)P(X|C).
    

#### Predicción

Para clasificar un nuevo dato XX:

- Se multiplica la probabilidad condicional de cada característica.
    
- Se elige la clase CC con mayor P(C∣X)P(C | X).
    

🔹 **Ejemplo**: Clasificación de correos como **spam** o **no spam**  
Supongamos que tenemos un correo con las palabras {gratis, oferta, descuento}.

- El modelo calcula la probabilidad de que el correo sea spam basándose en la frecuencia de esas palabras en correos spam vs. no spam.
    
- Se asigna la categoría con la mayor probabilidad.
    

---

### Tipos de Naïve Bayes

#### Naïve Bayes Gaussiano

- Se usa para **datos numéricos continuos**.
    
- Se asume que los datos siguen una **distribución normal (Gaussiana)**.
    
- **Ejemplo:** Clasificación de imágenes según colores o características físicas.
    

#### Naïve Bayes Multinomial

- Se usa para **conteo de datos discretos**, como palabras en un documento.
    
- Se aplica mucho en **Procesamiento de Lenguaje Natural (NLP)**.
    
- **Ejemplo:** Clasificación de textos (análisis de sentimientos, detección de spam).
    

#### Naïve Bayes de Bernoulli

- Se usa cuando los datos son **binarios (0 o 1)**.
    
- **Ejemplo:** Detección de palabras clave en correos electrónicos (si una palabra aparece o no).
    

---

### Ventajas de Naïve Bayes

✔ **Rápido y eficiente**: Funciona bien con grandes volúmenes de datos.  
✔ **Escalable**: Puede procesar miles o millones de datos con rapidez.  
✔ **Funciona bien con datos ruidosos o faltantes**.  
✔ **Especialmente útil en clasificación de texto y NLP**.

---

### Desventajas de Naïve Bayes

✖ **Suposición de independencia**: En la realidad, muchas características están correlacionadas, lo que puede afectar la precisión.  
✖ **Problema de probabilidad cero**: Si un valor no aparece en los datos de entrenamiento, su probabilidad será 0 y afectará el cálculo (se soluciona con **suavizado de Laplace**).  
✖ **No es bueno con datos numéricos sin distribución normal** (salvo en su versión Gaussiana).

---

### Ejemplos de Aplicación

🔹 **Filtrado de Spam** → Identificar si un correo es spam o no.  
🔹 **Clasificación de Sentimientos** → Analizar si un comentario es positivo o negativo.  
🔹 **Diagnóstico Médico** → Determinar enfermedades a partir de síntomas.  
🔹 **Reconocimiento de Texto** → Clasificación automática de documentos.
