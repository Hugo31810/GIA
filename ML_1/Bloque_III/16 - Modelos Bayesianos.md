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
    - Teorema de Bayes
    - Introduciendo el teorema de Bayes en el problema de maximización
- Modelo a priori de la clase
    - Bernoulli
    - Clasificación multiclase
- Modelos de verosimilitud
- Ejemplo para explicar el funcionamiento del Tº de Bayes
- Resumen
```
## Formulación probabilística de un problema de clasificación
$$
  y^* = \mathop{\arg\max}\limits_{y} p\big(y|{\bf x}\big)
$$

- Es decir, buscamos la etiqueta óptima $y^*$, que será la etiqueta más probable para el ejemplo dado.

- Evidentemente, esta formulación exige construir un modelo probabilístico de la etiqueta o clase condicionado a los ejemplos.

- Para ello es imprescindible el **Teorema de Bayes**.
### Teorema de Bayes
![[Captura de pantalla 2025-03-24 163021.png]]
### Introduciendo el teorema de Bayes en el problema de maximización
![[Captura de pantalla 2025-03-24 163547.png]]
- Para solucionar el problema de clasificación, desde el punto de vista probabilístico, tenemos que crear dos modelos:
	- Modelo a priori de la clase
	- El modelo de verosimilitud de los datos de cada clase
- Antes, debemos de reformular el problema usando logaritmos
	- para solucionar problemas con las números menores que 1

- Aplicamos los logaritmos: 
$$

\begin{aligned}
  y^* &= \mathop{\arg\max}\limits_{y} \log p(y \mid \mathbf{x}) \\[6pt]
  &= \mathop{\arg\max}\limits_{y} \log \left(\frac{p(y) \cdot p(\mathbf{x} \mid y)}{p(\mathbf{x})}\right) \\[6pt]
  &= \mathop{\arg\max}\limits_{y} \left( \log p(y) + \log p(\mathbf{x} \mid y) \right)
\end{aligned}

$$

En definitiva,

$$ \fbox{$y^* = \mathop{\arg\max}\limits_{y} \big( \log p(y) + \log p(\mathbf{x} \mid y) \big)$}
$$
## Modelo a priori de la clase
- Vamos a estimar o imponer una probabilidad para cada una de las posibles clases que se pueden dar.
	- En un problema de clasificación binaria habrá 2 clases es decir $y \in {0,1}$  
	- Por lo que un posible modelo a priori estimado a partir del conjunto de entrenamiento sería una tabla como la siguiente: 
	![[Pasted image 20250327105105.png|150]]

	- Pero también podemos imponer unas probabilidades a priori nosotros: 
		- Ej: todas las probabilidades son igualmente probables "a priori"
			- $p(y=0)=p(y=1) =0.5$ 

### Bernoulli
- Ya que estamos creando modelos probabilísticos, conviene aprender que la distribución de probabilidad de una variable aleatoria solo puede tomar dos valores, como es nuestro caso de $y \in \{0,1\} \rightarrow$ **Distribución de Bernoulli**
- En caso de que estimáramos esta distribución mediante el recuento de clases sería: 
$$
\mathrm{Ber}(y) =
\begin{cases}
\frac{N_{y=0}}{N}, & \text{si } y=0 \\[6pt]
\frac{N_{y=1}}{N}, & \text{si } y=1
\end{cases}
$$
que habitualmente se expresa de este otro modo más compacto:
$$
\mathrm{Ber}(y) =
\left(\frac{N_{y=0}}{N}\right)^{1-y}  
\left(\frac{N_{y=1}}{N}\right)^{y}
$$
### Clasificación multiclase
- En el casa de que  haya más de una clase (es decir, ya no es una clase binaria), tenemos que estimar o imponer la probabilidad de cada etiqueta posible.
- La distribución de probabilidad en este caso se llamada **distribución categórica**, y en caso de que la estimáramos mediante recuento su expresión, para $K$ clases sería: 
$$
\mathrm{Cat}(y) = \left\lbrace
\begin{array}{lll}
\frac{N_{y=k}}{N} & \text{para} & y=k\\
\end{array}
\right\rbrace_{k=0,1,2,\ldots, K}
$$


> En binario $\rightarrow$ Bernoulli
> En multiclase $\rightarrow$ Categórica
## Modelos de verosimilitud
- El modelo de verosimilitud es mucho más difícil ya que se trata de estimar la distribución de los ejemplos.
	- Estimar la distribución de una única variable aleatoria ya es difícil
	- Ahora los ejemplos tienen $D$ dimensiones 
	- El modelo de verosimilitud consiste en estimar la **distribución conjunta** de $D$ variables al mismo tiempo 
- Para minorizar toda esa dificultada podemos hacer varias suposiciones, que da lugar a modelos más sencillos y a otros más complejos.
	- Vamos a estudiar como modelos de verosimilitud:
		- Naive Bayes
		- Linear Discriminant Analysis (LDA)
		- Quadratic Discriminant Analysis (QDA)
		- Mezcla de gaussianas

## Ejemplo para explicar el funcionamiento del Tº de Bayes

-  Sabemos que la probabilidad de padecer **cáncer de mama** es del 15%
	- (Podemos asumir que nos hemos restringido a mujeres menores de 60 años por ejemplo.)
	- Por tanto, sin haber hecho ninguna prueba a nadie, podemos imponer una probabilidad _a priori_ $~p(y=1) = 0.15~$.
	- Evidentemente, la probabilidad _a priori_ de NO padecer cancer es $~p(y=0) = 0.85$

- Tenemos un test experimental que da positivo 80 de cada 100 veces que se ha probado en mujeres a las que se les detectó un tumor de mama por otros medios. <br>
Por tanto sabemos que
$~p(x=1|y=1) = 0.8~$, y evidentemente también sabemos que $~p(x=0|y=1) = 0.2~$<br>
Este mismo test, en mujeres sanas, se equivoca 15 de cada 100 veces; así que
$~p(x=1|y=0) = 0.15~$, y evidentemente también sabemos que $~p(x=0|y=0) = 0.85~$<br>
- Una mujer se hace un test y resulta positivo. Qué es más probable con estos datos: ¿que tenga cancer o que no?.

- **Modelo a priori:** (prevalencia del cáncer)

| $y$  | $p(y)$ |
|------|--------|
| $y=0$ | $0.85$ |
| $y=1$ | $0.15$ |

 - **Modelo de verosimilitud:** (Probabilidad de que el test sea positivo o negativo cuando la mujer tiene o no cáncer)

| $p(x \mid y)$ | $y=0$  | $y=1$  |
|---------------|--------|--------|
| $x=0$         | $0.85$ | $0.20$ |
| $x=1$         | $0.15$ | $0.80$ |

- Para elegir la **etiqueta a posteriori** tenemos que calcular:
![[Pasted image 20250404010806.png|325]]

- El resultado es: Cuando el ``test = 1``, la etiqueta más probable ``=0``
- Debemos darnos cuenta de que no hemos calculado la probabilidad de tener cáncer sino la etiqueta más probable según sea el resultado del test y la prevalencia del cáncer.
## Resumen
![[Pasted image 20250404005317.png]]