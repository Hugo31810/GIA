Es una manera de NO tener que decidir de antemano cuantos clusters queremos crear
Esta técnica comienza asignando a un cluster diferente a cada ejemplos del conjunto de datos. 
Después los clusters se agrupan entre sí por proximidad, formando otros más grandes. De esta manera se crea un árbol en el que podemos seleccionar por donde cortar una vez hemos terminado, siempre empezando por la unión más alta que haya.
![[Pasted image 20250519103252.png]]
- Si por ejemplos, eliminamos la línea azul, obtenemos 2 clusters
- Si a continuación cortamos en dos el grupo verde, tendremos 3: el grupo naranja, el verde de la izquierda y el verde de la derecha.
- Y así sucesivamente hasta el límite en el que recuperamos un cluster por cada ejemplo (que es como habíamos empezado)
- La figura de arriba es solo un ejemplos de lo que vamos a lograr

## Algoritmo par la creación de un cluster
1. Crear un cluster para cada ejemplo
	1. Si han N ejemplos comenzamos con N cluster de tamaño 1
2. Crear una tabla con las distancias entre cada par de clusters posibles
3. Localizar la fila y la columna que dan lugar al número más bajo
	1. Este cruce significa que la fila y la columna son los dos elementos más cercanos en este momento
4. Crear un nuevo cluster agrupando la fila y la columna localizadas
	1. Después se borran la fila y la columna de la tabla y se añade el nuevo cluster como una nueva fila y una nueva columna.
5. Volver a 2 y repetir hasta que ya solo haya 1 único cluster que agrupa todos los ejemplos.

![[Sin título.jpg|525]]

Finalmente podemos dibujar el deondograma representando las uniones de los ejemplos según se fueron obteniendo y en el eje vertical la distancia a la que se encuentran
![[Sin título 2.png]]

IMPORTANTE:
- En la figura se puede observar que se utiliza el min para calcular la distancia entre los dos clusters
- Se pude redefinir esta distancia de varias maneras. La habituales son:
	- la distancia que hay entre los dos ejemplos más próximos, uno de cada cluster obviamente demoniado en ingles **single linkage** (min)
	- la distancia que hay entre los dos ejemplos más alejados, también uno de cada cluster obviamente, denominado **complete linkage** (max)
	- la distancia promedio que hay entre cada par de ejemplos, también uno de cada cluster obviamente, denominado **average linkage** (average)
	- la distancia entre centroides, denominado, **centroid linkage**