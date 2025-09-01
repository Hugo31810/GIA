
- Es una técnica que se usa para reducir la dimensión de conjuntos de datos
- Se crean NUEVAS características que:
	- Sustituyen a las que había antes
	- Están ordenada, por lo que si elijo las $d$ primeras, estaré eligiendo las que contienen mayor información.
- Antes de hacer PCA **hay que estandarizar**. Porque es un método que surge de un problema de optimización y hay que estandarizar los autovalores. (Para ello en Python usamos `StandardScaled`)
	- Recordamos que estandarizar un conjunto de datos significa hacer que la media de los datos se acerque a 0 y que la desviación se acerque a 1.

## PCA
- ``n_components``:
	- $>1$ estamos indicado $d$.
	- $<1$ porcentaje de varianza explicada
		- El número $d$ se ajusta solo.

Las columnas que nos devuelve el PCA esta ordenadas de mayor a menos varianza explicada:
![[Pasted image 20250217154007.png]]
- En ``python`` tenemos dos opciones para hacer PDA:
	- Dándole el ``n_components`` que queramos que tenga.
	- Dándole el porcentaje de varianza explicada total acumulada mínima que debe de alcanzar