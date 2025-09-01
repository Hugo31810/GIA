En el filtrado se eliminan todas aquellas características cuya puntuación no supera un umbral (_threshold_) elegido por nostros. 
Por este motivo también se dice que **son técnicas de selección de características**.
Vamos a aprender a filtrar por :
- varianza
- correlación
- información mutua

## Filtrado por varianza
El filtrado por varianza consiste en eliminar todas aquellas columnas cuya varianza no supere un cierto umbral.
varianza es **sensible** al orden de magnitud de los datos.
Cálculo de la varianza en Pandas
  - Se utiliza el método `var()` de un dataframe.
  - Por defecto se calcula la _varianza de la muestra_, que divide entre $N-1$.
  - Se puede modificar con la opción `ddof=0`. En ese casi estaríamos dividiendo entre $N$ y calculando la _varianza de la población_.

Filtrado por varianza en Scikit-Learn
  - Se utiliza `sklearn.feature_selection.VarianceThreshold`.
  - El propio método calcula la varianza poblacional de los datos.


EN EL CONJUNTO DE TEST:
- primero `fit` y luego `transoform`
EN EL CONJUNTO DE ENTRENAMIENTO: 
- se puede usar `fit_transform`

## Filtrado por correlación
Si dos características están correlacionadas significa que entre ambas existe una relación lineal. 
Es decir, cuando una crece o decrece, la otra lo hace proporcionalmente.
Por tanto no necesitamos ambas, basta con una de ellas porque la otra nos aporta exactamente la misma información, salvo por una escala y, quizás, un desplazamiento (_offset_).

## Filtrado por información mutua