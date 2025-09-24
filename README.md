# PARCIAL-2-IA-APLICADA

# Desarrollo parcial

Integrantes del grupo: Andrés Mauricio Guerrero y Nicolás Torres. 

A continuación se presentará una explicación del proceso llevado a cabo para el desarrollo del parcial 2 que se sabe, tiene como objetivo, a partir de variables socieconomicas, predecir el nivel de ingreso de una persona, en este caso, una dummy con valores "<=50K" y "<50K". Para esto, primero, se describirá el procedimiento para depurar la base de datos y las características descriptivas de la misma. Segundo, se expondrán los ajustes realizados a la base para poder emplearla en la predicción de los dos modelos, regresión logística y MLP. Tercer, para cada uno de estos casos se especificarán los métodos aplicados, junto con los resultados conforme a las indicaciones del parcial. Finalmente, se efectuarán comparaciones entre estos resultados para extraer conclusiones.


# Base de datos y depuración

La base de datos utilizada fue un conjunto de información sobre caracteristicas socieconomicas y nivel de ingreso en Estados Unidos para 1994. Los grupos relevantes de información, y los que vamos a usar, eran los llamados adult.data y adult.test. En conjunto estos grupos tenian 48842 observaciones, con 15 variables, 14 descriptivas y 1 objetivo, el ingreso. Dentro de las descriptivas, de las cuales consideramos todas relevantes para el modelo ya que son condiciones socieconomicas y cuyos resultados pueden tener calculos con variables que sean relevantes para predecir, por lo que omitirlas seria un error. En este grupo se encontraban edad, clase de trabajo, peso final de la muestra (fnlwgt), nivel educativo, años de educación, estado civil, ocupación, relación familiar, raza, sexo, ganancia de capital, pérdida de capital, horas semanales de trabajo y país de origen. En cuanto a la variable objetivo, esta era una dummy con dos valores, "<=50K" y ">50K". Ahora bien, como se menciono, hay dos grupos de información, dado que, al ser un problema de machine learning se requiere un conjunto de entrenamiento (adult.data) y uno de prueba (adult.test). Los dos grupos tenian las mismas variables pero con 32561 y 16281 observaciones respectivamente. 

Ahora bien, una vez la base de datos habia que dejarla lista para usar. Lo primero, fue ajustar el titulo de las variables, ya que al ser archivo csv no se estaban identificando correctamente. Lo segundo, era garantizar que no utilizarán observaciones con missing values en variables para asi garantizar un entrenamiento optimo. Al observar la muestra en ambos grupos de información, nos dimos cuenta que no habia como tal celdas vacías, sino celdas con "?". Para este caso, decidimos ... Esto dejo la muestra con x observaciones, que representa una perdida de % sobre la muestra original, algo ...



Las siguientes variables tenían missing values marcados como "?": workclass, occupation, y native-country. 

# Estadisticas descriptivas

| **Variable**       | **Rol** | **Tipo**   | **Categoría Demográfica** | **Descripción**                                                                                                                                                                                                                      | **Unidades** | **Valores Faltantes** |
| ------------------ | ------- | ---------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------ | --------------------- |
| **age**            | Feature | Entero     | Edad                      | Edad del individuo                                                                                                                                                                                                                   | N/A          | No                    |
| **workclass**      | Feature | Categórica | Ingreso                   | Tipo de empleo (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)                                                                                                               | N/A          | Sí                    |
| **fnlwgt**         | Feature | Entero     | N/A                       | Peso final asignado por la muestra                                                                                                                                                                                                   | N/A          | No                    |
| **education**      | Feature | Categórica | Nivel educativo           | Nivel de educación alcanzado (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)                                                 | N/A          | No                    |
| **education-num**  | Feature | Entero     | Nivel educativo           | Años de educación completados                                                                                                                                                                                                        | N/A          | No                    |
| **marital-status** | Feature | Categórica | Otro                      | Estado civil (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)                                                                                                             | N/A          | No                    |
| **occupation**     | Feature | Categórica | Otro                      | Ocupación (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces) | N/A          | Sí                    |
| **relationship**   | Feature | Categórica | Otro                      | Relación familiar (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)                                                                                                                                               | N/A          | No                    |
| **race**           | Feature | Categórica | Raza                      | Grupo étnico (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)                                                                                                                                                           | N/A          | No                    |
| **sex**            | Feature | Binaria    | Sexo                      | Género biológico (Female, Male)                                                                                                                                                                                                      | N/A          | No                    |


Una vez la base lista para usar, era necesario conocer un poco de la misma. Para esto, se identificaron valores relevantes como media, moda, mediana, rangos o distribucion de las diferentes variables del dataframe, en esta caso, de entreanmiento, ya que este iba a ser usado como la base para enfrentarse al de prueba, por lo cual tiene más sentido ver la iformación del entrenamiento. Como variables categoricas se encuentran: ingreso, clase de trabajo, nivel educativo, estado civil, ocupación, relación familiar, raza, sexo, y país de origen. En cuanto a las numericas estan: edad, peso final de la muestra, años de educación, ganancia de capital, pérdida de capital, horas semanales de trabajo. A continuación los resultados para las categoricas

  <img width="864" height="445" alt="image" src="https://github.com/user-attachments/assets/81d7e554-3221-4e1c-996e-577615eac5df" />

Se puede ver como la mayoria, alrededor de 2/3 de la muestra, estan gradudados de high school y tienen algun nivel de universidad o se han graduado de la misma en pregrado. El mayor rubro se encuentra en personas graduadas de highschool, 1/3 de la muestra. 

  <img width="932" height="447" alt="image" src="https://github.com/user-attachments/assets/cff94307-beb2-48ba-82f6-c0895b9d8cb8" />

La mayoria estan casados o nunca se han casado (estos 1/3), lo que puede representar una población joven. 

<img width="854" height="420" alt="image" src="https://github.com/user-attachments/assets/9940bb08-57d8-40c5-8a73-c4e41097a335" />

Esto es un complemento a la información anterior, aquí se puede ver que en general hay una distribucion variada, entre casados y no casados, junto con elementos neuvos como ser hijo en la familia.

<img width="807" height="425" alt="image" src="https://github.com/user-attachments/assets/f181ae7b-bf0d-4fc9-9426-8fff8b58f5bb" />

Se ve una mayoria de hombres en la muestra, alrededor del 66%, pero consideramos que esto no implica un sesgo y hay suficiente cantidad demujeres para realizar la predicción

<img width="905" height="439" alt="image" src="https://github.com/user-attachments/assets/1c7b441a-b893-479d-8235-b04c09427101" />

Al ser Estados Unidos es esperable ver una distribución así

<img width="882" height="444" alt="image" src="https://github.com/user-attachments/assets/6f087cb2-c658-48d2-841b-8ccebcb3e6a5" />
? Work Class

<img width="896" height="434" alt="image" src="https://github.com/user-attachments/assets/0691c9c4-252f-44f8-9810-428d4cfd90f0" />
?Ocuppation

<img width="654" height="507" alt="image" src="https://github.com/user-attachments/assets/650ec463-a583-4cea-9635-6b70f7355356" />

Esta es la variable objetivo, la cual representa un conjunto desbalanceado, dado que solo 8000 de las 32 mil observaciones, un 25%, tienen ingresos superiores a los 50k. Esto hay que tenerlo en cuenta para el caculo de las metricas, y utilizar mejor el F1 score en vez del accuracy, ya que, para el modelo puede ser mas facil predecir, dado el sesgo, a aquellos con ingresos inferiores a 50k.

Ahora, en cuanto a las variables numericas

# Ajustes para realización del modelo

Dado que ya se tiene la base lista, ahora es necesario hacer trasnformaciones y definir parametros a usar. Dado que se van a usar todas las variables descriptivas para predecir el objetivo, income, se crea una matriz de las variables en este caso : edad, clase de trabajo, peso final de la muestra (fnlwgt), nivel educativo, años de educación, estado civil, ocupación, relación familiar, raza, sexo, ganancia de capital, pérdida de capital, horas semanales de trabajo y país de origen. Luego se hace necesario, para poder entrenar el modelo, hacer split al conjunto de prueba en datos a intentar predecir y datos de validación. Una vez hecho esto, lo que hacia falta era ajustar los datos para poder predecir, tanto en entrenamiento como en prueba. Habia que hacer dos cosas, normalizar los datos numericos, y dada la presencia de variables categoricas con más de 2 opciones, crear categorias distinas a modo de 0 y 1 para lograr que el modelo tomara los datos de manera correcta. Para el primer caso, se separaron las variables numericas de las categoricas, se transformaron mediante feature.scaler.fit en los datos de entrenamiento y se cuido el data leakage de los datos de prueba. Luego en las categoricas, se aplico one.hot encoder, que permite crear una serie de nuevas variables para cada categoria de cada variable, y si cumple con la categoria, se marca 1 en esa variable, lo que se hace para todas. Finalmente, para el objetivo, el income, tocaba cambiar el valor de "<=50k" por 0 y 1 para ">50k", lo cual se hizo mediante label encoder. De esta manera se logra evitar explosión en el gradiente, predicción de manera correcta y replibabilidad del modelo. 

# Missing values

En este proyecto se decidió tratar los valores '?' como una categoría separada ('unknown') en lugar de eliminarlos o imputarlos. Esta elección permite preservar la mayor cantidad de datos posible, evita introducir sesgos por eliminación no aleatoria, y reconoce que la ausencia de información puede ser en sí misma un patrón relevante para el modelo. Además, su implementación es sencilla mediante codificación one-hot, donde 'unknown' se maneja como cualquier otra categoría, otorgándole al modelo la capacidad de aprender un peso específico para ella. 

# Desarrollo modelo logístico

Una vez hecho los ajustes, se procede a hacer el modelo logístico. Se utilizaron como métricas, con el objetivo de aprender, un máximo de 1000 iteraciones. Los resultados son los siguientes

Métricas en el Conjunto de Prueba


               precision    recall  f1-score   support

       <=50K       0.94      0.80      0.87      6204
        >50K       0.57      0.85      0.68      1936

    accuracy                           0.81      8140
   macro avg       0.76      0.82      0.77      8140
weighted avg       0.85      0.81      0.82      8140

<img width="864" height="445" alt="image" src="https://images2.imgbox.com/93/ff/jKvAKieV_o.png" />

Métricas en el Conjunto de Validación



               precision    recall  f1-score   support

       <=50K       0.94      0.80      0.86      6231
        >50K       0.56      0.83      0.67      1910

    accuracy                           0.81      8141
   macro avg       0.75      0.82      0.77      8141
weighted avg       0.85      0.81      0.82      8141


<img width="864" height="445" alt="image" src="https://github.com/user-attachments/assets/81d7e554-3221-4e1c-996e-577615eac5df" />

Métricas en el Conjunto de Entrenamiento

               precision    recall  f1-score   support

       <=50K       0.94      0.80      0.87     24720
        >50K       0.58      0.85      0.69      7841

    accuracy                           0.81     32561
   macro avg       0.76      0.83      0.78     32561
weighted avg       0.86      0.81      0.82     32561


<img width="864" height="445" alt="image" src="https://github.com/user-attachments/assets/81d7e554-3221-4e1c-996e-577615eac5df" />




