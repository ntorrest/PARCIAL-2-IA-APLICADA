# PARCIAL-2-IA-APLICADA

# Desarrollo parcial

Integrantes del grupo: Andrés Mauricio Guerrero y Nicolás Torres. 

A continuación se presentará una explicación del proceso llevado a cabo para el desarrollo del parcial 2 que se sabe, tiene como objetivo, a partir de variables socioeconómicas, predecir el nivel de ingreso de una persona, en este caso, una dummy con valores "<=50K" y ">50K". Para esto, primero, se describirá el procedimiento para depurar la base de datos y las características descriptivas de la misma. Segundo, se expondrán los ajustes realizados a la base para poder emplearla en la predicción de los dos modelos, regresión logística y MLP. Tercero, para cada uno de estos casos se especificarán los métodos aplicados, junto con los resultados conforme a las indicaciones del parcial. Finalmente, se efectuarán comparaciones entre estos resultados para extraer conclusiones.


# Base de datos y depuración

La base de datos utilizada fue un conjunto de información sobre características socioeconómicas y nivel de ingreso en Estados Unidos para 1994. Los grupos relevantes de información, y los que vamos a usar, eran los llamados adult.data y adult.test. En conjunto, estos grupos tenían 48,842 observaciones, con 15 variables: 14 descriptivas y 1 objetivo, el ingreso. Dentro de las descriptivas, de las cuales consideramos todas relevantes para el modelo ya que son condiciones socioeconómicas y cuyos resultados pueden tener cálculos con variables que sean relevantes para predecir, por lo que omitirlas sería un error. En este grupo se encontraban edad, clase de trabajo, peso final de la muestra (fnlwgt), nivel educativo, años de educación, estado civil, ocupación, relación familiar, raza, sexo, ganancia de capital, pérdida de capital, horas semanales de trabajo y país de origen. En cuanto a la variable objetivo, esta era una dummy con dos valores, "<=50K" y ">50K". Ahora bien, como se mencionó, hay dos grupos de información, dado que, al ser un problema de machine learning, se requiere un conjunto de entrenamiento (adult.data) y uno de prueba (adult.test). Los dos grupos tenían las mismas variables pero con 32,561 y 16,281 observaciones respectivamente.

Ahora bien, una vez la base de datos habia que dejarla lista para usar. Lo primero, fue ajustar el titulo de las variables, ya que al ser archivo csv no se estaban identificando correctamente. Lo segundo, era garantizar que no utilizarán observaciones con missing values en variables para asi garantizar un entrenamiento optimo. Al observar la muestra en ambos grupos de información, nos dimos cuenta que no habia como tal celdas vacías, sino celdas con "?". Para este caso, decidimos tratar los valores '?' como una categoría separada ('unknown') en lugar de eliminarlos o imputarlos. Esta elección permite preservar la mayor cantidad de datos posible, evita introducir sesgos por eliminación no aleatoria, y reconoce que la ausencia de información puede ser en sí misma un patrón relevante para el modelo, por ejemplo en workclass, como trabajos no reconocidos que pueden influir en el ingreso devengado por una persona. Además, su implementación es sencilla mediante codificación one-hot, donde 'unknown' se maneja como cualquier otra categoría, otorgándole al modelo la capacidad de aprender un peso específico para ella.


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


Una vez la base lista para usar, era necesario conocer un poco de la misma. Para esto, se identificaron valores relevantes como media, moda, mediana, rangos o distribución de las diferentes variables del dataframe, en este caso, de entrenamiento, ya que este iba a ser usado como la base para enfrentarse al de prueba, por lo cual tiene más sentido ver la información del entrenamiento. Como variables categóricas se encuentran: ingreso, clase de trabajo, nivel educativo, estado civil, ocupación, relación familiar, raza, sexo y país de origen. En cuanto a las numéricas están: edad, peso final de la muestra, años de educación, ganancia de capital, pérdida de capital, horas semanales de trabajo. A continuación, los resultados para las categóricas.

  <img width="864" height="445" alt="image" src="https://github.com/user-attachments/assets/81d7e554-3221-4e1c-996e-577615eac5df" />

Se puede ver cómo la mayoría, alrededor de 2/3 de la muestra, están graduados de high school y tienen algún nivel de universidad o se han graduado de la misma en pregrado. El mayor rubro se encuentra en personas graduadas de high school, 1/3 de la muestra.

  <img width="932" height="447" alt="image" src="https://github.com/user-attachments/assets/cff94307-beb2-48ba-82f6-c0895b9d8cb8" />

La mayoria estan casados o nunca se han casado (estos 1/3), lo que puede representar una población joven. 

<img width="854" height="420" alt="image" src="https://github.com/user-attachments/assets/9940bb08-57d8-40c5-8a73-c4e41097a335" />

Esto es un complemento a la información anterior, aquí se puede ver que en general hay una distribución variada, entre casados y no casados, junto con elementos neuvos como ser hijo en la familia.

<img width="807" height="425" alt="image" src="https://github.com/user-attachments/assets/f181ae7b-bf0d-4fc9-9426-8fff8b58f5bb" />

Se ve una mayoria de hombres en la muestra, alrededor del 66%, pero consideramos que esto no implica un sesgo y hay suficiente cantidad demujeres para realizar la predicción

<img width="905" height="439" alt="image" src="https://github.com/user-attachments/assets/1c7b441a-b893-479d-8235-b04c09427101" />

Al ser Estados Unidos es esperable ver una distribución así

<img width="882" height="444" alt="image" src="https://github.com/user-attachments/assets/6f087cb2-c658-48d2-841b-8ccebcb3e6a5" />

Se ve una alta cantidad de personas en el sector privado, con una participacion relevante de observaciones con ?, que como se mencionaba anteriomente, pueden ser trabajos relevantes. 

<img width="896" height="434" alt="image" src="https://github.com/user-attachments/assets/0691c9c4-252f-44f8-9810-428d4cfd90f0" />

Al igual que workclass, se ve una participación importante de ?, junto con una distribución muy variada de actividades. Esto no da la perspectiva de sesgo por ningún lado

<img width="654" height="507" alt="image" src="https://github.com/user-attachments/assets/650ec463-a583-4cea-9635-6b70f7355356" />

Esta es la variable objetivo, la cual representa un conjunto desbalanceado, dado que solo 8000 de las 32 mil observaciones, un 25%, tienen ingresos superiores a los 50k. Esto hay que tenerlo en cuenta para el caculo de las metricas, y utilizar mejor el F1 score en vez del accuracy, ya que, para el modelo puede ser mas facil predecir, dado el sesgo, a aquellos con ingresos inferiores a 50k.

Ahora, en cuanto a las variables numericas

<img width="778" height="442" alt="image" src="https://github.com/user-attachments/assets/8de67c82-6ce0-4799-b36a-b30169bb2bc7" />

Se ve un conjunto de edades bastante diverso, lo que no implica sesgos

<img width="791" height="432" alt="image" src="https://github.com/user-attachments/assets/2760115d-613d-4c5f-8fdf-110c79661b10" />

La mayoria tienen 10 años o más de educación que va de la mano con lo que muestra la grafica de nivel educactivo en las categoricas

<img width="796" height="447" alt="image" src="https://github.com/user-attachments/assets/66810e82-6e45-492c-b19f-9a9e8199eb90" />

<img width="793" height="437" alt="image" src="https://github.com/user-attachments/assets/6df4460b-f1c4-4f28-b02d-90f3c5d96eaa" />

Se observa la distribución del capital-gain y capital-loss. La enorme mayoría de las personas tiene un gain menos a 20000 USD y, por tanto, al tener poquito capital, pierde poco capital. 

# Ajustes para realización del modelo

Dado que ya se tiene la base lista, ahora es necesario hacer transformaciones y definir parámetros a usar. Dado que se van a usar todas las variables descriptivas para predecir el objetivo, income, se crea una matriz de las variables en este caso: edad, clase de trabajo, peso final de la muestra (fnlwgt), nivel educativo, años de educación, estado civil, ocupación, relación familiar, raza, sexo, ganancia de capital, pérdida de capital, horas semanales de trabajo y país de origen. Luego se hace necesario, para poder entrenar el modelo, hacer split al conjunto de prueba en datos a intentar predecir y datos de validación. Una vez hecho esto, lo que hacía falta era ajustar los datos para poder predecir, tanto en entrenamiento como en prueba. Había que hacer dos cosas: normalizar los datos numéricos, y dada la presencia de variables categóricas con más de 2 opciones, crear categorías distintas a modo de 0 y 1 para lograr que el modelo tomara los datos de manera correcta. Para el primer caso, se separaron las variables numéricas de las categóricas, se transformaron mediante feature.scaler.fit en los datos de entrenamiento y se cuidó el data leakage de los datos de prueba. Luego, en las categóricas, se aplicó one hot encoder, que permite crear una serie de nuevas variables para cada categoría de cada variable, y si cumple con la categoría, se marca 1 en esa variable, lo que se hace para todas. Finalmente, para el objetivo, el income, tocaba cambiar el valor de "<=50K" por 0 y 1 para ">50K", lo cual se hizo mediante label encoder. De esta manera se logra evitar explosión en el gradiente, predicción de manera correcta y replicabilidad del modelo.


# Desarrollo modelo logístico

Una vez hecho los ajustes, se procede a hacer el modelo logístico. Se utilizaron como métricas, con el objetivo de aprender, un máximo de 1000 iteraciones. Para mitigar el riesgo de sobreajuste, se implementaron varias estrategias. En primer lugar, se entrenaron modelos de Regresión Logística con regularización L2, lo que introduce penalizaciones sobre los coeficientes y evita que el modelo se vuelva innecesariamente complejo. Además, el modelo evaluado utilizó el parámetro class_weight='balanced', lo que ayuda a manejar el desbalance de clases y previene que el modelo se incline excesivamente hacia la clase mayoritaria. De esta manera se logra tener un modelo sin sesgos y sin overfitting.

Los resultados son los siguientes

📊 Resultados del Modelo (L2 Logistic Regression)
Conjunto de Entrenamiento

Accuracy: 0.8129

Precision: 0.5755

Recall: 0.8494

F1 Score: 0.6861

Confusion Matrix:

<img width="864" height="864" alt="image" src="https://images2.imgbox.com/71/c5/4Tl4uCmv_o.png" />

Conjunto de Validación

Accuracy: 0.8064

Precision: 0.5587

Recall: 0.8319

F1 Score: 0.6685

Confusion Matrix:


<img width="864" height="864" alt="image" src="https://images2.imgbox.com/41/06/4nFsH22u_o.png" />

Conjunto de Prueba

Accuracy: 0.8114

Precision: 0.5695

Recall: 0.8481

F1 Score: 0.6815

Confusion Matrix:

<img width="864" height="864" alt="image" src="https://images2.imgbox.com/b7/16/UdSvEhy1_o.png" />

# Modelo de red neuronal

Descripción modelo sin early stopping y dropout: En este modelo, dado que se estaba prediciendo una variable binaria, tocaba utilizar como función de pérdida el BCE loss, dado que en este caso MSE no servía. Adicionalmente, tocaba definir una función de salida para que arrojara un resultado factible en la predicción, es decir, 0 y 1. La función corresponde a Sigmoid, la óptima para este tipo de modelo. Finalmente, cabe aclarar que se extrae el mejor modelo de las 5 iteraciones y se muestran los gráficos y métricas correspondientes. Al correr los distintos experimentos, nos dimos cuenta de que el F1-score del conjunto de entrenamiento específicamente estaba dando consistentemente en torno a 0.2 para todos los conjuntos de hiperparámetros. Para capturar este sesgo y poder seguir comparando los experimentos de forma más justa, se decidió transformar el F1-score aplicando la fórmula (1 – F1-score). De esta manera, un F1-score originalmente bajo se refleja como un valor alto en la métrica transformada, resaltando la magnitud de la discrepancia y facilitando la interpretación en conjunto con el resto de métricas.

Resultados

Hiper parametros : hidden_layers: 2, hidden_neurons: 64, learning_rate: 0.005, num_epochs: 25, batch_size: 32

| Dataset        | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| **Conjunto entrenamiento**      | 0.6578   | 0.7537    | 0.7956 | 0.7766   |
| **Conjunto validación** | 0.8461   | 0.7104    | 0.5806 | 0.6390   |
| **Conjunto de prueba**       | 0.8498   | 0.7263    | 0.5990 | 0.6517   |

<img width="963" height="607" alt="image" src="https://github.com/user-attachments/assets/2ce2a580-53cf-4bef-90ea-405b244cb4f6" />

Descripción modelo con early stopping y dropout: En este caso, se hizo lo mismo que el modelo anteiro solo que se añadio dropout rate y patience como parte del early stopping. 

Resultados

Hiperparametros: hidden_layers: 5, hidden_neurons: 512, learning_rate: 0.0001, dropout_rate: 0.6, num_epochs: 60, batch_size: 256, patience: 15

## 📊 Performance Metrics for Best Model

| Dataset        | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| **Conjunto entrenamiento**       | 0.6317   | 0.7573    | 0.7504 | 0.7539   |
| **Conjunto validación** | 0.8478   | 0.6697    | 0.6932 | 0.6812   |
| **Conjunto de prueba**       | 0.8565   | 0.6967    | 0.7025 | 0.6996   |

<img width="948" height="599" alt="image" src="https://github.com/user-attachments/assets/265bb80e-76bd-4310-88ab-6c3f82fb6f7b" />


# Comparacion de modelos neuronales

El mejor modelo, en cuanto al conjunto de prueba y validación, al comparar el F1-score, que es la mejor métrica dado el desbalance en la variable objetivo, y la gráfica de pérdida vs número de épocas, nos dimos cuenta de que el mejor es el modelo con early stopping e hiperparámetros: hidden_layers: 5, hidden_neurons: 512, learning_rate: 0.0001, dropout_rate: 0.6, num_epochs: 60, batch_size: 256, patience: 15

# Comparación mejor modelo MLP con la regresión logística

Al mirar el F1-score en test en ambos casos, tanto en prueba como en validación, se ve que el mejor modelo es el MLP con early stopping. Esto indica que el MLP, gracias al early stopping, logra generalizar mejor sin sobreajustarse y capturar patrones más complejos que la regresión logística no pudo utilizar por su peor desempeño.











