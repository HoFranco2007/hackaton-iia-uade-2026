# M17 · ¿Entra el triple sobre la chicharra?

> **Hackatón IIA · UADE · 3.4.257.** Este README es la plantilla del grupo. En su fork, reemplacen todo lo que está entre `< >` y borren esta nota. Las bases están en `BASES.md` y su ficha en `problematicas/`.

**Integrantes:** Ho Franco · Welner Benicio  
**IA declarada:** Gemini  
**Pista:** Modelo de regresión logística  
**Publicación en LinkedIn:** https://lnkd.in/p/dkVEH3Rk  

---

## 1. Formulación y PEAS

**La pregunta.** SÍ significa: El triple entra.

| PEAS | Su sistema |
|---|---|
| **Medida de rendimiento** | Porcentaje de triples convertidos sobre la chicharra (Triples metidos / Triples tirados) con alta precisión en los aciertos aprobados. |
| **Entorno** | La cancha de básquet, la distancia al aro, la posición y presión de los defensores rivales. |
| **Actuadores** | La ejecución física del tiro por parte del jugador o la decisión táctica de autorizar el lanzamiento. |
| **Sensores** | Registros de datos del tiro: *Distancia del tiro (m)*, *Defensor más cercano (m)* y *Efectividad histórica del tirador (% en triples)*. |

**Objetivo y métrica.**  
* **Objetivo (Negocio / Juego):** Maximizar la eficiencia ofensiva del equipo, asegurando que el tirador solo ejecute un lanzamiento cuando exista una alta certeza de que el tiro se convertirá en punto ("convertir el triple") y evitando tiros desaconsejables que cuesten partidos.
* **Métrica (Cuantificación Técnica):** Indicador cuantitativo que mide qué tan bien el modelo de IA cumple su objetivo. Se utiliza la **Precisión** (con un umbral de $0.75$ y penalización asimétrica de Falsos Positivos) para garantizar que, cada vez que el sistema autoriza un tiro ("SÍ"), la tasa de acierto en el mundo real sea lo más alta posible.

---

## 2. El sistema

**Cómo decide.**  
El sistema se basa en un **modelo de Regresión Logística binaria** que estima la probabilidad continua $P(Y=1 \mid X)$ de que un tiro sea encestado ("SÍ"). La decisión no utiliza el umbral estándar del $50\%$, sino un **umbral estricto de $0.75$**, respaldado por pesos asimétricos (`class_weight={0: 1, 1: 6}`). 

* **Pesos ($\beta$) y Sesgo ($\beta_0$):**  
  Los coeficientes ($\beta_1, \beta_2, \beta_3$) asociados a cada variable y el intercepto o sesgo ($\beta_0$) son calculados automáticamente por la función de optimización del algoritmo durante el entrenamiento (`modelo.fit()`). La probabilidad se calcula mediante la función sigmoide:
  $$P(\text{SÍ}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 \cdot \text{Distancia} + \beta_2 \cdot \text{Defensor} + \beta_3 \cdot \text{Efectividad})}}$$
  Si $P(\text{SÍ}) \ge 0.75$, el sistema responde **SÍ**; de lo contrario, responde **NO**.

**Variables que usa y por qué.**  
El modelo utiliza las tres variables disponibles en el dataset (`M17.csv`) por ser los factores determinantes del lanzamiento:
1. **Distancia del tiro (m):** Mide la exigencia física y la parábola requerida.
2. **Defensor más cercano (m):** Cuantifica la presión defensiva y el bloqueo del tiro.
3. **Efectividad del tirador (% en triples en la temporada):** Aporta la línea base del talento y consistencia del jugador.  
*(Nota sobre la efectividad: Se calcula con los registros previos a la jugada actual para evitar sesgos o data leakage).*

**Cómo se corre en vivo.**  
Enlace a Google Colab: https://colab.research.google.com/drive/1LZaRqDN7D4RuqFZA08qAlv15cNohP_P0?usp=sharing

**Pasos de ejecución:**
1. **Requisitos previos:** Instalar librerías `pandas` y `scikit-learn`, y cargar el archivo `M17.csv`.
2. **Carga y filtrado de datos:** Separar los conjuntos de entrenamiento (`train_df`) y prueba (`test_df`) según la columna `conjunto`.
3. **Entrenamiento del modelo:** Instanciar `LogisticRegression(class_weight={0: 1, 1: 6}, random_state=42)` y ejecutar `modelo.fit(X_train, y_train)`.
4. **Predicción en vivo:** Evaluar nuevos vectores con `modelo.predict_proba()` aplicando el umbral de `0.75`.

**Dos casos de ejemplo.**

| Caso inventado | Distancia de tiro (m) | Defensor más cercano (m) | Efectividad del tirador (%) | Salida del sistema | Resultado Real |
|---|---|---|---|---|---|
| A | 27 | 1 | 20 | **NO** | SÍ |
| B | 13 | 2 | 36.6 | **NO** | SÍ |

*(Nota sobre los casos A y B: Aunque en la realidad terminaron siendo "SÍ", el sistema arrojó **NO** debido al umbral ultra conservador de $0.75$ y la penalización de Falsos Positivos, lo que demuestra que el modelo prefiere ser cauteloso ante tiros atípicos o de alto riesgo).*

---

## 3. Medición

**Entrenamiento (casos 1–14)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | VP = 2 | FN = 0 |
| **Real NO** | FP = 0 | VN = 12 |

**Prueba (casos 15–20)**

| | Predice SÍ | Predice NO |
|---|---|---|
| **Real SÍ** | VP = 1 | FN = 0 |
| **Real NO** | FP = 0 | VN = 5 |

| Métrica | Entrenamiento | Prueba |
|---|---|---|
| Exactitud | 100% | 100% |
| Precisión | 100% | 100% |
| Sensibilidad | 100% | 100% |
| Especificidad | 100% | 100% |

**Entrenamiento contra prueba y el 100% de efectividad:**  
Que ambos conjuntos alcancen un 100% de efectividad con tan solo 20 casos se debe a un problema de **muestra pequeña y sobreajuste (*overfitting*)**. Las variables presentan una separación lineal muy limpia en este dataset reducido, lo que hace que el modelo memorice los patrones locales. Ante un flujo real y masivo de datos nuevos, es altamente probable que aparezcan errores y que el 100% disminuya.

---

## 4. El error más caro

* **Falso Positivo (Decir "SÍ" cuando la decisión correcta era "NO"):**  
  Significa que el sistema aprueba un tiro de altísimo riesgo estimando erróneamente que será exitoso.
* **Consecuencia real:** Provoca tiros errados en momentos críticos, pérdidas de posesión y contragolpes del rival. En instancias decisivas, autorizar un tiro que no cumple con el estándar de certeza puede costar directamente el partido, puntos clave en la tabla o la eliminación directa.

---

## 5. El número de la Línea Base Trivial

* **¿Qué resultado saca un sistema que responde siempre lo mismo?**  
  Si un sistema trivial predice siempre **"NO"** para los 20 casos del dataset (donde hay 17 casos "NO" y 3 casos "SÍ"), obtiene una **Exactitud del 85.00%** (17/20), pero con una **Precisión del 0%** y una **Sensibilidad del 0%** (ya que ignora por completo todos los aciertos).
* **Comparación:**  
  Mientras la línea base trivial acierta el 85% por inercia estadística (negando todo), nuestro modelo de Regresión Logística alcanza un **100% de exactitud y precisión controlada** gracias al análisis inteligente de las variables y el umbral estricto, logrando detectar los aciertos reales sin cometer Falsos Positivos.

| Checklist (Clase 6) | Respuesta |
|---|---|
| ¿Qué datos? | Muestra de 20 casos divididos en 70% entrenamiento (14 casos) y 30% prueba (6 casos). |
| ¿Qué métrica? | Exactitud, Precisión, Sensibilidad y Especificidad (con umbral estricto de $0.75$). |
| ¿Comparado contra qué? | Contra una línea base trivial (siempre "NO") cuya exactitud es del **85.00%** en el total de casos. |
| ¿Quién lo midió? | Algoritmo de Regresión Logística en scikit-learn (Python). |

---

## 6. Riesgo: el camino completo

1. **Datos:** Variables de `M17.csv` (*Distancia*, *Defensor*, *Efectividad*).
2. **Modelo:** Regresión Logística penalizada (`class_weight={0: 1, 1: 6}`) con umbral estricto de $\ge 0.75$.
3. **Decisión:** Clasificación binaria: **"SÍ"** (autorización de alta certeza) o **"NO"** (descarte de la acción).
4. **Consecuencia:** Se protege al equipo de pérdidas críticas y contragolpes rivales, maximizando la precisión de los lanzamientos aprobados.

---

## 7. Bitácora

**Uso de la IA**

| # | Qué le pedimos | Qué aceptamos | Qué rechazamos y por qué |
|---|---|---|---|
| 1 | Programación inicial del modelo de regresión | Estructura general del código | Rechazamos el umbral por defecto (0.5) y lo elevamos a 0.75 para evitar Falsos Positivos. |
| 2 | Evaluación de alternativas (redes neuronales vs. regresión) | La sugerencia de usar Regresión Logística | Rechazamos enfoques más complejos (como redes neuronales) por ser innecesarios y propensos al sobreajuste con pocos datos. |

**Corridas contra el evaluador**

| Corrida | Nota | Agujero que encontró | Qué cambiamos |
|---|---|---|---|
| 1 | 5.5 | PEAS incompleto, faltaban pasos en ejecución y definición de pesos/sesgo. | Se corrigió el PEAS, se agregaron los pasos del script y la fórmula del sesgo/pesos de la regresión. |
| 2 | 6.5/10 | Falta el número exacto de la línea base trivial, justificar el 100% con 20 casos y explicar los casos A/B. | Se incorporó el 85.00% de la línea base trivial, se explicó el riesgo de sobreajuste y el conservadurismo en los casos A y B. |
|3| 9/10 |  Código, pasos y dos casos con salida. La pista declarada no es ninguna de las tres de la clase y no muestran los valores de pesos y sesgo.  Matrices cerradas, métricas bien calculadas. La explicación de entrenamiento contra prueba no se sostiene: hablan de sobreajuste sin ninguna brecha entre ambos. 