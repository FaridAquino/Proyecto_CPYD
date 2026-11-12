# Guion de exposición oral — Proyecto 5: KNN paralelo con MPI

**Duración total:** 12–15 min · **Diapositivas:** 1–25 en la exposición; 26–30 son de anexo, solo para preguntas (códigos de Beta 1 y Beta 2, derivación de $s^*$ y tabla completa).

| Integrante | Bloque | Diapositivas | Tiempo |
|---|---|---|---|
| Juan David Velo Poma | Problema y diseño paralelo | 1–8 | 4–5 min |
| Farid Jack Aquino Castro | Implementación, speedup y normalización | 9–16 | 4–5 min |
| Josue Nehemias Velo Poma | Rendimiento, causas y conclusiones | 17–25 | 4–5 min |

**Reglas para el equipo**
- Hablar mirando al público y señalar la figura; la diapositiva es apoyo, no se lee.
- Cada uno cierra su bloque con una frase de transición hacia el siguiente.
- Los tres deben poder deducir $s^*$ en la pizarra y explicar por qué se satura el speedup.

---

## Bloque 1 — Juan David (diapositivas 1–8)

**1 · Portada**
- "Buenos días. Somos Juan David, Farid y Josue, y presentamos la paralelización de KNN con MPI."

**2 · Ruta de la presentación**
- "Vamos a responder una sola pregunta: cuánto se puede acelerar KNN con MPI y qué lo limita."
- "Les adelanto la respuesta: hasta 5.3 veces, con un óptimo de 12 procesos."
- "Y la sorpresa: el límite no es MPI, es la memoria del procesador."

**3 · Separador — Parte 1** (pasar directo)

**4 · El problema**
- "KNN no entrena: todo el costo está en clasificar, y cada consulta es independiente de las demás."
- "Cada distancia cuesta 3d operaciones; el trabajo total es W = 3·m·n·d."
- "El problema original es tan pequeño que el arranque de MPI dominaría; por eso escalamos los datos."

**5 · Metodología PCAM**
- "Partimos en tareas de una distancia, comunicamos con Bcast, Scatterv y Gatherv, y aglomeramos bloques de consultas por proceso."
- "Así la granularidad es alta: mucho cómputo por cada mensaje."
- "En la taxonomía de Flynn es MIMD con paradigma SPMD."

**6 · PRAM CREW y Brent**
- "Todos leen el entrenamiento a la vez (CR) y cada procesador escribe solo su resultado (EW)."
- "Brent dice que el tiempo es W/s más el span: el modelo predice escalamiento casi ideal."
- "Guarden esta predicción; en la parte final veremos por qué la realidad es distinta."

**7 · Algoritmo con pardo**
- "Aplicamos el feedback del profesor: todos los bucles concurrentes usan forall … pardo."
- "Hay dos niveles: forall p para los procesos y forall j para las consultas de cada proceso."
- "Los counts y displacements permiten repartir aunque m no sea divisible entre s."

**8 · Dos descomposiciones**
- "La Beta 3 replica el entrenamiento; la variante del DAG lo reparte y fusiona candidatos."
- "La variante DAG da el mismo resultado porque los k vecinos globales siempre están entre los candidatos locales."
- *Transición:* "Farid les mostrará cómo lo implementamos y qué medimos."

## Bloque 2 — Farid (diapositivas 9–16)

**9 · Separador — Parte 2** (pasar directo)

**10 · Tres versiones beta**
- "Beta 1 paraleliza el código original tal cual: 6.7 veces más rápido con 8 procesos."
- "Beta 2 vectoriza con NumPy y gana 156 veces; Beta 3 usa buffers contiguos y gana otro 33 %."
- "Todas dan 0.9833 de exactitud, y Beta 3 y DAG coinciden al 100 % con scikit-learn."

**11 · Entorno experimental**
- "Un i7 híbrido: 8 núcleos de rendimiento con SMT y 8 de eficiencia, 24 hilos en total."
- "291 ejecuciones de 1 a 24 procesos, con escalabilidad fuerte y débil."
- "Además medimos las colectivas sin cómputo y fijamos procesos a núcleos concretos."

**12 · Separador — Parte 3** (pasar directo)

**13 · Tiempos**
- "Con 200 mil muestras bajamos de 98.5 a 18.6 segundos."
- "Más del 99 % del tiempo es cómputo, así que la comunicación no explica nada."
- "Lo clave: desde 8 procesos el cómputo deja de bajar, aunque cada proceso haga menos trabajo."

**14 · Speedup y Amdahl**
- "El speedup se estanca en 5.3, y los tres tamaños de problema dan la misma curva."
- "Amdahl ajusta una fracción serial de 0.14, pero nuestro código serial real es menor al 1 %."
- "Esa fracción es efectiva: los procesos se turnan para acceder a la memoria."

**15 · Normalización y $s^*$**
- "Normalizamos el modelo como T = k1·W/s + k2·T_comm, con α y β medidos por separado."
- "Derivando e igualando a cero: s* = √(k1·W / (k2·c))." *(señalar el recuadro)*
- "El modelo lineal predice 11.8 y el mínimo medido es 12."

**16 · Interpretación**
- "1/k1 da unos 12 GFLOP/s, que es exactamente lo que rinde un núcleo P solo."
- "k2 vale unos 120 en vez de 1: el costo real está en el bus de memoria, el cuello de botella de von Neumann."
- *Transición:* "Recomendamos 12 procesos. Josue explicará por qué no conviene usar más."

## Bloque 3 — Josue (diapositivas 17–25)

**17 · FLOP/s y eficiencia**
- "Pasamos de 12.5 a 66 GFLOP/s, y la variante DAG llega a 70."
- "Eso es el 90 % del techo que medimos para esta máquina, 74 GFLOP/s."
- "La eficiencia cae a 0.21 con 24 procesos: no hay escalabilidad fuerte."

**18 · Escalabilidad débil y Gustafson**
- "Como el trabajo crece con N al cuadrado, para fijar el trabajo por proceso escalamos N con √s."
- "Aun así, el tiempo pasa de 5.9 a 29.4 segundos: eficiencia de 0.20."
- "Aumentar el problema no añade canales de memoria, por eso Gustafson tampoco se cumple."

**19 · Separador — Parte 4** (pasar directo)

**20 · Comunicación lineal vs. árbol**
- "Medimos el Bcast solo: de 2 a 24 procesos crece 24 veces, es lineal y no logarítmico."
- "El modelo lineal de clase ajusta con R² de 0.994; el de árbol, con 0.90."
- "En memoria compartida las copias compiten por el bus DDR5, pero en el KNN pesan menos del 1 %."

**21 · Justificación de hardware**
- "Lo que dice la diapositiva del profesor, lo medimos: el ancho de banda baja de 12.3 a 7.8 GFLOP/s por núcleo."
- "SMT solo suma 30 %, y un núcleo E rinde la mitad que uno P, lo que desbalancea la carga."
- "Por eso el óptimo, 12, es justo el punto intermedio que anticipa la clase."

**22 · Beta 3 vs. DAG y mejoras**
- "La variante DAG es 2 a 6 % más rápida: no difunde el entrenamiento y cada proceso lee menos memoria."
- "Proponemos repartir más consultas a los núcleos P, fijar afinidad y bloquear por caché."
- "En un clúster cada nodo tendría su propia memoria, y ahí sí escalaría."

**23 · Conclusiones**
- "Diseño correcto y validado al 100 %, y un modelo teórico que predice el óptimo con 6 % de error."
- "El speedup se satura en 5.3 por el hardware, no por MPI."
- "Recomendación final: usar 12 procesos, no los 24 hilos."

**24 · Fuentes y bibliografía destacada**
- "Nuestra base es el material del curso, la documentación de mpi4py, Amdahl y Gustafson, y la especificación del procesador; la bibliografía completa está en el informe."

**25 · Cierre**
- "Muchas gracias. Quedamos atentos a sus preguntas."

---

## Preguntas trampa esperadas del Prof. Fiestas y cómo responderlas

**1. ¿Por qué el Bcast escala linealmente, O(s), si en clase vimos que MPI usa árboles binomiales, O(log s)?**
> El árbol supone que cada etapa usa enlaces independientes, como en un clúster. En una sola máquina
> no los hay: Intel MPI copia el mensaje a cada proceso a través de memoria compartida, y las s−1
> copias compiten por el mismo bus DDR5. Por eso se comporta como la comunicación lineal de la
> Unidad 2: lo medimos con R² = 0.994, frente a 0.901 del árbol. *(Diap. 20)*

**2. ¿Por qué para Gustafson escalaron N como N0·√s y no como N0·s?**
> Porque el trabajo de KNN es W = 3·m·n·d, y m y n son proporcionales a N, así que W ∝ N². Para que
> el trabajo por proceso W/s quede constante se necesita N² ∝ s, es decir, N ∝ √s. Con N/s
> constante, el trabajo por proceso crecería con s. Medimos ambas variantes para mostrarlo.
> *(Diap. 18)*

**3. ¿Por qué el modelo teórico clásico predice un óptimo de miles de procesos y en el hardware real es 12?**
> El modelo clásico, con solo el costo de los mensajes (k2 ≈ 1), supone núcleos idénticos e
> independientes, sin competir por la memoria. Con esa comunicación tan barata, el óptimo sale en
> miles. Al normalizar con los datos, k2 vale unos 120 porque absorbe la contención del bus de
> memoria; con el modelo lineal el óptimo baja a s* = 11.8, igual al medido. *(Diap. 15–16 y 29)*

**4. ¿Por qué la variante DAG superó a la Beta 3 con muchos procesos?**
> Por dos razones: evita el Bcast de X_tr, que en memoria compartida cuesta (s−1)·n·d·β, y cada
> proceso recorre solo n/s puntos de entrenamiento, lo que reduce la presión sobre la caché L3 y
> la RAM, justo el recurso que limita. Su fusión serial crece con s, pero con k = 3 es
> despreciable: 0.056 s con 24 procesos. *(Diap. 22)*

**5. ¿Cómo validaron que no hay pérdida de precisión?**
> Todas las versiones dan exactamente 0.9833 de exactitud, igual que el secuencial. Además, Beta 3
> y DAG coinciden en el 100 % de las predicciones con KNeighborsClassifier de scikit-learn, también
> con repartos no divisibles, por ejemplo 360 consultas entre 7 procesos. *(Diap. 10; código en diap. 27–28)*

### Otras preguntas probables

| Pregunta | Respuesta breve |
|---|---|
| Deduzcan s* en la pizarra. | T = k1·W/s + k2·c·(s−1) → −k1·W/s² + k2·c = 0 → s* = √(k1·W/(k2·c)) = √(103.8/0.748) ≈ 11.8. *(Diap. 29)* |
| ¿Por qué s* da 11.8 y no 8, si hay 8 núcleos P? | Los núcleos E todavía suman algo de rendimiento hasta ~12; entre 12 y 24 el tiempo varía menos del 5 %. |
| ¿f_s = 0.14 significa que el 14 % del código es serial? | No: el código serial real es menor al 1 %. Es una fracción serial efectiva por contención de memoria. |
| ¿Qué es el pardo interno si cada proceso usa un solo núcleo? | Paralelismo de datos: las consultas son independientes y se calculan como producto matricial (BLAS/SIMD). |
| ¿Cómo midieron T_comm? | Con MPI.Wtime por fase, tomando el máximo entre procesos; incluye la espera por el proceso más lento. |
| ¿Cómo cuentan los FLOPs? | 3d por par: d restas, d multiplicaciones, d−1 sumas y una raíz. Total 3·m·n·d, dividido por T_comp. |
| ¿Por qué datos sintéticos? | El KNN vectorizado sobre load_digits tarda milisegundos; los sintéticos conservan la media y la covarianza de cada clase. |
| ¿Cómo usaron la IA? | Responder según la Sec. 5.3 del informe; los tres deben dar la misma respuesta. |
