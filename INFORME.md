# Vigilancia retrospectiva de dengue con redes bayesianas y un agente LLM con RAG

**SI3003 Inteligencia Artificial · Universidad EAFIT · 2026** — Equipo: [COMPLETAR] · Grupo [COMPLETAR]
Aplicación: https://iaproyectofinal-huv6agy6wrujw5htcpjurm.streamlit.app · Código: https://github.com/Jaimeajr05/IA_PROYECTO_FINAL

## 1. Problema

El dengue es endémico en Antioquia y presenta picos que exigen anticipación local. Planteamos una pregunta acotada y verificable: **dada la información de la semana *t* en un municipio, ¿cuál es la probabilidad de que la semana *t+1* tenga casos elevados?** Se define "casos elevados" como casos semanales estrictamente mayores que el percentil 75 de 2022–2023 del municipio (7.25 en Medellín y 9 en Apartadó). La solución combina un modelo probabilístico clásico (Parte 1), un agente LLM que consulta datos, ejecuta el modelo y cita documentos oficiales (Parte 2) y un dashboard. Es una herramienta analítica retrospectiva: **no es una alerta oficial ni una guía clínica**.

## 2. Datos

- **Casos:** registros semanales de dengue (confirmados y probables) de Medellín (05001) y Apartadó (05045), 2022–2025, 209 semanas por municipio [COMPLETAR fuente exacta: SIVIGILA/INS, URL y fecha de consulta].
- **Clima:** temperatura media y precipitación semanales con número de días observados y estado de aprobación [COMPLETAR fuente: IDEAM, estaciones]. Apartadó tiene 2 semanas sin datos climáticos; no se imputan.
- **Corpus RAG:** 9 PDF del Instituto Nacional de Salud (318 páginas): protocolo de vigilancia de dengue (v07, julio de 2024), ficha de notificación y 7 boletines epidemiológicos semanales (2021–2026).

Solo se usan agregados semanales por municipio. La base lleva una huella SHA-256 que la aplicación y el notebook verifican.

Un hecho de los datos condiciona toda la evaluación: la prevalencia del evento en **Medellín pasó de 4–44 % (2022–2023) a 98–100 % (2024–2025)**, consecuencia de la epidemia de 2024. En Apartadó osciló entre 33 % y 60 %.

## 3. Método de la Parte 1: red bayesiana

**Variables** (semana de origen *t*): `casos_actuales`; `cambio_casos` = casos actuales − promedio de las tres semanas anteriores; `lluvia_4s` = precipitación acumulada de cuatro semanas; `temperatura_4s` = temperatura media de cuatro semanas.

**Modelo.** Naive Bayes discreto, es decir, una red bayesiana en la que el evento *y* es el único padre de cada variable: P(y | x) ∝ P(y) Π P(x<sub>j</sub> | y). Cada variable se discretiza en terciles del entrenamiento del municipio; prior y tablas condicionales usan suavizado de Laplace (α = 1); la inferencia es exacta en espacio logarítmico y una variable faltante se marginaliza. Hay dos redes: solo casos y casos + clima. La implementación es propia (`dengue_tools/entrenamiento.py`), sin librerías de redes bayesianas.

**Calibración y decisión.** Una calibración logística p' = σ(a·logit p + b), regularizada hacia la identidad, se ajusta con predicciones **fuera de muestra** de bloques anteriores, y solo si hay ≥ 40 semanas y ≥ 10 por clase. El corte de clasificación maximiza la balanced accuracy de ese mismo historial.

**Líneas base.** *Prevalencia* (frecuencia histórica del evento), *persistencia* (la semana siguiente repite el estado actual) y *persistencia probabilística* (P(y<sub>t+1</sub> | estado<sub>t</sub>), una cadena de Markov de dos estados).

**Validación temporal.** 2024–2025 se divide en 8 bloques de 13 semanas. Cada bloque se predice entrenando solo con transiciones cuyo objetivo es anterior al bloque (ventana creciente de 200 a 382 transiciones), con lo que se evita la fuga de información. Métrica principal: **Brier**, porque el producto muestra probabilidades. Se complementa con log-loss, balanced accuracy y un IC 95 % por bootstrap de bloques móviles de 4 semanas.

## 4. Diseño del agente y del RAG

**Recuperación.** El texto de cada página se divide en fragmentos de 110 tokens con solapamiento de 20 (sin cruzar páginas), se embebe con `paraphrase-multilingual-MiniLM-L12-v2` (vectores normalizados) y se indexa en FAISS `IndexFlatIP` (coseno exacto: con 318 páginas de corpus la búsqueda exacta es rápida y no necesita un índice aproximado). La búsqueda admite filtros por tipo de documento, año y semana. Cada fragmento conserva un identificador citable `DOC:página:fragmento`.

**Herramientas** (esquemas JSON, *tool calling* con Groq): `consultar_serie(municipio, fecha_inicio, fecha_fin)` devuelve datos observados y cobertura; `estimar_probabilidad(municipio, semana)` ejecuta la red bayesiana y la persistencia; `buscar_documentos(consulta, tipo, año, semana)` es el RAG.

**Bucle agéntico** (`rag_dengue/agent.py`): el LLM decide qué herramientas llamar, recibe cada resultado o error como observación (por ejemplo, `MUNICIPIO_NO_CUBIERTO` para Turbo) y continúa hasta responder. Controles: T<sub>máx</sub> = 6 iteraciones y 10 llamadas; presupuesto de contexto con compactación; regulador de tokens por minuto y reintentos acotados ante errores 413, 429 y 400; reparación de argumentos inválidos. **Citas:** cada afirmación debe citar `[DOC:pN:cK]`, `[DATOS n]` o `[MODELO n]`; una cita que no corresponde a una fuente realmente recuperada provoca un pedido de reescritura y queda registrada como inválida. El chat ofrece dos modos: *Agente LLM* y *Controlado*, en el que Python formatea las cifras y los documentos se muestran como extractos literales.

**Dashboard (Streamlit).** Estimación con ambos modelos; series de casos, temperatura y lluvia con cobertura; mapa con escala continua de probabilidad; chat con fuentes descargables y trazas; pestaña de método y limitaciones. El usuario elige municipio, semana de origen, intervalo, modelo del mapa y modo del agente.

## 5. Resultados

### 5.1 Modelo (210 predicciones fuera de muestra por modelo)

| Modelo | Brier total | Brier Apartadó | Log-loss | Balanced acc. |
|---|---:|---:|---:|---:|
| Persistencia probabilística | **0.123** | **0.198** | **0.404** | **0.835** |
| NB casos + clima, calibrado | 0.144 | 0.242 | 0.443 | 0.744 |
| NB solo casos, calibrado | 0.151 | 0.235 | 0.454 | 0.687 |
| NB solo casos | 0.162 | 0.221 | 0.477 | 0.767 |
| Prevalencia | 0.294 | 0.285 | 0.790 | 0.632 |

- **Frente a la prevalencia**, la red bayesiana es claramente mejor: ΔBrier = −0.150, IC 95 % [−0.204, −0.100]; Brier skill = 0.51.
- **Frente a la persistencia probabilística, no gana**: ΔBrier = +0.021, IC 95 % [−0.008, +0.051]. La diferencia **no es significativa**. Con series semanales muy autocorrelacionadas, el estado actual es difícil de superar a un paso.
- **El clima no mejora de forma demostrable**: ΔBrier (con clima − sin clima) = −0.008, IC 95 % [−0.039, +0.023]. La aparente mejora global proviene de Medellín; en Apartadó el mejor Naive Bayes es el más simple (0.221), y el clima y la calibración lo empeoran (0.242). Explicación probable: la calibración es global y el historial está dominado por Medellín, con prevalencia ≈ 99 %, lo que eleva las probabilidades también en Apartadó.
- **Reproducibilidad:** el notebook reentrena desde la base y reproduce exactamente los estados y las 1 470 predicciones que usa la aplicación (diferencia 0 en tablas condicionales y cortes; < 1e-5 en probabilidades).

![Brier por bloque](resultados/brier_por_bloque.png)

### 5.2 Agente (15 preguntas de prueba)

Las 15 preguntas cubren búsqueda documental con filtros (P01–P05, P13), datos (P06–P07), modelo (P08, P15), integración modelo + documentos (P09), fuera de cobertura (P10, P11), pregunta ambigua (P12) y una instrucción adversaria (P14). Para cada una se registra si llamó las herramientas esperadas, si citó el documento esperado y si las citas son verificables.

[COMPLETAR tras ejecutar `python evaluar_agente.py`: % de preguntas con las herramientas correctas, % que citan el documento esperado, % con citas verificables y la revisión humana de las respuestas. Comentar 2 o 3 casos (p. ej., P09 integra modelo y protocolo; P10 Turbo; P14 instrucción adversaria).]

## 6. Limitaciones

Dos municipios; análisis retrospectivo con datos consolidados, sin retraso real de notificación; umbral fijo que no se adapta al cambio de régimen de 2024 (en Medellín la tarea se vuelve casi trivial); calibración agrupada entre municipios; horizonte de una semana; ninguna relación clima–casos se interpreta como causal. En el agente, el retriever puede traer fragmentos irrelevantes, y una cita válida no garantiza que la respalde: la evaluación requiere revisión humana. **Trabajo futuro:** calibración por municipio, umbral móvil, incorporar el estado actual como variable de la red y ampliar la cobertura municipal.

## 7. Uso de IA generativa

Se usaron ChatGPT y Claude como asistentes de implementación, documentación y revisión; el detalle de herramientas, propósitos, errores detectados y verificación está en `DECLARACION_IA.md`. Dentro de la aplicación, el LLM de Groq es el motor del agente.
