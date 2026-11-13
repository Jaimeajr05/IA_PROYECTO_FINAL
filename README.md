# Vigilancia retrospectiva de dengue — Medellín y Apartadó

Proyecto final de **SI3003 Inteligencia Artificial (EAFIT, 2026)**. Combina una **red bayesiana (Naive Bayes) implementada por el equipo** para estimar la probabilidad de semanas con casos elevados de dengue, un **agente LLM con RAG y tres herramientas** que consulta datos, ejecuta el modelo y cita protocolos y boletines del INS, y un **dashboard interactivo en Streamlit**.

- **Aplicación desplegada:** https://iaproyectofinal-huv6agy6wrujw5htcpjurm.streamlit.app
- **Repositorio:** https://github.com/Jaimeajr05/IA_PROYECTO_FINAL
- **Equipo:** [COMPLETAR: nombres de los integrantes] · Grupo [COMPLETAR]

> Herramienta analítica y académica. **No es una guía clínica ni un sistema de alerta oficial**: superar el umbral del proyecto no equivale a una declaración de brote.

## Cómo cumple los requisitos del curso

| Requisito | Dónde está |
|---|---|
| **Parte 1 — técnica clásica implementada por el equipo** | Red bayesiana Naive Bayes discreta (CPT con Laplace, inferencia exacta, marginalización de faltantes, calibración y corte propios) en `dengue_tools/entrenamiento.py`; entrenamiento y evaluación en `notebooks/01_red_bayesiana_entrenamiento_evaluacion.ipynb` |
| Métrica frente a línea base | Brier, log-loss y balanced accuracy contra **prevalencia** y **persistencia**, con validación temporal en 8 bloques e IC 95 % por bootstrap (ver *Resultados*) |
| **Parte 2 — agente LLM con RAG y ≥ 2 herramientas** | `rag_dengue/agent.py`: bucle agéntico con *tool calling* en Groq, herramientas `consultar_serie`, `estimar_probabilidad` y `buscar_documentos` (FAISS sobre 9 PDF del INS), T_máx de 6 iteraciones y 10 llamadas, errores de herramienta devueltos como observación y validación de citas |
| Evaluación del agente (10–15 preguntas) | 15 preguntas fijas en `rag_dengue/evaluation.py`; se ejecutan con `python evaluar_agente.py` → `resultados/evaluacion_agente_llm.csv` |
| Visualización e interacción | Dashboard: métricas, series de casos y clima, mapa, chat con fuentes y trazas; el usuario elige municipio, semana de origen, intervalo, modelo del mapa y modo del agente |

## Resultados de la Parte 1

Validación temporal *walk-forward*: 2024–2025 dividido en 8 bloques de 13 semanas; cada bloque se predice solo con datos anteriores (210 predicciones por modelo). El evento es "casos de la semana siguiente > percentil 75 de 2022–2023 del municipio". Brier: menor es mejor.

| Modelo | Brier (total) | Brier (Apartadó) | Balanced accuracy (total) |
|---|---:|---:|---:|
| Persistencia probabilística (línea base) | **0.1225** | **0.1976** | 0.835 |
| Red bayesiana con clima, calibrada | 0.1436 | 0.2418 | 0.744 |
| Red bayesiana solo casos, calibrada | 0.1514 | 0.2349 | 0.687 |
| Red bayesiana solo casos | 0.1622 | 0.2213 | 0.767 |
| Prevalencia (línea base) | 0.2938 | 0.2847 | 0.632 |

Conclusiones (detalle en el notebook):

- La red bayesiana **supera con claridad a la prevalencia**: ΔBrier = −0.150, IC 95 % [−0.204, −0.100].
- **No supera a la persistencia**: ΔBrier = +0.021, IC 95 % [−0.008, +0.051], diferencia **no significativa**. A una semana, el estado actual es un predictor muy fuerte.
- En **Medellín el 99 % de las semanas evaluadas superan el umbral** (epidemia de 2024), así que ahí la tarea es casi trivial; **Apartadó** (prevalencia ≈ 47 %) es el caso informativo, y en él el modelo más simple (solo casos) es el mejor Naive Bayes.
- El notebook **reentrena desde cero y reproduce exactamente** los estados (`estados_experimento.json`) y las 1 470 predicciones (`predicciones_temporales.csv`) que usa la aplicación.

## Datos y fuentes

Solo datos **agregados por semana y municipio**; ningún dato individual.

| Dato | Fuente | Archivo |
|---|---|---|
| Casos semanales de dengue (confirmados y probables), Medellín 05001 y Apartadó 05045, 2022–2025 | [COMPLETAR: p. ej., microdatos SIVIGILA del Instituto Nacional de Salud, evento y URL de descarga, fecha de consulta] | `dengue_tools/data/base_dengue_clima_semanal.csv` |
| Temperatura media y precipitación semanales con días observados y estado de aprobación | [COMPLETAR: p. ej., IDEAM (DHIME o datos.gov.co), estaciones usadas por municipio, URL y fecha de consulta] | mismo archivo |
| Corpus RAG: protocolo de vigilancia de dengue (v07, julio 2024), ficha de notificación y 7 boletines epidemiológicos semanales (BES 2021–2026) | Instituto Nacional de Salud (INS), https://www.ins.gov.co | `corpus/pdf/`, metadatos en `corpus/manifest.json` |

La base incluye la huella SHA-256 con la que se ajustó el modelo; la aplicación y el notebook la verifican al cargar. Los faltantes de clima no se rellenan con cero: se marginalizan en la inferencia.

## Arquitectura

```
base semanal (CSV) ──► dengue_tools/entrenamiento.py ──► estados del modelo + predicciones (dengue_tools/data/)
                                                              │
corpus PDF INS ──► rag_dengue/retrieval.py (MiniLM multilingüe + FAISS)
                                                              │
                         rag_dengue/agent.py  ◄── Groq (tool calling)
                          ├─ consultar_serie        (datos observados)
                          ├─ estimar_probabilidad   (red bayesiana vs persistencia)
                          └─ buscar_documentos      (RAG con citas DOC:página:fragmento)
                                                              │
                                   app.py (Streamlit): estimación · series · mapa · chat · método
```

## Ejecución

### Requisitos

Python 3.11 o 3.12. El dashboard numérico y el notebook no requieren internet ni claves. El chat necesita una clave de Groq y descarga el modelo de embeddings la primera vez.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Abre http://localhost:8501.

### Activar el chat

1. Copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y escribe tu `GROQ_API_KEY` (o defínela como variable de entorno). En Streamlit Community Cloud se pega en **App settings → Secrets**. El archivo con la clave está en `.gitignore`.
2. En la pestaña **Chat y documentos** elige el modo:
   - **Agente LLM (tool calling)** — el modelo decide qué herramientas llamar y redacta la respuesta con citas.
   - **Controlado** — Python enruta las consultas numéricas y la parte documental se muestra como extractos literales.
3. Escribe el modelo de Groq (por defecto `openai/gpt-oss-120b`) y pulsa **Activar chat**. La primera activación construye el índice FAISS (algunos minutos).

Ejemplos de preguntas:

- `Para Medellín, con origen 2025-SE10, estima la probabilidad de casos elevados.`
- `¿Cuántos casos hubo en Apartadó entre 2025-01-01 y 2025-03-31?`
- `Para Apartadó, con origen 2024-SE20, estima la siguiente semana y explica si eso equivale a una alerta oficial según el protocolo.`
- `¿Qué relación entre clima y dengue describe el boletín de 2021, semana 45?`

### Reproducir el modelo de la Parte 1

```bash
.venv/bin/python -m pip install jupyter
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_red_bayesiana_entrenamiento_evaluacion.ipynb
```

O abrirlo en Jupyter/Colab y ejecutar todo. Escribe tablas y gráficas en `resultados/`.

### Evaluar el agente (15 preguntas)

```bash
.venv/bin/python evaluar_agente.py              # modo agente LLM
.venv/bin/python evaluar_agente.py --modo controlado
```

Genera `resultados/evaluacion_agente_<modo>.csv` (herramientas esperadas vs llamadas, documentos citados, validez de citas), las trazas y un resumen. Las columnas `revision_humana_*` se completan a mano: la validación automática verifica que cada cita exista entre las fuentes recuperadas, no que la respalde.

## Estructura

| Ruta | Contenido |
|---|---|
| `app.py` | Interfaz Streamlit: filtros, gráficas, mapa, exportaciones y chat |
| `dashboard_core.py` | Catálogo de semanas, datos del mapa y tablas para la interfaz |
| `dengue_tools/core.py` | Servicio de consulta e inferencia con los estados ajustados (herramientas del agente) |
| `dengue_tools/entrenamiento.py` | **Entrenamiento**: variables, Naive Bayes, líneas base, calibración, corte, validación temporal y métricas |
| `dengue_tools/data/` | Base semanal, estados del modelo, predicciones temporales y auditoría de bloques |
| `notebooks/01_red_bayesiana_entrenamiento_evaluacion.ipynb` | Entrenamiento, verificación de reproducibilidad y evaluación de la Parte 1 |
| `rag_dengue/agent.py` | Agente LLM: bucle, herramientas, control de presupuesto, reintentos y validación de citas |
| `rag_dengue/controlado.py` | Modo controlado (cifras deterministas y extractos) |
| `rag_dengue/retrieval.py` | Fragmentación por tokens, embeddings e índice FAISS con filtros por tipo, año y semana |
| `rag_dengue/presupuesto.py` | Estimación de tokens y regulador de cuota de Groq |
| `rag_dengue/evaluation.py` · `evaluar_agente.py` | Preguntas de prueba y script de evaluación del agente |
| `corpus/` | PDF del INS, texto extraído por página y metadatos |
| `herramientas_datos.json` | Esquemas JSON de las herramientas de datos |
| `resultados/` | Métricas, gráficas y evaluación generadas |

## Limitaciones

- Solo Medellín y Apartadó; no hay cobertura del resto de Antioquia.
- Análisis retrospectivo con datos consolidados: no reproduce el retraso real de notificación.
- Umbral fijo (P75 2022–2023) que no se adapta a cambios de régimen; en Medellín 2024–2025 casi todas las semanas lo superan.
- La calibración es global (agrupa ambos municipios) y en Apartadó empeora el Brier; la mejora natural es calibrar por municipio.
- Horizonte de una semana; la probabilidad no es un pronóstico del número de casos.
- Una relación entre clima y casos en el modelo no demuestra causalidad.
- El agente puede recuperar fragmentos irrelevantes; las citas válidas no garantizan una respuesta suficiente. Requiere revisión humana.
- La aplicación está pensada para demostración: el regulador de cuota es por sesión y no hay autenticación.

## Declaración de uso de IA generativa

Ver [`DECLARACION_IA.md`](DECLARACION_IA.md).
