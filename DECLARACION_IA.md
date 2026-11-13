# Declaración de uso de IA generativa

Conforme al pacto pedagógico de SI3003, el equipo declara las herramientas de IA generativa usadas, su propósito y cómo se verificó lo producido. El equipo es responsable del contenido entregado y de entenderlo.

## Herramientas y propósito

| Herramienta | Propósito | Partes del repositorio |
|---|---|---|
| **ChatGPT (OpenAI)** | Asistencia en la implementación del servicio de consulta e inferencia, el agente V4 (bucle, herramientas, control de presupuesto, validación de citas), el dashboard, la documentación y las pruebas de integración | `dengue_tools/core.py`, `rag_dengue/`, `app.py`, `dashboard_core.py` [COMPLETAR: ajustar al uso real] |
| **Claude (Anthropic)** | Revisión del repositorio frente a la rúbrica; reconstrucción del código de entrenamiento a partir de los artefactos ya ajustados; notebook de entrenamiento y evaluación; selector del modo "Agente LLM" en el chat; script de evaluación del agente; reescritura del README; borrador del informe | `dengue_tools/entrenamiento.py`, `notebooks/01_red_bayesiana_entrenamiento_evaluacion.ipynb`, `evaluar_agente.py`, cambios en `app.py`, `README.md`, esta declaración |
| **Groq (`openai/gpt-oss-120b`)** | Modelo de lenguaje **dentro** de la aplicación: es el motor del agente, no una ayuda para escribir el código | `rag_dengue/agent.py` en tiempo de ejecución |
| **sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`)** | Embeddings del corpus para la recuperación (RAG) | `rag_dengue/retrieval.py` |

## Decisiones del equipo (no delegadas a la IA)

[COMPLETAR por el equipo: elección del problema y de los municipios, definición del evento (P75 de 2022–2023), selección de variables, diseño de la validación temporal por bloques, selección del corpus documental y criterios de las 15 preguntas de evaluación.]

## Caso concreto: el código de entrenamiento

Los estados del modelo (`estados_experimento.json`) y las predicciones (`predicciones_temporales.csv`) estaban en el repositorio, pero no el código que los generó. Con Claude se reconstruyó ese código en `dengue_tools/entrenamiento.py`, deduciendo cada regla a partir de los artefactos y de los metadatos que los acompañan (terciles para discretizar, Laplace α = 1, calibración logística regularizada hacia la identidad, corte que maximiza la balanced accuracy del historial). El notebook verifica con `assert` que el reentrenamiento desde la base produce **los mismos** estados (diferencia 0 en CPT y cortes) y **las mismas** 1 470 predicciones (diferencia < 1e-5). La reconstrucción no se acepta por confianza: se acepta porque reproduce los resultados.

## Errores de la IA detectados y corregidos

- La primera versión de la interpretación en el notebook afirmaba que "el clima y la calibración ayudan". Al revisar la tabla por municipio se vio que la mejora global venía de Medellín y que en Apartadó ambas **empeoran** el Brier; además, el bootstrap mostró que la diferencia no es significativa. El texto se corrigió para reflejar esos resultados.
- [COMPLETAR por el equipo: otros casos detectados durante el desarrollo con ChatGPT.]

## Verificación

- Notebook ejecutado de principio a fin sin errores; reproducción exacta de los artefactos comprobada con `assert`.
- Dashboard probado con Streamlit AppTest (carga, cambio de municipio, sin excepciones).
- Bucle del agente probado con un LLM simulado (llamadas a herramientas, error devuelto como observación, validación de citas) y con Groq en la aplicación desplegada.
- Evaluación con las 15 preguntas: `resultados/evaluacion_agente_llm.csv`, con revisión humana de las respuestas [COMPLETAR tras ejecutarla].
