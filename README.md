# Dashboard de dengue — Medellín y Apartadó

Aplicación local en Streamlit que integra la base semanal, los modelos históricos y el agente V4. El paquete incluye los datos, los estados del modelo, los módulos Python y los documentos originales. No requiere subir el ZIP anterior ni ejecutar notebooks para consultar las series y probabilidades.

## Inicio rápido en Windows (PowerShell)

Instala Python 3.11 o 3.12. Descomprime el ZIP y abre una terminal en la carpeta que contiene `app.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No hace falta activar el entorno ni cambiar la política de ejecución de PowerShell. Abre la dirección local que muestra la terminal, normalmente http://localhost:8501. Para detener la aplicación usa Ctrl+C.

## macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Activar el chat (opcional)

Las probabilidades, gráficas, tablas y exportaciones funcionan sin Groq. El mapa base requiere internet.

Para añadir el chat:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-chat.txt
```

Copia `.streamlit/secrets.toml.example` como `.streamlit/secrets.toml` y reemplaza `TU_CLAVE` por tu clave Groq. También puedes definir `GROQ_API_KEY` como variable de entorno. Nunca compartas el archivo con la clave ni lo subas a GitHub; está excluido en `.gitignore`.

Reinicia la aplicación si cambiaste dependencias. En la pestaña Chat selecciona un modelo Groq disponible con soporte de herramientas y pulsa **Activar chat**. El valor inicial es `openai/gpt-oss-120b`; su disponibilidad y límites dependen de tu cuenta. Se comprueba su presencia al activar.

La primera activación descarga el modelo de embeddings y construye el índice FAISS en CPU. Necesita internet, memoria y puede tardar varios minutos. Si falla, los apartados numéricos permanecen disponibles. La clave se usa solo en el servidor Python local, no se muestra en la interfaz ni se incluye en las descargas.

Las consultas pueden esperar aproximadamente un minuto por los controles de uso de V4. La aplicación está pensada para demostración local de un usuario: el regulador es por sesión y no coordina cuotas con otros programas o usuarios que compartan la misma clave. No se ha desplegado públicamente ni se incluye autenticación.

## Qué muestra

1. **Estimación:** probabilidades bayesiana y de persistencia, semana objetivo, umbral y cortes. Los porcentajes se calculan en Python.
2. **Series y cobertura:** casos, temperatura y lluvia en gráficas separadas; fechas filtradas por inicio semanal; tabla de cobertura diaria y descarga CSV.
3. **Mapa:** puntos urbanos aproximados de Medellín y Apartadó para una misma semana de origen. Color continuo según probabilidad, sin categorías oficiales. No representa estaciones ni límites municipales. Otros municipios no tienen estimación, no riesgo cero.
4. **Chat y documentos:** agente V4, historial visual de hasta diez consultas, fuentes PDF y exportación de trazas.
5. **Método:** alcance y limitaciones.

Los intervalos de las gráficas no se extienden más allá de la semana de origen elegida. Para explorar otro periodo elige primero una semana de origen posterior. La estimación de probabilidades siempre corresponde al selector de origen, no al intervalo de las gráficas.

Los selectores no modifican automáticamente las preguntas del chat. Incluye municipio y fecha: cada consulta se procesa independientemente, aunque las anteriores continúen visibles.

Ejemplos:

- `Para Medellín, con origen 2025-SE10, estima la probabilidad de casos elevados.`
- `Cuántos casos hubo en Apartadó entre 2025-01-01 y 2025-03-31?`
- `Consulta temperatura y lluvia de Medellín entre 2024-05-01 y 2024-05-31.`
- `¿Qué relación entre clima y dengue describe el boletín de 2021, semana 45?`

## Archivos

- `app.py`: interfaz completa, filtros, gráficas, mapa, exportaciones y chat.
- `dashboard_core.py`: integración del servicio, catálogo de semanas disponibles y datos del mapa.
- `dengue_tools/`: consulta e inferencia histórica; `data/` contiene base y estados del modelo.
- `rag_dengue/`: código del agente V4, índice documental y controles de contexto.
- `corpus/`: PDF, texto extraído y metadatos.
- `herramientas_datos.json`: esquemas de herramientas.
- `requirements.txt`: dependencias del dashboard numérico.
- `requirements-chat.txt`: dependencias adicionales del agente.
- `tests/test_dashboard.py`: pruebas de integración e interfaz.
- `VERIFICACION.txt`: resultado de las comprobaciones locales y límites de la verificación.
- `versiones_verificadas.json`: versiones usadas en las comprobaciones locales.

## Pruebas

Desde la carpeta de la aplicación:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Comprueban el total de 152 casos de Apartadó, probabilidades conocidas, transición 2023-SE52 a 2024-SE01, cobertura incompleta, mapa sin datos y cambios de filtros de la interfaz con Streamlit AppTest. No usan la API ni demuestran calidad semántica del RAG. La carga del mapa externo y el chat real se deben comprobar en tu navegador y con tu cuenta.

## Limitaciones que deben mantenerse visibles

- Solo Medellín y Apartadó; no cobertura de todo Antioquia.
- Análisis retrospectivo con datos consolidados, sin reproducir retrasos reales de notificación.
- La probabilidad de exceder el umbral no es el número de casos futuros ni una declaración oficial de brote.
- El calendario proviene de la base; no se sustituyen semanas epidemiológicas por ISO.
- Tener un agregado semanal de clima no implica siete días observados. Los faltantes no se rellenan con cero.
- V4 conserva pendientes: fragmentos irrelevantes, pregunta sobre ajuste 7 y ausencia de síntesis documental. La interfaz no corrige esas limitaciones y no las presenta como resueltas.
- Algunas consultas numéricas del chat son encaminadas por Python y no elegidas por el LLM. Las trazas distinguen su origen; debe declararse al evaluar el requisito de herramientas del agente.
- No debe contarse `evidencia_documental` como respuesta correcta automáticamente. Requiere revisión humana.
- La interfaz usa resultados de modelos previamente ajustados. No reentrena ni cambia sus probabilidades.

## Fuentes técnicas

- https://docs.streamlit.io/develop/api-reference/chat/st.chat_input
- https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource
- https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart
- https://docs.streamlit.io/develop/api-reference/app-testing

Uso de IA: asistencia de ChatGPT en implementación, documentación y pruebas; el equipo debe ejecutar y verificar la aplicación y los resultados antes de entregarlos.
