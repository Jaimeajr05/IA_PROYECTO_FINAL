"""Dashboard académico. Ejecutar: python -m streamlit run app.py"""
from pathlib import Path
import os,json,threading
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk
from dashboard_core import cargar_datos,datos_mapa,tabla_modelos,tabla_serie
from dengue_tools.core import ErrorConsulta

RAIZ=Path(__file__).resolve().parent
st.set_page_config(page_title='Dengue · Antioquia',page_icon='📊',layout='wide')

@st.cache_resource
def recursos():return cargar_datos()

@st.cache_resource
def indice_documental():
    # Importaciones pesadas solo al activar el chat.
    from rag_dengue.retrieval import IndiceDocumental
    indice=IndiceDocumental(RAIZ);indice.construir(reutilizar=True)
    return indice,threading.Lock()

def clave_configurada():
    clave=os.environ.get('GROQ_API_KEY','')
    if not clave:
        try:clave=st.secrets.get('GROQ_API_KEY','')
        except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError):pass
    return clave

def grafica(df,col,titulo,unidad,barras=False,umbral=None):
    fig=go.Figure()
    tipo=go.Bar if barras else go.Scatter
    parametros={'x':df['inicio_semana'],'y':df[col],'name':titulo}
    if not barras:parametros.update(mode='lines+markers',connectgaps=False,line={'color':'#137c91'})
    fig.add_trace(tipo(**parametros))
    if umbral is not None:fig.add_hline(y=umbral,line_dash='dash',line_color='#c97927',annotation_text='Umbral del proyecto')
    fig.update_layout(height=300,margin=dict(l=10,r=10,t=40,b=10),title=titulo,
                      xaxis_title='Inicio de semana epidemiológica',yaxis_title=unidad)
    return fig

def mostrar_respuesta(r,ident):
    estado=r.get('estado','desconocido')
    st.caption('Estado: '+estado)
    if estado in ('evidencia_documental','parcial'):
        st.info('El agente V4 muestra extractos. Su relevancia y suficiencia requieren revisión; no son una síntesis validada.')
    # No se permite HTML suministrado por el modelo o los documentos.
    st.markdown(r['respuesta'])
    with st.expander('Fuentes y trazabilidad'):
        for f in r.get('fuentes_citadas',[]):
            if f.get('tipo')=='documento':
                st.write(f"{f['id']} · {f['titulo']} · página PDF {f['pagina']}")
                ruta=(RAIZ/f['archivo_pdf']).resolve()
                if ruta.is_relative_to((RAIZ/'corpus/pdf').resolve()) and ruta.is_file():
                    st.download_button('Descargar PDF '+f['doc_id'],ruta.read_bytes(),file_name=ruta.name,
                        mime='application/pdf',key=f"pdf_{ident}_{f['id']}")
        st.dataframe(pd.DataFrame([{'herramienta':t['herramienta'],'argumentos':json.dumps(t['argumentos'],ensure_ascii=False),
                                   'correcta':t['salida'].get('ok'),'origen':t.get('origen_llamada','llm')}
                                  for t in r.get('traza',[])]),hide_index=True)
    st.download_button('Descargar respuesta y traza',json.dumps(r,ensure_ascii=False,indent=2),
        file_name='consulta_dengue.json',mime='application/json',key='json_'+str(ident))

st.title('Vigilancia retrospectiva de dengue')
st.caption('Medellín y Apartadó · Datos consolidados · Proyecto académico')
st.warning('Consulta histórica. Superar el umbral del proyecto no constituye una alerta oficial ni determina si hubo una declaración de brote.')
try:
    servicio,catalogo=recursos()
except Exception as exc:
    st.error('No fue posible cargar la base y los estados del modelo. Conserva íntegra la carpeta dengue_tools/data. '+str(exc))
    st.stop()

with st.sidebar:
    st.header('Consulta histórica')
    municipio=st.selectbox('Municipio',list(catalogo),key='municipio')
    semanas=list(catalogo[municipio])
    if not semanas:st.error('No hay semanas evaluadas.');st.stop()
    if st.session_state.get('semana') not in semanas:st.session_state['semana']=semanas[-1]
    origen=st.selectbox('Semana de origen (t)',semanas,key='semana')
    st.caption('El modelo estima la semana siguiente. Las opciones provienen del calendario y los estados disponibles.')

r=catalogo[municipio][origen]
fin_origen=pd.Timestamp(r['fin_semana_origen']).date()
min_fecha=servicio.base.loc[servicio.base.municipio.eq(municipio),'inicio_semana'].min().date()
with st.sidebar:
    st.subheader('Intervalo de las gráficas')
    # Cambiar semana/municipio reinicia este filtro para no mantener fechas incompatibles.
    firma=(municipio,origen)
    if st.session_state.get('firma_fechas')!=firma:
        st.session_state['rango']=(max(min_fecha,(pd.Timestamp(fin_origen)-pd.Timedelta(weeks=26)).date()),fin_origen)
        st.session_state['firma_fechas']=firma
    rango=st.date_input('Fechas de inicio semanal',min_value=min_fecha,max_value=fin_origen,key='rango')
    st.caption('El intervalo filtra el inicio de cada semana; se muestran datos hasta la semana de origen.')

resumen,series,mapa,chat,metodo=st.tabs(['Estimación','Series y cobertura','Mapa','Chat y documentos','Método'])
with resumen:
    st.subheader(f"{municipio}: {origen} → {r['semana_objetivo']}")
    st.caption(f"Objetivo: {r['inicio_semana_objetivo']} a {r['fin_semana_objetivo']}")
    a,b,c=st.columns(3)
    a.metric('Probabilidad bayesiana',f"{r['red_bayesiana']['probabilidad']:.2%}")
    b.metric('Persistencia probabilística',f"{r['referencia']['probabilidad']:.2%}")
    c.metric('Mínimo para casos elevados',f"{r['evento']['minimo_casos_elevados']} casos")
    st.write(f"Evento: casos de la semana objetivo estrictamente mayores que {r['evento']['umbral']:g}, percentil 75 de 2022–2023.")
    st.dataframe(tabla_modelos(r),hide_index=True)
    st.caption('Los cortes de probabilidad son distintos del umbral de número de casos. Una probabilidad mayor no demuestra mejor precisión.')
    e=r['evidencia']
    st.write(f"Casos observados en origen: **{e['casos_actuales']:g}**. No son un pronóstico del número de casos futuros.")
    st.write('Calibración bayesiana: '+('activa' if r['red_bayesiana']['calibracion_activa'] else 'inactiva'))
    if r['variables_marginalizadas']:st.info('Variables faltantes marginalizadas: '+', '.join(r['variables_marginalizadas']))
    with st.expander('Evidencia y auditoría del modelo'):
        st.json({'evidencia':e,'trazabilidad':r['trazabilidad']})
        st.caption('cambio_casos = casos actuales − promedio de las tres semanas anteriores. Lluvia y temperatura 4s incluyen la semana de origen.')
    st.download_button('Descargar estimación',json.dumps(r,ensure_ascii=False,indent=2),file_name=f'estimacion_{r["codigo_municipio"]}_{origen}.json',mime='application/json')

with series:
    if not isinstance(rango,(list,tuple)) or len(rango)!=2:
        st.info('Selecciona inicio y fin para consultar las series.')
    else:
        try:
            consulta,df=tabla_serie(servicio,municipio,rango[0],rango[1])
        except ErrorConsulta as exc:st.info(str(exc))
        else:
            x,y,z=st.columns(3)
            x.metric('Casos registrados en el intervalo',consulta['casos_registrados_total'])
            y.metric('Semanas',consulta['numero_semanas'])
            z.metric('Semanas sin temperatura',consulta['semanas_sin_temperatura'])
            st.plotly_chart(grafica(df,'casos_registrados','Casos registrados','Casos',umbral=r['evento']['umbral']),width='stretch')
            l,d=st.columns(2)
            with l:st.plotly_chart(grafica(df,'temperatura_media_c','Temperatura media','°C'),width='stretch')
            with d:st.plotly_chart(grafica(df,'precipitacion_total_mm','Precipitación total','mm',True),width='stretch')
            st.subheader('Cobertura y datos observados')
            incomplete=(df['dias_temperatura'].lt(7)|df['dias_lluvia'].lt(7)|df['temperatura_media_c'].isna()|df['precipitacion_total_mm'].isna())
            if incomplete.any():st.info(f'{int(incomplete.sum())} semanas tienen cobertura diaria incompleta o algún valor climático faltante. Los faltantes no se convierten en cero.')
            st.caption('Una semana con un agregado climático no implica siete días observados. Los datos climáticos pueden ser preliminares o estar en revisión.')
            st.dataframe(df,hide_index=True)
            st.download_button('Descargar serie CSV',df.to_csv(index=False).encode('utf-8-sig'),file_name=f'serie_{consulta["codigo_municipio"]}.csv',mime='text/csv')

with mapa:
    st.subheader('Probabilidades por municipio')
    seleccion=st.radio('Modelo del mapa',['Bayesiano','Persistencia'],horizontal=True,key='modelo_mapa')
    campo='red_bayesiana' if seleccion=='Bayesiano' else 'referencia'
    m=datos_mapa(catalogo,origen,campo)
    layer=pdk.Layer('ScatterplotLayer',data=m.to_dict('records'),get_position='[lon, lat]',
                    get_fill_color='color',get_radius=11000,pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[layer],initial_view_state=pdk.ViewState(latitude=7.05,longitude=-76.1,zoom=6),
        map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
        tooltip={'text':'{municipio}\nObjetivo: {objetivo}\nProbabilidad: {porcentaje}'}))
    st.caption(f'Origen común: {origen}. Azul: cercano a 0 %; rojo: cercano a 100 %; gris: sin estimación. Escala continua, sin categorías oficiales de riesgo.')
    st.caption('Puntos urbanos aproximados, no estaciones ni límites municipales. Los municipios no representados carecen de estimación; no equivalen a riesgo cero. El mapa base requiere internet.')
    st.dataframe(m[['municipio','objetivo','porcentaje']],hide_index=True)

with chat:
    st.subheader('Consulta al agente')
    MODOS={'Agente LLM (tool calling)':'llm','Controlado (cifras en Python + extractos)':'controlado'}
    modo_txt=st.radio('Modo del agente',list(MODOS),horizontal=True,key='modo_agente',
        help='Agente LLM: el modelo decide qué herramienta llamar (consultar_serie, estimar_probabilidad, buscar_documentos), '
             'recibe los errores como observación y debe citar cada afirmación. Controlado: Python enruta las cifras y '
             'la parte documental se muestra como extractos literales.')
    if MODOS[modo_txt]=='llm':
        st.info('El LLM elige las herramientas en un bucle agéntico (máximo 6 iteraciones y 10 llamadas). Las citas se validan contra las fuentes realmente recuperadas; su respaldo semántico requiere revisión humana.')
    else:
        st.info('Modo controlado: cifras formateadas por Python y extractos documentales literales, sin síntesis del LLM.')
    st.caption('Cada pregunta se procesa de forma independiente. Los selectores del dashboard no se añaden a la pregunta: incluye municipio y semana o fechas.')
    st.code(f'Para {municipio}, con origen {origen}, estima la probabilidad de casos elevados.\n¿Qué relación entre clima y dengue describe el boletín de 2021, semana 45?')
    modelo=st.text_input('Modelo Groq con herramientas',value=os.environ.get('GROQ_MODEL','openai/gpt-oss-120b'),key='modelo_chat')
    if st.session_state.get('agente') is not None and st.session_state.get('modo_activo')!=MODOS[modo_txt]:
        st.warning('Cambiaste el modo: pulsa **Activar chat** para aplicarlo.')
    if st.button('Activar chat',key='activar_chat'):
        clave=clave_configurada()
        if not clave:st.error('Configura GROQ_API_KEY en .streamlit/secrets.toml o como variable de entorno. Consulta README.md.')
        else:
            try:
                with st.spinner('Cargando embeddings y preparando el agente. La primera ejecución descarga el modelo de embeddings.'):
                    from groq import Groq
                    from rag_dengue.agent import AgenteDengue
                    from rag_dengue.controlado import AgenteControlado
                    cliente=Groq(api_key=clave,timeout=45,max_retries=0)
                    disponibles={x.id for x in cliente.models.list().data}
                    if modelo not in disponibles:raise ValueError('El modelo seleccionado no está disponible en esta cuenta.')
                    indice,lock=indice_documental()
                    clase=AgenteDengue if MODOS[modo_txt]=='llm' else AgenteControlado
                    st.session_state['agente']=clase(cliente,indice,RAIZ,modelo=modelo)
                    st.session_state['chat_lock']=lock
                    st.session_state['modelo_activo']=modelo
                    st.session_state['modo_activo']=MODOS[modo_txt]
                st.success('Chat activado.')
            except Exception as exc:
                st.error(f'No se pudo activar el chat ({type(exc).__name__}). Revisa dependencias, clave, modelo y conexión. La parte numérica del dashboard sigue disponible.')
    activo='agente' in st.session_state
    if activo:st.caption('Modelo activo: '+st.session_state['modelo_activo']+' · modo: '+st.session_state.get('modo_activo','controlado')+'. Para cambiarlos, vuelve a activar el chat.')
    if st.button('Limpiar historial visual',key='limpiar'):st.session_state['historial']=[]
    historial=st.session_state.setdefault('historial',[])
    for i,item in enumerate(historial):
        with st.chat_message('user'):st.write(item['pregunta'])
        with st.chat_message('assistant'):mostrar_respuesta(item,i)
    pregunta=st.chat_input('Incluye municipio y semana AAAA-SEnn o dos fechas AAAA-MM-DD',disabled=not activo,max_chars=4000)
    if pregunta:
        try:
            with st.spinner('Consultando. El regulador de Groq puede esperar aproximadamente un minuto entre llamadas.'):
                with st.session_state['chat_lock']:
                    respuesta=st.session_state['agente'].preguntar(pregunta)
            historial.append(respuesta)
            st.session_state['historial']=historial[-10:]
            st.rerun()
        except Exception as exc:st.error(f'La consulta no se completó ({type(exc).__name__}). No significa ausencia de datos ni riesgo cero.')

with metodo:
    st.subheader('Alcance y límites')
    st.markdown('''- Cobertura: Medellín y Apartadó; series epidemiológicas 2022–2025.
- Inferencia histórica con modelos ajustados previamente. No se entrena desde la interfaz.
- Semanas y objetivos seleccionados a partir del calendario y las predicciones disponibles, sin asumir semanas ISO.
- Casos elevados: excedencia del percentil 75 de entrenamiento 2022–2023 por municipio.
- Datos consolidados: no reproduce los retrasos reales de notificación.
- Chat, modo **Agente LLM**: bucle agéntico con *tool calling* (consultar_serie, estimar_probabilidad, buscar_documentos), errores de herramientas devueltos como observación, T_máx = 6 iteraciones y 10 llamadas, validación de citas contra las fuentes recuperadas.
- Chat, modo **Controlado**: enrutamiento numérico en Python y respuestas documentales extractivas; no es una síntesis validada.
- Entrenamiento y evaluación del modelo: `notebooks/01_red_bayesiana_entrenamiento_evaluacion.ipynb` (validación temporal en 8 bloques; métricas frente a prevalencia y persistencia).
- Los extractos pueden ser irrelevantes o incompletos. Las citas válidas no garantizan una respuesta suficiente.''')
    st.write('La interfaz calcula directamente los porcentajes como 100 × probabilidad. No permite al LLM reescribir las tarjetas ni las tablas.')
