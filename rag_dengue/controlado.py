"""Presentación controlada. No valida semántica de un resumen libre del LLM."""
import re,time,unicodedata
from .agent import AgenteDengue as AgenteBase

def normal(t):return ''.join(c for c in unicodedata.normalize('NFD',t.lower()) if unicodedata.category(c)!='Mn')
def porcentaje(p):return f'{100*float(p):.2f} %'

def presentar_modelo(r):
 b=r['red_bayesiana'];p=r['referencia'];e=r['evento'];v=r['evidencia'];c=r['cita']
 return f'''**{r['municipio']} — estimación retrospectiva** {c}

Origen: **{r['semana_origen']}** → objetivo: **{r['semana_objetivo']}**.
Evento: casos semanales > {e['umbral']:g}; mínimo entero: **{e['minimo_casos_elevados']} casos**.

| Modelo | Probabilidad | Porcentaje | Corte de clasificación | Corte histórico |
|---|---:|---:|---:|---:|
| Bayesiano | {b['probabilidad']:.4f} | {porcentaje(b['probabilidad'])} | {b['corte_clasificacion']:g} | {b['corte_historico']:g} |
| Persistencia | {p['probabilidad']:.4f} | {porcentaje(p['probabilidad'])} | {p['corte_clasificacion']:g} | {p['corte_historico']:g} |

Supera el corte histórico: bayesiano **{'sí' if b['alerta_corte_historico'] else 'no'}**; persistencia **{'sí' if p['alerta_corte_historico'] else 'no'}**.
Calibración bayesiana: **{'activa' if b['calibracion_activa'] else 'inactiva'}**.

**Datos observados en origen:** {v['casos_actuales']:g} casos. Cambio frente al promedio de las tres semanas anteriores: {v['cambio_casos']:.2f}. Estos son datos históricos, **no un pronóstico del número de casos**.
Lluvia acumulada de cuatro semanas: {v['lluvia_4s'] if v['lluvia_4s'] is not None else 'sin dato'} mm; temperatura media de cuatro semanas: {v['temperatura_4s'] if v['temperatura_4s'] is not None else 'sin dato'} °C.

Una probabilidad mayor no demuestra mayor precisión ni un efecto causal del clima. Esta consulta no devuelve métricas Brier, por lo que no se atribuye superioridad a ningún modelo.
Esta estimación no constituye una alerta oficial y no determina si la autoridad declaró una.'''

def presentar_serie(r):
 lineas=[f"**{r['municipio']} — datos históricos** {r['cita']}",f"Semanas con inicio entre {r['intervalo_solicitado'][0]} y {r['intervalo_solicitado'][1]}. Total: **{r['casos_registrados_total']} casos registrados** en {r['numero_semanas']} semanas.",
 '| Inicio | Casos | Temperatura °C | Lluvia mm | Días temperatura | Días lluvia |','|---|---:|---:|---:|---:|---:|']
 def fmt(x):return 'Sin dato' if x is None else f'{x:.2f}'
 for x in r['datos']:lineas.append(f"| {x['inicio_semana']} | {x['casos_registrados']} | {fmt(x['temperatura_media_c'])} | {fmt(x['precipitacion_total_mm'])} | {x['dias_temperatura']} | {x['dias_lluvia']} |")
 lineas.append('La cobertura diaria se indica por variable; tener un agregado semanal no implica siete días observados.')
 if r.get('filas_truncadas'):lineas.append(r['nota_truncamiento'])
 return '\n\n'.join(lineas[:2])+'\n\n'+'\n'.join(lineas[2:])

class AgenteControlado(AgenteBase):
 def preguntar(self,pregunta):
  if not isinstance(pregunta,str) or not pregunta.strip() or len(pregunta)>4000:raise ValueError('Pregunta vacía o demasiado larga.')
  inicio=time.monotonic();q=normal(pregunta);fuentes={};contadores={};traza=[];partes=[];errores=[]
  municipios=[nom for clave,nom in [('medellin','Medellín'),('apartado','Apartadó'),('turbo','Turbo')] if clave in q]
  fechas=re.findall(r'\b\d{4}-\d{2}-\d{2}\b',pregunta)
  semanas=re.findall(r'\b(\d{4})\s*[-–]?\s*SE\s*(\d{1,2})\b',pregunta,re.I)
  numerica=bool(re.search(r'casos|temperatura|precipitacion|lluvia|probabilidad|estima|riesgo',q))
  documental=bool(re.search(r'protocolo|boletin|ficha|ajuste|oficial',q))
  def ejecutar(nombre,args):
   salida=self._ejecutar(nombre,args,fuentes,contadores)
   traza.append({'paso':len(traza)+1,'herramienta':nombre,'argumentos':args,'salida':salida,'origen_llamada':'control_python'})
   if not salida['ok']:errores.append(salida.get('error'));return None
   return salida['resultado']
  if len(municipios)==1 and numerica:
   if len(semanas)==1:
    y,w=semanas[0];r=ejecutar('estimar_probabilidad',{'municipio':municipios[0],'semana':f'{y}-SE{int(w):02d}'})
    if r:partes.append(presentar_modelo(r))
   elif len(fechas)==2:
    r=ejecutar('consultar_serie',{'municipio':municipios[0],'fecha_inicio':fechas[0],'fecha_fin':fechas[1]})
    if r:partes.append(presentar_serie(r))
  resultado_llm=None
  if 'ficha' in q or 'ajuste' in q:
   ejecutar('buscar_documentos',{'consulta':pregunta,'tipo':'ficha'})
  elif documental or not traza:
   # Para consultas mixtas el LLM solo recibe la parte documental, nunca las cifras del modelo.
   consulta= ('Busca en el protocolo fragmentos sobre vigilancia y alerta oficial. No determines si hubo declaración oficial. No consultes modelos ni series numéricas.' if partes else pregunta)
   resultado_llm=super().preguntar(consulta)
   # Conservamos sus trazas, pero mostramos solo evidencia textual documental.
   for t in resultado_llm['traza']:
    if t['herramienta']=='buscar_documentos':
     t={**t,'paso':len(traza)+1,'origen_llamada':'llm'};traza.append(t)
  documentos={}
  for t in traza:
   if t['herramienta']=='buscar_documentos' and t['salida'].get('ok'):
    for f in t['salida']['resultado']['resultados']:documentos[f['id']]=f
  if documentos:
   partes.append('**Evidencia documental recuperada — lectura extractiva**\nLos siguientes textos son extractos, no una síntesis validada. Su recuperación no garantiza que respondan completamente la pregunta. No se deduce que otros documentos carezcan de información ni se declaran criterios oficiales adicionales.')
   for f in list(documentos.values())[:6]:
    partes.append(f"**{f['titulo']} — página {f['pagina']}** [{f['id']}]\n\n"+'\n'.join('> '+linea for linea in f['texto'].splitlines()))
    fuentes[f['id']]={**f,'tipo_documento':f['tipo'],'tipo':'documento'}
  if errores:partes.append('No se completó la consulta numérica: '+str(errores))
  estado='evidencia_documental' if documentos else 'completado'
  if errores:estado='error_herramienta'
  if resultado_llm and resultado_llm['estado']!='completado':
   estado='parcial' if partes else resultado_llm['estado'];partes.append('La fase LLM terminó con estado '+resultado_llm['estado']+'.')
  if not partes:
   if documental:partes=['No se recuperó evidencia documental suficiente para responder.'];estado='sin_evidencia'
   else:partes=['Indica un municipio (Medellín o Apartadó) y una semana de origen AAAA-SEnn o un intervalo con dos fechas AAAA-MM-DD.'];estado='requiere_aclaracion'
  texto='\n\n'.join(partes);citadas=self._citas(texto)
  return {'pregunta':pregunta,'respuesta':texto,'estado':estado,'modelo':self.modelo,
   'modo':'cifras_deterministas_y_documentos_extractivos','fuentes_citadas':[fuentes[k] for k in sorted(citadas) if k in fuentes],
   'citas_invalidas':sorted(citadas-set(fuentes)),'citas_verificables':bool(citadas) and citadas<=set(fuentes),
   'nota_verificacion':'Cifras formateadas en Python. Extractos literales; relevancia y suficiencia requieren revisión humana.',
   'traza':traza,'llamadas_herramientas':len(traza),'segundos':round(time.monotonic()-inicio,2),
   'tokens_totales':resultado_llm['tokens_totales'] if resultado_llm else 0,
   'diagnostico_api':resultado_llm.get('diagnostico_api',[]) if resultado_llm else [],
   'presupuestos':resultado_llm.get('presupuestos',[]) if resultado_llm else [],'incidencias':[],
   'estado_llm':resultado_llm['estado'] if resultado_llm else 'no_necesario'}
