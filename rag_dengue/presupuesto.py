"""Estimación con margen; no reproduce el tokenizer exacto del proveedor."""
import json,copy,math,time
try:
 import tiktoken
 ENCODER=tiktoken.get_encoding('o200k_base')
except (ImportError,Exception):
 ENCODER=None

def contar(mensajes,tools,salida):
 texto=json.dumps({'messages':mensajes,'tools':tools},ensure_ascii=False,separators=(',',':'))
 if ENCODER is None:
  entrada=len(texto.encode('utf-8'))+128;metodo='cota_bytes_utf8'
 else:
  entrada=math.ceil(len(ENCODER.encode(texto,disallowed_special=()))*1.3)+256;metodo='o200k_base_margen_30pct'
 return {'entrada_estimada':entrada,'reserva_salida':salida,'total_estimado':entrada+salida,'metodo':metodo}

def preparar(mensajes,tools,salida=1600,limite=6500):
 m=copy.deepcopy(mensajes)
 # No quitamos mensajes tool ni sus IDs: preserva el protocolo del chat.
 for grado in range(4):
  for msg in m:
   if msg['role']!='tool':continue
   try:d=json.loads(msg['content'])
   except (ValueError,TypeError):continue
   r=d.get('resultado',{})
   if 'resultados' in r:
    fs=r['resultados']
    if grado:
     fs=fs[:max(1,3-grado)]
    r['resultados']=[{k:v for k,v in f.items() if k in ('id','doc_id','pagina','texto','cita','contexto_completo','advertencia_contexto')} for f in fs]
    r['contexto_limitado']=True
   if grado and 'datos' in r:
    r['datos']=r['datos'][:max(1,8//grado)]
    r['filas_mostradas']=len(r['datos']);r['filas_truncadas']=r['numero_semanas']>len(r['datos'])
    r['nota_truncamiento']='Total y cobertura usan todo el intervalo; detalle parcial. Consultar rango menor si se requieren todas las filas.'
   msg['content']=json.dumps(d,ensure_ascii=False,separators=(',',':'))
  cuenta=contar(m,tools,salida)
  if cuenta['total_estimado']<=limite:
   cuenta.update(limite=limite,grado_compactacion=grado);return m,cuenta
 raise ValueError('Contexto excedido; no se envía a la API.')

class ReguladorTPM:
 def __init__(self,limite=8000):self.limite=limite;self.historial=[]
 def pausa(self,segundos):
  print('Esperando cupo de Groq:',round(segundos),'s',flush=True)
  fin=time.monotonic()+segundos
  while time.monotonic()<fin:time.sleep(min(15,max(0,fin-time.monotonic())))
 def esperar(self,tokens):
  if tokens>self.limite:raise ValueError('Solicitud mayor que el límite TPM local.')
  ahora=time.monotonic();self.historial=[(t,n) for t,n in self.historial if ahora-t<61]
  while sum(n for t,n in self.historial)+tokens>self.limite:
   self.pausa(max(.1,61-(time.monotonic()-self.historial[0][0])))
   ahora=time.monotonic();self.historial=[(t,n) for t,n in self.historial if ahora-t<61]
  self.historial.append((time.monotonic(),tokens))
