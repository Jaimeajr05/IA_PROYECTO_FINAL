"""Inferencia histórica con estados temporales; sin reentrenamiento ni llamadas externas."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import math
import re
import unicodedata
import numpy as np
import pandas as pd

ADVERTENCIA = ('Evaluación retrospectiva exploratoria con datos consolidados. '
               'Exceder el umbral del proyecto no equivale a un brote oficial. '
               'No reproduce la disponibilidad real de las notificaciones.')

class ErrorConsulta(ValueError):
    def __init__(self, codigo, mensaje):
        super().__init__(mensaje)
        self.codigo = codigo

def _normalizar(valor):
    return ''.join(c for c in unicodedata.normalize('NFD', str(valor).strip().lower())
                   if unicodedata.category(c) != 'Mn')

def _fecha(valor):
    if not isinstance(valor, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', valor):
        raise ErrorConsulta('FECHA_INVALIDA', 'Usa fechas YYYY-MM-DD.')
    try:
        return pd.Timestamp(valor)
    except ValueError as exc:
        raise ErrorConsulta('FECHA_INVALIDA', 'La fecha no existe.') from exc

def _iso(valor):
    return pd.Timestamp(valor).date().isoformat()

def _registros(tabla):
    # El JSON estricto convierte faltantes en null, nunca en NaN.
    return json.loads(tabla.to_json(orient='records', date_format='iso', force_ascii=False))

class ServicioDengue:
    def __init__(self, carpeta_datos=None):
        self.carpeta = Path(carpeta_datos) if carpeta_datos else Path(__file__).parent/'data'
        ruta = self.carpeta/'base_dengue_clima_semanal.csv'
        self.paquete = json.loads((self.carpeta/'estados_experimento.json').read_text(encoding='utf-8'))
        if hashlib.sha256(ruta.read_bytes()).hexdigest() != self.paquete['sha256_base']:
            raise ErrorConsulta('BASE_INCOMPATIBLE', 'La base no coincide con la utilizada para ajustar los modelos V2.')
        self.base = pd.read_csv(ruta, dtype={'codigo_municipio':str})
        self.base.codigo_municipio = self.base.codigo_municipio.str.zfill(5)
        for col in ['inicio_semana','fin_semana']:
            self.base[col] = pd.to_datetime(self.base[col])
        self.base = self.base.sort_values(['codigo_municipio','inicio_semana']).reset_index(drop=True)
        self.nombres = self.base.groupby('codigo_municipio').municipio.first().to_dict()
        self.umbrales = self.paquete['umbrales_fijos']
        self.estados = {x['bloque']:x for x in self.paquete['estados_por_bloque']}
        self.audit = pd.read_csv(self.carpeta/'auditoria_temporal.csv')
        for col in ['fin_entrenamiento','inicio_evaluacion','fin_historial']:
            self.audit[col] = pd.to_datetime(self.audit[col])
        if not (self.audit.fin_entrenamiento < self.audit.inicio_evaluacion).all():
            raise ErrorConsulta('AUDITORIA_INVALIDA','El entrenamiento invade la evaluación.')
        con_historial = self.audit.fin_historial.notna()
        if not (self.audit.loc[con_historial,'fin_historial'] < self.audit.loc[con_historial,'inicio_evaluacion']).all():
            raise ErrorConsulta('AUDITORIA_INVALIDA','El historial invade la evaluación.')
        self.variables = self._preparar()

    def _municipio(self, municipio):
        texto = _normalizar(municipio)
        for codigo,nombre in self.nombres.items():
            if texto in (codigo,str(int(codigo)),_normalizar(nombre)):
                return codigo
        raise ErrorConsulta('MUNICIPIO_NO_CUBIERTO','Solo hay cobertura para Medellín (05001) y Apartadó (05045).')

    def _preparar(self):
        partes=[]
        for codigo,g in self.base.groupby('codigo_municipio'):
            g=g.copy()
            g['casos_actuales']=g.casos_registrados
            g['cambio_casos']=g.casos_registrados-g.casos_registrados.shift(1).rolling(3,min_periods=3).mean()
            g['lluvia_4s']=g.precipitacion_total_mm.rolling(4,min_periods=4).sum()
            g['temperatura_4s']=g.temperatura_media_c.rolling(4,min_periods=4).mean()
            g['casos_objetivo']=g.casos_registrados.shift(-1)
            g['fecha_objetivo']=g.inicio_semana.shift(-1)
            g['y']=(g.casos_objetivo>self.umbrales[codigo]).astype(int)
            g['estado_actual']=(g.casos_actuales>self.umbrales[codigo]).astype(int)
            partes.append(g)
        # Los objetivos no conocidos quedan fuera del conjunto de transiciones.
        return pd.concat(partes,ignore_index=True)

    def consultar_serie(self, municipio, fecha_inicio, fecha_fin):
        """Semanas cuyo inicio está dentro del intervalo inclusivo indicado."""
        codigo=self._municipio(municipio);inicio=_fecha(fecha_inicio);fin=_fecha(fecha_fin)
        if inicio>fin:
            raise ErrorConsulta('RANGO_INVALIDO','fecha_inicio debe ser anterior o igual a fecha_fin.')
        g=self.base[self.base.codigo_municipio.eq(codigo)]
        z=g[g.inicio_semana.between(inicio,fin)].copy()
        if z.empty:
            raise ErrorConsulta('SIN_DATOS','No hay inicios de semana dentro del intervalo solicitado.')
        columnas=['codigo_municipio','municipio','ano_epi','semana_epi','inicio_semana','fin_semana',
                  'casos_registrados','casos_confirmados','casos_probables','temperatura_media_c',
                  'precipitacion_total_mm','dias_temperatura','dias_lluvia',
                  'temperatura_cobertura_suficiente','lluvia_completa','aprobacion_temperatura',
                  'aprobacion_lluvia','sin_registros_en_microdatos','cumplimiento_notificacion_verificado']
        for col in ['inicio_semana','fin_semana']:
            z[col]=z[col].dt.strftime('%Y-%m-%d')
        return {'municipio':self.nombres[codigo],'codigo_municipio':codigo,
                'criterio_intervalo':'inicio_semana dentro de [fecha_inicio, fecha_fin]',
                'intervalo_solicitado':[fecha_inicio,fecha_fin],
                'cobertura_disponible':[_iso(g.inicio_semana.min()),_iso(g.fin_semana.max())],
                'intervalo_parcial':bool(inicio<g.inicio_semana.min() or fin>g.fin_semana.max()),
                'numero_semanas':len(z),'casos_registrados_total':int(z.casos_registrados.sum()),
                'semanas_sin_temperatura':int(z.temperatura_media_c.isna().sum()),
                'semanas_sin_lluvia':int(z.precipitacion_total_mm.isna().sum()),
                'datos':_registros(z[columnas]),'advertencia':ADVERTENCIA}

    def estimar_probabilidad(self, municipio, semana):
        """semana='2025-SE10' es la semana de origen t; se estima t+1."""
        codigo=self._municipio(municipio)
        match=re.fullmatch(r'(\d{4})-SE(\d{1,2})',str(semana).upper().strip())
        if not match:
            raise ErrorConsulta('SEMANA_INVALIDA','Usa AAAA-SEnn (por ejemplo 2025-SE10). Es la semana de origen; estimamos la siguiente.')
        ano,num=map(int,match.groups())
        filas=self.variables[self.variables.codigo_municipio.eq(codigo)&self.variables.ano_epi.eq(ano)&self.variables.semana_epi.eq(num)]
        if len(filas)!=1:
            raise ErrorConsulta('SEMANA_NO_DISPONIBLE','La semana de origen no está en el calendario de la base.')
        fila=filas.iloc[0];objetivo=fila.inicio_semana+pd.Timedelta(days=7)
        destino=self.base[self.base.codigo_municipio.eq(codigo)&self.base.inicio_semana.eq(objetivo)]
        if destino.empty:
            raise ErrorConsulta('PREDICCION_NO_DISPONIBLE','No hay un bloque histórico evaluado para la semana siguiente; no se extrapola a fechas nuevas.')
        destino=destino.iloc[0];ano_obj=int(destino.ano_epi);sem_obj=int(destino.semana_epi)
        b=min((sem_obj-1)//13+1,4);clave=f'{ano_obj}_B{b}'
        if clave not in self.estados or pd.isna(fila.cambio_casos):
            raise ErrorConsulta('PREDICCION_NO_DISPONIBLE','Las semanas objetivo evaluadas cubren los años epidemiológicos 2024 y 2025.')
        estado=self.estados[clave];nombre='NB_casos_clima'
        aud=self.audit[self.audit.bloque.eq(clave)&self.audit.modelo.eq(nombre+'_cal')].iloc[0]
        if aud.fin_entrenamiento>=objetivo:
            raise ErrorConsulta('AUDITORIA_INVALIDA','Modelo no anterior al objetivo.')
        modelo=estado['redes'][nombre];est=modelo['estados'][codigo]
        logp=np.log(est['prior']);evidencia={};faltantes=[]
        for variable in modelo['variables']:
            valor=fila[variable]
            evidencia[variable]=None if pd.isna(valor) else float(valor)
            if pd.isna(valor):
                faltantes.append(variable);continue
            spec=est['variables'][variable]
            categoria=int(np.searchsorted(spec['cortes'],float(valor),side='right'))
            logp+=np.log(np.asarray(spec['cpt'])[:,categoria])
        posterior=np.exp(logp-logp.max());p_raw=float(posterior[1]/posterior.sum())
        cal=estado['calibradores'][nombre]
        if cal['activa']:
            q=float(np.clip(p_raw,1e-6,1-1e-6));z=cal['a']*math.log(q/(1-q))+cal['b']
            p=float(1/(1+math.exp(-z))) if z>=0 else float(math.exp(z)/(1+math.exp(z)))
        else: p=p_raw
        # Reconstrucción exacta de transiciones usadas por la referencia V2.
        ent=self.variables[self.variables.fecha_objetivo.lt(aud.inicio_evaluacion)&
                           self.variables.casos_objetivo.notna()&self.variables.cambio_casos.notna()]
        trans=ent[ent.codigo_municipio.eq(codigo)&ent.estado_actual.eq(fila.estado_actual)]
        ref=float((trans.y.sum()+1)/(len(trans)+2))
        corte=float(estado['cortes'][nombre+'_cal']);corte_ref=float(estado['cortes']['Persistencia_prob'])
        return {'municipio':self.nombres[codigo],'codigo_municipio':codigo,
                'modo':'inferencia_historica_retrospectiva',
                'semana_origen':f'{ano}-SE{num:02d}',
                'fin_semana_origen':_iso(fila.fin_semana),
                'semana_objetivo':f'{ano_obj}-SE{sem_obj:02d}',
                'inicio_semana_objetivo':_iso(objetivo),'fin_semana_objetivo':_iso(destino.fin_semana),
                'evento':{'definicion':'casos semanales estrictamente mayores que el percentil 75 de 2022–2023',
                          'umbral':float(self.umbrales[codigo]),'minimo_casos_elevados':math.floor(self.umbrales[codigo])+1},
                'red_bayesiana':{'modelo':nombre+'_cal','probabilidad':p,'probabilidad_sin_calibrar':p_raw,
                    'calibracion_activa':bool(cal['activa']),'corte_clasificacion':0.5,'alerta_corte_05':bool(p>=.5),
                    'corte_historico':corte,'alerta_corte_historico':bool(p>=corte)},
                'referencia':{'modelo':'Persistencia_prob','probabilidad':ref,'corte_clasificacion':0.5,'alerta_corte_05':bool(ref>=.5),
                    'corte_historico':corte_ref,'alerta_corte_historico':bool(ref>=corte_ref),
                    'transiciones_utilizadas':len(trans)},
                'evidencia':evidencia,'variables_marginalizadas':faltantes,
                'trazabilidad':{'bloque':clave,'ultimo_objetivo_entrenamiento':_iso(aud.fin_entrenamiento),
                    'ultimo_objetivo_calibracion':None if pd.isna(aud.fin_historial) else _iso(aud.fin_historial),
                    'sha256_base':self.paquete['sha256_base']},'advertencia':ADVERTENCIA}

@lru_cache(maxsize=1)
def _servicio():
    return ServicioDengue()

def consultar_serie(municipio, fecha_inicio, fecha_fin):
    return _servicio().consultar_serie(municipio,fecha_inicio,fecha_fin)

def estimar_probabilidad(municipio, semana):
    return _servicio().estimar_probabilidad(municipio,semana)

def ejecutar_herramienta(nombre, argumentos):
    """Adaptador JSON para un agente: lista permitida y errores de consulta estructurados."""
    permitidas={'consultar_serie':(consultar_serie,{'municipio','fecha_inicio','fecha_fin'}),
                'estimar_probabilidad':(estimar_probabilidad,{'municipio','semana'})}
    if nombre not in permitidas:
        return {'ok':False,'error':{'codigo':'HERRAMIENTA_INVALIDA','mensaje':'Herramienta no disponible.'}}
    fn,claves=permitidas[nombre]
    if not isinstance(argumentos,dict) or set(argumentos)!=claves or not all(isinstance(v,str) for v in argumentos.values()):
        return {'ok':False,'error':{'codigo':'ARGUMENTOS_INVALIDOS','mensaje':'Revisa los argumentos requeridos; deben ser cadenas de texto.'}}
    try:
        return {'ok':True,'resultado':fn(**argumentos)}
    except ErrorConsulta as exc:
        return {'ok':False,'error':{'codigo':exc.codigo,'mensaje':str(exc)}}
