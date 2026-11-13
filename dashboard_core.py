"""Funciones de integración independientes de la interfaz."""
import pandas as pd
from dengue_tools.core import ServicioDengue,ErrorConsulta

# Ubicaciones urbanas aproximadas; no representan estaciones ni polígonos municipales.
COORDENADAS={'Medellín':(6.2442,-75.5812),'Apartadó':(7.883,-76.633)}

def cargar_datos():
    servicio=ServicioDengue()
    catalogo={}
    for municipio in servicio.nombres.values():
        g=servicio.base[servicio.base.municipio.eq(municipio)]
        valores={}
        for fila in g.itertuples():
            semana=f'{int(fila.ano_epi)}-SE{int(fila.semana_epi):02d}'
            try:
                valores[semana]=servicio.estimar_probabilidad(municipio,semana)
            except ErrorConsulta as exc:
                if exc.codigo not in ('PREDICCION_NO_DISPONIBLE','SEMANA_NO_DISPONIBLE'):raise
        catalogo[municipio]=valores
    return servicio,catalogo

def datos_mapa(catalogo,origen,modelo):
    filas=[]
    for municipio,coords in COORDENADAS.items():
        r=catalogo.get(municipio,{}).get(origen)
        p=r[modelo]['probabilidad'] if r else None
        color=[145,150,160] if p is None else [int(30+210*p),int(145-75*p),int(205-130*p)]
        filas.append({'municipio':municipio,'lat':coords[0],'lon':coords[1],
                      'probabilidad':p,'porcentaje':'Sin estimación' if p is None else f'{p:.2%}',
                      'objetivo':r['semana_objetivo'] if r else 'Sin estimación','color':color})
    return pd.DataFrame(filas)

def tabla_modelos(r):
    return pd.DataFrame([{'Modelo':r[k]['modelo'],'Probabilidad':r[k]['probabilidad'],
                         'Porcentaje (%)':round(100*r[k]['probabilidad'],2),
                         'Corte de clasificación':r[k]['corte_clasificacion'],
                         'Corte histórico':r[k]['corte_historico'],
                         'Supera corte histórico':bool(r[k]['alerta_corte_historico'])}
                        for k in ['red_bayesiana','referencia']])

def tabla_serie(servicio,municipio,inicio,fin):
    r=servicio.consultar_serie(municipio,inicio.isoformat(),fin.isoformat())
    return r,pd.DataFrame(r['datos'])
