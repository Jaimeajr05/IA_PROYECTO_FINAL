"""Evaluación del agente con las 15 preguntas fijas de rag_dengue/evaluation.py.

Uso (desde la raíz del repositorio, con GROQ_API_KEY en el entorno o en .streamlit/secrets.toml):

    python evaluar_agente.py                  # modo agente LLM (tool calling)
    python evaluar_agente.py --modo controlado
    python evaluar_agente.py --modelo openai/gpt-oss-120b

Genera en resultados/:
    evaluacion_agente_<modo>.csv   una fila por pregunta (herramientas esperadas vs llamadas, citas)
    trazas_evaluacion_<modo>.json  respuesta y traza completa de cada pregunta
    resumen_evaluacion_<modo>.csv  porcentajes agregados

Las columnas revision_humana_* del CSV se llenan a mano: la validación automática comprueba
que las citas existan entre las fuentes recuperadas, no que respalden semánticamente la respuesta.
Puede tardar 15–25 minutos por el regulador de tokens por minuto de la capa gratuita de Groq.
"""
import argparse
import os
import shutil
import tomllib
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent


def leer_clave():
    clave = os.environ.get('GROQ_API_KEY')
    secretos = RAIZ / '.streamlit' / 'secrets.toml'
    if not clave and secretos.exists():
        clave = tomllib.loads(secretos.read_text(encoding='utf-8')).get('GROQ_API_KEY')
    if not clave:
        raise SystemExit('Falta GROQ_API_KEY (variable de entorno o .streamlit/secrets.toml).')
    return clave


def resumen(df):
    con_herr = df[df.herramientas_esperadas.fillna('') != '']
    con_docs = df[df.documentos_esperados_citados.notna()]
    return pd.DataFrame([
        {'métrica': 'preguntas evaluadas', 'valor': len(df)},
        {'métrica': '% con todas las herramientas esperadas llamadas', 'valor': 100 * con_herr.herramientas_esperadas_presentes.mean()},
        {'métrica': '% que citan el documento esperado (preguntas documentales)', 'valor': 100 * con_docs.documentos_esperados_citados.astype(bool).mean()},
        {'métrica': '% con citas verificables', 'valor': 100 * df.citas_verificables.mean()},
        {'métrica': 'preguntas con alguna cita inválida', 'valor': int((df.citas_invalidas.fillna('') != '').sum())},
        {'métrica': 'errores de herramientas devueltos como observación', 'valor': int(df.errores_herramientas.sum())},
        {'métrica': 'segundos promedio por pregunta', 'valor': df.segundos.mean()},
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modo', choices=['llm', 'controlado'], default='llm')
    ap.add_argument('--modelo', default=os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b'))
    ap.add_argument('--pausa', type=float, default=2.0, help='segundos entre preguntas')
    args = ap.parse_args()

    from groq import Groq
    from rag_dengue.agent import AgenteDengue
    from rag_dengue.controlado import AgenteControlado
    from rag_dengue.evaluation import evaluar_agente
    from rag_dengue.retrieval import IndiceDocumental

    cliente = Groq(api_key=leer_clave(), timeout=60, max_retries=0)
    if args.modelo not in {m.id for m in cliente.models.list().data}:
        raise SystemExit(f'El modelo {args.modelo} no está disponible en esta cuenta de Groq.')
    indice = IndiceDocumental(RAIZ)
    print('Índice:', indice.construir(reutilizar=True))
    clase = AgenteDengue if args.modo == 'llm' else AgenteControlado
    agente = clase(cliente, indice, RAIZ, modelo=args.modelo)

    salida = RAIZ / 'resultados'
    tmp = salida / f'_tmp_{args.modo}'
    df = evaluar_agente(agente, tmp, pausa=args.pausa)
    salida.mkdir(exist_ok=True)
    shutil.move(tmp / 'evaluacion_agente.csv', salida / f'evaluacion_agente_{args.modo}.csv')
    shutil.move(tmp / 'trazas_evaluacion.json', salida / f'trazas_evaluacion_{args.modo}.json')
    tmp.rmdir()
    r = resumen(df)
    r.to_csv(salida / f'resumen_evaluacion_{args.modo}.csv', index=False)
    print(f'\nModo {args.modo} · modelo {args.modelo}')
    print(r.round(1).to_string(index=False))
    print(f'\nResultados en {salida}. Completa a mano las columnas revision_humana_*.')


if __name__ == '__main__':
    main()
