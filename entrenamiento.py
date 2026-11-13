"""Entrenamiento y validación temporal de la red bayesiana (Parte 1 del proyecto).

Este módulo reconstruye, desde la base semanal, todo lo que la aplicación usa ya
ajustado: umbrales del evento, tablas de probabilidad condicional (CPT), líneas
base, calibración, cortes de clasificación y predicciones fuera de muestra.

Modelo: Naive Bayes discreto (red bayesiana con la clase como único padre):

    P(y | x1..xk)  ∝  P(y) · Π_j P(xj | y)

- y = 1 si los casos de la semana siguiente (t+1) superan el umbral del municipio
  (percentil 75 de los casos semanales 2022–2023).
- Cada variable continua se discretiza en 3 categorías con los terciles del
  conjunto de entrenamiento de ese municipio y bloque (cuantil lineal de NumPy).
- Suavizado de Laplace con alfa = 1 en el prior y en cada CPT.
- Un faltante se marginaliza: esa variable no aporta factor a la verosimilitud.

Validación temporal "walk-forward": 2024 y 2025 se dividen en bloques de 13
semanas (B1..B4). Para cada bloque se entrena solo con transiciones cuyo objetivo
es anterior al inicio del bloque y se evalúa en el bloque. La calibración y el
corte de clasificación se eligen con las predicciones fuera de muestra de los
bloques anteriores (historial), nunca con el bloque evaluado.
"""
from pathlib import Path
import hashlib
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

VARIABLES = {
    'NB_casos': ['casos_actuales', 'cambio_casos'],
    'NB_casos_clima': ['casos_actuales', 'cambio_casos', 'lluvia_4s', 'temperatura_4s'],
}
MODELOS = ['Prevalencia', 'Persistencia', 'Persistencia_prob',
           'NB_casos', 'NB_casos_cal', 'NB_casos_clima', 'NB_casos_clima_cal']
BINS = 3
ALPHA = 1.0
LAMBDA_CAL = 0.01                       # regularización de la calibración hacia la identidad
GRID_CORTES = np.linspace(0.05, 0.95, 19)
MIN_HISTORIAL, MIN_POR_CLASE = 40, 10   # requisitos para calibrar y para mover el corte de 0.5


# ---------------------------------------------------------------- datos

def cargar_base(ruta):
    base = pd.read_csv(ruta, dtype={'codigo_municipio': str})
    base['codigo_municipio'] = base['codigo_municipio'].str.zfill(5)
    for col in ['inicio_semana', 'fin_semana']:
        base[col] = pd.to_datetime(base[col])
    return base.sort_values(['codigo_municipio', 'inicio_semana']).reset_index(drop=True)


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def calcular_umbrales(base, anos=(2022, 2023), q=0.75):
    """Umbral del evento por municipio: percentil 75 de casos semanales del periodo base."""
    h = base[base.ano_epi.isin(anos)]
    return {c: float(np.quantile(g.casos_registrados, q)) for c, g in h.groupby('codigo_municipio')}


def construir_variables(base, umbrales):
    """Una fila por semana de origen t con sus predictores y el objetivo en t+1."""
    partes = []
    for codigo, g in base.groupby('codigo_municipio'):
        g = g.copy()
        g['casos_actuales'] = g.casos_registrados
        # Casos actuales menos el promedio de las TRES semanas anteriores.
        g['cambio_casos'] = g.casos_registrados - g.casos_registrados.shift(1).rolling(3, min_periods=3).mean()
        g['lluvia_4s'] = g.precipitacion_total_mm.rolling(4, min_periods=4).sum()
        g['temperatura_4s'] = g.temperatura_media_c.rolling(4, min_periods=4).mean()
        g['casos_objetivo'] = g.casos_registrados.shift(-1)
        g['fecha_objetivo'] = g.inicio_semana.shift(-1)
        g['ano_objetivo'] = g.ano_epi.shift(-1)
        g['semana_objetivo'] = g.semana_epi.shift(-1)
        g['y'] = (g.casos_objetivo > umbrales[codigo]).astype(int)
        g['estado_actual'] = (g.casos_actuales > umbrales[codigo]).astype(int)
        partes.append(g)
    v = pd.concat(partes, ignore_index=True)
    # Bloque de 13 semanas según la semana epidemiológica objetivo (la B4 absorbe la SE53).
    con_obj = v.semana_objetivo.notna()
    b = np.minimum((v.loc[con_obj, 'semana_objetivo'].astype(int) - 1) // 13 + 1, 4)
    v.loc[con_obj, 'bloque'] = v.loc[con_obj, 'ano_objetivo'].astype(int).astype(str) + '_B' + b.astype(str)
    return v


def transiciones_validas(v):
    """Transiciones con objetivo conocido y con historial suficiente para cambio_casos."""
    return v[v.casos_objetivo.notna() & v.cambio_casos.notna()]


# ---------------------------------------------------------------- Naive Bayes

def ajustar_nb(g, variables, bins=BINS, alpha=ALPHA):
    """Prior y CPT con Laplace; cortes = terciles del entrenamiento (faltantes excluidos por variable)."""
    n, n1 = len(g), int(g.y.sum())
    estado = {'prior': [(n - n1 + alpha) / (n + 2 * alpha), (n1 + alpha) / (n + 2 * alpha)], 'variables': {}}
    for var in variables:
        x = g[[var, 'y']].dropna()
        cortes = [float(c) for c in np.quantile(x[var], [k / bins for k in range(1, bins)])]
        categoria = np.searchsorted(cortes, x[var].to_numpy(), side='right')
        cpt = []
        for clase in (0, 1):
            c = categoria[x.y.to_numpy() == clase]
            conteo = np.bincount(c, minlength=bins)
            cpt.append([float(z) for z in (conteo + alpha) / (len(c) + bins * alpha)])
        estado['variables'][var] = {'cortes': cortes, 'cpt': cpt}
    return estado


def predecir_nb(estado, fila, variables):
    """P(y=1 | evidencia) en espacio logarítmico; las variables faltantes se marginalizan."""
    logp = np.log(estado['prior'])
    for var in variables:
        valor = fila[var]
        if pd.isna(valor):
            continue
        spec = estado['variables'][var]
        k = int(np.searchsorted(spec['cortes'], float(valor), side='right'))
        logp = logp + np.log(np.asarray(spec['cpt'])[:, k])
    post = np.exp(logp - logp.max())
    return float(post[1] / post.sum())


# ---------------------------------------------------------------- líneas base

def prob_prevalencia(entrenamiento):
    """Frecuencia histórica del evento en el municipio (Laplace)."""
    return float((entrenamiento.y.sum() + 1) / (len(entrenamiento) + 2))


def prob_persistencia(fila):
    """La semana siguiente repite el estado actual (0 o 1)."""
    return float(fila.estado_actual)


def prob_persistencia_prob(entrenamiento, fila):
    """P(y_{t+1}=1 | estado_t) estimada con las transiciones históricas (Laplace)."""
    t = entrenamiento[entrenamiento.estado_actual.eq(fila.estado_actual)]
    return float((t.y.sum() + 1) / (len(t) + 2))


# ---------------------------------------------------------------- calibración y corte

def _logit(p):
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def historial_suficiente(y):
    y = np.asarray(y)
    return len(y) >= MIN_HISTORIAL and min((y == 1).sum(), (y == 0).sum()) >= MIN_POR_CLASE


def ajustar_calibrador(p, y, lam=LAMBDA_CAL):
    """Calibración logística global: p' = σ(a·logit(p) + b), log-loss medio + λ·((a−1)² + b²)."""
    z, y = _logit(p), np.asarray(y, dtype=float)

    def objetivo(t):
        s = t[0] * z + t[1]
        return (np.logaddexp(0, s) - y * s).mean() + lam * ((t[0] - 1) ** 2 + t[1] ** 2)

    a, b = minimize(objetivo, [1.0, 0.0], method='BFGS', options={'gtol': 1e-10}).x
    return float(a), float(b)


def aplicar_calibrador(p, a, b):
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))


def balanced_accuracy(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    sens = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    esp = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return (sens + esp) / 2


def elegir_corte(p, y):
    """Corte que maximiza la balanced accuracy del historial; empates -> el más cercano a 0.5."""
    p, y = np.asarray(p), np.asarray(y)
    if not historial_suficiente(y):
        return 0.5
    valores = np.array([balanced_accuracy(y, (p >= c).astype(int)) for c in GRID_CORTES])
    mejores = np.flatnonzero(np.isclose(valores, valores.max()))
    return float(GRID_CORTES[mejores[np.argmin(abs(GRID_CORTES[mejores] - 0.5))]])


# ---------------------------------------------------------------- validación temporal

def validacion_temporal(base):
    """Entrena y evalúa los 7 modelos bloque a bloque. Devuelve (estados, predicciones, auditoría)."""
    umbrales = calcular_umbrales(base)
    v = construir_variables(base, umbrales)
    validas = transiciones_validas(v)
    evaluables = validas[validas.ano_objetivo.isin([2024, 2025])]
    bloques = list(evaluables.sort_values('fecha_objetivo').bloque.unique())

    estados, predicciones, auditoria = [], [], []
    for i, clave in enumerate(bloques):
        ev = evaluables[evaluables.bloque.eq(clave)].sort_values(['codigo_municipio', 'fecha_objetivo'])
        inicio_eval = ev.fecha_objetivo.min()
        ent = validas[validas.fecha_objetivo.lt(inicio_eval)]
        historial = pd.concat(predicciones) if predicciones else None
        hist_bloques = historial[historial.bloque.isin(bloques[:i])] if historial is not None else None

        redes = {}
        for red, variables in VARIABLES.items():
            redes[red] = {'variables': variables, 'bins': BINS, 'alpha': ALPHA, 'estados': {
                cod: ajustar_nb(ent[ent.codigo_municipio.eq(cod)], variables) for cod in sorted(ev.codigo_municipio.unique())}}

        # Probabilidades crudas de cada modelo en el bloque evaluado.
        crudas = {m: [] for m in ['Prevalencia', 'Persistencia', 'Persistencia_prob', 'NB_casos', 'NB_casos_clima']}
        for _, fila in ev.iterrows():
            e = ent[ent.codigo_municipio.eq(fila.codigo_municipio)]
            crudas['Prevalencia'].append(prob_prevalencia(e))
            crudas['Persistencia'].append(prob_persistencia(fila))
            crudas['Persistencia_prob'].append(prob_persistencia_prob(e, fila))
            for red, variables in VARIABLES.items():
                crudas[red].append(predecir_nb(redes[red]['estados'][fila.codigo_municipio], fila, variables))

        # Calibración con el historial fuera de muestra de la red sin calibrar.
        calibradores, probs = {}, {m: np.array(p) for m, p in crudas.items()}
        for red in VARIABLES:
            h = hist_bloques[hist_bloques.modelo.eq(red)] if hist_bloques is not None else None
            activa = h is not None and historial_suficiente(h.y)
            a, b = ajustar_calibrador(h.p, h.y) if activa else (1.0, 0.0)
            calibradores[red] = {'a': a, 'b': b, 'activa': bool(activa)}
            probs[red + '_cal'] = aplicar_calibrador(probs[red], a, b) if activa else probs[red].copy()

        # Corte de cada modelo con su propio historial fuera de muestra.
        cortes, filas_bloque = {}, []
        for modelo in MODELOS:
            h = hist_bloques[hist_bloques.modelo.eq(modelo)] if hist_bloques is not None else None
            cortes[modelo] = elegir_corte(h.p, h.y) if h is not None else 0.5
            tabla = ev[['codigo_municipio', 'municipio', 'fecha_objetivo', 'ano_objetivo', 'semana_objetivo',
                        'casos_objetivo', 'y']].copy()
            tabla['umbral'] = tabla.codigo_municipio.map(umbrales)
            tabla['bloque'], tabla['modelo'], tabla['p'] = clave, modelo, probs[modelo]
            tabla['corte'] = cortes[modelo]
            tabla['pred_05'] = (tabla.p >= 0.5).astype(int)
            tabla['pred_ajustado'] = (tabla.p >= cortes[modelo]).astype(int)
            filas_bloque.append(tabla)
            cal = calibradores.get(modelo.replace('_cal', '')) if modelo.endswith('_cal') else None
            auditoria.append({'bloque': clave, 'modelo': modelo, 'n_entrenamiento': len(ent), 'n_evaluacion': len(ev),
                              'fin_entrenamiento': ent.fecha_objetivo.max(), 'inicio_evaluacion': inicio_eval,
                              'n_historial': 0 if h is None else len(h),
                              'fin_historial': None if h is None else h.fecha_objetivo.max(),
                              'calibracion_activa': bool(cal['activa']) if cal else False,
                              'a': cal['a'] if cal else np.nan, 'b': cal['b'] if cal else np.nan,
                              'corte': cortes[modelo]})
        predicciones.extend(filas_bloque)
        estados.append({'bloque': clave, 'redes': redes, 'calibradores': calibradores, 'cortes': cortes})

    paquete = {'version': 2, 'umbrales_fijos': umbrales, 'bins': BINS, 'alpha': ALPHA,
               'estados_por_bloque': estados}
    return paquete, pd.concat(predicciones, ignore_index=True), pd.DataFrame(auditoria)


# ---------------------------------------------------------------- métricas

def metricas(pred, por=('modelo',)):
    """Brier, log-loss, balanced accuracy (corte ajustado), sensibilidad, especificidad y accuracy a 0.5."""
    def calc(g):
        y, p, pa = g.y.to_numpy(), g.p.to_numpy(), g.pred_ajustado.to_numpy()
        q = np.clip(p, 1e-6, 1 - 1e-6)
        sens = ((pa == 1) & (y == 1)).sum() / (y == 1).sum() if (y == 1).any() else math.nan
        esp = ((pa == 0) & (y == 0)).sum() / (y == 0).sum() if (y == 0).any() else math.nan
        return pd.Series({'n': len(g), 'prevalencia': y.mean(), 'brier': ((p - y) ** 2).mean(),
                          'log_loss': -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean(),
                          'balanced_accuracy': (sens + esp) / 2, 'sensibilidad': sens, 'especificidad': esp,
                          'accuracy_05': (g.pred_05.to_numpy() == y).mean()})
    return pred.groupby(list(por)).apply(calc, include_groups=False)


def bootstrap_diferencia_brier(pred, modelo_a, modelo_b, n=2000, bloque=4, semilla=0):
    """IC 95 % de Brier(a) − Brier(b) con bootstrap por bloques móviles (respeta autocorrelación semanal)."""
    rng = np.random.default_rng(semilla)
    diffs = []
    for _, g in pred[pred.modelo.isin([modelo_a, modelo_b])].groupby('codigo_municipio'):
        w = g.pivot_table(index='fecha_objetivo', columns='modelo', values='p').join(
            g.drop_duplicates('fecha_objetivo').set_index('fecha_objetivo').y)
        diffs.append(((w[modelo_a] - w.y) ** 2 - (w[modelo_b] - w.y) ** 2).to_numpy())
    puntual = float(np.concatenate(diffs).mean())
    muestras = []
    for _ in range(n):
        rem = []
        for d in diffs:
            inicios = rng.integers(0, len(d), size=math.ceil(len(d) / bloque))
            idx = np.concatenate([(np.arange(s, s + bloque) % len(d)) for s in inicios])[:len(d)]
            rem.append(d[idx])
        muestras.append(np.concatenate(rem).mean())
    lo, hi = np.percentile(muestras, [2.5, 97.5])
    return puntual, float(lo), float(hi)
