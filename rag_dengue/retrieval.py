from pathlib import Path
import hashlib,json,re
import numpy as np
import faiss

MODELO_EMBEDDINGS='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

class IndiceDocumental:
    def __init__(self, raiz, encoder=None):
        self.raiz=Path(raiz)
        self.manifest=json.loads((self.raiz/'corpus/manifest.json').read_text(encoding='utf-8'))
        self.documentos={x['doc_id']:x for x in self.manifest}
        self.paginas=[json.loads(s) for s in (self.raiz/'corpus/texto/paginas.jsonl').read_text(encoding='utf-8').splitlines() if s]
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder=SentenceTransformer(MODELO_EMBEDDINGS,device='cpu')
        self.encoder=encoder
        self.fragmentos=[];self.indice=None

    def _fragmentar(self, tokens=110, solapamiento=20):
        tokenizer=self.encoder.tokenizer
        limite=min(tokens,int(self.encoder.max_seq_length)-2)
        if limite<=solapamiento:raise ValueError('Límite de tokens incompatible.')
        resultado=[]
        for p in self.paginas:
            texto=p['texto']
            if not texto:continue
            # Offsets conservan exactamente el texto de la página, sin cruzar páginas.
            posiciones=tokenizer(texto,add_special_tokens=False,return_offsets_mapping=True,truncation=False)['offset_mapping']
            for numero,inicio in enumerate(range(0,len(posiciones),limite-solapamiento),1):
                fin=min(inicio+limite,len(posiciones))
                a=posiciones[inicio][0];b=posiciones[fin-1][1];trozo=texto[a:b].strip()
                if not trozo:continue
                doc=self.documentos[p['doc_id']]
                resultado.append({'id':f"{p['doc_id']}:p{p['pagina']}:c{numero}",
                    'doc_id':p['doc_id'],'pagina':p['pagina'],'texto':trozo,'titulo':doc['titulo'],
                    'tipo':doc['tipo'],'ano':doc['ano'],'semana':doc['semana'],'archivo_pdf':doc['archivo_pdf']})
                if fin==len(posiciones):break
        return resultado

    def construir(self, reutilizar=True):
        # Un encoder de pruebas no puede escribir ni leer el índice semántico real.
        es_prueba=bool(getattr(self.encoder,'es_prueba',False))
        firma=hashlib.sha256((self.raiz/'corpus/texto/paginas.jsonl').read_bytes() +
            (self.raiz/'corpus/manifest.json').read_bytes()+f'{MODELO_EMBEDDINGS}|110|20|v1'.encode()).hexdigest()
        cache=self.raiz/'indice';config=cache/'config.json'
        if reutilizar and not es_prueba and config.exists() and json.loads(config.read_text()).get('firma')==firma:
            self.fragmentos=json.loads((cache/'fragmentos.json').read_text(encoding='utf-8'))
            vectores=np.load(cache/'vectores.npy',allow_pickle=False)
        else:
            self.fragmentos=self._fragmentar()
            if not self.fragmentos:raise ValueError('No hay texto para indexar.')
            vectores=self.encoder.encode([x['texto'] for x in self.fragmentos],batch_size=32,
                normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True)
            vectores=np.asarray(vectores,dtype=np.float32)
            if not es_prueba:
                cache.mkdir(exist_ok=True)
                np.save(cache/'vectores.npy',vectores,allow_pickle=False)
                (cache/'fragmentos.json').write_text(json.dumps(self.fragmentos,ensure_ascii=False),encoding='utf-8')
                (cache/'config.json').write_text(json.dumps({'firma':firma,'modelo':MODELO_EMBEDDINGS,
                    'tokens_fragmento':110,'solapamiento':20,'dimension':int(vectores.shape[1]),
                    'fragmentos':len(self.fragmentos),'normalizacion':True},indent=2))
        if vectores.ndim!=2 or len(vectores)!=len(self.fragmentos) or not np.isfinite(vectores).all():
            raise ValueError('Índice inconsistente; reconstruye con reutilizar=False.')
        self.indice=faiss.IndexFlatIP(vectores.shape[1])
        self.indice.add(np.ascontiguousarray(vectores,dtype=np.float32))
        return {'documentos':len(self.documentos),'paginas':len(self.paginas),'fragmentos':len(self.fragmentos),
                'dimension':int(vectores.shape[1]),'es_prueba':es_prueba}

    def buscar(self, consulta, tipo='todos', ano=0, semana=0, k=5):
        if self.indice is None:raise RuntimeError('Construye primero el índice.')
        if not isinstance(consulta,str) or not consulta.strip() or len(consulta)>2000:raise ValueError('Consulta vacía o demasiado larga.')
        if tipo not in ['todos','protocolo','ficha','boletin']:raise ValueError('Tipo documental inválido.')
        for v in [ano,semana,k]:
            if isinstance(v,bool) or not isinstance(v,int):raise ValueError('Los filtros numéricos deben ser enteros.')
        if not (ano==0 or 1900<=ano<=2100) or not 0<=semana<=53 or not 1<=k<=6:raise ValueError('Filtro fuera de rango.')
        q=np.asarray(self.encoder.encode([consulta],normalize_embeddings=True,convert_to_numpy=True),dtype=np.float32)
        # Corpus pequeño: ordenamos todo para aplicar filtros exactos sin perder candidatos.
        scores,indices=self.indice.search(q,len(self.fragmentos))
        encontrados=[]
        for score,idx in zip(scores[0],indices[0]):
            f=self.fragmentos[int(idx)]
            if tipo!='todos' and f['tipo']!=tipo:continue
            if ano and f['ano']!=ano:continue
            if semana and f['semana']!=semana:continue
            pagina=next(p['texto'] for p in self.paginas if p['doc_id']==f['doc_id'] and p['pagina']==f['pagina'])
            # Contexto contiguo acotado; límites de oración cuando están disponibles.
            inicio=pagina.find(f['texto']);fin=inicio+len(f['texto'])
            limites=[0]+[m.end() for m in re.finditer(r'[.!?]\s+|\n\s*\n',pagina)]+[len(pagina)]
            izquierda=max([x for x in limites if x<=inicio] or [0])
            derecha=min([x for x in limites if x>=fin] or [len(pagina)])
            completo=derecha-izquierda<=1600
            if not completo:
                izquierda=max(0,inicio-220);derecha=min(len(pagina),max(fin,izquierda+1400))
            texto=pagina[izquierda:derecha]
            if any(x['doc_id']==f['doc_id'] and x['pagina']==f['pagina'] and x['texto']==texto for x in encontrados):continue
            encontrados.append({**f,'texto':texto,'texto_ancla':f['texto'],
                'contexto_completo':completo,'alcance_cita':'contexto_local',
                'advertencia_contexto':None if completo else 'Ventana recortada: no inferir etiquetas, definiciones o denominadores ausentes.',
                'similitud':float(score),'cita':f"[{f['id']}]"})
            if len(encontrados)==k:break
        return {'resultados':encontrados,'filtros':{'tipo':tipo,'ano':ano,'semana':semana},
                'nota':'Similitud no es probabilidad ni garantía de respaldo. Verificar el contenido. Página = posición física en el PDF, empezando en 1.',
                'estado':'candidatos_recuperados' if encontrados else 'sin_documentos_con_esos_filtros'}
