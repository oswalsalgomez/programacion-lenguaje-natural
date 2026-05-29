"""
retriever_utils.py — utilidades para usar el motor de retrieval ANLA en otros notebooks.
Generado por Etapa 3 (v2: con fecha_ordinal para filtros de rango y metadatos robustos).

Uso típico:
    from retriever_utils import Retriever
    r = Retriever(chroma_dir="/content/drive/.../Retrieval/chroma_db")
    resultados = r.buscar("incumplimiento plan de restauración", top_k=5)
"""
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import chromadb
from chromadb.config import Settings
from transformers import LongformerTokenizerFast, LongformerModel


def _fecha_a_ordinal(fecha_str: str) -> int:
    if not fecha_str:
        return 0
    try:
        return int(fecha_str.replace("-", ""))
    except (ValueError, AttributeError):
        return 0


class Retriever:
    def __init__(
        self,
        chroma_dir: str,
        collection_name: str = "autos_anla_longformer",
        model_name: str = "allenai/longformer-base-4096",
        max_model_len: int = 4096,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_model_len = max_model_len
        self.tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
        self.model = LongformerModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.client = chromadb.PersistentClient(
            path=chroma_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(collection_name)

    def encodear_consulta(self, query: str) -> np.ndarray:
        token_ids = self.tokenizer.encode(query, add_special_tokens=False)
        ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
        if len(ids) > self.max_model_len:
            ids = ids[: self.max_model_len]
        input_ids = torch.tensor([ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).squeeze(0).cpu().numpy().astype(np.float32)

    def buscar(
        self,
        query: str,
        top_k: int = 5,
        tipo_chunk: Optional[str] = None,
        filename: Optional[str] = None,
        numeral: Optional[str] = None,
        fecha_desde: Optional[str] = None,
        fecha_hasta: Optional[str] = None,
        filtros_extra: Optional[dict] = None,
    ) -> List[Dict]:
        conds = []
        if tipo_chunk:
            conds.append({"tipo_chunk": {"$eq": tipo_chunk}})
        if filename:
            conds.append({"filename": {"$eq": filename}})
        if numeral:
            conds.append({"numeral": {"$eq": str(numeral)}})
        if fecha_desde:
            conds.append({"fecha_ordinal": {"$gte": _fecha_a_ordinal(fecha_desde)}})
        if fecha_hasta:
            conds.append({"fecha_ordinal": {"$lte": _fecha_a_ordinal(fecha_hasta)}})

        # Manejo robusto de filtros extra
        if filtros_extra:
            if isinstance(filtros_extra, list):
                for f in filtros_extra:
                    if f: conds.append(f)
            elif isinstance(filtros_extra, dict) and filtros_extra:
                conds.append(filtros_extra)

        if not conds:
            where = None
        elif len(conds) == 1:
            where = conds[0]
        else:
            where = {"$and": conds}

        query_emb = self.encodear_consulta(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb], n_results=top_k, where=where
        )

        out = []
        # Validación preventiva si no hay coincidencias
        if not results or not results["ids"] or len(results["ids"][0]) == 0:
            return out

        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            out.append({
                "chunk_id":  results["ids"][0][i],
                "texto":     results["documents"][0][i],
                "metadata":  results["metadatas"][0][i],
                "distancia": dist,
                "similitud": 1 - dist,
            })
        return out

    def traer_bloque_padre(self, chunk_resultado: dict) -> Optional[Dict]:
        md = chunk_resultado["metadata"]
        if md.get("tipo_chunk") != "tabla":
            return None
        bloque_padre_id = md.get("bloque_id_padre")
        id_doc = md.get("id_documento")
        if not bloque_padre_id:
            return None

        # Filtro compuesto plano seguro para .get()
        res = self.collection.get(
            where={"$and": [
                {"id_documento": {"$eq": id_doc}},
                {"bloque_id":    {"$eq": bloque_padre_id}},
                {"tipo_chunk":   {"$eq": "texto"}},
            ]},
            limit=50,
        )
        if not res or not res["ids"]:
            return None
        items = list(zip(res["metadatas"], res["documents"]))
        items.sort(key=lambda x: x[0].get("chunk_index", 0))
        return {
            "bloque_id": bloque_padre_id,
            "numeral":   items[0][0].get("numeral"),
            "titulo":    items[0][0].get("titulo_bloque"),
            "texto_completo": " ".join(t for _, t in items),
        }
