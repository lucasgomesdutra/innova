"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/embeddings.py — Geração e cache de embeddings semânticos
=============================================================================

Responsabilidades:
  - Carregar o modelo sentence-transformers multilíngue (singleton)
  - Gerar embeddings densos (384 dimensões) para listas de textos
  - Cachear embeddings em disco (.npy) para evitar recálculo
  - Calcular similaridade cosseno entre textos
  - Encontrar os N textos mais similares a uma query

Modelo: paraphrase-multilingual-MiniLM-L12-v2
  - ~120MB, multilíngue (português nativo)
  - 384 dimensões, rápido na CPU
  - Entende contexto semântico: "Pix não chegou" ≈ "transferência sumiu"
"""

import os
import hashlib
import numpy as np
import pandas as pd
from typing import List, Tuple

# =============================================================================
# SINGLETON DO MODELO
# =============================================================================

_modelo_st = None
NOME_MODELO = "paraphrase-multilingual-MiniLM-L12-v2"

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR  = os.path.join(BASE_DIR, "models", "embeddings_cache")


def _carregar_modelo():
    """Carrega o modelo sentence-transformers uma única vez (singleton)."""
    global _modelo_st
    if _modelo_st is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embeddings] Carregando modelo '{NOME_MODELO}'...")
        _modelo_st = SentenceTransformer(NOME_MODELO)
        print("[embeddings] Modelo carregado.")
    return _modelo_st


# =============================================================================
# GERAÇÃO DE EMBEDDINGS
# =============================================================================

def gerar_embeddings(textos: List[str] | pd.Series,
                     batch_size: int = 64,
                     show_progress: bool = False) -> np.ndarray:
    """
    Gera embeddings semânticos para uma lista de textos.

    Args:
        textos: lista ou Series de strings
        batch_size: tamanho do lote para inference
        show_progress: exibe barra de progresso (útil para bases grandes)

    Returns:
        np.ndarray de shape (n_textos, 384)
    """
    if isinstance(textos, pd.Series):
        textos = textos.fillna("").tolist()

    modelo = _carregar_modelo()
    embeddings = modelo.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalizado → similaridade cosseno = produto escalar
    )
    return embeddings.astype(np.float32)


def gerar_embeddings_com_cache(textos: List[str] | pd.Series,
                                chave_cache: str,
                                forcar_recalculo: bool = False) -> np.ndarray:
    """
    Gera embeddings com cache em disco.

    Se o arquivo de cache existir e a chave bater, carrega do disco.
    Caso contrário, calcula e salva.

    Args:
        textos: textos para vetorizar
        chave_cache: identificador único (ex: "avaliacoes_reais", "base_treino")
        forcar_recalculo: ignora cache e recalcula

    Returns:
        np.ndarray de shape (n_textos, 384)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if isinstance(textos, pd.Series):
        textos_list = textos.fillna("").tolist()
    else:
        textos_list = list(textos)

    # Hash do conteúdo para invalidar cache automaticamente se os textos mudarem
    conteudo_hash = hashlib.md5("|".join(textos_list).encode("utf-8")).hexdigest()[:12]
    nome_arquivo  = f"{chave_cache}_{conteudo_hash}.npy"
    caminho_cache = os.path.join(CACHE_DIR, nome_arquivo)

    if not forcar_recalculo and os.path.exists(caminho_cache):
        print(f"[embeddings] Carregando do cache: {nome_arquivo}")
        return np.load(caminho_cache)

    print(f"[embeddings] Calculando embeddings para '{chave_cache}' ({len(textos_list)} textos)...")
    embeddings = gerar_embeddings(textos_list, show_progress=len(textos_list) > 500)
    np.save(caminho_cache, embeddings)
    print(f"[embeddings] Cache salvo: {nome_arquivo}")
    return embeddings


# =============================================================================
# SIMILARIDADE
# =============================================================================

def similaridade_cosseno(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Similaridade cosseno entre dois vetores L2-normalizados.
    Como os embeddings já são normalizados, é simplesmente o produto escalar.
    """
    return float(np.dot(emb_a, emb_b))


def buscar_similares(query: str,
                     textos_base: List[str] | pd.Series,
                     embeddings_base: np.ndarray,
                     top_n: int = 5,
                     limiar: float = 0.60) -> List[Tuple[str, float]]:
    """
    Encontra os textos mais similares semanticamente à query.

    Args:
        query: texto de busca
        textos_base: lista de textos do corpus
        embeddings_base: embeddings pré-calculados do corpus (L2-normalizado)
        top_n: número máximo de resultados
        limiar: similaridade mínima (0-1) para incluir no resultado

    Returns:
        Lista de tuplas (texto, score) ordenada por score decrescente
    """
    modelo = _carregar_modelo()
    emb_query = modelo.encode([query], normalize_embeddings=True)[0]

    scores = embeddings_base @ emb_query  # produto escalar = cosseno (normalizado)

    if isinstance(textos_base, pd.Series):
        textos_list = textos_base.tolist()
    else:
        textos_list = list(textos_base)

    resultados = [
        (textos_list[i], float(scores[i]))
        for i in np.argsort(scores)[::-1]
        if float(scores[i]) >= limiar
    ]

    return resultados[:top_n]
