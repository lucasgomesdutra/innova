"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/agrupamento_semantico.py — Clustering semântico com sentence-transformers
=============================================================================

Responsabilidades:
  - Agrupar avaliações por similaridade semântica (não por palavras-chave)
  - Reconhecer que frases diferentes descrevem o mesmo problema
  - Nomear clusters automaticamente a partir dos textos mais centrais
  - Calcular coesão dos clusters (silhouette)
  - Detectar problemas recorrentes mesmo com vocabulário variado

Diferença em relação a agrupamento.py (TF-IDF):
  TF-IDF:  "Pix não chegou" ≠ "transferência sumiu" (palavras diferentes)
  Semantic: "Pix não chegou" ≈ "transferência sumiu" (significado próximo) ✅
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embeddings import gerar_embeddings_com_cache, gerar_embeddings


# =============================================================================
# SELEÇÃO AUTOMÁTICA DE K
# =============================================================================

def escolher_k_semantico(embeddings: np.ndarray,
                          k_min: int = 3,
                          k_max: int = 12,
                          random_state: int = 42) -> int:
    """
    Escolhe o número ótimo de clusters usando Silhouette Score sobre embeddings.

    Args:
        embeddings: matriz (n, 384) L2-normalizada
        k_min: mínimo de clusters
        k_max: máximo de clusters
        random_state: semente

    Returns:
        k ótimo
    """
    k_max = min(k_max, embeddings.shape[0] - 1)
    melhor_k     = k_min
    melhor_score = -1.0

    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=5)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(
            embeddings, labels,
            metric="cosine",
            sample_size=min(2000, embeddings.shape[0]),
        )
        if score > melhor_score:
            melhor_score = score
            melhor_k = k

    print(f"  [semântico] K ótimo: {melhor_k}  (silhouette coseno = {melhor_score:.4f})")
    return melhor_k


# =============================================================================
# CLUSTERING SEMÂNTICO
# =============================================================================

def agrupar_semanticamente(df: pd.DataFrame,
                            coluna_texto: str = "texto",
                            n_clusters: Optional[int] = None,
                            chave_cache: str = "clustering",
                            random_state: int = 42) -> tuple:
    """
    Agrupa comentários usando embeddings semânticos + K-Means.

    Reconhece frases semanticamente equivalentes mesmo com vocabulário diferente:
      - "meu Pix não caiu"
      - "fiz uma transferência e o dinheiro não chegou"
      - "o valor saiu da conta mas não apareceu no destino"
      → mesmo cluster ✅

    Args:
        df: DataFrame com coluna de texto
        coluna_texto: nome da coluna de texto original (não pré-processado)
        n_clusters: número de clusters (None = automático)
        chave_cache: identificador para cache de embeddings
        random_state: semente

    Returns:
        (df_com_cluster, km, embeddings)
    """
    textos = df[coluna_texto].fillna("").tolist()

    if len(textos) < 6:
        df = df.copy()
        df["cluster_semantico"] = 0
        return df, None, None

    print("[agrupamento_semantico] Gerando embeddings...")
    embeddings = gerar_embeddings_com_cache(textos, chave_cache=chave_cache)

    if n_clusters is None:
        print("[agrupamento_semantico] Selecionando k ótimo...")
        n_clusters = escolher_k_semantico(embeddings)

    print(f"[agrupamento_semantico] K-Means com k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=10, max_iter=300)
    labels = km.fit_predict(embeddings)

    df = df.copy()
    df["cluster_semantico"] = labels
    return df, km, embeddings


# =============================================================================
# ANÁLISE DOS CLUSTERS
# =============================================================================

def texto_central_por_cluster(df: pd.DataFrame,
                               embeddings: np.ndarray,
                               km: KMeans,
                               coluna_texto: str = "texto",
                               top_n: int = 3) -> dict:
    """
    Encontra os textos mais próximos ao centróide de cada cluster.

    Estes textos são os mais "representativos" do grupo — melhores que
    palavras-chave para comunicar o que é o problema.

    Returns:
        dict {cluster_id: [textos_centrais]}
    """
    textos = df[coluna_texto].fillna("").tolist()
    centroides = km.cluster_centers_
    resultado = {}

    for cluster_id in range(km.n_clusters):
        mascara = df["cluster_semantico"] == cluster_id
        indices = np.where(mascara)[0]

        if len(indices) == 0:
            resultado[cluster_id] = []
            continue

        embs_cluster = embeddings[indices]
        centroide    = centroides[cluster_id]

        # Similaridade cosseno com o centróide
        sims = embs_cluster @ centroide
        top_indices = indices[np.argsort(sims)[::-1][:top_n]]
        resultado[cluster_id] = [textos[i] for i in top_indices]

    return resultado


def resumo_clusters_semanticos(df: pd.DataFrame,
                                embeddings: np.ndarray,
                                km: KMeans,
                                coluna_texto: str = "texto",
                                top_exemplos: int = 3) -> list:
    """
    Gera resumo legível de cada cluster semântico.

    Returns:
        Lista de dicts ordenada por tamanho (maior primeiro)
    """
    centrais = texto_central_por_cluster(df, embeddings, km,
                                          coluna_texto=coluna_texto,
                                          top_n=top_exemplos)
    resumos = []

    for cluster_id in sorted(df["cluster_semantico"].unique()):
        subset = df[df["cluster_semantico"] == cluster_id]

        categoria_principal = None
        if "categoria" in df.columns:
            moda = subset["categoria"].mode()
            categoria_principal = moda.iloc[0] if len(moda) > 0 else None

        sentimento_principal = None
        if "sentimento" in df.columns:
            moda = subset["sentimento"].mode()
            sentimento_principal = moda.iloc[0] if len(moda) > 0 else None

        # Coesão interna do cluster (média de similaridade dos membros ao centróide)
        indices  = np.where(df["cluster_semantico"] == cluster_id)[0]
        coesao   = float(np.mean(embeddings[indices] @ km.cluster_centers_[cluster_id])) if len(indices) > 0 else 0.0

        resumos.append({
            "cluster_id": int(cluster_id),
            "tamanho": len(subset),
            "categoria_predominante": categoria_principal,
            "sentimento_predominante": sentimento_principal,
            "textos_representativos": centrais.get(cluster_id, []),
            "coesao": round(coesao, 4),
        })

    return sorted(resumos, key=lambda x: x["tamanho"], reverse=True)


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho  = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")

    if not os.path.exists(caminho):
        print("Execute src/treinamento.py primeiro.")
    else:
        df = pd.read_csv(caminho, encoding="utf-8-sig")
        df_neg = df[df["sentimento"] == "negativo"].reset_index(drop=True)
        print(f"[demo] {len(df_neg)} comentários negativos para agrupar semanticamente.")

        resultado, km, embeddings = agrupar_semanticamente(
            df_neg, coluna_texto="texto", chave_cache="demo_negativos"
        )
        resumos = resumo_clusters_semanticos(resultado, embeddings, km)

        print("\n" + "="*65)
        print("  GRUPOS SEMÂNTICOS DE PROBLEMAS")
        print("="*65)
        for r in resumos:
            print(f"\n  Cluster {r['cluster_id']:>2} | {r['tamanho']:>4} avaliações"
                  f" | Categoria: {r['categoria_predominante']}"
                  f" | Coesão: {r['coesao']:.3f}")
            for ex in r["textos_representativos"]:
                print(f"    → {ex}")
