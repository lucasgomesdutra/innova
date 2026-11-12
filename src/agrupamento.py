"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/agrupamento.py — Clustering semântico de comentários
=============================================================================

Responsabilidades:
  - Gerar representações vetoriais (embeddings) dos comentários via TF-IDF
  - Aplicar K-Means para agrupar comentários semelhantes
  - Identificar os termos mais representativos de cada grupo
  - Calcular similaridade entre comentários
  - Detectar comentários similares ao longo do tempo (agrupamento temporal)

Nota técnica:
  Utiliza TF-IDF + K-Means em vez de sentence-transformers por padrão,
  pois é mais leve e não exige download de modelos grandes.
  O módulo aceita embeddings externos como argumento se disponíveis.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocessar_serie

# =============================================================================
# VETORIZAÇÃO
# =============================================================================

def vetorizar_textos(textos: list[str] | pd.Series,
                     max_features: int = 10000,
                     ngram_range: tuple = (1, 2)) -> tuple:
    """
    Transforma textos em matriz TF-IDF L2-normalizada.

    Returns:
        (matriz_normalizada, vectorizer)
    """
    if isinstance(textos, pd.Series):
        textos = textos.tolist()

    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )
    X = vec.fit_transform(textos)
    X_norm = normalize(X, norm="l2")
    return X_norm, vec


# =============================================================================
# SELEÇÃO AUTOMÁTICA DE K
# =============================================================================

def escolher_k_otimo(X, k_min: int = 3, k_max: int = 15, random_state: int = 42) -> int:
    """
    Usa o Silhouette Score para escolher automaticamente o número de clusters.

    Avalia valores de k de k_min a k_max e retorna o k com maior score.
    Limitado para não demorar demais em bases grandes.

    Returns:
        int: k ótimo encontrado
    """
    k_max = min(k_max, X.shape[0] - 1)  # k não pode ser maior que n_amostras-1
    melhor_k = k_min
    melhor_score = -1.0

    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=5)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="cosine", sample_size=min(2000, X.shape[0]))
        if score > melhor_score:
            melhor_score = score
            melhor_k = k

    print(f"  K ótimo encontrado: {melhor_k}  (silhouette={melhor_score:.4f})")
    return melhor_k


# =============================================================================
# CLUSTERING
# =============================================================================

def agrupar_comentarios(df: pd.DataFrame,
                        coluna_texto_limpo: str = "texto_limpo",
                        n_clusters: int | None = None,
                        random_state: int = 42) -> pd.DataFrame:
    """
    Agrupa comentários semelhantes usando TF-IDF + K-Means.

    Args:
        df: DataFrame com coluna de textos pré-processados
        coluna_texto_limpo: nome da coluna com texto limpo
        n_clusters: número de grupos (None = automático via silhouette)
        random_state: semente aleatória

    Returns:
        DataFrame com coluna `cluster_id` adicionada
    """
    textos = df[coluna_texto_limpo].fillna("").tolist()

    if len(textos) < 6:
        df = df.copy()
        df["cluster_id"] = 0
        return df

    print("[agrupamento] Vetorizando textos...")
    X, vec = vetorizar_textos(textos)

    if n_clusters is None:
        print("[agrupamento] Selecionando k ótimo...")
        n_clusters = escolher_k_otimo(X)

    print(f"[agrupamento] Aplicando K-Means com k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    df = df.copy()
    df["cluster_id"] = labels
    return df, km, vec


# =============================================================================
# ANÁLISE DOS CLUSTERS
# =============================================================================

def termos_por_cluster(km: KMeans, vec: TfidfVectorizer, top_n: int = 8) -> dict:
    """
    Retorna os termos mais representativos de cada cluster.

    Usa os centróides do K-Means para identificar quais termos
    têm maior peso médio em cada grupo.

    Returns:
        dict {cluster_id: [termos]}
    """
    feature_names = np.array(vec.get_feature_names_out())
    centros = km.cluster_centers_
    resultado = {}

    for i, centro in enumerate(centros):
        indices = np.argsort(centro)[::-1][:top_n]
        resultado[i] = feature_names[indices].tolist()

    return resultado


def resumo_clusters(df: pd.DataFrame,
                    km: KMeans,
                    vec: TfidfVectorizer,
                    coluna_texto_original: str = "texto",
                    top_exemplos: int = 3) -> list[dict]:
    """
    Gera um resumo legível de cada cluster.

    Returns:
        Lista de dicts com id, tamanho, termos principais e exemplos de comentários
    """
    termos = termos_por_cluster(km, vec)
    resumos = []

    for cluster_id in sorted(df["cluster_id"].unique()):
        subset = df[df["cluster_id"] == cluster_id]
        exemplos = subset[coluna_texto_original].dropna().head(top_exemplos).tolist()

        # Categoria mais frequente no cluster (se disponível)
        categoria_principal = None
        if "categoria" in df.columns:
            categoria_principal = subset["categoria"].mode().iloc[0] if len(subset) > 0 else None

        resumos.append({
            "cluster_id": int(cluster_id),
            "tamanho": len(subset),
            "termos_principais": termos.get(cluster_id, []),
            "categoria_predominante": categoria_principal,
            "exemplos": exemplos,
        })

    return sorted(resumos, key=lambda x: x["tamanho"], reverse=True)


# =============================================================================
# BRIDGE SEMÂNTICO — delega para agrupamento_semantico.py quando disponível
# =============================================================================

def agrupar_com_fallback(df: pd.DataFrame,
                          coluna_texto: str = "texto",
                          n_clusters: int | None = None,
                          usar_semantico: bool = True,
                          random_state: int = 42):
    """
    Agrupa comentários usando embeddings semânticos quando disponível,
    com fallback automático para TF-IDF caso sentence-transformers não esteja instalado.

    Args:
        df: DataFrame com coluna de texto
        coluna_texto: nome da coluna de texto
        n_clusters: número de clusters (None = automático)
        usar_semantico: se True, tenta usar sentence-transformers
        random_state: semente

    Returns:
        (df_com_cluster, km, objeto_vetorizador)
        - coluna adicionada: 'cluster_semantico' (semântico) ou 'cluster_id' (TF-IDF)
    """
    if usar_semantico:
        try:
            from agrupamento_semantico import agrupar_semanticamente
            return agrupar_semanticamente(
                df,
                coluna_texto=coluna_texto,
                n_clusters=n_clusters,
                chave_cache="dashboard",
                random_state=random_state,
            )
        except ImportError:
            print("[agrupamento] sentence-transformers indisponível, usando TF-IDF.")

    # Fallback: TF-IDF original
    coluna_limpa = "texto_limpo"
    if coluna_limpa not in df.columns:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from preprocessing import preprocessar_serie
        df = df.copy()
        df[coluna_limpa] = preprocessar_serie(df[coluna_texto])

    return agrupar_comentarios(df, coluna_texto_limpo=coluna_limpa,
                                n_clusters=n_clusters, random_state=random_state)


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")

    if not os.path.exists(caminho):
        print("Execute src/treinamento.py primeiro.")
    else:
        df = pd.read_csv(caminho, encoding="utf-8-sig")

        # Filtra apenas negativos para clustering de problemas
        df_neg = df[df["sentimento"] == "negativo"].reset_index(drop=True)
        print(f"[agrupamento] {len(df_neg)} comentários negativos para agrupar.")

        resultado, km, vec = agrupar_comentarios(df_neg, n_clusters=None)
        resumos = resumo_clusters(resultado, km, vec)

        print("\nRESUMO DOS GRUPOS DE PROBLEMAS:")
        for r in resumos:
            print(f"\n  Cluster {r['cluster_id']:>2} | {r['tamanho']:>4} comentários"
                  f" | Categoria: {r['categoria_predominante']}")
            print(f"  Termos   : {', '.join(r['termos_principais'])}")
            for ex in r["exemplos"]:
                print(f"    → {ex}")
