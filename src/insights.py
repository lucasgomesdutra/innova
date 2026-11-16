"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/insights.py — Camada de geração de insights para o dashboard
=============================================================================

Responsabilidades:
  - Combinar resultados de classificação, clustering semântico e análise temporal
  - Transformar dados brutos em textos de insight acionáveis
  - Identificar padrões não óbvios (ex: mesmo problema com vocabulário diferente)
  - Calcular score de urgência por cluster/categoria
  - Fornecer função única que o dashboard pode chamar
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analise_problemas import rankear_problemas, detectar_aumento_por_categoria
from analise_sentimentos import sentimentos_por_periodo


# =============================================================================
# FREQUÊNCIA E EVOLUÇÃO DE RECLAMAÇÕES
# =============================================================================

def frequencia_por_categoria_e_periodo(df: pd.DataFrame,
                                        coluna_categoria: str = "categoria",
                                        coluna_data: str = "data",
                                        coluna_sentimento: str = "sentimento") -> pd.DataFrame:
    """
    Contagem de reclamações negativas por categoria e mês.

    Returns:
        DataFrame pivot: linhas = meses, colunas = categorias, valores = contagem
    """
    df = df.copy()
    df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")
    df = df.dropna(subset=[coluna_data])
    df["periodo"] = df[coluna_data].dt.to_period("M")

    df_neg = df[df[coluna_sentimento] == "negativo"]
    pivot = (
        df_neg.groupby(["periodo", coluna_categoria])
        .size()
        .unstack(fill_value=0)
    )
    pivot.index = pivot.index.astype(str)
    return pivot


def variacao_mensal_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a variação percentual mês a mês por categoria de problema.

    Returns:
        DataFrame com variações percentuais (NaN no primeiro período)
    """
    freq = frequencia_por_categoria_e_periodo(df)
    if freq.empty or len(freq) < 2:
        return pd.DataFrame()
    return freq.pct_change() * 100


# =============================================================================
# SCORE DE URGÊNCIA
# =============================================================================

def calcular_urgencia(df: pd.DataFrame,
                       resumos_clusters: Optional[list] = None) -> list:
    """
    Calcula um score de urgência (0-100) por categoria de problema.

    Combina:
      - Volume absoluto de reclamações (50% do score)
      - Tendência de crescimento recente (30%)
      - Proporção negativo/total da categoria (20%)

    Args:
        df: DataFrame com todas as avaliações
        resumos_clusters: resultado de resumo_clusters_semanticos (opcional)

    Returns:
        Lista de dicts com categoria, score_urgencia e componentes
    """
    if df.empty:
        return []

    ranking = rankear_problemas(df)
    if ranking is None or ranking.empty:
        return []

    total_neg = ranking["total_reclamacoes"].sum()
    alertas   = detectar_aumento_por_categoria(df)
    alertas_dict = {a["categoria"]: a["variacao_percentual"] for a in alertas}

    resultados = []
    vol_max = ranking["total_reclamacoes"].max() if total_neg > 0 else 1

    for _, row in ranking.iterrows():
        cat = row["categoria"]
        vol = row["total_reclamacoes"]

        # Componente 1: volume relativo (0-50)
        score_vol = (vol / vol_max) * 50

        # Componente 2: crescimento (0-30)
        var = alertas_dict.get(cat, 0)
        score_cresc = min(var / 100 * 30, 30) if var > 0 else 0

        # Componente 3: proporção negativo/total da categoria
        total_cat = len(df[df["categoria"] == cat])
        neg_cat   = len(df[(df["categoria"] == cat) & (df["sentimento"] == "negativo")])
        prop_neg  = neg_cat / total_cat if total_cat > 0 else 0
        score_prop = prop_neg * 20

        score_total = round(score_vol + score_cresc + score_prop, 1)

        resultados.append({
            "categoria": cat,
            "score_urgencia": min(score_total, 100),
            "volume_reclamacoes": int(vol),
            "variacao_mensal": round(var, 1),
            "proporcao_negativo": round(prop_neg * 100, 1),
        })

    return sorted(resultados, key=lambda x: x["score_urgencia"], reverse=True)


# =============================================================================
# INSIGHTS DE CLUSTERS SEMÂNTICOS
# =============================================================================

def insights_de_clusters(resumos_clusters: list) -> list:
    """
    Transforma resumos de clusters semânticos em textos de insight.

    Gera frases prontas para exibir no dashboard, descrevendo o que
    cada grupo de reclamações tem em comum.

    Args:
        resumos_clusters: saída de agrupamento_semantico.resumo_clusters_semanticos()

    Returns:
        Lista de dicts com insight textual por cluster
    """
    insights = []

    for r in resumos_clusters:
        if r["tamanho"] < 2:
            continue

        cat   = r.get("categoria_predominante") or "Não classificada"
        sent  = r.get("sentimento_predominante") or "indefinido"
        tamanho = r["tamanho"]
        coesao  = r.get("coesao", 0)
        exemplos = r.get("textos_representativos", [])

        # Grau de coesão em linguagem natural
        if coesao >= 0.80:
            descricao_coesao = "muito específico"
        elif coesao >= 0.65:
            descricao_coesao = "bem definido"
        else:
            descricao_coesao = "variado"

        # Monta o texto de insight
        texto = (
            f"Grupo de {tamanho} avaliações sobre '{cat}' "
            f"({sent}), tema {descricao_coesao}."
        )
        if exemplos:
            texto += f" Exemplo representativo: \"{exemplos[0][:120]}\""

        insights.append({
            "cluster_id": r["cluster_id"],
            "insight": texto,
            "categoria": cat,
            "sentimento": sent,
            "tamanho": tamanho,
            "coesao": coesao,
            "exemplos": exemplos,
        })

    return insights


# =============================================================================
# FUNÇÃO PRINCIPAL — chamada pelo dashboard
# =============================================================================

def gerar_insights_completos(df: pd.DataFrame,
                              resumos_clusters: Optional[list] = None) -> dict:
    """
    Gera todos os insights disponíveis a partir dos dados.

    Esta é a função que o app.py deve chamar. Combina:
      - Ranking de urgência por categoria
      - Insights semânticos dos clusters (se disponíveis)
      - Evolução temporal de reclamações
      - Alertas de aumento

    Args:
        df: DataFrame completo de avaliações
        resumos_clusters: resultado de resumo_clusters_semanticos (ou None)

    Returns:
        dict com todos os componentes de insight prontos para o dashboard
    """
    if df.empty:
        return {
            "urgencia": [],
            "insights_clusters": [],
            "alertas_aumento": [],
            "evolucao_mensal": pd.DataFrame(),
            "total_reclamacoes": 0,
            "percentual_negativo": 0.0,
            "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    total = len(df)
    total_neg = int((df["sentimento"] == "negativo").sum())
    pct_neg   = round(total_neg / total * 100, 1) if total > 0 else 0.0

    urgencia         = calcular_urgencia(df, resumos_clusters)
    alertas          = detectar_aumento_por_categoria(df)
    evolucao         = sentimentos_por_periodo(df)
    insights_clusters = insights_de_clusters(resumos_clusters) if resumos_clusters else []

    # Insights textuais gerais (sempre disponíveis, sem depender de clusters)
    alertas_texto = []
    if urgencia:
        top = urgencia[0]
        alertas_texto.append(
            f"⚠️ '{top['categoria']}' é a categoria mais crítica "
            f"(score {top['score_urgencia']:.0f}/100, "
            f"{top['volume_reclamacoes']} reclamações)."
        )
    for a in alertas[:2]:
        alertas_texto.append(
            f"📈 '{a['categoria']}' cresceu {a['variacao_percentual']:.1f}% "
            f"em {a['periodo_atual']} vs período anterior."
        )

    return {
        "urgencia": urgencia,
        "insights_clusters": insights_clusters,
        "alertas_aumento": alertas,
        "alertas_texto": alertas_texto,
        "evolucao_mensal": evolucao,
        "total_reclamacoes": total_neg,
        "percentual_negativo": pct_neg,
        "gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


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
        resultado = gerar_insights_completos(df)

        print("="*60)
        print("  INSIGHTS GERADOS")
        print("="*60)
        print(f"  Total de reclamações : {resultado['total_reclamacoes']}")
        print(f"  % Negativos          : {resultado['percentual_negativo']:.1f}%")

        print("\n  RANKING DE URGÊNCIA:")
        for u in resultado["urgencia"]:
            print(f"    [{u['score_urgencia']:>5.1f}] {u['categoria']}")

        print("\n  ALERTAS:")
        for a in resultado["alertas_texto"]:
            print(f"    {a}")
