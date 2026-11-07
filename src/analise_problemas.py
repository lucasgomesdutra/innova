"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/analise_problemas.py — Identificação de problemas recorrentes
=============================================================================

Responsabilidades:
  - Identificar categorias com volume acima do esperado (outliers de frequência)
  - Detectar aumentos súbitos de reclamações por categoria e período
  - Rankear problemas por frequência, crescimento e impacto
  - Sugerir pontos que merecem investigação (sem afirmar causa-raiz)
  - Manter rastreabilidade: cada problema aponta para os comentários originais
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# IDENTIFICAÇÃO DE PROBLEMAS RECORRENTES
# =============================================================================

CATEGORIAS_PROBLEMA = {
    "Problemas no aplicativo",
    "Problemas com Pix e transferências",
    "Cobranças indevidas",
    "Demora no atendimento",
    "Problemas de acesso à conta",
    "Problemas com cartões",
    "Dificuldades com serviços bancários",
}


def rankear_problemas(df: pd.DataFrame,
                      coluna_categoria: str = "categoria",
                      coluna_sentimento: str = "sentimento") -> pd.DataFrame:
    """
    Classifica as categorias de problema por frequência de comentários negativos.

    Returns:
        DataFrame ordenado por contagem decrescente com percentuais
    """
    df_neg = df[
        (df[coluna_sentimento] == "negativo") &
        (df[coluna_categoria].isin(CATEGORIAS_PROBLEMA))
    ].copy()

    contagem = df_neg[coluna_categoria].value_counts().reset_index()
    contagem.columns = ["categoria", "total_reclamacoes"]
    contagem["percentual"] = (contagem["total_reclamacoes"] / len(df_neg) * 100).round(2)
    contagem["rank"] = range(1, len(contagem) + 1)

    return contagem


def detectar_aumento_por_categoria(df: pd.DataFrame,
                                    coluna_categoria: str = "categoria",
                                    coluna_data: str = "data",
                                    coluna_sentimento: str = "sentimento",
                                    limiar_variacao: float = 0.25) -> list[dict]:
    """
    Detecta categorias com aumento significativo de reclamações mês a mês.

    Compara o volume de reclamações de cada categoria nos últimos dois períodos.
    Retorna alertas quando a variação percentual supera `limiar_variacao`.

    Returns:
        Lista de dicts com alertas de aumento
    """
    df = df.copy()
    df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")
    df = df.dropna(subset=[coluna_data])
    df["periodo"] = df[coluna_data].dt.to_period("M")

    df_neg = df[
        (df[coluna_sentimento] == "negativo") &
        (df[coluna_categoria].isin(CATEGORIAS_PROBLEMA))
    ]

    pivot = df_neg.groupby(["periodo", coluna_categoria]).size().unstack(fill_value=0)
    periodos = pivot.index.tolist()

    if len(periodos) < 2:
        return []

    alertas = []
    p_atual = periodos[-1]
    p_ant   = periodos[-2]

    for cat in pivot.columns:
        vol_atual = pivot.loc[p_atual, cat]
        vol_ant   = pivot.loc[p_ant, cat]

        if vol_ant == 0:
            continue

        variacao = (vol_atual - vol_ant) / vol_ant

        if variacao >= limiar_variacao:
            alertas.append({
                "categoria": cat,
                "periodo_atual": str(p_atual),
                "periodo_anterior": str(p_ant),
                "volume_atual": int(vol_atual),
                "volume_anterior": int(vol_ant),
                "variacao_percentual": round(variacao * 100, 2),
                "sugestao": (
                    f"Aumento de {variacao:.1%} em '{cat}' no período {p_atual}. "
                    "Recomenda-se investigar se há mudanças recentes relacionadas a este tema."
                ),
            })

    return sorted(alertas, key=lambda x: x["variacao_percentual"], reverse=True)


def obter_exemplos_por_categoria(df: pd.DataFrame,
                                  categoria: str,
                                  coluna_categoria: str = "categoria",
                                  coluna_texto: str = "texto",
                                  coluna_sentimento: str = "sentimento",
                                  top_n: int = 5) -> list[str]:
    """
    Retorna exemplos de comentários negativos de uma categoria específica.

    Mantém rastreabilidade entre o problema identificado e os relatos originais.
    """
    subset = df[
        (df[coluna_categoria] == categoria) &
        (df[coluna_sentimento] == "negativo")
    ][coluna_texto].dropna()

    # Deduplica para não repetir o mesmo texto
    unicos = subset.drop_duplicates()
    return unicos.head(top_n).tolist()


def gerar_relatorio_problemas(df: pd.DataFrame) -> dict:
    """
    Gera um relatório completo de problemas identificados.

    Returns:
        dict com:
          - data_relatorio: timestamp
          - total_atendimentos: int
          - total_reclamacoes: int
          - total_positivos: int
          - percentual_negativo: float
          - ranking_problemas: list
          - alertas_aumento: list
          - exemplos_por_categoria: dict {categoria: [exemplos]}
          - sugestoes_investigacao: list[str]
    """
    total = len(df)
    total_neg = int((df["sentimento"] == "negativo").sum())
    total_pos = int((df["sentimento"] == "positivo").sum())
    pct_neg   = round(total_neg / total * 100, 2) if total > 0 else 0.0

    ranking = rankear_problemas(df)
    alertas = detectar_aumento_por_categoria(df)

    exemplos = {}
    for cat in CATEGORIAS_PROBLEMA:
        exemplos[cat] = obter_exemplos_por_categoria(df, cat, top_n=5)

    # Sugestões de investigação baseadas nos dados
    sugestoes = []

    if ranking is not None and len(ranking) > 0:
        top_cat = ranking.iloc[0]["categoria"]
        top_pct = ranking.iloc[0]["percentual"]
        sugestoes.append(
            f"'{top_cat}' representa {top_pct:.1f}% das reclamações. "
            "Sugere-se revisar os processos relacionados a esta categoria."
        )

    if pct_neg > 60:
        sugestoes.append(
            f"{pct_neg:.1f}% dos comentários são negativos. "
            "Sugere-se uma revisão ampla da experiência do cliente."
        )

    for alerta in alertas[:3]:
        sugestoes.append(alerta["sugestao"])

    if not sugestoes:
        sugestoes.append("Nenhum ponto crítico identificado com os dados disponíveis.")

    return {
        "data_relatorio": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_atendimentos": total,
        "total_reclamacoes": total_neg,
        "total_positivos": total_pos,
        "percentual_negativo": pct_neg,
        "ranking_problemas": ranking.to_dict("records") if ranking is not None else [],
        "alertas_aumento": alertas,
        "exemplos_por_categoria": exemplos,
        "sugestoes_investigacao": sugestoes,
    }


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
        relatorio = gerar_relatorio_problemas(df)

        print("=" * 60)
        print("  RELATÓRIO DE PROBLEMAS IDENTIFICADOS")
        print("=" * 60)
        print(f"  Data         : {relatorio['data_relatorio']}")
        print(f"  Total        : {relatorio['total_atendimentos']}")
        print(f"  Reclamações  : {relatorio['total_reclamacoes']}  ({relatorio['percentual_negativo']}%)")
        print(f"  Positivos    : {relatorio['total_positivos']}")

        print("\n  RANKING DE PROBLEMAS:")
        for r in relatorio["ranking_problemas"]:
            print(f"    #{r['rank']} {r['categoria']:<45} {r['total_reclamacoes']:>4}  ({r['percentual']:.1f}%)")

        print("\n  SUGESTÕES DE INVESTIGAÇÃO:")
        for s in relatorio["sugestoes_investigacao"]:
            print(f"    → {s}")
