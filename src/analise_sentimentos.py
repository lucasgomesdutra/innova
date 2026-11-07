"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/analise_sentimentos.py — Análise de sentimentos baseada no classificador
=============================================================================

Responsabilidades:
  - Fornecer funções de análise de sentimentos reutilizáveis
  - Agregar estatísticas de sentimento por período e categoria
  - Detectar mudanças na proporção de sentimentos ao longo do tempo
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classificacao import classificar_dataframe

# =============================================================================
# ANÁLISE AGREGADA
# =============================================================================

def resumo_sentimentos(df: pd.DataFrame, coluna_sentimento: str = "sentimento") -> dict:
    """
    Gera um resumo com contagem e percentual de cada sentimento.

    Args:
        df: DataFrame com coluna de sentimento
        coluna_sentimento: nome da coluna (padrão: "sentimento")

    Returns:
        dict com contagens e percentuais por sentimento
    """
    contagem = df[coluna_sentimento].value_counts().to_dict()
    total = len(df)
    percentual = {k: round(v / total * 100, 2) for k, v in contagem.items()}

    return {
        "total": total,
        "contagem": contagem,
        "percentual": percentual,
    }


def sentimentos_por_categoria(df: pd.DataFrame,
                               coluna_categoria: str = "categoria",
                               coluna_sentimento: str = "sentimento") -> pd.DataFrame:
    """
    Tabela cruzada de sentimentos por categoria.

    Returns:
        DataFrame com percentuais de cada sentimento por categoria
    """
    tabela = pd.crosstab(
        df[coluna_categoria],
        df[coluna_sentimento],
        normalize="index",
    ).round(4) * 100

    tabela.columns = [f"% {c}" for c in tabela.columns]
    tabela["total"] = df[coluna_categoria].value_counts()

    return tabela.sort_values("total", ascending=False)


def sentimentos_por_periodo(df: pd.DataFrame,
                             coluna_data: str = "data",
                             coluna_sentimento: str = "sentimento",
                             freq: str = "ME") -> pd.DataFrame:
    """
    Evolução do sentimento ao longo do tempo, agrupado por mês.

    Args:
        df: DataFrame com colunas de data e sentimento
        coluna_data: nome da coluna de datas
        coluna_sentimento: nome da coluna de sentimento
        freq: frequência de agrupamento (padrão: 'ME' = fim de mês)

    Returns:
        DataFrame com contagem de sentimentos por período
    """
    df = df.copy()
    df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")
    df = df.dropna(subset=[coluna_data])

    df["periodo"] = df[coluna_data].dt.to_period("M")
    tabela = df.groupby(["periodo", coluna_sentimento]).size().unstack(fill_value=0)
    tabela.index = tabela.index.astype(str)

    return tabela


def detectar_aumento_negativos(df: pd.DataFrame,
                                coluna_data: str = "data",
                                coluna_sentimento: str = "sentimento",
                                limiar_variacao: float = 0.20) -> list[dict]:
    """
    Detecta períodos com aumento significativo de sentimentos negativos.

    Um período é marcado quando a proporção de negativos cresce mais de
    `limiar_variacao` em relação ao período anterior.

    Returns:
        Lista de dicts descrevendo cada período com aumento detectado
    """
    evolucao = sentimentos_por_periodo(df, coluna_data, coluna_sentimento)

    if "negativo" not in evolucao.columns:
        return []

    alertas = []
    total_por_periodo = evolucao.sum(axis=1)
    prop_neg = evolucao["negativo"] / total_por_periodo

    for i in range(1, len(prop_neg)):
        periodo_atual = prop_neg.index[i]
        periodo_ant   = prop_neg.index[i - 1]
        variacao = prop_neg.iloc[i] - prop_neg.iloc[i - 1]

        if variacao >= limiar_variacao:
            alertas.append({
                "periodo": periodo_atual,
                "periodo_anterior": periodo_ant,
                "prop_negativos_atual": round(prop_neg.iloc[i], 4),
                "prop_negativos_anterior": round(prop_neg.iloc[i - 1], 4),
                "variacao": round(variacao, 4),
                "mensagem": (
                    f"Aumento de {variacao:.1%} nos comentários negativos em {periodo_atual} "
                    f"(de {prop_neg.iloc[i-1]:.1%} para {prop_neg.iloc[i]:.1%})"
                ),
            })

    return alertas


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_proc = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")

    if not os.path.exists(caminho_proc):
        print("Arquivo processado não encontrado. Execute src/treinamento.py primeiro.")
    else:
        df = pd.read_csv(caminho_proc, encoding="utf-8-sig")

        print("RESUMO DE SENTIMENTOS (base original rotulada):")
        resumo = resumo_sentimentos(df, coluna_sentimento="sentimento")
        for sent, pct in resumo["percentual"].items():
            print(f"  {sent:<12}: {resumo['contagem'][sent]:>5} ({pct:.1f}%)")

        print("\nSENTIMENTOS POR CATEGORIA:")
        tabela = sentimentos_por_categoria(df, coluna_sentimento="sentimento")
        print(tabela.to_string())

        print("\nEVOLUÇÃO MENSAL:")
        evolucao = sentimentos_por_periodo(df, coluna_sentimento="sentimento")
        print(evolucao.to_string())
