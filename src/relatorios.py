"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
src/relatorios.py — Geração de relatórios e exportação
=============================================================================

Responsabilidades:
  - Gerar relatório em texto (.txt)
  - Gerar relatório em CSV com classificações
  - Gerar gráficos de distribuição (categorias, sentimentos, evolução temporal)
  - Salvar todos os artefatos na pasta reports/
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analise_problemas import gerar_relatorio_problemas, rankear_problemas
from analise_sentimentos import sentimentos_por_periodo

# =============================================================================
# CAMINHOS
# =============================================================================

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

# =============================================================================
# GRÁFICOS
# =============================================================================

def grafico_categorias(df: pd.DataFrame, coluna: str = "categoria") -> str:
    """Gera gráfico de barras horizontal com contagem por categoria."""
    contagem = df[coluna].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    cores = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(contagem)))
    contagem.plot(kind="barh", ax=ax, color=cores, edgecolor="white")

    ax.set_title("Distribuição por Categoria de Atendimento", fontsize=14, pad=12)
    ax.set_xlabel("Quantidade de comentários", fontsize=11)
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.bar_label(ax.containers[0], padding=4, fontsize=9)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    caminho = os.path.join(REPORTS_DIR, "grafico_categorias.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[relatorios] Gráfico salvo: {caminho}")
    return caminho


def grafico_sentimentos(df: pd.DataFrame, coluna: str = "sentimento") -> str:
    """Gera gráfico de pizza de sentimentos."""
    contagem = df[coluna].value_counts()
    cores_map = {"positivo": "#4caf50", "negativo": "#f44336", "neutro": "#ff9800"}
    cores = [cores_map.get(s, "#9e9e9e") for s in contagem.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        contagem.values,
        labels=contagem.index,
        autopct="%1.1f%%",
        colors=cores,
        startangle=90,
        pctdistance=0.80,
    )
    for at in autotexts:
        at.set_fontsize(11)

    ax.set_title("Distribuição de Sentimentos", fontsize=14, pad=14)
    plt.tight_layout()
    caminho = os.path.join(REPORTS_DIR, "grafico_sentimentos.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[relatorios] Gráfico salvo: {caminho}")
    return caminho


def grafico_evolucao_temporal(df: pd.DataFrame,
                               coluna_data: str = "data",
                               coluna_sentimento: str = "sentimento") -> str:
    """Gera gráfico de linha com evolução mensal dos sentimentos."""
    evolucao = sentimentos_por_periodo(df, coluna_data, coluna_sentimento)

    if evolucao.empty:
        return ""

    cores_map = {"positivo": "#4caf50", "negativo": "#f44336", "neutro": "#ff9800"}

    fig, ax = plt.subplots(figsize=(12, 5))

    for col in evolucao.columns:
        cor = cores_map.get(col, "#607d8b")
        ax.plot(evolucao.index, evolucao[col], marker="o", linewidth=2,
                color=cor, label=col.capitalize())

    ax.set_title("Evolução Mensal dos Sentimentos", fontsize=14, pad=12)
    ax.set_xlabel("Período", fontsize=11)
    ax.set_ylabel("Quantidade de comentários", fontsize=11)
    ax.legend(fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    caminho = os.path.join(REPORTS_DIR, "grafico_evolucao_temporal.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[relatorios] Gráfico salvo: {caminho}")
    return caminho


def grafico_ranking_problemas(df: pd.DataFrame) -> str:
    """Gera gráfico de barras com o ranking de problemas."""
    ranking = rankear_problemas(df)

    if ranking is None or ranking.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    cores = plt.cm.Reds_r(np.linspace(0.2, 0.7, len(ranking)))
    ax.barh(ranking["categoria"][::-1], ranking["total_reclamacoes"][::-1],
            color=cores[::-1], edgecolor="white")

    ax.set_title("Ranking de Problemas por Volume de Reclamações", fontsize=14, pad=12)
    ax.set_xlabel("Total de reclamações", fontsize=11)
    ax.bar_label(ax.containers[0], padding=4, fontsize=9)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    caminho = os.path.join(REPORTS_DIR, "grafico_ranking_problemas.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[relatorios] Gráfico salvo: {caminho}")
    return caminho


# =============================================================================
# RELATÓRIO TEXTO
# =============================================================================

def gerar_relatorio_txt(relatorio: dict) -> str:
    """Gera relatório em formato texto legível."""
    linhas = [
        "=" * 70,
        "  DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI",
        f"  Relatório gerado em: {relatorio['data_relatorio']}",
        "=" * 70,
        "",
        "VISÃO GERAL",
        "-" * 40,
        f"  Total de atendimentos analisados : {relatorio['total_atendimentos']}",
        f"  Comentários negativos            : {relatorio['total_reclamacoes']}  "
        f"({relatorio['percentual_negativo']:.1f}%)",
        f"  Comentários positivos            : {relatorio['total_positivos']}",
        "",
        "RANKING DE PROBLEMAS (por volume de reclamações)",
        "-" * 40,
    ]

    for r in relatorio["ranking_problemas"]:
        linhas.append(
            f"  #{r['rank']:>2}  {r['categoria']:<45} "
            f"{r['total_reclamacoes']:>4} reclamações  ({r['percentual']:.1f}%)"
        )

    linhas += [
        "",
        "ALERTAS DE AUMENTO",
        "-" * 40,
    ]

    if relatorio["alertas_aumento"]:
        for a in relatorio["alertas_aumento"]:
            linhas.append(
                f"  ⚠  {a['categoria']}: +{a['variacao_percentual']:.1f}% "
                f"em {a['periodo_atual']} (era {a['volume_anterior']}, passou para {a['volume_atual']})"
            )
    else:
        linhas.append("  Nenhum aumento significativo detectado nos últimos períodos.")

    linhas += [
        "",
        "SUGESTÕES DE INVESTIGAÇÃO",
        "-" * 40,
    ]
    for s in relatorio["sugestoes_investigacao"]:
        linhas.append(f"  → {s}")

    linhas += [
        "",
        "EXEMPLOS DE COMENTÁRIOS POR CATEGORIA",
        "-" * 40,
    ]
    for cat, exemplos in relatorio["exemplos_por_categoria"].items():
        if exemplos:
            linhas.append(f"\n  [{cat}]")
            for ex in exemplos:
                linhas.append(f"    • {ex}")

    linhas += [
        "",
        "=" * 70,
        "  AVISO: Este relatório é gerado automaticamente por um modelo de IA.",
        "  Os resultados são indicativos e devem ser revisados por profissionais.",
        "  Dados utilizados são exclusivamente fictícios para fins acadêmicos.",
        "=" * 70,
    ]

    return "\n".join(linhas)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def gerar_todos_relatorios(df: pd.DataFrame) -> dict:
    """
    Executa a geração completa de relatórios e gráficos.

    Args:
        df: DataFrame processado com colunas categoria, sentimento e data

    Returns:
        dict com caminhos dos arquivos gerados e conteúdo do relatório
    """
    print("\n[relatorios] Gerando relatório de problemas...")
    relatorio = gerar_relatorio_problemas(df)

    print("[relatorios] Gerando gráficos...")
    caminhos = {
        "grafico_categorias":      grafico_categorias(df),
        "grafico_sentimentos":     grafico_sentimentos(df),
        "grafico_evolucao":        grafico_evolucao_temporal(df),
        "grafico_ranking":         grafico_ranking_problemas(df),
    }

    # Salva texto
    texto = gerar_relatorio_txt(relatorio)
    caminho_txt = os.path.join(REPORTS_DIR, "relatorio_problemas.txt")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write(texto)
    caminhos["relatorio_txt"] = caminho_txt
    print(f"[relatorios] Relatório texto salvo: {caminho_txt}")

    # Salva JSON
    caminho_json = os.path.join(REPORTS_DIR, "relatorio_problemas.json")
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)
    caminhos["relatorio_json"] = caminho_json
    print(f"[relatorios] Relatório JSON salvo: {caminho_json}")

    return {"relatorio": relatorio, "caminhos": caminhos, "texto": texto}


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    caminho_proc = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")

    if not os.path.exists(caminho_proc):
        print("Execute src/treinamento.py primeiro.")
    else:
        df = pd.read_csv(caminho_proc, encoding="utf-8-sig")
        resultado = gerar_todos_relatorios(df)

        print("\n" + resultado["texto"])
        print(f"\nArquivos gerados:")
        for k, v in resultado["caminhos"].items():
            if v:
                print(f"  {k}: {v}")
