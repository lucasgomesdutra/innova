"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
app.py — Servidor Flask com front-end HTML/CSS
"""

import os
import sys
import json
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from classificacao import classificar_comentario
from analise_problemas import gerar_relatorio_problemas, rankear_problemas
from analise_sentimentos import sentimentos_por_periodo

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV      = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")
CAMINHO_METRICAS = os.path.join(BASE_DIR, "models", "metricas_treinamento.json")
STATIC_DIR       = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

# =============================================================================
# HELPERS
# =============================================================================

def carregar_df():
    if not os.path.exists(CAMINHO_CSV):
        return None
    return pd.read_csv(CAMINHO_CSV, encoding="utf-8-sig")


def carregar_metricas():
    if not os.path.exists(CAMINHO_METRICAS):
        return None
    with open(CAMINHO_METRICAS, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# ROTAS — FRONT-END
# =============================================================================

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# =============================================================================
# ROTAS — API
# =============================================================================

@app.route("/api/dashboard")
def api_dashboard():
    df = carregar_df()
    if df is None:
        return jsonify({"erro": "Base de dados não encontrada. Execute python src/treinamento.py"}), 404

    total     = len(df)
    neg       = int((df["sentimento"] == "negativo").sum())
    pos       = int((df["sentimento"] == "positivo").sum())
    neu       = int((df["sentimento"] == "neutro").sum())
    pct_neg   = round(neg / total * 100, 1) if total else 0

    # Distribuição por categoria
    cat_counts = df["categoria"].value_counts().to_dict()

    # Sentimento por categoria (para gráfico stacked)
    sent_cat = df.groupby(["categoria", "sentimento"]).size().reset_index(name="n")
    sent_cat_list = sent_cat.to_dict("records")

    # Evolução mensal
    df2 = df.copy()
    df2["data"] = pd.to_datetime(df2["data"], errors="coerce")
    df2 = df2.dropna(subset=["data"])
    df2["periodo"] = df2["data"].dt.to_period("M").astype(str)
    evo = df2.groupby(["periodo", "sentimento"]).size().reset_index(name="n")
    evo_list = evo.to_dict("records")

    # Ranking de problemas
    ranking_df = rankear_problemas(df)
    ranking = ranking_df.to_dict("records") if ranking_df is not None else []

    return jsonify({
        "total": total,
        "negativos": neg,
        "positivos": pos,
        "neutros": neu,
        "pct_negativo": pct_neg,
        "categorias": cat_counts,
        "sentimento_por_categoria": sent_cat_list,
        "evolucao_mensal": evo_list,
        "ranking_problemas": ranking,
    })


@app.route("/api/problemas")
def api_problemas():
    df = carregar_df()
    if df is None:
        return jsonify({"erro": "Base não encontrada"}), 404

    relatorio = gerar_relatorio_problemas(df)
    # Serializa só o necessário (sem DataFrames)
    return jsonify({
        "data_relatorio": relatorio["data_relatorio"],
        "total_atendimentos": relatorio["total_atendimentos"],
        "total_reclamacoes": relatorio["total_reclamacoes"],
        "percentual_negativo": relatorio["percentual_negativo"],
        "ranking_problemas": relatorio["ranking_problemas"],
        "alertas_aumento": relatorio["alertas_aumento"],
        "exemplos_por_categoria": relatorio["exemplos_por_categoria"],
        "sugestoes_investigacao": relatorio["sugestoes_investigacao"],
    })


@app.route("/api/metricas")
def api_metricas():
    m = carregar_metricas()
    if m is None:
        return jsonify({"erro": "Métricas não encontradas"}), 404
    return jsonify(m)


@app.route("/api/classificar", methods=["POST"])
def api_classificar():
    data = request.get_json(force=True)
    texto = data.get("texto", "").strip()
    if not texto:
        return jsonify({"erro": "Texto vazio"}), 400
    resultado = classificar_comentario(texto)
    return jsonify(resultado)


# =============================================================================
# INICIALIZAÇÃO
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Detector de Problemas — Sicredi")
    print("  Acesse: http://127.0.0.1:8080")
    print("=" * 55)
    app.run(debug=False, port=8080, host="127.0.0.1")
