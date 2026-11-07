"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
app.py — Painel de visualização com Streamlit
=============================================================================

Execução:
    streamlit run app.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Garante que src/ está no path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from classificacao import classificar_comentario
from analise_problemas import gerar_relatorio_problemas, rankear_problemas
from analise_sentimentos import (
    resumo_sentimentos,
    sentimentos_por_periodo,
    sentimentos_por_categoria,
)
from relatorios import gerar_todos_relatorios

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Detector de Problemas — Sicredi",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")
CAMINHO_METRICAS = os.path.join(BASE_DIR, "models", "metricas_treinamento.json")

# =============================================================================
# CORES
# =============================================================================

COR_NEGATIVO = "#ef4444"
COR_POSITIVO = "#22c55e"
COR_NEUTRO   = "#f59e0b"
COR_PRIMARIA = "#1e40af"

# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================

@st.cache_data(show_spinner=False)
def carregar_dados_cache():
    if not os.path.exists(CAMINHO_CSV):
        return None
    return pd.read_csv(CAMINHO_CSV, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def carregar_metricas_cache():
    if not os.path.exists(CAMINHO_METRICAS):
        return None
    with open(CAMINHO_METRICAS, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Sicredi_logo.svg/320px-Sicredi_logo.svg.png",
        width=160,
        use_container_width=False,
    )
    st.markdown("## 🏦 Sicredi — IA de Atendimento")
    st.markdown("---")
    st.markdown(
        """
        **Projeto Acadêmico**

        Detector de Problemas no Atendimento ao Cliente.

        Utiliza modelos de Machine Learning para:
        - Classificar categorias
        - Analisar sentimentos
        - Identificar problemas recorrentes

        ---
        ⚠️ **Dados fictícios** para fins acadêmicos.
        """
    )

    pagina = st.radio(
        "Navegação",
        [
            "📊 Dashboard",
            "🔍 Analisar Comentário",
            "📈 Evolução Temporal",
            "🚨 Problemas Detectados",
            "🤖 Métricas do Modelo",
        ],
    )

# =============================================================================
# VERIFICAÇÃO DE PRÉ-REQUISITOS
# =============================================================================

df = carregar_dados_cache()
metricas = carregar_metricas_cache()

if df is None:
    st.error(
        "⚠️ **Base de dados não encontrada.**\n\n"
        "Execute os seguintes comandos no terminal:\n"
        "```powershell\n"
        "python data/raw/gerar_base.py\n"
        "python src/treinamento.py\n"
        "```"
    )
    st.stop()

# =============================================================================
# PÁGINA: DASHBOARD PRINCIPAL
# =============================================================================

if pagina == "📊 Dashboard":
    st.title("📊 Dashboard — Detector de Problemas no Atendimento")
    st.caption("Análise de atendimentos e pesquisas de satisfação — Sicredi")
    st.markdown("---")

    # Métricas principais
    total = len(df)
    total_neg = int((df["sentimento"] == "negativo").sum())
    total_pos = int((df["sentimento"] == "positivo").sum())
    total_neu = int((df["sentimento"] == "neutro").sum())
    pct_neg = total_neg / total * 100 if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Total de Atendimentos", f"{total:,}")
    col2.metric("😠 Reclamações", f"{total_neg:,}", f"{pct_neg:.1f}%")
    col3.metric("😊 Avaliações Positivas", f"{total_pos:,}")
    col4.metric("😐 Neutros", f"{total_neu:,}")

    st.markdown("---")

    col_esq, col_dir = st.columns(2)

    # Gráfico de categorias
    with col_esq:
        st.subheader("📂 Comentários por Categoria")
        contagem_cat = df["categoria"].value_counts().reset_index()
        contagem_cat.columns = ["Categoria", "Quantidade"]
        fig_cat = px.bar(
            contagem_cat,
            x="Quantidade",
            y="Categoria",
            orientation="h",
            color="Quantidade",
            color_continuous_scale="RdYlGn_r",
            text="Quantidade",
        )
        fig_cat.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=400,
            yaxis={"categoryorder": "total ascending"},
        )
        fig_cat.update_traces(textposition="outside")
        st.plotly_chart(fig_cat, use_container_width=True)

    # Gráfico de sentimentos
    with col_dir:
        st.subheader("💬 Distribuição de Sentimentos")
        cont_sent = df["sentimento"].value_counts().reset_index()
        cont_sent.columns = ["Sentimento", "Quantidade"]
        cores_sent = {"negativo": COR_NEGATIVO, "positivo": COR_POSITIVO, "neutro": COR_NEUTRO}
        fig_sent = px.pie(
            cont_sent,
            values="Quantidade",
            names="Sentimento",
            color="Sentimento",
            color_discrete_map=cores_sent,
            hole=0.4,
        )
        fig_sent.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400)
        st.plotly_chart(fig_sent, use_container_width=True)

    # Top problemas
    st.markdown("---")
    st.subheader("🚨 Principais Motivos de Reclamação")

    ranking = rankear_problemas(df)
    if ranking is not None and not ranking.empty:
        fig_rank = px.bar(
            ranking,
            x="total_reclamacoes",
            y="categoria",
            orientation="h",
            color="total_reclamacoes",
            color_continuous_scale="Reds",
            text="total_reclamacoes",
            labels={"total_reclamacoes": "Reclamações", "categoria": "Categoria"},
        )
        fig_rank.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
            yaxis={"categoryorder": "total ascending"},
        )
        fig_rank.update_traces(textposition="outside")
        st.plotly_chart(fig_rank, use_container_width=True)
    else:
        st.info("Nenhum dado de reclamações disponível.")


# =============================================================================
# PÁGINA: ANALISAR COMENTÁRIO
# =============================================================================

elif pagina == "🔍 Analisar Comentário":
    st.title("🔍 Analisar Novo Comentário")
    st.caption("Insira um texto e veja como a IA classifica o atendimento.")
    st.markdown("---")

    texto_input = st.text_area(
        "💬 Digite o comentário do cliente:",
        placeholder="Ex: Não consigo entrar no aplicativo desde ontem...",
        height=120,
    )

    if st.button("🤖 Analisar", type="primary", use_container_width=True):
        if not texto_input.strip():
            st.warning("Por favor, insira um comentário para analisar.")
        else:
            with st.spinner("Analisando..."):
                resultado = classificar_comentario(texto_input)

            st.markdown("---")

            # Exibe resultado
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 📂 Categoria Identificada")
                st.info(f"**{resultado['categoria']}**")
                conf_cat = resultado["confianca_categoria"]
                st.progress(conf_cat, text=f"Confiança: {conf_cat:.1%}")

            with col2:
                st.markdown("#### 💬 Sentimento Detectado")
                sent = resultado["sentimento"]
                if sent == "negativo":
                    st.error(f"**{sent.capitalize()}** 😠")
                elif sent == "positivo":
                    st.success(f"**{sent.capitalize()}** 😊")
                else:
                    st.warning(f"**{sent.capitalize()}** 😐")
                conf_sent = resultado["confianca_sentimento"]
                st.progress(conf_sent, text=f"Confiança: {conf_sent:.1%}")

            with col3:
                st.markdown("#### ⚠️ Revisão Necessária?")
                if resultado["requer_revisao"]:
                    st.warning("**Sim** — Baixa confiança.\nRecomenda-se revisão humana.")
                else:
                    st.success("**Não** — Classificação com boa confiança.")

            st.markdown("---")

            # Probabilidades por categoria
            with st.expander("🔬 Ver probabilidades detalhadas (categorias)"):
                probs_cat = resultado["prob_categorias"]
                df_probs = pd.DataFrame(
                    list(probs_cat.items()), columns=["Categoria", "Probabilidade"]
                ).sort_values("Probabilidade", ascending=True)
                fig_probs = px.bar(
                    df_probs, x="Probabilidade", y="Categoria", orientation="h",
                    color="Probabilidade", color_continuous_scale="Blues",
                    text=df_probs["Probabilidade"].apply(lambda x: f"{x:.2%}"),
                )
                fig_probs.update_layout(height=400, coloraxis_showscale=False,
                                        margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_probs, use_container_width=True)

            with st.expander("🔬 Ver probabilidades detalhadas (sentimentos)"):
                probs_sent = resultado["prob_sentimentos"]
                df_ps = pd.DataFrame(
                    list(probs_sent.items()), columns=["Sentimento", "Probabilidade"]
                )
                fig_ps = px.bar(
                    df_ps, x="Sentimento", y="Probabilidade",
                    color="Sentimento",
                    color_discrete_map={"negativo": COR_NEGATIVO, "positivo": COR_POSITIVO, "neutro": COR_NEUTRO},
                    text=df_ps["Probabilidade"].apply(lambda x: f"{x:.2%}"),
                )
                fig_ps.update_layout(showlegend=False, height=300,
                                     margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_ps, use_container_width=True)

            # Texto processado
            with st.expander("🔤 Ver texto pré-processado pela IA"):
                st.code(resultado["texto_limpo"], language=None)


# =============================================================================
# PÁGINA: EVOLUÇÃO TEMPORAL
# =============================================================================

elif pagina == "📈 Evolução Temporal":
    st.title("📈 Evolução Temporal dos Atendimentos")
    st.caption("Acompanhe como os sentimentos e categorias evoluíram ao longo do tempo.")
    st.markdown("---")

    if "data" not in df.columns:
        st.warning("Coluna 'data' não encontrada no dataset.")
    else:
        df_temp = df.copy()
        df_temp["data"] = pd.to_datetime(df_temp["data"], errors="coerce")
        df_temp = df_temp.dropna(subset=["data"])
        df_temp["periodo"] = df_temp["data"].dt.to_period("M").astype(str)

        # Evolução de sentimentos
        st.subheader("💬 Evolução Mensal de Sentimentos")
        evolucao = df_temp.groupby(["periodo", "sentimento"]).size().unstack(fill_value=0).reset_index()
        evolucao_long = evolucao.melt(id_vars="periodo", var_name="sentimento", value_name="quantidade")

        cores_sent = {"negativo": COR_NEGATIVO, "positivo": COR_POSITIVO, "neutro": COR_NEUTRO}
        fig_evo = px.line(
            evolucao_long, x="periodo", y="quantidade", color="sentimento",
            color_discrete_map=cores_sent,
            markers=True,
            labels={"periodo": "Período", "quantidade": "Comentários", "sentimento": "Sentimento"},
        )
        fig_evo.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_evo, use_container_width=True)

        # Evolução de categorias de problema
        st.subheader("🚨 Evolução das Principais Reclamações por Mês")

        CATS_PROBLEMA = [
            "Problemas no aplicativo",
            "Problemas com Pix e transferências",
            "Cobranças indevidas",
            "Demora no atendimento",
            "Problemas de acesso à conta",
            "Problemas com cartões",
        ]
        cats_disponiveis = [c for c in CATS_PROBLEMA if c in df_temp["categoria"].unique()]
        cats_sel = st.multiselect("Selecione as categorias:", cats_disponiveis, default=cats_disponiveis[:3])

        if cats_sel:
            df_neg_temp = df_temp[
                (df_temp["sentimento"] == "negativo") &
                (df_temp["categoria"].isin(cats_sel))
            ]
            evo_cat = df_neg_temp.groupby(["periodo", "categoria"]).size().reset_index(name="quantidade")
            fig_cat_evo = px.line(
                evo_cat, x="periodo", y="quantidade", color="categoria",
                markers=True,
                labels={"periodo": "Período", "quantidade": "Reclamações", "categoria": "Categoria"},
            )
            fig_cat_evo.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cat_evo, use_container_width=True)


# =============================================================================
# PÁGINA: PROBLEMAS DETECTADOS
# =============================================================================

elif pagina == "🚨 Problemas Detectados":
    st.title("🚨 Problemas Identificados pela IA")
    st.caption("Análise automática de reclamações recorrentes com exemplos dos comentários originais.")
    st.markdown("---")

    with st.spinner("Gerando análise de problemas..."):
        relatorio = gerar_relatorio_problemas(df)

    # Sugestões de investigação
    st.subheader("🔎 Sugestões de Investigação")
    for s in relatorio["sugestoes_investigacao"]:
        st.warning(f"→ {s}")

    st.markdown("---")

    # Ranking
    st.subheader("📊 Ranking de Problemas")
    df_rank = pd.DataFrame(relatorio["ranking_problemas"])
    if not df_rank.empty:
        st.dataframe(
            df_rank[["rank", "categoria", "total_reclamacoes", "percentual"]].rename(columns={
                "rank": "#",
                "categoria": "Categoria",
                "total_reclamacoes": "Reclamações",
                "percentual": "% do Total",
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")

    # Alertas de aumento
    st.subheader("⚡ Alertas de Aumento de Reclamações")
    if relatorio["alertas_aumento"]:
        for a in relatorio["alertas_aumento"]:
            with st.expander(f"⚠️ {a['categoria']} — +{a['variacao_percentual']:.1f}% em {a['periodo_atual']}"):
                st.write(f"**Período anterior ({a['periodo_anterior']}):** {a['volume_anterior']} reclamações")
                st.write(f"**Período atual ({a['periodo_atual']}):** {a['volume_atual']} reclamações")
                st.info(a["sugestao"])
    else:
        st.success("Nenhum aumento significativo detectado nos últimos períodos.")

    st.markdown("---")

    # Exemplos por categoria
    st.subheader("💬 Exemplos de Comentários por Categoria")
    for cat, exemplos in relatorio["exemplos_por_categoria"].items():
        if exemplos:
            with st.expander(f"📂 {cat} ({len(exemplos)} exemplos)"):
                for ex in exemplos:
                    st.markdown(f"• {ex}")

    st.markdown("---")
    st.caption(
        "⚠️ Os resultados acima são gerados automaticamente por modelos de IA. "
        "São indicativos e devem ser revisados por profissionais antes de qualquer decisão."
    )


# =============================================================================
# PÁGINA: MÉTRICAS DO MODELO
# =============================================================================

elif pagina == "🤖 Métricas do Modelo":
    st.title("🤖 Métricas de Desempenho dos Modelos")
    st.caption("Resultados reais do treinamento e avaliação dos modelos de IA.")
    st.markdown("---")

    if metricas is None:
        st.warning(
            "Métricas não encontradas. Execute `python src/treinamento.py` para treinar os modelos."
        )
    else:
        tab1, tab2 = st.tabs(["📂 Categorias", "💬 Sentimentos"])

        for tab, chave, nome in [
            (tab1, "categorias", "Classificador de Categorias"),
            (tab2, "sentimentos", "Classificador de Sentimentos"),
        ]:
            with tab:
                st.subheader(f"📊 {nome}")
                m = metricas[chave]

                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Accuracy",    f"{m['accuracy']:.2%}")
                c2.metric("📏 F1 Macro",    f"{m['f1_macro']:.2%}")
                c3.metric("⚖️ F1 Weighted", f"{m['f1_weighted']:.2%}")

                st.markdown("---")
                st.subheader("📋 Relatório Detalhado por Classe")

                report_data = []
                for classe, vals in m["report"].items():
                    if isinstance(vals, dict) and "precision" in vals:
                        report_data.append({
                            "Classe": classe,
                            "Precision": f"{vals['precision']:.2%}",
                            "Recall":    f"{vals['recall']:.2%}",
                            "F1-Score":  f"{vals['f1-score']:.2%}",
                            "Suporte":   int(vals["support"]),
                        })

                if report_data:
                    st.dataframe(pd.DataFrame(report_data), hide_index=True, use_container_width=True)

                # Imagens das matrizes de confusão
                nomes_img = {
                    "categorias": "matriz_confusao_categorias.png",
                    "sentimentos": "matriz_confusao_sentimentos.png",
                }
                caminho_img = os.path.join(BASE_DIR, "reports", nomes_img[chave])
                if os.path.exists(caminho_img):
                    st.markdown("---")
                    st.subheader("🔲 Matriz de Confusão")
                    st.image(caminho_img, use_container_width=True)

    st.markdown("---")
    st.info(
        "**Sobre os modelos:**\n\n"
        "- **Classificador de Categorias:** TF-IDF (bigramas, 15.000 features) + "
        "Regressão Logística (C=5.0, class_weight='balanced')\n"
        "- **Classificador de Sentimentos:** TF-IDF (bigramas) + "
        "Regressão Logística (C=3.0)\n"
        "- Treinados com dados sintéticos. A qualidade dos resultados depende "
        "diretamente do volume e diversidade dos dados de treinamento."
    )
