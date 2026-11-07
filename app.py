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
from datetime import datetime

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
CAMINHO_AVALIACOES  = os.path.join(BASE_DIR, "data", "processed", "avaliacoes_reais.csv")
CAMINHO_FICTICIOS   = os.path.join(BASE_DIR, "data", "processed", "atendimentos_processados.csv")
CAMINHO_METRICAS    = os.path.join(BASE_DIR, "models", "metricas_treinamento.json")

COLUNAS_AVALIACOES = ["id", "data", "texto", "categoria", "sentimento",
                      "confianca_categoria", "confianca_sentimento", "requer_revisao"]

# =============================================================================
# CORES
# =============================================================================

COR_NEGATIVO = "#ef4444"
COR_POSITIVO = "#22c55e"
COR_NEUTRO   = "#f59e0b"
COR_PRIMARIA = "#1e40af"

# =============================================================================
# PERSISTÊNCIA DE AVALIAÇÕES REAIS
# =============================================================================

def _garantir_arquivo():
    """Cria o CSV de avaliações se ainda não existir."""
    os.makedirs(os.path.dirname(CAMINHO_AVALIACOES), exist_ok=True)
    if not os.path.exists(CAMINHO_AVALIACOES):
        pd.DataFrame(columns=COLUNAS_AVALIACOES).to_csv(
            CAMINHO_AVALIACOES, index=False, encoding="utf-8-sig"
        )


def salvar_avaliacao(resultado: dict):
    """Adiciona uma nova avaliação classificada ao CSV de avaliações reais."""
    _garantir_arquivo()
    df_existente = pd.read_csv(CAMINHO_AVALIACOES, encoding="utf-8-sig")
    novo_id = int(df_existente["id"].max() + 1) if not df_existente.empty else 1

    nova_linha = pd.DataFrame([{
        "id": novo_id,
        "data": datetime.now().strftime("%Y-%m-%d"),
        "texto": resultado["texto_original"],
        "categoria": resultado["categoria"],
        "sentimento": resultado["sentimento"],
        "confianca_categoria": resultado["confianca_categoria"],
        "confianca_sentimento": resultado["confianca_sentimento"],
        "requer_revisao": resultado["requer_revisao"],
    }])

    df_atualizado = pd.concat([df_existente, nova_linha], ignore_index=True)
    df_atualizado.to_csv(CAMINHO_AVALIACOES, index=False, encoding="utf-8-sig")
    # Invalida cache para o dashboard refletir a nova entrada
    carregar_avaliacoes.clear()


@st.cache_data(show_spinner=False)
def carregar_avaliacoes():
    """
    Combina avaliações reais (inseridas pelo usuário) com dados fictícios de base.
    Retorna None se não houver nenhum dado.
    """
    _garantir_arquivo()
    df_reais = pd.read_csv(CAMINHO_AVALIACOES, encoding="utf-8-sig")

    partes = []

    # Dados fictícios como base histórica (se existirem)
    if os.path.exists(CAMINHO_FICTICIOS):
        df_fict = pd.read_csv(CAMINHO_FICTICIOS, encoding="utf-8-sig")[
            ["texto", "categoria", "sentimento", "data"]
        ].copy()
        df_fict["id"]                  = range(-len(df_fict), 0)   # IDs negativos = fictícios
        df_fict["confianca_categoria"]  = 1.0
        df_fict["confianca_sentimento"] = 1.0
        df_fict["requer_revisao"]       = False
        partes.append(df_fict)

    # Avaliações reais inseridas pelo usuário
    if not df_reais.empty:
        partes.append(df_reais)

    if not partes:
        return None

    df_combined = pd.concat(partes, ignore_index=True)
    return df_combined


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
    st.markdown("### Detector de Problemas no Atendimento")
    st.caption("Análise inteligente de atendimentos e pesquisas de satisfação.")
    st.markdown("---")

    st.markdown(
        """
        **Capacidades do sistema**

        - Classificação automática de categorias
        - Análise de sentimentos do cliente
        - Identificação de problemas recorrentes
        - Monitoramento de tendências ao longo do tempo
        """
    )

    st.markdown("---")

    pagina = st.radio(
        "Navegação",
        [
            "📊 Dashboard",
            "🔍 Analisar Comentário",
            "📈 Evolução Temporal",
            "🚨 Problemas Detectados",
            "🤖 Métricas do Modelo",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Contador na sidebar — só conta avaliações reais
    _garantir_arquivo()
    df_side_reais = pd.read_csv(CAMINHO_AVALIACOES, encoding="utf-8-sig")
    n_reais = len(df_side_reais)
    tem_ficticios = os.path.exists(CAMINHO_FICTICIOS)
    if n_reais > 0:
        st.caption(f"📋 {n_reais} avaliação(ões) real(is) registrada(s)")
    else:
        st.caption("📋 Nenhuma avaliação real ainda.")
    if tem_ficticios:
        st.caption("📦 Base fictícia carregada como histórico.")


# =============================================================================
# CARREGA DADOS PARA AS PÁGINAS
# =============================================================================

df      = carregar_avaliacoes()
metricas = carregar_metricas_cache()

# =============================================================================
# COMPONENTE: ESTADO VAZIO
# =============================================================================

def _mostrar_estado_vazio():
    st.info(
        "**Nenhum dado disponível ainda.**\n\n"
        "Execute `python src/treinamento.py` para gerar a base fictícia de histórico, "
        "ou vá até **🔍 Analisar Comentário** e insira avaliações reais."
    )


# =============================================================================
# PÁGINA: DASHBOARD PRINCIPAL
# =============================================================================

if pagina == "📊 Dashboard":
    st.title("📊 Dashboard — Detector de Problemas no Atendimento")
    st.caption("Resumo das avaliações analisadas pela IA.")
    st.markdown("---")

    if df is None:
        _mostrar_estado_vazio()
    else:
        total     = len(df)
        total_neg = int((df["sentimento"] == "negativo").sum())
        total_pos = int((df["sentimento"] == "positivo").sum())
        total_neu = int((df["sentimento"] == "neutro").sum())
        pct_neg   = total_neg / total * 100 if total > 0 else 0
        pct_pos   = total_pos / total * 100 if total > 0 else 0
        indice_satisfacao = round(pct_pos - pct_neg, 1)

        cores_sent = {"negativo": COR_NEGATIVO, "positivo": COR_POSITIVO, "neutro": COR_NEUTRO}

        def _layout_base(altura=340):
            return dict(
                showlegend=False,
                margin=dict(l=10, r=80, t=30, b=10),
                height=altura,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(size=13),
                xaxis=dict(visible=False, fixedrange=True),
                yaxis=dict(title="", tickfont=dict(size=12), fixedrange=True),
            )

        # --- Métricas principais ---
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📋 Total", f"{total:,}")
        col2.metric("😠 Negativas", f"{total_neg:,}", f"{pct_neg:.1f}%")
        col3.metric("😊 Positivas", f"{total_pos:,}", f"{pct_pos:.1f}%")
        col4.metric("😐 Neutras", f"{total_neu:,}")
        col5.metric("⭐ Satisfação", f"{indice_satisfacao:+.1f}%",
                    help="Positivas (%) menos Negativas (%)")

        st.markdown("---")

        # --- Linha 1: Sentimentos | Categorias ---
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown("#### 💬 Sentimentos")
            cont_sent = (
                df["sentimento"]
                .value_counts()
                .reindex(["negativo", "neutro", "positivo"])
                .dropna()
                .reset_index()
            )
            cont_sent.columns = ["Sentimento", "Qtd"]
            total_sent = cont_sent["Qtd"].sum()
            cont_sent["pct"] = (cont_sent["Qtd"] / total_sent * 100).round(1)
            cont_sent["label"] = cont_sent.apply(
                lambda r: f"{int(r['Qtd'])} ({r['pct']:.0f}%)", axis=1
            )

            fig_sent = px.bar(
                cont_sent,
                x="Qtd", y="Sentimento",
                orientation="h",
                color="Sentimento",
                color_discrete_map=cores_sent,
                text="label",
            )
            layout = _layout_base(altura=220)
            layout["yaxis"]["categoryorder"] = "array"
            layout["yaxis"]["categoryarray"] = ["negativo", "neutro", "positivo"]
            layout["bargap"] = 0.35
            layout["margin"]["r"] = 20
            fig_sent.update_layout(**layout)
            fig_sent.update_traces(
                textposition="inside",
                textfont=dict(size=12, color="white"),
                insidetextanchor="middle",
                marker_line_width=0,
            )
            st.plotly_chart(fig_sent, use_container_width=True, config={"displayModeBar": False})

        with col_b:
            st.markdown("#### 📂 Avaliações por Categoria")
            contagem_cat = df["categoria"].value_counts().reset_index()
            contagem_cat.columns = ["Categoria", "Qtd"]

            def _cor_cat(cat):
                sub = df[df["categoria"] == cat]["sentimento"].value_counts()
                dom = sub.idxmax() if not sub.empty else "neutro"
                return cores_sent.get(dom, COR_NEUTRO)

            contagem_cat["cor"] = contagem_cat["Categoria"].apply(_cor_cat)
            contagem_cat = contagem_cat.sort_values("Qtd", ascending=True)

            fig_cat = px.bar(
                contagem_cat,
                x="Qtd", y="Categoria",
                orientation="h",
                color="cor",
                color_discrete_map={c: c for c in contagem_cat["cor"].unique()},
                text="Qtd",
            )
            n_cats = len(contagem_cat)
            altura_cat = max(260, n_cats * 52)
            layout_cat = _layout_base(altura=altura_cat)
            layout_cat["bargap"] = 0.3
            fig_cat.update_layout(**layout_cat)
            fig_cat.update_traces(
                textposition="outside",
                textfont=dict(size=13),
                marker_line_width=0,
            )
            st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")

        # --- Linha 2: Reclamações | Últimas avaliações ---
        col_c, col_d = st.columns([1, 1])

        with col_c:
            st.markdown("#### 🚨 Principais Reclamações")
            ranking = rankear_problemas(df)
            if ranking is not None and not ranking.empty:
                ranking_plot = ranking.sort_values("total_reclamacoes", ascending=True)
                fig_rank = px.bar(
                    ranking_plot,
                    x="total_reclamacoes", y="categoria",
                    orientation="h",
                    color="total_reclamacoes",
                    color_continuous_scale=[[0, "#fca5a5"], [1, "#ef4444"]],
                    text="total_reclamacoes",
                    labels={"total_reclamacoes": "", "categoria": ""},
                )
                n_rank = len(ranking_plot)
                altura_rank = max(260, n_rank * 52)
                layout_rank = _layout_base(altura=altura_rank)
                layout_rank["coloraxis_showscale"] = False
                layout_rank["bargap"] = 0.3
                fig_rank.update_layout(**layout_rank)
                fig_rank.update_traces(
                    textposition="outside",
                    textfont=dict(size=13),
                    marker_line_width=0,
                )
                st.plotly_chart(fig_rank, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Nenhuma reclamação registrada ainda.")

        with col_d:
            st.markdown("#### 🕓 Últimas Avaliações")
            df_recentes = df.sort_values("id", ascending=False).head(8)[
                ["data", "texto", "sentimento"]
            ].rename(columns={
                "data": "Data",
                "texto": "Comentário",
                "sentimento": "Sentimento",
            })
            st.dataframe(df_recentes, hide_index=True, use_container_width=True, height=320)


# =============================================================================
# PÁGINA: ANALISAR COMENTÁRIO
# =============================================================================

elif pagina == "🔍 Analisar Comentário":
    st.title("🔍 Analisar Novo Comentário")
    st.caption("Insira um texto e veja como a IA classifica o atendimento. O resultado é salvo automaticamente no dashboard.")
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
                salvar_avaliacao(resultado)

            st.success("✅ Avaliação registrada e dashboard atualizado.")
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
    st.caption("Histórico combinado de dados fictícios e avaliações reais inseridas.")
    st.markdown("---")

    if df is None:
        _mostrar_estado_vazio()
    elif "data" not in df.columns:
        st.warning("Coluna 'data' não encontrada no dataset.")
    else:
        df_temp = df.copy()
        df_temp["data"] = pd.to_datetime(df_temp["data"], errors="coerce")
        df_temp = df_temp.dropna(subset=["data"])
        df_temp["periodo"] = df_temp["data"].dt.to_period("M").astype(str)
        df_temp["origem"]  = df_temp["id"].apply(lambda x: "Real" if x >= 0 else "Fictício")

        periodos_unicos = df_temp["periodo"].nunique()
        cores_sent = {"negativo": COR_NEGATIVO, "positivo": COR_POSITIVO, "neutro": COR_NEUTRO}

        # Filtro de origem
        origens_disp = df_temp["origem"].unique().tolist()
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            origens_sel = st.multiselect(
                "Exibir dados de:", origens_disp, default=origens_disp,
                help="Filtre entre avaliações reais e a base fictícia de histórico."
            )
        df_temp = df_temp[df_temp["origem"].isin(origens_sel)] if origens_sel else df_temp

        if periodos_unicos < 2:
            st.info(
                "Os dados estão concentrados em poucos períodos. "
                "O gráfico ficará mais informativo conforme avaliações reais forem adicionadas em meses diferentes."
            )

        st.markdown("---")

        # --- Gráfico 1: Evolução de sentimentos ---
        st.subheader("💬 Evolução Mensal de Sentimentos")

        # Filtra apenas períodos que realmente têm dados (evita o "buraco" e queda a zero)
        evolucao = (
            df_temp.groupby(["periodo", "sentimento"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        # Remove períodos onde todos os sentimentos são zero
        cols_sent = [c for c in evolucao.columns if c != "periodo"]
        evolucao = evolucao[evolucao[cols_sent].sum(axis=1) > 0]

        evolucao_long = evolucao.melt(id_vars="periodo", var_name="sentimento", value_name="quantidade")
        # Remove linhas com quantidade zero para não conectar com o eixo
        evolucao_long = evolucao_long[evolucao_long["quantidade"] > 0]

        fig_evo = px.line(
            evolucao_long, x="periodo", y="quantidade", color="sentimento",
            color_discrete_map=cores_sent,
            markers=True,
            labels={"periodo": "Período", "quantidade": "Avaliações", "sentimento": "Sentimento"},
        )
        fig_evo.update_layout(
            height=360,
            margin=dict(l=10, r=20, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            xaxis=dict(tickangle=-30, type="category"),
        )
        fig_evo.update_traces(line=dict(width=2.5), marker=dict(size=7),
                              connectgaps=False)
        st.plotly_chart(fig_evo, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")

        # --- Gráfico 2: Proporção de negativos por mês (barra empilhada) ---
        st.subheader("📊 Composição de Sentimentos por Mês")
        comp = (
            df_temp.groupby(["periodo", "sentimento"])
            .size()
            .reset_index(name="qtd")
        )
        fig_comp = px.bar(
            comp, x="periodo", y="qtd", color="sentimento",
            color_discrete_map=cores_sent,
            barmode="stack",
            labels={"periodo": "Período", "qtd": "Avaliações", "sentimento": "Sentimento"},
            text="qtd",
        )
        fig_comp.update_layout(
            height=320,
            margin=dict(l=10, r=20, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
            xaxis=dict(tickangle=-30),
        )
        fig_comp.update_traces(textposition="inside", textfont_size=11)
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")

        # --- Gráfico 3: Evolução de categorias de problema ---
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

        if not cats_disponiveis:
            st.info("Nenhuma reclamação nas categorias monitoradas para o filtro selecionado.")
        else:
            cats_sel = st.multiselect(
                "Selecione as categorias:", cats_disponiveis, default=cats_disponiveis[:4]
            )
            if cats_sel:
                df_neg_temp = df_temp[
                    (df_temp["sentimento"] == "negativo") &
                    (df_temp["categoria"].isin(cats_sel))
                ]
                evo_cat = (
                    df_neg_temp.groupby(["periodo", "categoria"])
                    .size()
                    .reset_index(name="quantidade")
                )
                fig_cat_evo = px.line(
                    evo_cat, x="periodo", y="quantidade", color="categoria",
                    markers=True,
                    labels={"periodo": "Período", "quantidade": "Reclamações", "categoria": "Categoria"},
                )
                fig_cat_evo.update_layout(
                    height=360,
                    margin=dict(l=10, r=20, t=20, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=-0.25),
                    xaxis=dict(tickangle=-30),
                )
                fig_cat_evo.update_traces(line=dict(width=2.5), marker=dict(size=7))
                st.plotly_chart(fig_cat_evo, use_container_width=True, config={"displayModeBar": False})


# =============================================================================
# PÁGINA: PROBLEMAS DETECTADOS
# =============================================================================

elif pagina == "🚨 Problemas Detectados":
    st.title("🚨 Problemas Identificados pela IA")
    st.caption("Análise automática de reclamações recorrentes com exemplos dos comentários originais.")
    st.markdown("---")

    if df is None:
        _mostrar_estado_vazio()
    else:
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
        else:
            st.info("Nenhuma reclamação registrada ainda para gerar o ranking.")

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
            st.success("Nenhum aumento significativo detectado nos períodos disponíveis.")

        st.markdown("---")

        # Exemplos por categoria
        st.subheader("💬 Exemplos de Comentários por Categoria")
        exemplos_com_dados = {
            cat: exs for cat, exs in relatorio["exemplos_por_categoria"].items() if exs
        }
        if not exemplos_com_dados:
            st.info("Nenhum comentário negativo registrado ainda.")
        else:
            for cat, exemplos in exemplos_com_dados.items():
                with st.expander(f"📂 {cat} ({len(exemplos)} exemplo(s))"):
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
